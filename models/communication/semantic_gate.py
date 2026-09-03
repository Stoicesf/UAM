"""Semantic gate: send L1/L2/L3 when gain > cost (ToA upgrade)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def semantic_gain(prior_entropy: torch.Tensor, posterior_entropy: torch.Tensor) -> torch.Tensor:
    """Gain = H(prior) - H(posterior | semantic)."""
    return (prior_entropy - posterior_entropy).clamp(min=0)


class SemanticGate(nn.Module):
    """MLP: [gain, snr, bandwidth_norm, u] → send? + level logits.

    Level selection defaults to gain-binned rules (trainable MLP optional via
    ``use_mlp_levels=True``). Low-bandwidth bias is soft (+0.3 L1), not a hard L1 lock.
    """

    def __init__(self, hidden: int = 32, n_roles: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.send_head = nn.Linear(hidden, 1)
        self.level_head = nn.Linear(hidden, 3)  # L1 L2 L3
        # role → level bias (n_roles, 3); trained in 2.2, zero by default
        self.role_level_bias = nn.Parameter(torch.zeros(n_roles, 3))

    def forward(
        self,
        gain: torch.Tensor,
        snr_norm: torch.Tensor,
        bw_norm: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns send_prob (...,), level_logits (...,3)."""
        x = torch.stack([gain, snr_norm, bw_norm, uncertainty], dim=-1)
        h = self.net(x)
        send = torch.sigmoid(self.send_head(h)).squeeze(-1)
        levels = self.level_head(h)
        return send, levels

    @staticmethod
    def level_from_gain(gain: torch.Tensor) -> torch.Tensor:
        """Rule levels: low gain→L1, mid→L2, high→L3."""
        level = torch.ones_like(gain, dtype=torch.long)
        level = torch.where(gain >= 0.4, torch.full_like(level, 2), level)
        level = torch.where(gain >= 0.8, torch.full_like(level, 3), level)
        return level

    @torch.no_grad()
    def decide(
        self,
        gain: torch.Tensor,
        snr_db: torch.Tensor,
        bandwidth_hz: float,
        bandwidth_max: float,
        uncertainty: torch.Tensor,
        send_thresh: float = 0.5,
        *,
        roles: torch.Tensor | None = None,
        use_mlp_levels: bool = False,
        soft_bw_bias: float = 0.3,
        trace: list[dict[str, Any]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        snr_n = ((snr_db - 0.0) / 20.0).clamp(0, 1)
        bw_ratio = bandwidth_hz / max(bandwidth_max, 1.0)
        bw_n = torch.full_like(gain, bw_ratio)
        send, level_logits = self.forward(gain, snr_n, bw_n, uncertainty)
        is_hard_biased = bw_ratio < 0.5
        # soft low-bandwidth bias (was +1.0/-1.0 → near-hard L1 lock)
        if is_hard_biased and soft_bw_bias != 0.0:
            level_logits = level_logits.clone()
            level_logits[..., 0] = level_logits[..., 0] + soft_bw_bias
            level_logits[..., 2] = level_logits[..., 2] - soft_bw_bias
        if roles is not None:
            bias = self.role_level_bias[roles.clamp(0, self.role_level_bias.shape[0] - 1)]
            level_logits = level_logits + bias

        do_send = send >= send_thresh
        cost = 0.2 * (1.0 - bw_n) + 0.1
        do_send = do_send | (gain > cost)

        if use_mlp_levels:
            level = level_logits.argmax(dim=-1) + 1
        else:
            level = self.level_from_gain(gain)
            # still allow role bias to bump level when strong
            if roles is not None:
                bump = bias.argmax(dim=-1) + 1
                strong = bias.max(dim=-1).values > 0.2
                level = torch.where(strong, bump, level)

        level = torch.where(do_send, level, torch.zeros_like(level))

        if trace is not None:
            g = gain.detach().cpu()
            c = cost.detach().cpu()
            sp = send.detach().cpu()
            lv = level.detach().cpu()
            lg = level_logits.detach().cpu()
            for i in range(g.numel()):
                trace.append(
                    {
                        "gain": float(g.view(-1)[i]),
                        "cost": float(c.view(-1)[i]),
                        "bandwidth_ratio": float(bw_ratio),
                        "send_prob": float(sp.view(-1)[i]),
                        "selected_level": int(lv.view(-1)[i]),
                        "is_hard_biased": bool(is_hard_biased),
                        "level_logits": [float(x) for x in lg.view(-1, 3)[i].tolist()],
                    }
                )
        return do_send, level


def self_check() -> None:
    g = SemanticGate()
    gain = torch.tensor([0.8, 0.1])
    snr = torch.tensor([20.0, 5.0])
    u = torch.tensor([0.7, 0.2])
    tr: list[dict[str, Any]] = []
    send, lvl = g.decide(gain, snr, 5e6, 20e6, u, trace=tr)
    assert send.shape == (2,)
    assert len(tr) == 2
    assert tr[0]["is_hard_biased"] is True
    # gain 0.8 → L3 under rule levels when sending
    assert int(lvl[0]) in (0, 3)
    print(f"semantic_gate: OK (send={send.tolist()}, lvl={lvl.tolist()})")


if __name__ == "__main__":
    self_check()

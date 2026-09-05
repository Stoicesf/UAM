"""High-level role selector (slow timescale), conditioned on UAV type."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HighLevelRoleSelector(nn.Module):
    def __init__(
        self,
        global_obs_dim: int,
        n_roles: int = 3,
        n_types: int = 3,
        decision_interval: int = 50,
        type_embed_dim: int = 8,
    ):
        super().__init__()
        self.n_roles = n_roles
        self.decision_interval = decision_interval
        self.type_embed = nn.Embedding(n_types, type_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim + type_embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_roles),
        )
        self.register_buffer("_step", torch.tensor(-10**9, dtype=torch.long))
        # last per-agent logits [n_agents, n_roles]; not a buffer (size varies)
        self._logits: torch.Tensor | None = None

    def update_logits(
        self,
        global_obs: torch.Tensor,
        uav_types: list[int] | torch.Tensor,
        step: int,
        force: bool = False,
    ) -> torch.Tensor:
        if self._logits is None or force or int(step - int(self._step)) >= self.decision_interval:
            if isinstance(uav_types, torch.Tensor):
                types = uav_types.long().cpu()
            else:
                types = torch.tensor(list(uav_types), dtype=torch.long)
            n = types.shape[0]
            g = global_obs.detach().float().view(1, -1).expand(n, -1)
            x = torch.cat([g, self.type_embed(types)], dim=-1)
            self._logits = self.net(x)
            self._step.fill_(int(step))
        assert self._logits is not None
        return F.softmax(self._logits, dim=-1)

    def get_roles(self, n_agents: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample per-agent roles. Returns (indices [n], log_prob sum)."""
        if self._logits is None or self._logits.shape[0] != n_agents:
            raise RuntimeError("call update_logits(...) before get_roles")
        dist = torch.distributions.Categorical(logits=self._logits)
        indices = dist.sample()
        log_prob = dist.log_prob(indices).sum()
        return indices, log_prob

    @property
    def mean_logits(self) -> torch.Tensor:
        """Mean logits over agents (for entropy logging)."""
        if self._logits is None:
            return torch.zeros(self.n_roles)
        return self._logits.detach().mean(dim=0)


def self_check() -> None:
    m = HighLevelRoleSelector(6, n_roles=3, decision_interval=2)
    g = torch.randn(6)
    types = [0, 1, 2, 2]
    m.update_logits(g, types, step=0, force=True)
    roles, lp = m.get_roles(4)
    assert roles.shape == (4,) and lp.ndim == 0
    print("high_level: OK")


if __name__ == "__main__":
    self_check()

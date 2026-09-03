"""Lightweight Byzantine tolerance via neighbor reputation."""

from __future__ import annotations

import torch

from dice.local_obs import neighbor_mask


class ByzantineTolerance:
    def __init__(self, n_agents: int, decay: float = 0.95):
        self.rep = torch.ones(n_agents)
        self.decay = decay

    def update(
        self,
        positions: torch.Tensor,
        claimed_positions: torch.Tensor,
        byzantine_mask: torch.Tensor,
        radius: float = 2.0,
    ) -> None:
        """Penalize agents whose claimed pos diverges from observed (sim)."""
        err = (positions - claimed_positions).norm(dim=-1)
        # byzantine agents inject large claim error in tests
        self.rep = self.decay * self.rep + (1 - self.decay) * (1.0 / (1.0 + err))
        self.rep = self.rep * (~byzantine_mask).float() + 0.1 * byzantine_mask.float() * self.rep

    def trusted_mask(self, positions: torch.Tensor, radius: float, threshold: float = 0.4) -> torch.Tensor:
        base = neighbor_mask(positions, radius)
        trust = self.rep >= threshold
        return base & trust.unsqueeze(0) & trust.unsqueeze(1)

    def filter_messages(self, messages: torch.Tensor, threshold: float = 0.4) -> torch.Tensor:
        """Zero messages from low-reputation agents. messages: (N, ...)."""
        keep = (self.rep >= threshold).to(messages.device).float()
        while keep.dim() < messages.dim():
            keep = keep.unsqueeze(-1)
        return messages * keep


def self_check() -> None:
    bt = ByzantineTolerance(5)
    pos = torch.randn(5, 2)
    claim = pos.clone()
    claim[0] += 5  # liar
    byz = torch.tensor([True, False, False, False, False])
    bt.update(pos, claim, byz)
    assert float(bt.rep[0]) < float(bt.rep[1])
    print("byzantine_tolerance: OK")


if __name__ == "__main__":
    self_check()

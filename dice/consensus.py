"""Lightweight distributed consensus (majority / max-bid)."""

from __future__ import annotations

import torch

from dice.local_obs import neighbor_mask


def consensus_max_bid(
    bids: torch.Tensor,
    positions: torch.Tensor,
    radius: float,
    alive: torch.Tensor | None = None,
    iters: int = 3,
) -> torch.Tensor:
    """Propagate max bid over the comm graph. bids: (N,) values."""
    v = bids.clone()
    for _ in range(iters):
        mask = neighbor_mask(positions, radius, alive)
        # include self
        mask = mask | torch.eye(positions.shape[0], device=positions.device, dtype=torch.bool)
        # max over neighbors
        nv = v.clone()
        for i in range(positions.shape[0]):
            nv[i] = v[mask[i]].max()
        v = nv
    return v


def self_check() -> None:
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    bids = torch.tensor([1.0, 5.0, 2.0])
    out = consensus_max_bid(bids, pos, radius=1.2, iters=3)
    assert float(out[0]) >= 4.9  # 5 propagates
    print("consensus: OK")


if __name__ == "__main__":
    self_check()

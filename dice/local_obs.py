"""Neighbor discovery and local observation helpers."""

from __future__ import annotations

import torch


def neighbor_mask(pos: torch.Tensor, radius: float, alive: torch.Tensor | None = None) -> torch.Tensor:
    dist = torch.cdist(pos, pos)
    mask = (dist < radius) & (dist > 0)
    if alive is not None:
        mask = mask & alive.unsqueeze(0) & alive.unsqueeze(1)
    return mask


def gather_neighbor_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of neighbor features; x: (N,D), mask: (N,N)."""
    deg = mask.float().sum(dim=-1, keepdim=True).clamp(min=1)
    return (mask.float() @ x) / deg


def self_check() -> None:
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])
    m = neighbor_mask(pos, 2.0)
    assert m[0, 1] and not m[0, 2]
    print("local_obs: OK")


if __name__ == "__main__":
    self_check()

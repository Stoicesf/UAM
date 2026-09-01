"""Communication cost C = sum g_ij (directed, no self-loops)."""

from __future__ import annotations

import torch


def communication_cost(g: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    Args:
        g: (B, N, N) or (N, N) gates
        reduction: 'mean' | 'sum' | 'none'
    """
    if g.dim() == 2:
        g = g.unsqueeze(0)
    per_env = g.sum(dim=(-2, -1))
    if reduction == "none":
        return per_env
    if reduction == "sum":
        return per_env.sum()
    return per_env.mean()


def edge_count(g: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Hard count of active directed edges after thresholding."""
    if g.dim() == 2:
        g = g.unsqueeze(0)
    return (g >= threshold).float().sum(dim=(-2, -1)).mean()

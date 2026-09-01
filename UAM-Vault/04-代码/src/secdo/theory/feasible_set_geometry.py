"""Geometry of sum-budget feasible sets."""

from __future__ import annotations

import torch


def delta_capacity(c_true: torch.Tensor, c_hat: torch.Tensor) -> torch.Tensor:
    return (c_true - c_hat).abs()


def hausdorff_bound_sum_budget(delta: torch.Tensor, lipschitz_k: float = 1.0) -> torch.Tensor:
    return lipschitz_k * delta

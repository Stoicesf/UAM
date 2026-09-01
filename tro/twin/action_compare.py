"""Action discrepancy: Δa = ||a* - a|| on shared state."""

from __future__ import annotations

import torch


def action_l2(a_full: torch.Tensor, a_sparse: torch.Tensor) -> torch.Tensor:
    diff = a_full - a_sparse
    return diff.reshape(diff.shape[0], -1).norm(dim=-1)


def action_mean_l2(a_full: torch.Tensor, a_sparse: torch.Tensor) -> float:
    return float(action_l2(a_full, a_sparse).mean().item())


def actions_from_residual_actor(
    actor: torch.nn.Module,
    obs: torch.Tensor,
    phi_full: torch.Tensor,
    phi_sparse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean actions (loc) under ResidualGuidanceActor on shared obs."""
    loc_f, _ = actor(obs, phi_full)
    loc_s, _ = actor(obs, phi_sparse)
    return loc_f, loc_s

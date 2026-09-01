"""One-step / multi-step value influence for counterfactual ΔV."""

from __future__ import annotations

from typing import Callable

import torch


def next_obs_under_action(
    obs: torch.Tensor,
    action: torch.Tensor,
    dt: float = 1.0,
) -> torch.Tensor:
    """Lightweight next-obs proxy (pos / vel / goal_rel only)."""
    next_obs = obs.clone()
    a_xy = action[..., :2]
    next_obs[..., 0:2] = obs[..., 0:2] + a_xy * dt
    if obs.shape[-1] >= 4:
        next_obs[..., 2:4] = a_xy
    if obs.shape[-1] >= 6:
        next_obs[..., 4:6] = obs[..., 4:6] - a_xy * dt
    return next_obs


def multi_step_delta_value(
    obs: torch.Tensor,
    action_with: torch.Tensor,
    action_without: torch.Tensor,
    value_fn: Callable[[torch.Tensor], torch.Tensor],
    horizon: int = 10,
    gamma: float = 0.95,
    dt: float = 1.0,
) -> torch.Tensor:
    """U_V = Σ_{k=1}^H γ^k (V(ŝ_{t+k}^m) − V(ŝ_{t+k}^0))."""
    obs_m = obs
    obs_0 = obs
    total = torch.zeros(obs.shape[:-1], device=obs.device, dtype=obs.dtype)
    for k in range(1, horizon + 1):
        obs_m = next_obs_under_action(obs_m, action_with, dt=dt)
        obs_0 = next_obs_under_action(obs_0, action_without, dt=dt)
        total = total + (gamma**k) * (value_fn(obs_m) - value_fn(obs_0))
    return total

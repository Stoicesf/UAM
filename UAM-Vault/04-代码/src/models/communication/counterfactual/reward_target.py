"""Instantaneous navigation reward proxy for counterfactual ΔR.

Uses VMAS-style obs layout: [pos(2), vel(2), goal_rel(2), ...].
"""

from __future__ import annotations

import torch


def nav_progress_reward(
    obs: torch.Tensor,
    action: torch.Tensor,
) -> torch.Tensor:
    """Approximate one-step goal progress: ‖g‖ − ‖g − a_xy‖."""
    if obs.shape[-1] >= 6:
        goal_rel = obs[..., 4:6]
    else:
        goal_rel = obs[..., -2:]
    a_xy = action[..., :2]
    dist0 = goal_rel.norm(dim=-1)
    dist1 = (goal_rel - a_xy).norm(dim=-1)
    return dist0 - dist1


def delta_reward_raw(
    obs: torch.Tensor,
    action_with: torch.Tensor,
    action_without: torch.Tensor,
) -> torch.Tensor:
    """Signed ΔR = R(s, a^m) − R(s, a^0) (no clamp)."""
    return nav_progress_reward(obs, action_with) - nav_progress_reward(
        obs, action_without
    )


def multi_step_delta_reward(
    obs: torch.Tensor,
    action_with: torch.Tensor,
    action_without: torch.Tensor,
    horizon: int = 10,
    gamma: float = 0.95,
    dt: float = 1.0,
) -> torch.Tensor:
    """Accumulate discounted progress under repeated a^m vs a^0."""
    from models.communication.counterfactual.value_target import next_obs_under_action

    obs_m = obs
    obs_0 = obs
    total = torch.zeros(obs.shape[:-1], device=obs.device, dtype=obs.dtype)
    for k in range(1, horizon + 1):
        r_m = nav_progress_reward(obs_m, action_with)
        r_0 = nav_progress_reward(obs_0, action_without)
        total = total + (gamma**k) * (r_m - r_0)
        obs_m = next_obs_under_action(obs_m, action_with, dt=dt)
        obs_0 = next_obs_under_action(obs_0, action_without, dt=dt)
    return total

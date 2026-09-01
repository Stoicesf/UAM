"""Task-aware Causal Utility — CAU-v2 (label-collapse fix).

Fixes CAU-v1 positive-clamp starvation:

  1) Signed batch-normalized sigmoid (no max(Δ,0))
  2) Multi-step discounted ΔV credit (H=10)
  3) Weights: 0.6 ΔV + 0.35 ΔR + 0.05 ΔA

Utility *network* unchanged; only U* supervision changes.
"""

from __future__ import annotations

from typing import Callable

import torch

from models.communication.counterfactual.reward_target import multi_step_delta_reward
from models.communication.counterfactual.value_target import multi_step_delta_value
from utils.counterfactual import compute_action_difference


def batch_sigmoid_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """σ((x − μ) / (σ + ε)) over the full tensor — keeps dynamic range."""
    flat = x.reshape(-1)
    if flat.numel() < 2:
        return torch.sigmoid(x)
    mu = flat.mean()
    std = flat.std(unbiased=False)
    return torch.sigmoid((x - mu) / (std + eps))


def compose_cau_v2(
    delta_r: torch.Tensor,
    delta_v: torch.Tensor,
    delta_a: torch.Tensor,
    alpha_v: float = 0.6,
    alpha_r: float = 0.35,
    alpha_a: float = 0.05,
) -> torch.Tensor:
    """Mix already-normalized [0,1]-ish components."""
    return alpha_v * delta_v + alpha_r * delta_r + alpha_a * delta_a


def compute_cau_v2_agent_utility(
    obs: torch.Tensor,
    action_with: torch.Tensor,
    action_without: torch.Tensor,
    value_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    alpha_v: float = 0.6,
    alpha_r: float = 0.35,
    alpha_a: float = 0.05,
    horizon: int = 10,
    gamma: float = 0.95,
) -> torch.Tensor:
    """Per-agent U*_j ∈ (0, 1) with non-collapsed labels."""
    delta_a = compute_action_difference(action_without, action_with, max_norm=1.0)
    delta_a_n = delta_a  # already [0, 1]

    delta_r_raw = multi_step_delta_reward(
        obs, action_with, action_without, horizon=horizon, gamma=gamma
    )
    delta_r_n = batch_sigmoid_norm(delta_r_raw)

    if value_fn is None:
        # Fold value mass into reward when critic unavailable
        av, ar = 0.0, alpha_v + alpha_r
        delta_v_n = torch.zeros_like(delta_r_n)
    else:
        av, ar = alpha_v, alpha_r
        delta_v_raw = multi_step_delta_value(
            obs,
            action_with,
            action_without,
            value_fn=value_fn,
            horizon=horizon,
            gamma=gamma,
        )
        delta_v_n = batch_sigmoid_norm(delta_v_raw)

    return compose_cau_v2(
        delta_r_n, delta_v_n, delta_a_n, alpha_v=av, alpha_r=ar, alpha_a=alpha_a
    )


# Backward-compat alias used by older CAU-v1 path
def compute_cau_agent_utility(
    obs: torch.Tensor,
    action_with: torch.Tensor,
    action_without: torch.Tensor,
    value_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    alpha: float = 0.5,
    beta: float = 0.4,
    gamma: float = 0.1,
    reward_scale: float = 0.5,
    value_scale: float = 2.0,
) -> torch.Tensor:
    """Legacy CAU-v1 (positive clamp) — prefer compute_cau_v2_agent_utility."""
    from models.communication.counterfactual.reward_target import nav_progress_reward

    delta_a = compute_action_difference(action_without, action_with, max_norm=1.0)
    delta_r = torch.clamp(
        (
            nav_progress_reward(obs, action_with)
            - nav_progress_reward(obs, action_without)
        )
        / max(float(reward_scale), 1e-6),
        0.0,
        1.0,
    )
    if value_fn is None:
        a, b = alpha + beta, 0.0
        delta_v = torch.zeros_like(delta_r)
    else:
        from models.communication.counterfactual.value_target import next_obs_under_action

        a, b = alpha, beta
        obs_m = next_obs_under_action(obs, action_with)
        obs_0 = next_obs_under_action(obs, action_without)
        delta_v = torch.clamp(
            (value_fn(obs_m) - value_fn(obs_0)) / max(float(value_scale), 1e-6),
            0.0,
            1.0,
        )
    return a * delta_r + b * delta_v + gamma * delta_a

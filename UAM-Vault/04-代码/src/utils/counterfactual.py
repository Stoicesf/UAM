"""Counterfactual action-difference labels for AC-DSGF++.

U* ≈ clamp(‖a^m − a^0‖, 0, 1)

a^0: action / guidance under zero communication
a^m: action / guidance under gated communication
"""

from __future__ import annotations

import torch


def compute_action_difference(
    action_without: torch.Tensor,
    action_with: torch.Tensor,
    max_norm: float = 1.0,
) -> torch.Tensor:
    """Per-agent utility target from action change.

    Args:
        action_without: (..., A) actions without communication
        action_with: (..., A) actions with communication
        max_norm: clamp upper bound (default 1)

    Returns:
        u_star_agent: (...) in [0, max_norm]
    """
    diff = torch.norm(action_with - action_without, dim=-1)
    return torch.clamp(diff, 0.0, max_norm)


def broadcast_receiver_utility(
    u_agent: torch.Tensor,
) -> torch.Tensor:
    """Expand per-receiver Δa_j to pairwise U*_ij = Δa_j.

    Args:
        u_agent: (B, N) or (N,)

    Returns:
        U_star: (B, N, N) with U_ij = u_j, diagonal 0
    """
    if u_agent.dim() == 1:
        u_agent = u_agent.unsqueeze(0)
    b, n = u_agent.shape
    # sender i, receiver j → attribute change at j
    u = u_agent.unsqueeze(1).expand(b, n, n)
    eye = torch.eye(n, device=u.device, dtype=u.dtype).unsqueeze(0)
    return u * (1.0 - eye)


def phi_as_action_proxy(
    phi_without: torch.Tensor,
    phi_with: torch.Tensor,
    max_norm: float = 1.0,
) -> torch.Tensor:
    """Fallback label when residual actor is unavailable: use ‖ΔΦ‖."""
    return compute_action_difference(phi_without, phi_with, max_norm=max_norm)


def pearson_correlation(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> float:
    """Pearson r on flattened masked elements (returns 0 if undefined)."""
    xf = x.reshape(-1).float()
    yf = y.reshape(-1).float()
    if mask is not None:
        m = mask.reshape(-1).bool()
        xf = xf[m]
        yf = yf[m]
    if xf.numel() < 2:
        return 0.0
    vx = xf - xf.mean()
    vy = yf - yf.mean()
    denom = vx.std(unbiased=False) * vy.std(unbiased=False)
    if float(denom) < 1e-8:
        return 0.0
    return float((vx * vy).mean() / denom)

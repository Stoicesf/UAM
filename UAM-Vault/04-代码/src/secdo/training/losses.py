"""Theory-aligned losses (platform)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dynamics_loss(s_hat: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(s_hat, s_true)


def constraint_loss(c_hat: torch.Tensor, c_true: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(c_hat.view_as(c_true), c_true)


def predictor_loss(
    s_hat: torch.Tensor | None,
    s_true: torch.Tensor | None,
    c_hat: torch.Tensor,
    c_true: torch.Tensor,
    *,
    lambda_s: float = 0.5,
    lambda_c: float = 1.0,
    lambda_u: float = 0.1,
    u_hat: torch.Tensor | None = None,
    u_true: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = lambda_c * constraint_loss(c_hat, c_true)
    if s_hat is not None and s_true is not None:
        loss = loss + lambda_s * dynamics_loss(s_hat, s_true)
    if u_hat is not None and u_true is not None:
        loss = loss + lambda_u * F.mse_loss(u_hat, u_true)
    return loss


def violation(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return (x.sum(dim=-1) - c.view(-1)).clamp(min=0).mean()


def allocation_objective(x: torch.Tensor, pref: torch.Tensor) -> torch.Tensor:
    return 0.5 * ((x - pref) ** 2).sum(dim=-1)


def eps_l2(s_hat: torch.Tensor, s_true: torch.Tensor) -> float:
    return float((s_hat.float() - s_true.float()).norm(dim=-1).mean())


def optimization_loss(
    F_vals: torch.Tensor,
    V_vals: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:
    return F_vals.mean() + gamma * V_vals.mean()


def secdo_joint_loss(
    L_opt: torch.Tensor,
    L_pred: torch.Tensor,
    lambda_pred: float = 1.0,
) -> torch.Tensor:
    return L_opt + lambda_pred * L_pred

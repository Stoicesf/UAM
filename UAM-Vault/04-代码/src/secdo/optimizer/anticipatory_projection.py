"""Anticipatory projection Π_{B̂}(y) with ĉ.detach() discipline."""

from __future__ import annotations

from typing import Any

import torch


def project_sum_budget(y: torch.Tensor, budget: torch.Tensor) -> torch.Tensor:
    if budget.dim() > 1:
        budget = budget.squeeze(-1)
    x = y.clamp(min=0.0)
    s = x.sum(dim=-1)
    over = s > budget + 1e-8
    if not over.any():
        return x
    x_over = x[over]
    b_over = budget[over]
    u, _ = torch.sort(x_over, dim=-1, descending=True)
    cssv = torch.cumsum(u, dim=-1) - b_over.unsqueeze(-1)
    ind = torch.arange(1, x_over.shape[-1] + 1, device=y.device, dtype=y.dtype)
    cond = u - cssv / ind > 0
    rho = (cond.sum(dim=-1) - 1).clamp(min=0)
    theta = cssv.gather(1, rho.unsqueeze(1)).squeeze(1) / (rho + 1).clamp(min=1).to(y.dtype)
    x_proj = (x_over - theta.unsqueeze(-1)).clamp(min=0.0)
    x = x.clone()
    x[over] = x_proj
    return x


def project_budget(x: torch.Tensor, budget: torch.Tensor) -> torch.Tensor:
    return project_sum_budget(x, budget)


def anticipatory_project_spec(y: torch.Tensor, spec: Any) -> torch.Tensor:
    if spec.kind == "sum_budget":
        return project_sum_budget(y, spec.c)
    if spec.kind == "box":
        return y.clamp(min=0.0, max=spec.c.max())
    raise ValueError(f"Unknown feasible kind: {spec.kind}")


def anticipatory_project(
    y: torch.Tensor,
    budget_or_spec: Any,
    *,
    detach_budget: bool = True,
) -> torch.Tensor:
    if hasattr(budget_or_spec, "kind"):
        return anticipatory_project_spec(y, budget_or_spec)
    c = budget_or_spec.detach() if detach_budget else budget_or_spec
    return project_sum_budget(y, c)

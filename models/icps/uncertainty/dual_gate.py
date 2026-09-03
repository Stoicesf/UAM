"""Dual gate: (u > τ) OR (q_perception < θ) → allow topology projection."""

from __future__ import annotations

import torch

from models.uncertainty.uncertainty_driven_comm import (
    project_topology,
    uncertainty_gated_topology,
)


def dual_gated_topology(
    uncertainty: torch.Tensor,
    perception_q: torch.Tensor,
    scores: torch.Tensor,
    radius_mask: torch.Tensor,
    *,
    tau: float,
    theta: float,
    budget_ratio: float | None = 0.5,
    fixed_k: int | None = None,
) -> torch.Tensor:
    """Row gate ON if uncertain OR perception quality low.

    Does not replace Π_Bt; only decides who may open edges before projection.
    """
    if uncertainty.dim() == 1:
        uncertainty = uncertainty.unsqueeze(0)
    if perception_q.dim() == 1:
        perception_q = perception_q.unsqueeze(0)

    row_on = ((uncertainty > tau) | (perception_q < theta)).float().unsqueeze(-1)
    g = project_topology(
        scores, radius_mask, budget_ratio=budget_ratio, fixed_k=fixed_k
    )
    g = g * row_on
    if float(row_on.detach().max()) <= 0:
        return torch.zeros_like(g)
    return g


def self_check() -> None:
    b, n = 1, 4
    scores = torch.rand(b, n, n)
    mask = torch.ones(b, n, n) - torch.eye(n)
    # calm + good perception → zero
    g0 = dual_gated_topology(
        torch.zeros(b, n),
        torch.ones(b, n),
        scores,
        mask,
        tau=0.5,
        theta=0.3,
        budget_ratio=0.5,
    )
    assert torch.count_nonzero(g0) == 0
    # low perception triggers
    g1 = dual_gated_topology(
        torch.zeros(b, n),
        torch.full((b, n), 0.1),
        scores,
        mask,
        tau=0.5,
        theta=0.3,
        budget_ratio=0.5,
    )
    assert torch.count_nonzero(g1) > 0
    # pure ToA path still works
    g2 = uncertainty_gated_topology(
        torch.ones(b, n), scores, mask, threshold=0.5, budget_ratio=0.5
    )
    assert torch.count_nonzero(g2) > 0
    print("dual_gate: OK")


if __name__ == "__main__":
    self_check()

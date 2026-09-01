"""Feasible region B(c) = { x ≥ 0 : 1ᵀx ≤ c }."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FeasibleSpec:
    kind: str  # "sum_budget" | "box"
    c: torch.Tensor


class FeasibleRegion:
    def __init__(self, kind: str = "sum_budget"):
        self.kind = kind

    def __call__(self, c: torch.Tensor) -> FeasibleSpec:
        if c.dim() == 1:
            c = c.unsqueeze(-1)
        return FeasibleSpec(kind=self.kind, c=c)


def sum_budget_set(c: torch.Tensor) -> FeasibleSpec:
    return FeasibleRegion("sum_budget")(c)


# Legacy aliases
FeasibleHead = FeasibleRegion

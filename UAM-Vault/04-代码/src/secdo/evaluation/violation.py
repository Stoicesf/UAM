"""Constraint violation metrics + anticipatory vs reactive certificates."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from secdo.training.losses import violation as viol_fn


@dataclass
class ViolationTracker:
    viol: list[float] = field(default_factory=list)
    delta: list[float] = field(default_factory=list)
    chi: list[float] = field(default_factory=list)
    PI: list[float] = field(default_factory=list)
    alpha: list[float] = field(default_factory=list)

    def update(
        self,
        x: torch.Tensor,
        c_true: torch.Tensor,
        *,
        delta: float,
        chi: float,
        PI: float | None = None,
        alpha: float | None = None,
    ) -> None:
        self.viol.append(float(viol_fn(x, c_true)))
        self.delta.append(float(delta))
        self.chi.append(float(chi))
        self.PI.append(float(PI if PI is not None else delta / (chi + 1e-8)))
        if alpha is not None:
            self.alpha.append(float(alpha))

    def summary(self) -> dict[str, float]:
        def m(xs):
            return float(sum(xs) / max(len(xs), 1))

        return {
            "violation": m(self.viol),
            "delta": m(self.delta),
            "chi": m(self.chi),
            "PI": m(self.PI),
            "frac_PI_lt_1": float(sum(1 for p in self.PI if p < 1) / max(len(self.PI), 1)),
            "mean_alpha": m(self.alpha) if self.alpha else 0.0,
            "sum_violation": float(sum(self.viol)),
        }

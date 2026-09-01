"""Dynamic / variational regret logging for Thm2 validation."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from secdo.training.losses import allocation_objective


@dataclass
class RegretTracker:
    instant: list[float] = field(default_factory=list)
    cumulative: list[float] = field(default_factory=list)
    path_variation: list[float] = field(default_factory=list)  # ||x*_{t+1}-x*_t||
    _x_star_prev: torch.Tensor | None = None
    _cum: float = 0.0

    def update(
        self,
        x: torch.Tensor,
        x_star: torch.Tensor,
        pref: torch.Tensor,
    ) -> None:
        # variational gap as instantaneous regret proxy under static F
        r = float((allocation_objective(x, pref) - allocation_objective(x_star, pref)).clamp(min=0).mean())
        self.instant.append(r)
        self._cum += r
        self.cumulative.append(self._cum)
        if self._x_star_prev is not None:
            self.path_variation.append(float((x_star - self._x_star_prev).norm(dim=-1).mean()))
        self._x_star_prev = x_star.detach().clone()

    def summary(self) -> dict[str, float]:
        P = float(sum(self.path_variation))
        T = max(len(self.instant), 1)
        return {
            "Reg_T": self._cum,
            "mean_instant_regret": self._cum / T,
            "P_T": P,
            "T": float(T),
        }

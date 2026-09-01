"""Theory-aligned metrics for SECDO Phase 2C."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


def allocation_objective(x: torch.Tensor, pref: torch.Tensor | None = None) -> torch.Tensor:
    """
    Static convex surrogate F(x) = 0.5 ‖x - pref‖²  (pref defaults to uniform direction).
    Minimizer on B(c) is a Euclidean projection of pref onto the simplex-cap.
    """
    if pref is None:
        pref = torch.ones_like(x) / x.shape[-1]
    return 0.5 * ((x - pref) ** 2).sum(dim=-1)


def instantaneous_optimum(c: torch.Tensor, dim: int, pref: torch.Tensor | None = None) -> torch.Tensor:
    """x* = Π_{B(c)}(pref)."""
    from secdo.optimizer.anticipatory_projection import project_sum_budget

    B = c.shape[0]
    if pref is None:
        pref = torch.ones(B, dim, device=c.device, dtype=c.dtype) / dim
    return project_sum_budget(pref, c)


@dataclass
class TheoryLogger:
    eps: list[float] = field(default_factory=list)
    delta: list[float] = field(default_factory=list)
    rho: list[float] = field(default_factory=list)
    viol: list[float] = field(default_factory=list)
    gap: list[float] = field(default_factory=list)
    F: list[float] = field(default_factory=list)

    def update(
        self,
        *,
        eps_t: float,
        delta_t: float,
        rho_t: float,
        viol_t: float,
        gap_t: float,
        F_t: float,
    ) -> None:
        self.eps.append(eps_t)
        self.delta.append(delta_t)
        self.rho.append(rho_t)
        self.viol.append(viol_t)
        self.gap.append(gap_t)
        self.F.append(F_t)

    def summary(self, *, include_series: bool = False) -> dict:
        def m(xs):
            return float(sum(xs) / max(len(xs), 1))

        mean_pred = m([e + d for e, d in zip(self.eps, self.delta)])
        out = {
            "T": float(len(self.gap)),
            "mean_eps": m(self.eps),
            "mean_delta": m(self.delta),
            "mean_rho": m(self.rho),
            "mean_viol": m(self.viol),
            "mean_gap": m(self.gap),
            "mean_eps_plus_delta": mean_pred,
            "frac_delta_lt_rho": float(
                sum(1 for d, r in zip(self.delta, self.rho) if d < r) / max(len(self.delta), 1)
            ),
        }
        if include_series:
            out["series"] = {
                "eps": list(self.eps),
                "delta": list(self.delta),
                "rho": list(self.rho),
                "viol": list(self.viol),
                "gap": list(self.gap),
            }
        return out

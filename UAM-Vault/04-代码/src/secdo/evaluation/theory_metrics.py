"""Theory metrics: ε, δ, ρ, Gap, Violation → results.json."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EpisodeMetrics:
    eps: list[float] = field(default_factory=list)
    delta: list[float] = field(default_factory=list)
    rho: list[float] = field(default_factory=list)
    gap: list[float] = field(default_factory=list)
    viol: list[float] = field(default_factory=list)

    def update(self, *, eps: float, delta: float, rho: float, gap: float, viol: float) -> None:
        self.eps.append(float(eps))
        self.delta.append(float(delta))
        self.rho.append(float(rho))
        self.gap.append(float(gap))
        self.viol.append(float(viol))

    def summary(self) -> dict[str, float]:
        def m(xs: list[float]) -> float:
            return float(sum(xs) / max(len(xs), 1))

        return {
            "T": float(len(self.gap)),
            "epsilon": m(self.eps),
            "delta": m(self.delta),
            "rho": m(self.rho),
            "gap": m(self.gap),
            "violation": m(self.viol),
            "mean_eps_plus_delta": m([e + d for e, d in zip(self.eps, self.delta)]),
            "frac_delta_lt_rho": float(
                sum(1 for d, r in zip(self.delta, self.rho) if d < r) / max(len(self.delta), 1)
            ),
        }

    def series(self) -> dict[str, list[float]]:
        return {
            "eps": list(self.eps),
            "delta": list(self.delta),
            "rho": list(self.rho),
            "gap": list(self.gap),
            "viol": list(self.viol),
        }


class TheoryAccumulator:
    """Multi-episode aggregation."""

    def __init__(self):
        self.episodes: list[dict[str, float]] = []

    def add(self, summary: dict[str, float]) -> None:
        self.episodes.append(summary)

    def mean(self) -> dict[str, float]:
        if not self.episodes:
            return {}
        keys = self.episodes[0].keys()
        return {k: float(sum(e[k] for e in self.episodes) / len(self.episodes)) for k in keys}


def results_payload(
    *,
    method: str,
    regime: str,
    seed: int,
    summary: dict[str, float],
    series: dict[str, list[float]] | None = None,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "method": method,
        "regime": regime,
        "seed": seed,
        **summary,
    }
    if series is not None:
        out["series"] = series
    if extras:
        out["extras"] = extras
    return out

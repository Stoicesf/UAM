"""TensorBoard-friendly meters for ε, δ, ρ, Gap, V."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TheoryMeters:
    eps: list[float] = field(default_factory=list)
    delta: list[float] = field(default_factory=list)
    rho: list[float] = field(default_factory=list)
    gap: list[float] = field(default_factory=list)
    viol: list[float] = field(default_factory=list)
    loss_c: list[float] = field(default_factory=list)
    loss_s: list[float] = field(default_factory=list)
    loss_total: list[float] = field(default_factory=list)

    def update(self, **kwargs: float) -> None:
        for k, v in kwargs.items():
            if hasattr(self, k) and v is not None:
                getattr(self, k).append(float(v))

    def mean(self, key: str) -> float:
        xs = getattr(self, key)
        return float(sum(xs) / max(len(xs), 1))

    def as_tb_dict(self, prefix: str = "") -> dict[str, float]:
        return {
            f"{prefix}metric/epsilon": self.mean("eps"),
            f"{prefix}metric/delta": self.mean("delta"),
            f"{prefix}metric/rho": self.mean("rho"),
            f"{prefix}metric/gap": self.mean("gap"),
            f"{prefix}metric/violation": self.mean("viol"),
            f"{prefix}train/loss_constraint": self.mean("loss_c"),
            f"{prefix}train/loss_state": self.mean("loss_s"),
            f"{prefix}train/loss_total": self.mean("loss_total"),
        }

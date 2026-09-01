from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Meter:
    values: list[float] = field(default_factory=list)

    def update(self, v: float) -> None:
        self.values.append(float(v))

    def mean(self) -> float:
        return float(sum(self.values) / max(len(self.values), 1))


@dataclass
class TheoryLog:
    eps: Meter = field(default_factory=Meter)
    delta: Meter = field(default_factory=Meter)
    rho: Meter = field(default_factory=Meter)
    gap: Meter = field(default_factory=Meter)
    viol: Meter = field(default_factory=Meter)
    loss_c: Meter = field(default_factory=Meter)
    loss_s: Meter = field(default_factory=Meter)
    loss_total: Meter = field(default_factory=Meter)

    def as_dict(self) -> dict[str, float]:
        return {
            "epsilon": self.eps.mean(),
            "delta": self.delta.mean(),
            "rho": self.rho.mean(),
            "gap": self.gap.mean(),
            "violation": self.viol.mean(),
            "constraint_MSE": self.loss_c.mean(),
            "loss_state": self.loss_s.mean(),
            "loss_total": self.loss_total.mean(),
        }


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[misc, assignment]


def make_writer(log_dir: str | None):
    if log_dir and SummaryWriter is not None:
        return SummaryWriter(log_dir)
    return None


def log_scalars(writer, tag_prefix: str, metrics: dict[str, Any], step: int) -> None:
    if writer is None:
        return
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            writer.add_scalar(f"{tag_prefix}/{k}", float(v), step)

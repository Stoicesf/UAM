"""Track ε_t, δ_t, violation_t for theory-aligned logging."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class ErrorTracker:
    eps: list[float] = field(default_factory=list)
    delta: list[float] = field(default_factory=list)
    violation: list[float] = field(default_factory=list)

    def update(
        self,
        *,
        s_next: torch.Tensor,
        s_hat: torch.Tensor,
        c_true: torch.Tensor,
        c_hat: torch.Tensor,
        x: torch.Tensor,
    ) -> dict[str, float]:
        eps_t = (s_next - s_hat).norm(dim=-1).mean().item()
        delta_t = (c_true - c_hat).abs().mean().item()
        c = c_true.view(-1)
        viol = (x.sum(dim=-1) - c).clamp(min=0.0).mean().item()
        self.eps.append(eps_t)
        self.delta.append(delta_t)
        self.violation.append(viol)
        return {"eps_t": eps_t, "delta_t": delta_t, "violation_t": viol}

    def sums(self) -> dict[str, float]:
        return {
            "sum_eps": float(sum(self.eps)),
            "sum_delta": float(sum(self.delta)),
            "sum_violation": float(sum(self.violation)),
            "T": float(len(self.eps)),
        }

"""Common solver interface — no if-method branching in experiment scripts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch

from experiments.secdo_uav.env_uav_bandwidth import UAVBandwidthEnv
from secdo.optimizer.anticipatory_projection import project_sum_budget
from secdo.optimizer.reactive_projection import reactive_project
from secdo.training.losses import allocation_objective


@dataclass
class SolveResult:
    method: str
    metrics: dict[str, float]
    series: dict[str, list[float]] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


class Solver(ABC):
    name: str = "base"

    def __init__(self, eta: float = 0.25, device: str | torch.device = "cuda:0"):
        self.eta = eta
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

    @abstractmethod
    def solve(self, env: UAVBandwidthEnv, *, batch: int = 8) -> SolveResult:
        ...

    def _init_alloc(self, env: UAVBandwidthEnv, batch: int) -> tuple[dict, torch.Tensor, torch.Tensor]:
        obs = env.reset(batch=batch)
        n = env.cfg.n_agents
        pref = torch.ones(batch, n, device=self.device) / n
        x = reactive_project(pref.clone(), obs["c_teacher"].to(self.device))
        return obs, pref, x

    def _gap(self, x: torch.Tensor, pref: torch.Tensor, c: torch.Tensor) -> float:
        x_star = project_sum_budget(pref, c)
        return float(
            (allocation_objective(x, pref) - allocation_objective(x_star, pref)).clamp(min=0).mean()
        )

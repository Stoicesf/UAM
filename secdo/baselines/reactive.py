"""Reactive baseline: Π_{B(c_t)} — no prediction."""

from __future__ import annotations

import torch

from experiments.secdo_uav.env_uav_bandwidth import UAVBandwidthEnv
from secdo.baselines.base import SolveResult, Solver
from secdo.evaluation.theory_metrics import EpisodeMetrics
from secdo.optimizer.projected_gradient import gradient_step
from secdo.optimizer.reactive_projection import reactive_project
from secdo.training.losses import violation


class ReactiveSolver(Solver):
    name = "reactive"

    @torch.no_grad()
    def solve(self, env: UAVBandwidthEnv, *, batch: int = 8) -> SolveResult:
        obs, pref, x = self._init_alloc(env, batch)
        met = EpisodeMetrics()
        while not env.done:
            c_t = obs["c_teacher"].to(self.device)
            y = gradient_step(x, pref, self.eta)
            x = reactive_project(y, c_t)
            obs = env.step()
            c_next = obs["c_teacher"].to(self.device)
            rho = float((c_next - c_t).abs().mean())
            delta = rho  # certificate: reactive inherits drift
            met.update(
                eps=0.0,
                delta=delta,
                rho=rho,
                gap=self._gap(x, pref, c_next),
                viol=float(violation(x, c_next)),
            )
        return SolveResult(method=self.name, metrics=met.summary(), series=met.series())

"""Oracle predictive upper bound: Π_{B(c_{t+1})} with teacher peek."""

from __future__ import annotations

import torch

from experiments.secdo_uav.env_uav_bandwidth import UAVBandwidthEnv
from secdo.baselines.base import SolveResult, Solver
from secdo.datasets.uav.mobility import step_positions
from secdo.datasets.uav.teacher import capacity_teacher
from secdo.evaluation.theory_metrics import EpisodeMetrics
from secdo.optimizer.anticipatory_projection import anticipatory_project
from secdo.optimizer.projected_gradient import gradient_step
from secdo.training.losses import violation


class OracleSolver(Solver):
    name = "oracle"

    def _peek_c_next(self, env: UAVBandwidthEnv) -> torch.Tensor:
        assert env.pos is not None and env.vel is not None
        pos2, vel2 = env.pos.clone(), env.vel.clone()
        scale = env._regime_velocity_scale()
        speed = vel2.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        vel2 = vel2 / speed * scale
        pos2, _ = step_positions(pos2, vel2, dt=env.cfg.dt, box=env.cfg.box)
        return capacity_teacher(pos2, env.channel)

    @torch.no_grad()
    def solve(self, env: UAVBandwidthEnv, *, batch: int = 8) -> SolveResult:
        obs, pref, x = self._init_alloc(env, batch)
        met = EpisodeMetrics()
        while not env.done:
            c_t = obs["c_teacher"].to(self.device)
            c_hat = self._peek_c_next(env).to(self.device)
            y = gradient_step(x, pref, self.eta)
            x = anticipatory_project(y, c_hat, detach_budget=True)
            obs = env.step()
            c_next = obs["c_teacher"].to(self.device)
            met.update(
                eps=0.0,
                delta=float((c_hat - c_next).abs().mean()),
                rho=float((c_next - c_t).abs().mean()),
                gap=self._gap(x, pref, c_next),
                viol=float(violation(x, c_next)),
            )
        return SolveResult(method=self.name, metrics=met.summary(), series=met.series())

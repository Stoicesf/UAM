"""Synthetic convex dynamic-constraint environment for Thm2 validation.

F_t(x) = 0.5 x^T Q_t x + b_t^T x
B_t = { x ≥ 0 : 1^T x ≤ c_t }
x*_t = Π_{B_t}(Q_t^{-1} (-b_t))  (for Q≈I use pref = -b)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from secdo.optimizer.anticipatory_projection import project_sum_budget


@dataclass
class SynthConfig:
    dim: int = 8
    horizon: int = 200
    seed: int = 0
    c0: float = 2.0
    drift_amp: float = 0.15  # χ scale
    noise_sigma: float = 0.05  # prediction noise on features / c
    device: str = "cpu"


class SyntheticConvexEnv:
    def __init__(self, cfg: SynthConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.t = 0
        self.c: torch.Tensor | None = None
        self.b: torch.Tensor | None = None
        g = torch.Generator(device=self.device)
        g.manual_seed(cfg.seed)
        self._g = g

    def reset(self, batch: int = 16) -> dict[str, torch.Tensor]:
        self.t = 0
        B, d = batch, self.cfg.dim
        self.c = torch.full((B, 1), self.cfg.c0, device=self.device)
        self.b = torch.randn(B, d, generator=self._g, device=self.device) * 0.3
        return self._obs()

    def _obs(self) -> dict[str, torch.Tensor]:
        assert self.c is not None and self.b is not None
        # feat = [b, c] as "state"
        feat = torch.cat([self.b, self.c], dim=-1)
        pref = -self.b  # for Q=I, unconstrained min is -b; project onto B(c)
        x_star = project_sum_budget(pref, self.c)
        return {
            "feat": feat,
            "c_teacher": self.c.clone(),
            "pref": pref,
            "x_star": x_star,
            "t": torch.tensor(self.t, device=self.device),
        }

    def step(self) -> dict[str, torch.Tensor]:
        assert self.c is not None and self.b is not None
        # drift capacity and linear term
        chi = self.cfg.drift_amp * (2 * torch.rand_like(self.c) - 1)
        self.c = (self.c + chi).clamp(min=0.3, max=5.0)
        self.b = self.b + 0.02 * torch.randn_like(self.b)
        self.t += 1
        return self._obs()

    @property
    def done(self) -> bool:
        return self.t >= self.cfg.horizon

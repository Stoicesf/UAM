"""UAV dynamic bandwidth environment for SECDO.

Canonical teacher: secdo.datasets.uav.teacher (no models/__init__ side effects).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from secdo.datasets.uav.channel import ChannelConfig
from secdo.datasets.uav.mobility import step_positions
from secdo.datasets.uav.teacher import pack_state, capacity_teacher


@dataclass
class UAVBandwidthConfig:
    n_agents: int = 8
    dim: int = 2
    horizon: int = 128
    dt: float = 0.12
    box: float = 3.0
    vel_scale: float = 0.4
    regime: str = "fast"  # slow | fast | stress
    stress_delta: float = 0.0
    stress_eps: float = 0.0
    seed: int = 0
    channel: ChannelConfig | None = None


class UAVBandwidthEnv:
    def __init__(self, cfg: UAVBandwidthConfig, device: torch.device | str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.channel = cfg.channel or ChannelConfig(
            bandwidth_hz=1.0,
            comm_radius=2.8 if cfg.regime != "slow" else 3.5,
            path_loss_exp=2.2,
        )
        self.t = 0
        self.pos: torch.Tensor | None = None
        self.vel: torch.Tensor | None = None
        self._c_prev = None

    def _regime_velocity_scale(self) -> float:
        if self.cfg.regime == "slow":
            return 0.05 * self.cfg.vel_scale
        if self.cfg.regime == "stress":
            return 1.2 * self.cfg.vel_scale
        return self.cfg.vel_scale

    def reset(self, batch: int = 1) -> dict[str, torch.Tensor]:
        g = torch.Generator(device=self.device)
        g.manual_seed(self.cfg.seed)
        B, N, D = batch, self.cfg.n_agents, self.cfg.dim
        self.pos = (torch.rand(B, N, D, generator=g, device=self.device) - 0.5) * 2.0 * self.cfg.box
        scale = self._regime_velocity_scale()
        self.vel = (torch.rand(B, N, D, generator=g, device=self.device) - 0.5) * 2.0 * scale
        self.t = 0
        self._c_prev = None
        return self._observe()

    def _observe(self) -> dict[str, torch.Tensor]:
        assert self.pos is not None and self.vel is not None
        pack = pack_state(self.pos, self.vel, cfg=self.channel)
        c = pack["c_teacher"]
        feat = pack["feat"]
        if self.cfg.stress_eps > 0:
            feat = feat + self.cfg.stress_eps * torch.randn_like(feat)
        c_obs = c
        if self.cfg.stress_delta > 0:
            c_obs = (c + self.cfg.stress_delta * torch.randn_like(c).abs()).clamp(min=1e-3)
        rho = torch.zeros_like(c)
        if self._c_prev is not None:
            rho = (c - self._c_prev).abs()
        self._c_prev = c.detach().clone()
        return {
            "feat": feat,
            "c_teacher": c,
            "c_obs": c_obs,
            "rho": rho,
            "position": self.pos,
            "velocity": self.vel,
            "t": torch.tensor(self.t, device=self.device),
        }

    def step(self) -> dict[str, torch.Tensor]:
        assert self.pos is not None and self.vel is not None
        scale = self._regime_velocity_scale()
        speed = self.vel.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        self.vel = self.vel / speed * scale * (0.5 + torch.rand_like(speed))
        self.pos, self.vel = step_positions(self.pos, self.vel, dt=self.cfg.dt, box=self.cfg.box)
        self.t += 1
        return self._observe()

    @property
    def done(self) -> bool:
        return self.t >= self.cfg.horizon

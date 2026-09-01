"""UAV mobility models: p_{t+1} = p_t + v_t Δt (+ bounce / regime scaling)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class MobilityConfig:
    n_agents: int = 8
    dim: int = 2
    dt: float = 0.12
    box: float = 3.0
    vel_scale: float = 0.4
    seed: int = 0


class MobilityModel(ABC):
    name: str = "base"

    def __init__(self, cfg: MobilityConfig):
        self.cfg = cfg

    def init_state(
        self, batch: int, device: torch.device, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, D = batch, self.cfg.n_agents, self.cfg.dim
        g = generator
        pos = (torch.rand(B, N, D, generator=g, device=device) - 0.5) * 2.0 * self.cfg.box
        scale = self.velocity_scale()
        vel = (torch.rand(B, N, D, generator=g, device=device) - 0.5) * 2.0 * scale
        return pos, vel

    @abstractmethod
    def velocity_scale(self) -> float:
        ...

    def step(
        self, pos: torch.Tensor, vel: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self.velocity_scale()
        speed = vel.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        vel = vel / speed * scale * (0.5 + torch.rand_like(speed))
        return step_positions(pos, vel, dt=self.cfg.dt, box=self.cfg.box)


def step_positions(
    pos: torch.Tensor,
    vel: torch.Tensor,
    dt: float = 0.1,
    box: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bounded kinematics (identical to legacy smoke)."""
    pos = pos + vel * dt
    over = pos.abs() > box
    vel = torch.where(over, -vel, vel)
    pos = pos.clamp(-box, box)
    vel = vel + 0.05 * torch.randn_like(vel)
    vel = vel.clamp(-1.0, 1.0)
    return pos, vel


class SlowDrift(MobilityModel):
    name = "slow_drift"

    def velocity_scale(self) -> float:
        return 0.05 * self.cfg.vel_scale


class FastDrift(MobilityModel):
    name = "fast_drift"

    def velocity_scale(self) -> float:
        return self.cfg.vel_scale


class RandomWaypoint(MobilityModel):
    """Stress-like mobility: larger speed + occasional heading redraw."""

    name = "random_waypoint"

    def velocity_scale(self) -> float:
        return 1.2 * self.cfg.vel_scale

    def step(
        self, pos: torch.Tensor, vel: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # occasional full re-sample of direction
        redraw = torch.rand(vel.shape[0], vel.shape[1], 1, device=vel.device) < 0.05
        noise = (torch.rand_like(vel) - 0.5) * 2.0 * self.velocity_scale()
        vel = torch.where(redraw, noise, vel)
        return super().step(pos, vel)


# Legacy regime aliases
_REGIME_MAP = {
    "slow": SlowDrift,
    "slow_drift": SlowDrift,
    "fast": FastDrift,
    "fast_drift": FastDrift,
    "stress": RandomWaypoint,
    "random_waypoint": RandomWaypoint,
}


def get_mobility(regime: str, cfg: MobilityConfig | None = None) -> MobilityModel:
    cfg = cfg or MobilityConfig()
    cls = _REGIME_MAP.get(regime.lower())
    if cls is None:
        raise ValueError(f"Unknown regime/mobility: {regime}")
    return cls(cfg)

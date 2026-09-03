"""Pursuit / encirclement helpers for DICEUAVScenario."""

from __future__ import annotations

import math

import torch


def sample_evader(boundary: float, generator: torch.Generator | None, device: str | torch.device) -> torch.Tensor:
    """Return (1, 4): x, y, priority=1, time_window placeholder."""
    xy = (torch.rand(1, 2, generator=generator, device=device) * 2 - 1) * (boundary * 0.5)
    pri = torch.ones(1, 1, device=device)
    tw = torch.full((1, 1), 1e9, device=device)
    return torch.cat([xy, pri, tw], dim=-1)


def step_evader(
    evader_xy: torch.Tensor,
    uav_pos: torch.Tensor,
    alive: torch.Tensor,
    *,
    speed: float,
    boundary: float,
    dt: float,
) -> torch.Tensor:
    """Flee from nearest alive UAV; stay in bounds."""
    d = uav_pos - evader_xy  # (N,2) relative from evader? wait: uav - e
    dist = d.norm(dim=-1)
    dist = torch.where(alive, dist, torch.full_like(dist, 1e6))
    i = int(dist.argmin())
    away = evader_xy - uav_pos[i]
    nrm = away.norm().clamp(min=1e-6)
    vel = (away / nrm) * speed
    nxt = evader_xy + vel * dt
    return nxt.clamp(-boundary, boundary)


def encirclement_score(uav_pos: torch.Tensor, evader_xy: torch.Tensor, alive: torch.Tensor, min_agents: int = 3) -> float:
    """True if ≥min_agents UAVs surround evader with pairwise angle gaps covering the circle."""
    rel = uav_pos[alive] - evader_xy
    if rel.shape[0] < min_agents:
        return 0.0
    # keep nearby agents only
    dist = rel.norm(dim=-1)
    near = dist < 2.5
    rel = rel[near]
    if rel.shape[0] < min_agents:
        return 0.0
    ang = torch.atan2(rel[:, 1], rel[:, 0])
    ang, _ = torch.sort(ang)
    gaps = torch.diff(ang)
    wrap = (ang[0] + 2 * math.pi) - ang[-1]
    gaps = torch.cat([gaps, wrap.unsqueeze(0)])
    # encircled if no gap > 120° (2π/3) and ≥3 agents
    if float(gaps.max()) < (2 * math.pi / 3) and rel.shape[0] >= min_agents:
        return 1.0
    return 0.0


def pursuit_rewards(
    uav_pos: torch.Tensor,
    alive: torch.Tensor,
    evader_xy: torch.Tensor,
    trap_center: torch.Tensor,
    *,
    capture_radius: float = 2.0,
) -> tuple[torch.Tensor, bool, dict]:
    """Return per-agent rewards, captured flag, metrics."""
    n = uav_pos.shape[0]
    device = uav_pos.device
    rewards = torch.zeros(n, device=device)
    d_e = (uav_pos - evader_xy).norm(dim=-1)
    rewards = rewards - 0.05 * d_e  # approach
    enc = encirclement_score(uav_pos, evader_xy, alive)
    if enc > 0:
        rewards = rewards + 5.0 * alive.float()
    in_trap = float((evader_xy - trap_center).norm()) <= capture_radius
    mean_d = float(d_e[alive].mean()) if alive.any() else 99.0
    # capture: tight encirclement near prey, or prey pushed into trap while surrounded
    captured = bool((enc > 0 and mean_d < 1.0) or (in_trap and enc > 0))
    if captured:
        rewards = rewards + 20.0 * alive.float()
    return rewards, captured, {"encirclement": enc, "in_trap": in_trap, "capture": captured, "mean_d": mean_d}

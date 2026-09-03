"""Adversarial pursuit: scripted or RL evader on top of pursuit helpers."""

from __future__ import annotations

from pathlib import Path

import torch

from environments.scenarios.pursuit import encirclement_score, pursuit_rewards, sample_evader
from models.evader_ppo import EvaderPPO

ROOT = Path(__file__).resolve().parents[2]


def load_evader(n_uav: int, ckpt: str | Path | None = None) -> EvaderPPO:
    ppo = EvaderPPO(EvaderPPO.obs_dim_for(n_uav))
    path = Path(ckpt) if ckpt else ROOT / "experiment_results" / "adversarial" / "evader.pt"
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        ppo.load_state_dict(payload)
    return ppo


def rl_evader_step(
    ppo: EvaderPPO,
    evader_xy: torch.Tensor,
    evader_vel: torch.Tensor,
    uav_pos: torch.Tensor,
    *,
    boundary: float,
    dt: float,
    max_speed: float = 1.4,
    deterministic: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return new_xy, new_vel, action, logp, value."""
    obs = EvaderPPO.build_obs(evader_xy, evader_vel, uav_pos)
    action, logp, value = ppo.act(obs, deterministic=deterministic)
    vel = evader_vel + action * dt * 4.0
    speed = vel.norm().clamp(min=1e-6)
    if float(speed) > max_speed:
        vel = vel / speed * max_speed
    xy = (evader_xy + vel * dt).clamp(-boundary, boundary)
    return xy, vel, action, logp, value


__all__ = [
    "EvaderPPO",
    "load_evader",
    "rl_evader_step",
    "sample_evader",
    "pursuit_rewards",
    "encirclement_score",
]

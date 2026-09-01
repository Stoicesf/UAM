"""Verify navigation_obstacle scenario spawns for all test densities."""
from __future__ import annotations

import time

from env.vmas_env import make_torchrl_env

for n_obs in (0, 2, 4, 6, 8):
    cfg = {
        "scenario": "navigation_obstacle",
        "num_agents": 4,
        "num_envs": 2,
        "num_obstacles": n_obs,
        "max_steps": 128,
        "device": "cpu",
        "continuous_actions": True,
        "world_spawning_x": 1.5 + 0.05 * n_obs,
        "world_spawning_y": 1.5 + 0.05 * n_obs,
    }
    t0 = time.time()
    env = make_torchrl_env(cfg)
    td = env.reset()
    obs_dim = td["agents", "observation"].shape[-1]
    dt = time.time() - t0
    print(f"obs={n_obs}: reset OK in {dt:.2f}s, obs_dim={obs_dim}")

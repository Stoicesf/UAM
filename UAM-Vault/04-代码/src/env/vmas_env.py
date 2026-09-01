"""VMAS 环境 — 基于 TorchRL 官方 VmasEnv 封装。"""

from __future__ import annotations

from typing import Any

import torch
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv


def _resolve_scenario(cfg: dict[str, Any]):
    name = cfg.get("scenario", "navigation")
    if name == "navigation_obstacle":
        from env.scenarios.navigation_obstacle import Scenario

        return Scenario()
    return name


def make_torchrl_env(cfg: dict[str, Any]) -> TransformedEnv:
    """创建 TorchRL VMAS 环境 (navigation / navigation_obstacle)。"""
    device = cfg.get("device", "cpu")
    if isinstance(device, str) and device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    scenario = _resolve_scenario(cfg)
    scenario_kwargs = dict(cfg.get("scenario_kwargs") or {})
    if cfg.get("num_obstacles") is not None:
        scenario_kwargs.setdefault("n_obstacles", cfg["num_obstacles"])
    if cfg.get("world_spawning_x") is not None:
        scenario_kwargs.setdefault("world_spawning_x", cfg["world_spawning_x"])
    if cfg.get("world_spawning_y") is not None:
        scenario_kwargs.setdefault("world_spawning_y", cfg["world_spawning_y"])

    base = VmasEnv(
        scenario=scenario,
        num_envs=cfg["num_envs"],
        continuous_actions=cfg.get("continuous_actions", True),
        max_steps=cfg["max_steps"],
        device=device,
        n_agents=cfg["num_agents"],
        scenario_kwargs=scenario_kwargs or None,
    )
    return TransformedEnv(
        base,
        RewardSum(
            in_keys=[base.reward_key],
            out_keys=[("agents", "episode_reward")],
        ),
    )

"""Baseline MAPPO 训练入口。"""

from __future__ import annotations

from algorithms.baseline.runner import train_mappo


def run_baseline(train_cfg: dict, env_cfg: dict, run_name: str = "mappo_baseline"):
    return train_mappo(train_cfg, env_cfg, run_name=run_name)

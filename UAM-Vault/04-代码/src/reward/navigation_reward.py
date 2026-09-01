"""导航奖励 — 到达目标。"""

from __future__ import annotations

import torch


def goal_reward(reached: torch.Tensor, reward_scale: float = 1.0) -> torch.Tensor:
    """VMAS navigation 场景通常内置 goal reward，此处供自定义扩展。"""
    return reached.float() * reward_scale

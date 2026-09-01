"""DSGF-HRL — 在 MAPPO 基础上仅增加 Graph + DSGF + Guidance Reward。

训练流程:
    VMAS → 状态 → 构建通信图 → DSGF → Φ → Actor(obs+Φ) → Action
         → Environment → Reward(Goal+Collision+Guide) → MAPPO Update

MAPPO 更新逻辑不变，真正新增的只有 guidance/ 和 reward/guidance_reward.py。
"""

from __future__ import annotations

from typing import Any

import torch

from guidance.dsfg_encoder import DSGFEncoder
from guidance.guidance_field import fuse_obs_guidance
from reward.guidance_reward import guidance_alignment_reward


class DSFGHRLTrainer:
    """DSGF 层级强化学习训练器 — 包装 baseline MAPPO。"""

    def __init__(
        self,
        mappo_trainer,
        dsfg_encoder: DSGFEncoder,
        cfg: dict[str, Any],
    ):
        self.mappo = mappo_trainer
        self.dsfg = dsfg_encoder
        self.cfg = cfg
        self.use_guidance_reward = cfg.get("use_guidance_reward", True)
        self.guidance_reward_coef = cfg.get("guidance_reward_coef", 0.1)

    def compute_actor_input(
        self,
        obs: torch.Tensor,
        node_features: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """obs + Φ → Actor 输入 (Day 4 核心改动)。"""
        phi = self.dsfg(node_features, positions)
        return fuse_obs_guidance(obs, phi), phi

    def augment_reward(
        self,
        base_reward: torch.Tensor,
        actions: torch.Tensor,
        phi: torch.Tensor,
    ) -> torch.Tensor:
        """R = Goal + Collision + Guide (Day 6)。"""
        if not self.use_guidance_reward:
            return base_reward
        guide = guidance_alignment_reward(actions, phi)
        return base_reward + self.guidance_reward_coef * guide

    def train(self, total_timesteps: int):
        raise NotImplementedError("Week 2-3: 接入 baseline MAPPO 后实现")

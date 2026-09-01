"""策略封装 — Actor + Critic，支持 baseline / DSGF 两种输入维度。"""

from __future__ import annotations

import torch.nn as nn

from guidance.guidance_field import fuse_obs_guidance
from models.actor import Actor
from models.critic import Critic


class MAPPOPolicy(nn.Module):
    """MAPPO 策略 — Baseline 用 obs_dim，DSGF 用 obs_dim + guidance_dim。"""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        guidance_dim: int = 0,
        hidden_dim: int = 128,
        centralized_critic: bool = True,
    ):
        super().__init__()
        self.use_guidance = guidance_dim > 0
        actor_input = obs_dim + guidance_dim
        self.actor = Actor(actor_input, action_dim, hidden_dim)
        critic_input = obs_dim  # TODO: centralized critic 拼接所有 agent obs
        self.critic = Critic(critic_input, hidden_dim)

    def get_action(self, obs, phi=None):
        x = fuse_obs_guidance(obs, phi) if phi is not None else obs
        dist = self.actor(x)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        value = self.critic(x if self.use_guidance else obs)
        return action, log_prob, value

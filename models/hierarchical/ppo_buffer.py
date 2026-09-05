"""Rollout buffer for PPO (decision- or step-level)."""

from __future__ import annotations

from typing import Any

import torch


class RolloutBuffer:
    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.states: list[torch.Tensor] = []
        self.types: list[list[int]] = []
        self.actions: list[torch.Tensor] = []
        self.log_probs: list[torch.Tensor] = []
        self.rewards: list[float] = []
        self.dones: list[float] = []
        self.values: list[torch.Tensor] = []
        self.obs: list[torch.Tensor] = []
        self.roles: list[torch.Tensor] = []
        self.rewards_vec: list[torch.Tensor] = []
        self.values_vec: list[torch.Tensor] = []
        self.log_probs_vec: list[torch.Tensor] = []

    def add_high(
        self,
        state: torch.Tensor,
        uav_types: list[int],
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: float,
        done: bool,
        value: torch.Tensor,
    ) -> None:
        self.states.append(state.detach().float().cpu())
        self.types.append(list(uav_types))
        self.actions.append(action.detach().cpu())
        self.log_probs.append(log_prob.detach().cpu())
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.values.append(value.detach().float().cpu().view(()))

    def add_low(
        self,
        obs: torch.Tensor,
        roles: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        done: bool,
        value: torch.Tensor,
    ) -> None:
        self.obs.append(obs.detach().float().cpu())
        self.roles.append(roles.detach().cpu())
        self.actions.append(action.detach().cpu())
        self.log_probs.append(log_prob.detach().cpu())
        # store mean reward for GAE scalar path; keep per-agent in rewards_vec if needed
        self.rewards.append(float(reward.mean().item()))
        self.dones.append(float(done))
        self.values.append(value.detach().float().mean().cpu().view(()))
        self.rewards_vec.append(reward.detach().float().cpu())
        self.values_vec.append(value.detach().float().cpu())
        self.log_probs_vec.append(log_prob.detach().float().cpu())

    def compute_gae(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        last_value: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """GAE over scalar reward/value sequence. Returns (advantages, returns)."""
        T = len(self.rewards)
        advantages = torch.zeros(T)
        gae = 0.0
        values = [float(v.item()) for v in self.values] + [last_value]
        for t in reversed(range(T)):
            mask = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * values[t + 1] * mask - values[t]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages[t] = gae
        values_t = torch.tensor([float(v.item()) for v in self.values], dtype=torch.float32)
        returns = advantages + values_t
        return advantages, returns

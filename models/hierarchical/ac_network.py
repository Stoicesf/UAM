"""Actor-Critic networks for hierarchical PPO (type-conditioned high + low)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class RoleActorCritic(nn.Module):
    """High-level role selector with shared trunk + critic; conditioned on UAV type."""

    def __init__(
        self,
        global_obs_dim: int,
        n_roles: int = 3,
        n_types: int = 3,
        type_embed_dim: int = 8,
        decision_interval: int = 50,
        hidden: int = 128,
    ):
        super().__init__()
        self.n_roles = n_roles
        self.decision_interval = decision_interval
        self.type_embed = nn.Embedding(n_types, type_embed_dim)
        self.shared = nn.Sequential(
            nn.Linear(global_obs_dim + type_embed_dim, hidden),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, n_roles)
        self.critic = nn.Linear(hidden, 1)

    def _features(self, global_obs: torch.Tensor, uav_types: list[int] | torch.Tensor) -> torch.Tensor:
        if isinstance(uav_types, torch.Tensor):
            types = uav_types.long().to(global_obs.device)
        else:
            types = torch.tensor(list(uav_types), dtype=torch.long, device=global_obs.device)
        n = types.shape[0]
        g = global_obs.float().view(1, -1).expand(n, -1)
        return self.shared(torch.cat([g, self.type_embed(types)], dim=-1))

    def forward(
        self, global_obs: torch.Tensor, uav_types: list[int] | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns logits [n, n_roles], values [n, 1]."""
        h = self._features(global_obs, uav_types)
        return self.actor(h), self.critic(h)

    def get_action(
        self,
        global_obs: torch.Tensor,
        uav_types: list[int] | torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample roles. Returns (indices, log_prob_sum, entropy_sum, value_mean)."""
        logits, values = self.forward(global_obs, uav_types)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            indices = logits.argmax(dim=-1)
        else:
            indices = dist.sample()
        log_prob = dist.log_prob(indices).sum()
        entropy = dist.entropy().sum()
        value = values.mean()
        return indices, log_prob, entropy, value

    def evaluate(
        self,
        global_obs: torch.Tensor,
        uav_types: list[int] | torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (log_prob_sum, entropy_sum, value_mean) for PPO ratio."""
        logits, values = self.forward(global_obs, uav_types)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions).sum(), dist.entropy().sum(), values.mean()


class LowLevelActorCritic(nn.Module):
    """Low-level Gaussian motion policy + value head, conditioned on role."""

    def __init__(
        self,
        obs_dim: int,
        role_embed_dim: int = 16,
        n_roles: int = 3,
        action_std: float = 0.1,
        hidden: int = 256,
    ):
        super().__init__()
        self.role_embed = nn.Embedding(n_roles, role_embed_dim)
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + role_embed_dim, hidden),
            nn.ReLU(),
        )
        self.motion = nn.Linear(hidden, 2)
        self.critic = nn.Linear(hidden, 1)
        self.register_buffer("log_std", torch.full((2,), math.log(action_std)))

    def _trunk(self, obs: torch.Tensor, role_indices: torch.Tensor) -> torch.Tensor:
        return self.trunk(torch.cat([obs, self.role_embed(role_indices)], dim=-1))

    def mean(self, obs: torch.Tensor, role_indices: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.motion(self._trunk(obs, role_indices)))

    def forward(self, obs: torch.Tensor, role_indices: torch.Tensor) -> torch.Tensor:
        return self.mean(obs, role_indices)

    def value(self, obs: torch.Tensor, role_indices: torch.Tensor) -> torch.Tensor:
        return self.critic(self._trunk(obs, role_indices)).squeeze(-1)

    def act(
        self, obs: torch.Tensor, role_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (dxdy, log_prob [n], value [n])."""
        h = self._trunk(obs, role_indices)
        mu = torch.tanh(self.motion(h))
        dist = torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))
        raw = dist.rsample()
        actions = raw.clamp(-1.0, 1.0)
        log_prob = dist.log_prob(raw).sum(dim=-1)
        value = self.critic(h).squeeze(-1)
        return actions, log_prob, value

    def evaluate(
        self, obs: torch.Tensor, role_indices: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (log_prob [n], entropy [n], value [n])."""
        h = self._trunk(obs, role_indices)
        mu = torch.tanh(self.motion(h))
        dist = torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(h).squeeze(-1)
        return log_prob, entropy, value


def self_check() -> None:
    ac = RoleActorCritic(6, n_roles=3)
    g = torch.randn(6)
    types = [0, 1, 2, 2]
    roles, lp, ent, v = ac.get_action(g, types)
    assert roles.shape == (4,) and lp.ndim == 0
    low = LowLevelActorCritic(32, n_roles=3)
    a, lp2, val = low.act(torch.randn(4, 32), roles)
    assert a.shape == (4, 2) and lp2.shape == (4,) and val.shape == (4,)
    print("ac_network: OK")


if __name__ == "__main__":
    self_check()

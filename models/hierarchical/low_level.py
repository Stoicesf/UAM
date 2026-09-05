"""Low-level motion policy conditioned on role embedding."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LowLevelConditionalPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        role_embed_dim: int = 16,
        n_roles: int = 3,
        action_std: float = 0.1,
    ):
        super().__init__()
        self.role_embed = nn.Embedding(n_roles, role_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(obs_dim + role_embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )
        # ponytail: fixed isotropic Gaussian; upgrade to state-dependent std if needed
        self.register_buffer("log_std", torch.full((2,), math.log(action_std)))

    def mean(self, obs: torch.Tensor, role_indices: torch.Tensor) -> torch.Tensor:
        role_emb = self.role_embed(role_indices)
        return torch.tanh(self.net(torch.cat([obs, role_emb], dim=-1)))

    def forward(self, obs: torch.Tensor, role_indices: torch.Tensor) -> torch.Tensor:
        return self.mean(obs, role_indices)

    def act(self, obs: torch.Tensor, role_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample dxdy + per-agent log_prob (sum over action dims)."""
        mu = self.mean(obs, role_indices)
        dist = torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))
        raw = dist.rsample()
        actions = raw.clamp(-1.0, 1.0)
        log_prob = dist.log_prob(raw).sum(dim=-1)
        return actions, log_prob


def self_check() -> None:
    p = LowLevelConditionalPolicy(32, n_roles=3)
    obs = torch.randn(4, 32)
    roles = torch.zeros(4, dtype=torch.long)
    a, lp = p.act(obs, roles)
    assert a.shape == (4, 2) and lp.shape == (4,)
    print("low_level: OK")


if __name__ == "__main__":
    self_check()

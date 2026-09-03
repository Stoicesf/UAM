"""Lightweight role + motion policy head."""

from __future__ import annotations

import torch
import torch.nn as nn


class RolePolicy(nn.Module):
    def __init__(self, obs_dim: int, n_roles: int = 3, hidden: int = 64):
        super().__init__()
        self.n_roles = n_roles
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.motion = nn.Linear(hidden, 2)
        self.role = nn.Linear(hidden, n_roles)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(obs)
        dxdy = torch.tanh(self.motion(h))
        role_logits = self.role(h)
        actions = torch.cat([dxdy, role_logits], dim=-1)
        roles = role_logits.argmax(dim=-1)
        return actions, roles


def self_check() -> None:
    p = RolePolicy(32, 3)
    a, r = p(torch.randn(4, 32))
    assert a.shape == (4, 5) and r.shape == (4,)
    print("role_policy: OK")


if __name__ == "__main__":
    self_check()

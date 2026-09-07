"""Attractive-potential formation (XDU §3.2/3.3), 2D."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class AttractivePotentialFormation(nn.Module):
    """
    Desired follower acceleration from leader + formation offsets.
    delta is relative to leader (payload / ring center), shape [N, 2].
    """

    def __init__(
        self,
        n_agents: int,
        eta: float = 3.0,
        beta: float = 2.0,
        formation_delta: torch.Tensor | None = None,
        radius: float = 1.5,
    ):
        super().__init__()
        self.n = int(n_agents)
        self.eta = float(eta)
        self.beta = float(beta)
        if formation_delta is None:
            angles = torch.linspace(0, 2 * math.pi, self.n + 1)[:-1]
            delta = torch.stack(
                [radius * torch.cos(angles), radius * torch.sin(angles)], dim=-1
            )
        else:
            delta = formation_delta.float()
        self.register_buffer("delta", delta)

    def set_delta(self, delta: torch.Tensor) -> None:
        """Update formation offsets (e.g. from env._ideal_ring - leader)."""
        self.delta = delta.detach().float()

    def forward(
        self,
        leader_pos: torch.Tensor,
        leader_vel: torch.Tensor,
        follower_pos: torch.Tensor,
        follower_vel: torch.Tensor,
    ) -> torch.Tensor:
        """Return desired acceleration u_i [N, 2]."""
        desired_pos = leader_pos.view(1, 2) + self.delta
        desired_vel = leader_vel.view(1, 2).expand(self.n, -1)
        pos_error = desired_pos - follower_pos
        vel_error = desired_vel - follower_vel
        return self.eta * pos_error + self.beta * vel_error


def self_check() -> None:
    n = 4
    form = AttractivePotentialFormation(n, radius=2.0)
    leader = torch.zeros(2)
    lvel = torch.zeros(2)
    # followers offset from desired
    pos = form.delta + 0.5
    vel = torch.zeros(n, 2)
    acc = form(leader, lvel, pos, vel)
    assert acc.shape == (n, 2)
    # should pull back toward delta (negative if pos = delta+0.5)
    assert float(acc.mean()) < 0
    print("attractive_potential: OK")


if __name__ == "__main__":
    self_check()

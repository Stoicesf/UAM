"""Learnable Boids-style local interaction rules."""

from __future__ import annotations

import torch
import torch.nn as nn

from dice.local_obs import neighbor_mask
from dice.safety_shield import SafetyShield


class LocalRules(nn.Module):
    def __init__(self, comm_radius: float = 2.0, min_distance: float = 0.5):
        super().__init__()
        self.w_separation = nn.Parameter(torch.tensor(1.5))
        self.w_alignment = nn.Parameter(torch.tensor(1.0))
        self.w_cohesion = nn.Parameter(torch.tensor(1.0))
        self.w_goal = nn.Parameter(torch.tensor(1.2))
        self.comm_radius = comm_radius
        self.min_distance = min_distance

    def forward(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor,
        goal: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is None:
            mask = neighbor_mask(pos, self.comm_radius)
        sep = self._separation(pos, mask)
        ali = self._alignment(vel, mask)
        coh = self._cohesion(pos, mask)
        glo = self._goal_attraction(pos, goal)
        return (
            self.w_separation * sep
            + self.w_alignment * ali
            + self.w_cohesion * coh
            + self.w_goal * glo
        )

    def _separation(self, pos: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)
        dist = diff.norm(dim=-1, keepdim=True).clamp(min=1e-4)
        close = (dist.squeeze(-1) < self.min_distance * 2) & mask
        force = (diff / (dist**2)) * close.unsqueeze(-1).float()
        return force.sum(dim=1)

    def _alignment(self, vel: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        deg = mask.float().sum(-1, keepdim=True).clamp(min=1)
        mean_v = (mask.float() @ vel) / deg
        return mean_v - vel

    def _cohesion(self, pos: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        deg = mask.float().sum(-1, keepdim=True).clamp(min=1)
        center = (mask.float() @ pos) / deg
        return center - pos

    def _goal_attraction(self, pos: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return goal - pos


def simulate_stable(
    n: int = 16,
    steps: int = 500,
    seed: int = 0,
) -> bool:
    """Hardcoded rules (no training) collision-stable for `steps`."""
    torch.manual_seed(seed)
    rules = LocalRules()
    pos = (torch.rand(n, 2) * 2 - 1) * 3
    vel = torch.zeros(n, 2)
    goal = torch.zeros(n, 2)
    for _ in range(steps):
        acc = rules(pos, vel, goal).clamp(-2, 2)
        vel = 0.9 * vel + 0.1 * acc
        pos = pos + 0.1 * vel
        pos, vel = SafetyShield.apply(pos, vel, boundary=5.0, min_distance=0.4)
        d = torch.cdist(pos, pos) + torch.eye(n) * 10
        if (d < 0.25).any():
            return False
    return True


def self_check() -> None:
    assert simulate_stable(16, 500)
    r = LocalRules()
    assert any(p.requires_grad for p in r.parameters())
    print("local_rules: OK (500-step stable)")


if __name__ == "__main__":
    self_check()

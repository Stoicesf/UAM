"""Joint allocation of [bandwidth, sensing_freq, compute_quota] (+ optional semantic tier)."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
import torch.nn as nn


@dataclass
class ResourceState:
    bandwidth_hz: float = 20e6
    sense_hz: float = 10.0
    tops_quota: float = 0.5
    budget_ratio: float = 0.5
    model_tier: int = 1
    semantic_level: int = 1  # 1=L1, 2=L2, 3=L3 (SEM extension)


@dataclass
class ResourceLimits:
    bandwidth_max: float = 20e6
    sense_min: float = 5.0
    sense_max: float = 30.0
    tops_max: float = 100.0
    budget_max: float = 1.0


def to_vector(state: ResourceState, limits: ResourceLimits | None = None) -> torch.Tensor:
    """Normalized 3D resource vector [bw, sense, compute]."""
    limits = limits or ResourceLimits()
    return torch.tensor(
        [
            state.bandwidth_hz / limits.bandwidth_max,
            state.sense_hz / limits.sense_max,
            state.tops_quota / limits.tops_max,
        ]
    )


def joint_allocate(
    u: torch.Tensor,
    q_perception: torch.Tensor,
    latency_ms: float,
    *,
    state: ResourceState | None = None,
    limits: ResourceLimits | None = None,
    latency_limit_ms: float = 50.0,
) -> ResourceState:
    state = state or ResourceState()
    limits = limits or ResourceLimits()
    u_m = float(u.float().mean())
    q_m = float(q_perception.float().mean())

    budget = min(limits.budget_max, max(0.1, state.budget_ratio + 0.4 * (u_m - 0.5)))
    sense = min(limits.sense_max, max(limits.sense_min, state.sense_hz + 15.0 * (0.5 - q_m)))
    bw = min(limits.bandwidth_max, state.bandwidth_hz * (0.5 + budget))
    tops = min(limits.tops_max, max(0.5, state.tops_quota * (1.2 if latency_ms > latency_limit_ms else 1.0)))
    tier = state.model_tier
    if latency_ms > latency_limit_ms:
        tier = max(0, tier - 1)
    elif latency_ms < 0.5 * latency_limit_ms and u_m > 0.6:
        tier = min(2, tier + 1)
    # low bandwidth → prefer L1 semantic
    sem = 1 if bw < 0.5 * limits.bandwidth_max else (3 if bw > 0.8 * limits.bandwidth_max else 2)

    return ResourceState(
        bandwidth_hz=bw,
        sense_hz=sense,
        tops_quota=tops,
        budget_ratio=budget,
        model_tier=tier,
        semantic_level=sem,
    )


def resource_efficiency(state: ResourceState, limits: ResourceLimits | None = None) -> float:
    limits = limits or ResourceLimits()
    v = to_vector(state, limits)
    # target mid-high usage ~0.7–0.9
    return float(v.mean().clamp(0, 1))


def reward_nav_minus_resources(
    nav_reward: torch.Tensor,
    bandwidth_usage: float,
    compute_usage: float,
    *,
    lambda1: float = 0.05,
    lambda2: float = 0.05,
    semantic_loss: float = 0.0,
    alpha_sem: float = 0.0,
) -> torch.Tensor:
    """R = R_nav - λ1*bw - λ2*compute - α*semantic_loss."""
    return nav_reward - lambda1 * bandwidth_usage - lambda2 * compute_usage - alpha_sem * semantic_loss


def reward_with_resource(
    nav_reward: torch.Tensor,
    efficiency: float,
    lambda_r: float = 0.01,
) -> torch.Tensor:
    return nav_reward + lambda_r * efficiency


class JointResourcePolicy(nn.Module):
    """Maps obs → Δresource (3) + optional semantic level logits (3). Extends action space."""

    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.delta = nn.Linear(hidden, 3)  # bw, sense, compute deltas in [-1,1]
        self.sem = nn.Linear(hidden, 3)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(obs)
        return torch.tanh(self.delta(h)), self.sem(h)

    def apply_delta(
        self,
        state: ResourceState,
        delta: torch.Tensor,
        sem_logits: torch.Tensor,
        limits: ResourceLimits | None = None,
    ) -> ResourceState:
        limits = limits or ResourceLimits()
        d = delta.detach().cpu().flatten()
        bw_n = max(0.05, min(1.0, state.bandwidth_hz / limits.bandwidth_max + 0.1 * float(d[0])))
        sense_n = max(
            limits.sense_min / limits.sense_max,
            min(1.0, state.sense_hz / limits.sense_max + 0.1 * float(d[1])),
        )
        tops_n = max(0.01, min(1.0, state.tops_quota / limits.tops_max + 0.1 * float(d[2])))
        bw = bw_n * limits.bandwidth_max
        sense = sense_n * limits.sense_max
        tops = tops_n * limits.tops_max
        sem = int(sem_logits.detach().cpu().flatten().argmax()) + 1
        return ResourceState(
            bandwidth_hz=bw,
            sense_hz=sense,
            tops_quota=tops,
            budget_ratio=min(1.0, bw / limits.bandwidth_max),
            model_tier=state.model_tier,
            semantic_level=sem,
        )


def self_check() -> None:
    u = torch.tensor([0.9, 0.8])
    q = torch.tensor([0.1, 0.2])
    s = joint_allocate(u, q, latency_ms=80.0)
    assert s.sense_hz >= ResourceLimits().sense_min
    v = to_vector(s)
    assert v.shape == (3,)
    pol = JointResourcePolicy(16)
    d, sem = pol(torch.randn(2, 16))
    s2 = pol.apply_delta(s, d[0], sem[0])
    r = reward_nav_minus_resources(torch.tensor(1.0), 0.5, 0.2)
    assert float(r) < 1.0
    _ = fields(ResourceState)
    print(f"joint_allocator: OK (vec={v.tolist()}, sem={s2.semantic_level})")


if __name__ == "__main__":
    self_check()

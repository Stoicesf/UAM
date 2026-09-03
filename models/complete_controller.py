"""Unified ICPS + DICE + Semantic controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from dice.local_rules import LocalRules
from dice.safety_shield import SafetyShield
from models.communication.semantic_gate import SemanticGate, semantic_gain
from models.communication.semantic_reconstructor import SemanticReconstructor
from models.communication.comm_delay import CommDelayBuffer
from models.hierarchical_controller import HierarchicalController
from models.icps.compute.resource_simulator import ComputeBudgetEnv, TIER_PROFILES
from models.icps.resource.joint_allocator import (
    JointResourcePolicy,
    ResourceLimits,
    ResourceState,
    joint_allocate,
    resource_efficiency,
    reward_nav_minus_resources,
    to_vector,
)
from models.semantic.encoder import SemanticEncoder


@dataclass
class StepResult:
    dxdy: torch.Tensor
    roles: torch.Tensor
    assignment: torch.Tensor
    resource: ResourceState
    send: torch.Tensor
    semantic_level: torch.Tensor
    delay_ms: float
    efficiency: float


class CompleteController(nn.Module):
    def __init__(
        self,
        n_agents: int,
        n_tasks: int,
        obs_dim: int,
        *,
        n_roles: int = 3,
        compute_tier: str = "mid",
        use_semantic: bool = True,
        use_icps_resource: bool = True,
        use_dice: bool = True,
        use_projection_shield: bool = False,
        shield_type: str | None = None,  # None→from use_projection_shield; hard|cbf|none
        comm_delay: int = 0,
    ):
        super().__init__()
        self.use_semantic = use_semantic
        self.use_icps = use_icps_resource
        self.use_dice = use_dice
        if shield_type is None:
            self.shield_type = "hard" if use_projection_shield else "none"
        else:
            self.shield_type = shield_type
        self.use_projection_shield = self.shield_type in ("hard", "cbf")
        self.hier = HierarchicalController(n_agents, n_tasks, n_roles)
        self.rules = LocalRules()
        self.encoder = SemanticEncoder(obs_dim, z_dim=16)
        self.gate = SemanticGate(n_roles=n_roles)
        self.recon = SemanticReconstructor(z_dim=16)
        self.res_policy = JointResourcePolicy(obs_dim)
        self.compute = ComputeBudgetEnv(compute_tier, model_params=5e5, obs_dim=obs_dim)
        self.limits = ResourceLimits()
        self.resource = ResourceState(tops_quota=TIER_PROFILES[compute_tier].tops)
        self.comm_delay = CommDelayBuffer(comm_delay)
        self._cbf = None
        if self.shield_type == "cbf":
            from dice.cbf_projection import CBFProjector

            self._cbf = CBFProjector(min_distance=0.5, alpha=1.0)

    def step(
        self,
        obs: torch.Tensor,
        pos: torch.Tensor,
        vel: torch.Tensor,
        tasks: torch.Tensor,
        roles: torch.Tensor,
        alive: torch.Tensor,
        done_mask: torch.Tensor,
        assignment: torch.Tensor,
        *,
        snr_db: float = 20.0,
        u: torch.Tensor | None = None,
        q_perc: torch.Tensor | None = None,
        gate_trace: list[dict[str, Any]] | None = None,
        soft_bw_bias: float | None = None,
        use_mlp_levels: bool | None = None,
    ) -> StepResult:
        delay = self.compute.before_decision(real_sleep=False)
        n = pos.shape[0]
        if u is None:
            u = torch.rand(n, device=pos.device) * 0.5
        if q_perc is None:
            q_perc = torch.ones(n, device=pos.device) * 0.7

        if self.use_dice:
            dxdy, roles, assignment, _ = self.hier.step(
                pos, vel, tasks, roles, alive, done_mask, assignment
            )
        else:
            goals = torch.zeros_like(pos)
            dxdy = self.rules(pos, vel, goals).clamp(-1, 1)

        if self.use_icps:
            self.resource = joint_allocate(u, q_perc, delay, state=self.resource, limits=self.limits)
            # learnable residual resource tweak from mean obs
            delta, sem_logits = self.res_policy(obs.mean(dim=0, keepdim=True))
            self.resource = self.res_policy.apply_delta(self.resource, delta[0], sem_logits[0], self.limits)

        send = torch.zeros(n, dtype=torch.bool, device=pos.device)
        levels = torch.zeros(n, dtype=torch.long, device=pos.device)
        if self.use_semantic:
            z, recon, mu, logvar = self.encoder(obs)
            # entropy proxies
            prior_h = torch.full((n,), 1.5, device=pos.device)
            post_h = 0.5 + 0.01 * (obs - recon).norm(dim=-1)
            gain = semantic_gain(prior_h, post_h)
            snr_t = torch.full((n,), snr_db, device=pos.device)
            kw: dict[str, Any] = dict(
                roles=roles,
                trace=gate_trace,
            )
            if soft_bw_bias is not None:
                kw["soft_bw_bias"] = soft_bw_bias
            if use_mlp_levels is not None:
                kw["use_mlp_levels"] = use_mlp_levels
            send, levels = self.gate.decide(
                gain,
                snr_t,
                self.resource.bandwidth_hz,
                self.limits.bandwidth_max,
                u,
                **kw,
            )
            send, levels = self.comm_delay.push(send, levels)
            # sync resource semantic level with majority
            if send.any():
                self.resource.semantic_level = int(levels[send].float().mean().round().clamp(1, 3))

        # apply shield to dxdy (was: apply then discard)
        if self.shield_type == "cbf":
            from dice.cbf_projection import project_dxdy_cbf

            dxdy = project_dxdy_cbf(dxdy, pos, vel)
        elif self.shield_type == "hard":
            from dice.safety_shield_projection import project_action

            dxdy = project_action(dxdy, pos)
        else:
            proposed = pos + dxdy
            pos_s, _ = SafetyShield.apply(proposed, vel)
            dxdy = (pos_s - pos).clamp(-1.0, 1.0)

        eff = resource_efficiency(self.resource, self.limits)
        return StepResult(
            dxdy=dxdy,
            roles=roles,
            assignment=assignment,
            resource=self.resource,
            send=send,
            semantic_level=levels,
            delay_ms=delay,
            efficiency=eff,
        )


def self_check() -> None:
    n, obs_dim = 8, 32
    ctl = CompleteController(n, 3, obs_dim)
    obs = torch.randn(n, obs_dim)
    pos = torch.randn(n, 2)
    vel = torch.zeros(n, 2)
    tasks = torch.randn(3, 4)
    tasks[:, 2] = 1
    roles = torch.zeros(n, dtype=torch.long)
    alive = torch.ones(n, dtype=torch.bool)
    done = torch.zeros(3, dtype=torch.bool)
    asn = torch.full((n,), -1, dtype=torch.long)
    out = ctl.step(obs, pos, vel, tasks, roles, alive, done, asn, snr_db=5.0)
    assert out.dxdy.shape == (n, 2)
    ctl_cbf = CompleteController(n, 3, obs_dim, shield_type="cbf")
    out2 = ctl_cbf.step(obs, pos, vel, tasks, roles, alive, done, asn, snr_db=5.0)
    assert out2.dxdy.shape == (n, 2)
    r = reward_nav_minus_resources(torch.tensor(1.0), 0.4, 0.2)
    _ = to_vector(out.resource)
    print(f"complete_controller: OK (eff={out.efficiency:.2f}, delay={out.delay_ms:.2f}, cbf_ok)")


if __name__ == "__main__":
    self_check()

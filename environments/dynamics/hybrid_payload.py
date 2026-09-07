"""Hybrid slack/taut cable payload dynamics (2D, HIT-style impact reset)."""

from __future__ import annotations

import torch
import torch.nn as nn


class HybridPayloadDynamics(nn.Module):
    """Per-cable slack/taut hybrid; g=0 aligns with 2D top-down transport demo."""

    def __init__(
        self,
        mass: float = 5.0,
        spring_k: float = 50.0,
        damping_d: float = 5.0,
        L0: float = 2.0,
        hysteresis: float = 0.05,
        g: float = 0.0,
    ):
        super().__init__()
        self.mass = float(mass)
        self.k = float(spring_k)
        self.d = float(damping_d)
        self.L0 = float(L0)
        self.hysteresis = float(hysteresis)
        self.g = float(g)
        # resized on first forward / reset(n)
        self.register_buffer("state", torch.ones(1, dtype=torch.long))

    def reset_state(self, n: int, device: torch.device | None = None) -> None:
        """Reset all cables to taut."""
        self.state = torch.ones(n, dtype=torch.long, device=device or self.state.device)

    def _ensure_state(self, n: int, device: torch.device) -> None:
        if self.state.numel() != n or self.state.device != device:
            self.state = torch.ones(n, dtype=torch.long, device=device)

    def forward(
        self,
        uav_pos: torch.Tensor,
        uav_vel: torch.Tensor,
        payload_pos: torch.Tensor,
        payload_vel: torch.Tensor,
        dt: float = 0.05,
    ) -> dict[str, torch.Tensor]:
        """
        uav_pos/vel: [n, 2]; payload_pos/vel: [2]
        Returns keys compatible with PayloadDynamics plus state/transition.
        """
        n = uav_pos.shape[0]
        self._ensure_state(n, uav_pos.device)

        diff = uav_pos - payload_pos.view(1, 2)
        dist = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-6)

        slack_cond = (dist < (self.L0 - self.hysteresis)).squeeze(-1)
        taut_cond = (dist > (self.L0 + self.hysteresis)).squeeze(-1)
        new_state = self.state.clone()
        new_state[slack_cond] = 0
        new_state[taut_cond] = 1

        trans_mask = (self.state == 0) & (new_state == 1)
        if bool(trans_mask.any()):
            payload_vel = self._impact_reset(payload_vel, uav_vel, diff, dist, trans_mask)
        self.state = new_state

        tension = torch.zeros_like(dist)
        taut_idx = self.state == 1
        if bool(taut_idx.any()):
            delta_L = dist[taut_idx] - self.L0
            rel_vel = uav_vel[taut_idx] - payload_vel.view(1, 2)
            direction = diff[taut_idx] / dist[taut_idx]
            delta_v = (rel_vel * direction).sum(dim=-1, keepdim=True)
            tension_taut = torch.clamp(self.k * delta_L + self.d * delta_v, min=0.0)
            tension[taut_idx] = tension_taut

        direction_all = diff / dist
        force_vectors = tension * direction_all
        total_force = force_vectors.sum(dim=0)
        # g reserved for future 3D; 2D demo keeps g=0
        gravity = payload_pos.new_tensor([0.0, -self.g]) * self.mass
        drag = -0.5 * payload_vel
        acc = (total_force + gravity + drag) / self.mass

        new_vel = payload_vel + acc * dt
        new_pos = payload_pos + new_vel * dt
        return {
            "new_payload_pos": new_pos,
            "new_payload_vel": new_vel,
            "tensions": tension,
            "forces": force_vectors,
            "payload_acc": acc,
            "state": self.state.clone(),
            "transition": trans_mask,
        }

    def _impact_reset(
        self,
        payload_vel: torch.Tensor,
        uav_vel: torch.Tensor,
        diff: torch.Tensor,
        dist: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Inelastic along-cable velocity reset (2D projection)."""
        q = diff[mask] / dist[mask]
        v_uav_par = (uav_vel[mask] * q).sum(dim=-1, keepdim=True) * q
        v_load_par = (payload_vel.view(1, 2) * q).sum(dim=-1, keepdim=True) * q
        return payload_vel + (v_uav_par - v_load_par).mean(dim=0)


def self_check() -> None:
    pd = HybridPayloadDynamics(L0=2.0, hysteresis=0.05)
    n = 4
    ang = torch.linspace(0, 2 * 3.14159, n + 1)[:-1]
    # taut: ring slightly beyond L0
    uav_pos = torch.stack([2.2 * torch.cos(ang), 2.2 * torch.sin(ang)], dim=-1)
    uav_vel = torch.zeros(n, 2)
    payload_pos = torch.zeros(2)
    payload_vel = torch.zeros(2)
    out = pd(uav_pos, uav_vel, payload_pos, payload_vel, dt=0.05)
    assert out["tensions"].shape == (n, 1)
    assert out["forces"].shape == (n, 2)
    assert bool((out["tensions"] >= 0).all())
    assert bool((out["state"] == 1).all())
    assert not bool(torch.isnan(out["new_payload_pos"]).any())

    # slack: pull UAVs inward
    uav_slack = torch.stack([1.0 * torch.cos(ang), 1.0 * torch.sin(ang)], dim=-1)
    out2 = pd(uav_slack, uav_vel, payload_pos, payload_vel, dt=0.05)
    assert bool((out2["state"] == 0).all())
    assert bool((out2["tensions"] == 0).all())

    # re-taut: move out again with relative inward UAV vel for impact path
    uav_taut = torch.stack([2.3 * torch.cos(ang), 2.3 * torch.sin(ang)], dim=-1)
    uav_vel_in = -0.5 * (uav_taut / torch.norm(uav_taut, dim=-1, keepdim=True))
    out3 = pd(uav_taut, uav_vel_in, out2["new_payload_pos"], out2["new_payload_vel"], dt=0.05)
    assert bool((out3["state"] == 1).all())
    assert not bool(torch.isnan(out3["new_payload_vel"]).any())
    print("hybrid_payload: OK")


if __name__ == "__main__":
    self_check()

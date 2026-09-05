"""Cable-suspended payload dynamics (spring-damper, tension ≥ 0)."""

from __future__ import annotations

import torch
import torch.nn as nn


class PayloadDynamics(nn.Module):
    def __init__(
        self,
        mass: float = 5.0,
        spring_k: float = 50.0,
        damping_d: float = 5.0,
        L0: float = 2.0,
        # ponytail: 2D top-down — no world -Y gravity (would break ring lift); g kept for load share. Upgrade: 3D.
        gravity_scale: float = 0.0,
    ):
        super().__init__()
        self.mass = float(mass)
        self.k = float(spring_k)
        self.d = float(damping_d)
        self.L0 = float(L0)
        self.g = 9.81
        self.gravity_scale = float(gravity_scale)

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
        """
        diff = uav_pos - payload_pos.view(1, 2)
        dist = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-6)
        rel_vel = uav_vel - payload_vel.view(1, 2)
        delta_L = dist - self.L0
        delta_v = (rel_vel * diff / dist).sum(dim=-1, keepdim=True)

        tension = self.k * delta_L + self.d * delta_v
        tension = torch.clamp(tension, min=0.0)

        direction = diff / dist
        force_vectors = tension * direction
        total_force = force_vectors.sum(dim=0)
        gravity = payload_pos.new_tensor([0.0, -self.g * self.gravity_scale]) * self.mass
        # light viscous drag so free payload does not coast forever
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
        }


def self_check() -> None:
    pd = PayloadDynamics()
    n = 4
    # UAVs above payload on a ring of radius ~2 (near L0)
    ang = torch.linspace(0, 2 * 3.14159, n + 1)[:-1]
    uav_pos = torch.stack([2.2 * torch.cos(ang), 2.2 * torch.sin(ang)], dim=-1)
    uav_vel = torch.zeros(n, 2)
    payload_pos = torch.zeros(2)
    payload_vel = torch.zeros(2)
    out = pd(uav_pos, uav_vel, payload_pos, payload_vel, dt=0.05)
    assert out["tensions"].shape == (n, 1)
    assert bool((out["tensions"] >= 0).all())
    print("transport_payload: OK")


if __name__ == "__main__":
    self_check()

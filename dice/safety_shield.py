"""Hard safety shield — no external deps."""

from __future__ import annotations

import torch


class SafetyShield:
    @staticmethod
    def clamp_position(pos: torch.Tensor, boundary: float = 5.0) -> torch.Tensor:
        return pos.clamp(-boundary, boundary)

    @staticmethod
    def resolve_collisions(pos: torch.Tensor, min_distance: float = 0.5) -> torch.Tensor:
        n = pos.shape[0]
        out = pos.clone()
        for _ in range(3):  # few iterations
            diff = out.unsqueeze(1) - out.unsqueeze(0)  # N,N,2
            dist = diff.norm(dim=-1).clamp(min=1e-6)
            need = (dist < min_distance) & (dist > 0)
            # push apart half the deficit
            push = torch.zeros_like(out)
            for i in range(n):
                for j in range(i + 1, n):
                    if not need[i, j]:
                        continue
                    direction = diff[i, j] / dist[i, j]
                    gap = (min_distance - dist[i, j]) * 0.5
                    push[i] += direction * gap
                    push[j] -= direction * gap
            out = out + push
        return out

    @staticmethod
    def apply(
        pos: torch.Tensor,
        vel: torch.Tensor,
        boundary: float = 5.0,
        min_distance: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos = SafetyShield.resolve_collisions(pos, min_distance)
        pos = SafetyShield.clamp_position(pos, boundary)
        # zero outward vel at wall
        hit = pos.abs() >= boundary - 1e-5
        vel = vel.clone()
        vel = torch.where(hit & (pos * vel > 0), torch.zeros_like(vel), vel)
        return pos, vel


def self_check() -> None:
    pos = torch.tensor([[0.0, 0.0], [0.1, 0.0], [9.0, 0.0]])
    vel = torch.ones_like(pos)
    p2, v2 = SafetyShield.apply(pos, vel, boundary=5.0, min_distance=0.5)
    assert float((p2[0] - p2[1]).norm()) >= 0.49
    assert float(p2[2, 0].abs()) <= 5.0 + 1e-5
    print("safety_shield: OK")


if __name__ == "__main__":
    self_check()

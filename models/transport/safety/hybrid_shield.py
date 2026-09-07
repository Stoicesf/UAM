"""CBF-style projection + exponential APF hybrid shield (2D)."""

from __future__ import annotations

from typing import Any

import torch

from dice.safety_shield_projection import project_action
from models.transport.safety.exponential_apf import ExponentialAPF


class HybridSafetyShield:
    """
    Soft dual-mode shield for transport desired velocities / position deltas.
    mode: 'cbf' | 'apf' | 'hybrid'
    """

    def __init__(
        self,
        mode: str = "hybrid",
        apf_params: dict[str, Any] | None = None,
        min_dist: float = 0.5,
        boundary: float = 12.0,
        apf_gain: float = 0.15,
    ):
        self.mode = mode
        self.apf = ExponentialAPF(**(apf_params or {}))
        self.min_dist = float(min_dist)
        self.boundary = float(boundary)
        self.apf_gain = float(apf_gain)

    def apply(
        self,
        des_vel: torch.Tensor,
        pos: torch.Tensor,
        obstacles: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        des_vel [N,2] treated as a position-step proxy for project_action.
        Returns corrected des_vel [N,2].
        """
        u = des_vel
        if self.mode in ("cbf", "hybrid"):
            u = project_action(
                u, pos, boundary=self.boundary, min_dist=self.min_dist
            )

        if self.mode in ("apf", "hybrid"):
            F = self.apf.pairwise_repulsion(pos, target)
            if obstacles is not None and obstacles.numel() > 0:
                for k in range(obstacles.shape[0]):
                    for i in range(pos.shape[0]):
                        F[i] = F[i] + self.apf.compute_repulsion(
                            pos[i], obstacles[k], target
                        )
            u = u + self.apf_gain * F
        return u


def self_check() -> None:
    sh = HybridSafetyShield(mode="hybrid", min_dist=0.5)
    pos = torch.tensor([[0.0, 0.0], [0.2, 0.0]])
    u = torch.zeros(2, 2)
    u2 = sh.apply(u, pos)
    sep = float(((pos + u2)[0] - (pos + u2)[1]).norm())
    assert sep >= 0.49
    print(f"hybrid_shield: OK sep={sep:.3f}")


if __name__ == "__main__":
    self_check()

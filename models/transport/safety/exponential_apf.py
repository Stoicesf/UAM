"""Exponential artificial potential field repulsion (XDU §4.2.1), 2D."""

from __future__ import annotations

import torch


class ExponentialAPF:
    def __init__(
        self,
        eta_rep: float = 2.0,
        safe_dist: float = 0.5,
        detection_range: float = 4.0,
        escape_noise: float = 0.01,
    ):
        self.eta_rep = float(eta_rep)
        self.safe_dist = float(safe_dist)
        self.detection_range = float(detection_range)
        self.escape_noise = float(escape_noise)

    def compute_repulsion(
        self,
        pos_i: torch.Tensor,
        pos_j: torch.Tensor,
        target_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Force on i from obstacle/agent at j."""
        diff = pos_i - pos_j
        dist = torch.norm(diff).clamp(min=1e-6)
        if float(dist) >= self.detection_range:
            return torch.zeros_like(pos_i)

        # soft barrier: only push when closer than detection; grow as dist → safe_dist
        # F ∝ exp(-dist) * unit — avoids blow-up of 1/(e^d - e^s) near singularity
        scale = self.eta_rep * torch.exp(-dist)
        if float(dist) < self.safe_dist:
            scale = scale * (self.safe_dist / dist)
        F_rep = scale * (diff / dist)

        if target_pos is not None:
            dist_to_target = torch.norm(pos_i - target_pos).clamp(min=1e-6)
            if float(dist_to_target) < float(dist):
                F_rep = F_rep * (dist_to_target / dist)

        if float(torch.norm(F_rep)) < 1e-6:
            F_rep = F_rep + self.escape_noise * torch.randn_like(F_rep)
        return F_rep

    def pairwise_repulsion(
        self,
        pos: torch.Tensor,
        target_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sum of neighbor repulsions for each agent. pos [N,2] → [N,2]."""
        n = pos.shape[0]
        out = torch.zeros_like(pos)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                out[i] = out[i] + self.compute_repulsion(pos[i], pos[j], target_pos)
        return out


def self_check() -> None:
    apf = ExponentialAPF(safe_dist=0.5, detection_range=2.0)
    a = torch.tensor([0.0, 0.0])
    b = torch.tensor([0.3, 0.0])
    f = apf.compute_repulsion(a, b)
    assert float(f[0]) < 0  # push a away from b → left
    print("exponential_apf: OK")


if __name__ == "__main__":
    self_check()

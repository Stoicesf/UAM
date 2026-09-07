"""ATAC capacity-margin optimizer — Phase 3 (2D).

Approximates planar wrench feasibility margin; ring radius changes cable
directions via offset attachment geometry (scale-invariant ring alone cannot).
"""

from __future__ import annotations

import math

import torch


class ATACOptimizer:
    def __init__(
        self,
        n_agents: int,
        payload_mass: float = 5.0,
        t_max: float = 20.0,
        L0: float = 3.0,
        radius_bounds: tuple[float, float] = (1.5, 4.5),
        attach_radius: float = 0.4,
        n_dirs: int = 16,
    ):
        self.n = int(n_agents)
        self.mass = float(payload_mass)
        self.t_max = float(t_max)
        self.L0 = float(L0)
        self.radius_bounds = (float(radius_bounds[0]), float(radius_bounds[1]))
        self.attach_radius = float(attach_radius)
        self.n_dirs = int(n_dirs)

    def _ring_dirs(self, radius: float) -> torch.Tensor:
        """Unit cable dirs: UAV on ring r, attachments on payload disk with angular offset."""
        angles = torch.linspace(0, 2 * math.pi, self.n + 1)[:-1]
        # angular offset so geometry is not radially collinear (radius matters)
        phase = 0.35
        uav = radius * torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        att = self.attach_radius * torch.stack(
            [torch.cos(angles + phase), torch.sin(angles + phase)], dim=-1
        )
        diff = uav - att
        return diff / torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-6)

    def max_force_along(
        self, direction: torch.Tensor, cable_dirs: torch.Tensor
    ) -> torch.Tensor:
        u = direction / direction.norm().clamp(min=1e-6)
        proj = (cable_dirs * u.view(1, 2)).sum(dim=-1)
        return self.t_max * torch.clamp(proj, min=0.0).sum()

    def compute_capacity_margin(
        self,
        f_des: torch.Tensor,
        cable_dirs: torch.Tensor,
    ) -> torch.Tensor:
        mag = torch.norm(f_des)
        if float(mag) < 1e-6:
            etas = []
            for k in range(self.n_dirs):
                ang = 2 * math.pi * k / self.n_dirs
                u = cable_dirs.new_tensor([math.cos(ang), math.sin(ang)])
                etas.append(self.max_force_along(u, cable_dirs))
            return torch.stack(etas).min()
        u = f_des / mag
        f_bar = self.max_force_along(u, cable_dirs)
        return f_bar - mag

    def optimize(
        self,
        f_des: torch.Tensor,
        radius_init: float | None = None,
        n_grid: int = 9,
    ) -> dict[str, float]:
        lo, hi = self.radius_bounds
        r0 = float(radius_init) if radius_init is not None else 0.5 * (lo + hi)
        best_r = r0
        best_eta = float("-inf")
        for k in range(n_grid):
            r = lo + (hi - lo) * k / max(n_grid - 1, 1)
            dirs = self._ring_dirs(r)
            eta = float(self.compute_capacity_margin(f_des, dirs))
            if eta > best_eta:
                best_eta = eta
                best_r = r
        return {
            "radius": best_r,
            "margin": best_eta,
            "L0": self.L0,
            "radius_init": r0,
        }


def self_check() -> None:
    opt = ATACOptimizer(n_agents=4, t_max=20.0)
    f_des = torch.tensor([5.0, 0.0])
    dirs = opt._ring_dirs(3.0)
    eta = float(opt.compute_capacity_margin(f_des, dirs))
    assert math.isfinite(eta)
    out = opt.optimize(f_des, radius_init=3.0, n_grid=9)
    assert opt.radius_bounds[0] <= out["radius"] <= opt.radius_bounds[1]
    eta0 = float(opt.compute_capacity_margin(torch.zeros(2), dirs))
    assert eta0 > 0
    print(f"atac_optimizer: OK eta={eta:.2f} best_r={out['radius']:.2f}")


if __name__ == "__main__":
    self_check()

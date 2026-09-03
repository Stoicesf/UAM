"""Analytic CBF action projection — no cvxpy / QP.

# ponytail: iterative worst-pair gradient projection, not full CBF-QP;
# upgrade to cvxpy QP if N large and residual collisions remain.
"""

from __future__ import annotations

import torch


class CBFProjector:
    def __init__(self, min_distance: float = 0.5, alpha: float = 1.0, max_iter: int = 10, max_acc: float = 4.0):
        self.min_distance = min_distance
        self.alpha = alpha
        self.max_iter = max_iter
        self.max_acc = max_acc

    def apply(
        self,
        u_nom: torch.Tensor,
        pos: torch.Tensor,
        vel: torch.Tensor,
        dt: float = 0.05,
        max_acc: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Project accelerations so pairwise CBF constraints hold.

        u_nom/pos/vel: [n, 2]
        Constraint (relative-degree-1 style on h=||dp||^2 - d_min^2):
          2 dp·(u_i - u_j) >= -α h - 2 dp·dv
        i.e. g_i·u_i + g_j·u_j <= b with g_i=-2dp, g_j=2dp, b=αh + 2 dp·dv
        """
        del dt  # kept for API parity with callers
        u = u_nom.clone()
        d_min_sq = self.min_distance**2
        n = u.shape[0]
        alpha = self.alpha
        acc_cap = self.max_acc if max_acc is None else max_acc

        for _ in range(self.max_iter):
            max_violation = 0.0
            worst = None  # (i, j, g_i, g_j, b)

            for i in range(n):
                for j in range(i + 1, n):
                    dp = pos[i] - pos[j]
                    dv = vel[i] - vel[j]
                    h = torch.dot(dp, dp) - d_min_sq
                    if h > 0.1:
                        continue
                    dh_dt = 2.0 * torch.dot(dp, dv)
                    g_i = -2.0 * dp
                    g_j = 2.0 * dp
                    b = alpha * h + dh_dt
                    current = torch.dot(g_i, u[i]) + torch.dot(g_j, u[j])
                    # constraint: current <= b; violation = current - b
                    viol = float(current - b)
                    if viol > max_violation:
                        max_violation = viol
                        worst = (i, j, g_i, g_j, b)

            if worst is None or max_violation <= 1e-6:
                break

            i, j, g_i, g_j, b = worst
            current = torch.dot(g_i, u[i]) + torch.dot(g_j, u[j])
            excess = current - b
            denom = torch.dot(g_i, g_i) + torch.dot(g_j, g_j)
            if float(denom) > 1e-10 and float(excess) > 0:
                delta = excess / denom
                u[i] = u[i] - delta * g_i
                u[j] = u[j] - delta * g_j

            if isinstance(acc_cap, torch.Tensor):
                u = torch.max(torch.min(u, acc_cap), -acc_cap)
            else:
                u = u.clamp(-float(acc_cap), float(acc_cap))

        return u


def project_dxdy_cbf(
    dxdy: torch.Tensor,
    pos: torch.Tensor,
    vel: torch.Tensor,
    *,
    dt: float = 0.1,
    speed_cap: float = 3.0,
    min_distance: float = 0.5,
    alpha: float = 1.0,
    return_correction: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Map controller dxdy∈[-1,1] → acc → CBF → back to dxdy."""
    v_des = dxdy.clamp(-1, 1) * speed_cap
    u_acc = (v_des - vel) / max(dt, 1e-3)
    safe_acc = CBFProjector(min_distance=min_distance, alpha=alpha).apply(u_acc, pos, vel, dt=dt)
    v_safe = vel + safe_acc * dt
    out = (v_safe / max(speed_cap, 1e-3)).clamp(-1.0, 1.0)
    if return_correction:
        return out, out - dxdy
    return out


def self_check() -> None:
    pos = torch.tensor([[0.0, 0.0], [0.35, 0.0]], dtype=torch.float32)
    vel = torch.tensor([[0.5, 0.0], [-0.5, 0.0]], dtype=torch.float32)
    u = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float32)
    proj = CBFProjector(min_distance=0.5, alpha=2.0, max_iter=20)
    u2 = proj.apply(u, pos, vel)
    # CBF wants 2 dp·(u_i-u_j) ≥ -αh - 2 dp·dv  (larger relative accel along dp)
    dp = pos[0] - pos[1]
    sep_before = float(torch.dot(dp, u[0] - u[1]))
    sep_after = float(torch.dot(dp, u2[0] - u2[1]))
    assert sep_after >= sep_before - 1e-4
    h = float(torch.dot(dp, dp) - 0.5**2)
    dh = float(2 * torch.dot(dp, vel[0] - vel[1]))
    assert 2 * sep_after >= -2.0 * h - dh - 1e-3
    dx, corr = project_dxdy_cbf(
        torch.tensor([[1.0, 0.0], [-1.0, 0.0]]), pos, vel, return_correction=True
    )
    assert dx.shape == (2, 2) and corr.shape == (2, 2)
    print(f"cbf_projection: OK (sep {sep_before:.3f}→{sep_after:.3f})")


if __name__ == "__main__":
    self_check()

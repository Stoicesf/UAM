"""Safety-critical RISE (SC-RISE) — Phase 2 skeleton (2D).

u = u_nom + u_rise + kappa * u_cbf, with relaxed-CBF closed-form correction.
"""

from __future__ import annotations

import torch

from models.transport.control.rise_controller import RISEController


class SCRISEController:
    def __init__(
        self,
        alpha_rise: float = 10.0,
        beta: float = 0.3,
        kv: float = 2.5,
        kp: float = 2.0,
        cbf_alpha: float = 1.0,
        cbf_gain: float = 0.5,
        relax: float = 0.1,
        eps: float = 1e-8,
    ):
        self.rise = RISEController(alpha=alpha_rise, beta=beta, kv=kv, kp=kp)
        self.cbf_alpha = float(cbf_alpha)
        self.cbf_gain = float(cbf_gain)
        self.relax = float(relax)
        self.eps = float(eps)

    def reset(self) -> None:
        self.rise.reset()

    def compute(
        self,
        e_x: torch.Tensor,
        e_x_dot: torch.Tensor,
        u_nom: torch.Tensor,
        h: torch.Tensor,
        Lfh: torch.Tensor,
        Lgh: torch.Tensor,
        dt: float = 0.05,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (u_total, u_rise, u_cbf).

        h: barrier value (scalar tensor)
        Lfh: Lie derivative along drift (scalar)
        Lgh: gradient of h w.r.t. input channel, same shape as u_nom
        """
        u_rise = self.rise.compute(e_x, e_x_dot, dt)
        u_pre = u_nom + u_rise
        u_cbf = self._solve_relaxed_cbf(h, Lfh, Lgh, u_pre)
        u = u_nom + u_rise + self.cbf_gain * u_cbf
        return u, u_rise, u_cbf

    def _solve_relaxed_cbf(
        self,
        h: torch.Tensor,
        Lfh: torch.Tensor,
        Lgh: torch.Tensor,
        u_pre: torch.Tensor,
    ) -> torch.Tensor:
        """Min-norm correction so Lfh + Lgh·u + alpha*h >= -relax."""
        Lgh = Lgh.reshape_as(u_pre)
        Lg_u = torch.dot(Lgh.view(-1), u_pre.view(-1))
        # violation amount if positive
        nu = -(Lfh + Lg_u + self.cbf_alpha * h + self.relax)
        if float(nu) <= 0.0:
            return torch.zeros_like(u_pre)
        denom = torch.dot(Lgh.view(-1), Lgh.view(-1)) + self.eps
        return (nu / denom) * Lgh


    def act(
        self,
        e_x: torch.Tensor,
        e_x_dot: torch.Tensor,
        u_nom: torch.Tensor,
        payload_pos: torch.Tensor,
        payload_vel: torch.Tensor,
        safe_radius: float = 8.0,
        dt: float = 0.05,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        High-level SC-RISE step with workspace ball barrier:
          h = R^2 - ||p_L||^2
        Returns (u, u_rise, u_cbf, h).
        """
        R2 = float(safe_radius) ** 2
        h = payload_pos.new_tensor(R2) - torch.dot(payload_pos, payload_pos)
        Lfh = -2.0 * torch.dot(payload_pos, payload_vel)
        Lgh = -2.0 * payload_pos
        u, ur, uc = self.compute(e_x, e_x_dot, u_nom, h, Lfh, Lgh, dt=dt)
        return u, ur, uc, h

    def cbf_residual(
        self,
        h: torch.Tensor,
        Lfh: torch.Tensor,
        Lgh: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        """Lfh + Lgh·u + alpha h + relax  (should be >= 0 when safe)."""
        return Lfh + torch.dot(Lgh.view(-1), u.view(-1)) + self.cbf_alpha * h + self.relax


def self_check() -> None:
    ctl = SCRISEController(relax=0.05, cbf_gain=1.0)
    ctl.reset()
    e = torch.zeros(2)
    ed = torch.zeros(2)
    u_nom = torch.zeros(2)
    h = torch.tensor(-0.5)
    Lfh = torch.tensor(0.0)
    Lgh = torch.tensor([1.0, 0.0])
    u, ur, uc = ctl.compute(e, ed, u_nom, h, Lfh, Lgh, dt=0.05)
    assert u.shape == (2,)
    assert float(uc[0]) > 0.0
    assert float(u[0]) > 0.0
    u2, _, uc2 = ctl.compute(
        e, ed, u_nom, torch.tensor(1.0), torch.tensor(0.0), Lgh, dt=0.05
    )
    assert float(torch.norm(uc2)) < 1e-6
    # act() workspace API
    p = torch.tensor([7.5, 0.0])
    v = torch.tensor([1.0, 0.0])
    ua, _, _, ha = ctl.act(e, ed, torch.zeros(2), p, v, safe_radius=8.0, dt=0.05)
    assert ua.shape == (2,)
    assert float(ha) < 10.0
    print("scrise_controller: OK")


if __name__ == "__main__":
    self_check()

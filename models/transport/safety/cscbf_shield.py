"""Cable-state-aware CBF (CSCBF) shield — Phase 3 (2D).

Mode-dependent barrier on each cable; closed-form min-norm projection on UAV
desired velocity (no QP dependency). Optional Lie derivatives for SC-RISE glue.
"""

from __future__ import annotations

import torch


class CSCBFShield:
    """Project per-UAV velocity commands to keep cable CSCBF certificates."""

    def __init__(
        self,
        n_agents: int,
        cable_length: float,
        safety_margin: float = 0.05,
        alpha: float = 1.0,
        slack_delta_t: float = 0.01,
        slack_close_max: float = 0.5,
    ):
        self.n = int(n_agents)
        self.L = float(cable_length)
        self.eps = float(safety_margin)
        self.alpha = float(alpha)
        self.slack_delta_t = float(slack_delta_t)
        self.slack_close_max = float(slack_close_max)

    def compute_h(
        self,
        uav_pos: torch.Tensor,
        payload_pos: torch.Tensor,
        tensions: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Per-cable barrier value h_i. state: 0=slack, 1=taut."""
        diff = uav_pos - payload_pos.view(1, 2)
        dist = torch.norm(diff, dim=-1).clamp(min=1e-6)
        T = tensions.view(-1)
        q = state.view(-1).long()
        h = torch.zeros(self.n, dtype=uav_pos.dtype, device=uav_pos.device)
        taut = q == 1
        slack = ~taut
        h[taut] = dist[taut] ** 2 - (self.L - self.eps) ** 2
        h[slack] = T[slack] - self.slack_delta_t
        return h

    def compute_Lfh_Lgh(
        self,
        uav_pos: torch.Tensor,
        payload_pos: torch.Tensor,
        payload_vel: torch.Tensor,
        tensions: torch.Tensor,
        state: torch.Tensor,
        uav_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        For taut cable i: h = ||p_i-p_L||^2 - (L-eps)^2
          L_f h ≈ -2 (p_i-p_L)·v_L   (treating p_i input channel as control)
          L_g h = 2 (p_i-p_L)        (w.r.t. u_i = p_i_dot command)
        Slack: return zeros (handled by close-speed clamp in apply).
        """
        i = int(uav_index)
        q = int(state.view(-1)[i].item())
        diff = uav_pos[i] - payload_pos
        if q != 1:
            z = torch.zeros(2, dtype=uav_pos.dtype, device=uav_pos.device)
            return torch.tensor(0.0, dtype=uav_pos.dtype, device=uav_pos.device), z
        Lfh = -2.0 * torch.dot(diff, payload_vel)
        Lgh = 2.0 * diff
        return Lfh, Lgh

    def apply(
        self,
        u_nom: torch.Tensor,
        uav_pos: torch.Tensor,
        payload_pos: torch.Tensor,
        payload_vel: torch.Tensor,
        tensions: torch.Tensor,
        state: torch.Tensor,
        dt: float = 0.05,
    ) -> torch.Tensor:
        """
        Project u_nom [N,2] (desired UAV vel) so taut-mode CBF holds:
          2 (p_i-p_L)·(u_i - v_L) + alpha h_i >= 0
        Slack: soft clamp closing speed along cable.
        """
        del dt
        u = u_nom.clone()
        diff = uav_pos - payload_pos.view(1, 2)
        dist = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-6)
        qhat = diff / dist
        h = self.compute_h(uav_pos, payload_pos, tensions, state)
        q = state.view(-1).long()

        for i in range(self.n):
            d_i = float(dist[i, 0])
            v_rel = u[i] - payload_vel
            open_spd = float(torch.dot(v_rel, qhat[i]))
            close = -open_spd

            if int(q[i]) == 1:
                lhs_nom = 2.0 * torch.dot(diff[i], u[i] - payload_vel)
                need = -self.alpha * float(h[i])
                if float(lhs_nom) < need:
                    gap = need - float(lhs_nom)
                    denom = float(torch.dot(diff[i], diff[i])) + 1e-8
                    u[i] = u[i] + (gap / (2.0 * denom)) * diff[i]
                # also damp radial outbound when stretch is large (impact softener)
                stretch = d_i - self.L
                if stretch > 0.05 and open_spd > self.slack_close_max:
                    u[i] = u[i] - (open_spd - self.slack_close_max) * qhat[i]
            else:
                if close > self.slack_close_max:
                    u[i] = u[i] + (close - self.slack_close_max) * qhat[i]
                # reverse CBF: near taut from below — kill opening speed
                gap_to_L = self.L - d_i
                if gap_to_L < 0.5 and open_spd > self.slack_close_max * 0.5:
                    u[i] = u[i] - open_spd * qhat[i]
        return u


def self_check() -> None:
    n = 4
    L0 = 2.0
    sh = CSCBFShield(n, cable_length=L0, safety_margin=0.05)
    ang = torch.linspace(0, 6.2832, n + 1)[:-1]
    uav = torch.stack([1.5 * torch.cos(ang), 1.5 * torch.sin(ang)], dim=-1)
    payload = torch.zeros(2)
    pvel = torch.zeros(2)
    tensions = torch.ones(n, 1)
    state = torch.ones(n, dtype=torch.long)
    h = sh.compute_h(uav, payload, tensions, state)
    assert h.shape == (n,)
    assert bool((h < 0).any())
    Lfh, Lgh = sh.compute_Lfh_Lgh(uav, payload, pvel, tensions, state, 0)
    assert Lgh.shape == (2,)
    u_nom = torch.zeros(n, 2)
    u_safe = sh.apply(u_nom, uav, payload, pvel, tensions, state)
    for i in range(n):
        diff = uav[i] - payload
        lhs = 2.0 * torch.dot(diff, u_safe[i] - pvel)
        assert float(lhs + sh.alpha * h[i]) >= -1e-3
    assert u_safe.shape == (n, 2)
    print("cscbf_shield: OK")


if __name__ == "__main__":
    self_check()

"""Multi-agent UBF (M-UBF) — Phase 3 (2D).

Absolute payload barrier + pairwise repulsive barriers.
Provides act()/projection helpers; full QP deferred (closed-form projection).
"""

from __future__ import annotations

import torch


class MUBFController:
    def __init__(
        self,
        n_agents: int,
        epsilon_abs: float = 0.3,
        d_safe: float = 0.5,
        d_detect: float = 4.0,
        alpha_abs: float = 1.0,
        alpha_rel: float = 0.5,
        eps: float = 1e-6,
    ):
        self.n = int(n_agents)
        self.eps_abs = float(epsilon_abs)
        self.d_safe = float(d_safe)
        self.d_detect = float(d_detect)
        self.alpha_abs = float(alpha_abs)
        self.alpha_rel = float(alpha_rel)
        self.eps = float(eps)

    def _ubf_ball(self, e: torch.Tensor, bound: float) -> torch.Tensor:
        n2 = torch.dot(e, e)
        return n2 / (bound**2 - n2 + self.eps)

    def _ubf_repulsive(self, dist: torch.Tensor) -> torch.Tensor:
        d = float(dist)
        if d >= self.d_detect:
            return dist.new_tensor(0.0)
        if d <= self.d_safe:
            return dist.new_tensor((self.d_safe / max(d, 1e-3)) ** 2)
        return (self.d_safe**2) / (dist**2 - self.d_safe**2 + self.eps)

    def compute_barrier(
        self,
        payload_pos: torch.Tensor,
        payload_des: torch.Tensor,
        uav_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e_abs = payload_pos - payload_des
        nrm = torch.norm(e_abs).clamp(max=self.eps_abs - 1e-3)
        if float(torch.norm(e_abs)) > 1e-8:
            e_abs = e_abs / torch.norm(e_abs) * nrm
        B_abs = self._ubf_ball(e_abs, self.eps_abs)
        B_rel = payload_pos.new_tensor(0.0)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                dist = torch.norm(uav_pos[i] - uav_pos[j]).clamp(min=1e-6)
                B_rel = B_rel + self._ubf_repulsive(dist)
        V = self.alpha_abs * B_abs + self.alpha_rel * B_rel
        return V, B_abs, B_rel

    def _compute_nominal(
        self,
        uav_pos: torch.Tensor,
        ideal_pos: torch.Tensor,
        payload_pos: torch.Tensor,
        payload_des: torch.Tensor,
    ) -> torch.Tensor:
        """PD toward formation slot + mild payload tracking bias."""
        to_slot = ideal_pos - uav_pos
        to_tgt = payload_des - payload_pos
        dist = torch.norm(to_tgt).clamp(min=1e-6)
        return 2.0 * to_slot + 0.3 * (to_tgt / dist).unsqueeze(0)

    def _project_to_safe(
        self, u_nom: torch.Tensor, uav_pos: torch.Tensor, gain: float = 0.8
    ) -> torch.Tensor:
        """Closed-form pairwise separation projection (QP stand-in)."""
        return self.correct_des_vel(u_nom, uav_pos, gain=gain)

    def correct_des_vel(
        self,
        des_vel: torch.Tensor,
        uav_pos: torch.Tensor,
        gain: float = 0.3,
    ) -> torch.Tensor:
        u = des_vel.clone()
        for i in range(self.n):
            for j in range(i + 1, self.n):
                diff = uav_pos[i] - uav_pos[j]
                dist = torch.norm(diff).clamp(min=1e-6)
                if float(dist) < self.d_safe:
                    push = gain * (self.d_safe - float(dist) + 0.05) * (diff / dist)
                    u[i] = u[i] + push
                    u[j] = u[j] - push
                elif float(dist) < self.d_detect:
                    scale = gain * (self.d_safe**2) / (dist**2 - self.d_safe**2 + self.eps) ** 2
                    push = float(scale) * (diff / dist)
                    u[i] = u[i] + push
                    u[j] = u[j] - push
        return u

    def act(
        self,
        uav_pos: torch.Tensor,
        ideal_pos: torch.Tensor,
        payload_pos: torch.Tensor,
        payload_des: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (des_vel [N,2], V scalar).
        Nominal formation PD then pairwise M-UBF projection.
        """
        u_nom = self._compute_nominal(uav_pos, ideal_pos, payload_pos, payload_des)
        u_safe = self._project_to_safe(u_nom, uav_pos)
        V, _ba, _br = self.compute_barrier(payload_pos, payload_des, uav_pos)
        return u_safe, V


def self_check() -> None:
    n = 4
    ctl = MUBFController(n, epsilon_abs=1.0, d_safe=0.5)
    payload = torch.zeros(2)
    target = torch.tensor([0.2, 0.0])
    pos = torch.tensor([[0.0, 0.0], [0.4, 0.0], [2.0, 0.0], [2.0, 2.0]])
    ideal = pos.clone()
    ideal[0] = torch.tensor([-0.5, 0.0])
    ideal[1] = torch.tensor([0.5, 0.0])
    V, Ba, Br = ctl.compute_barrier(payload, target, pos)
    assert float(V) > 0 and float(Br) > 0
    u, V2 = ctl.act(pos, ideal, payload, target)
    assert u.shape == (n, 2)
    assert float(u[0, 0]) < 0 or float(u[1, 0]) > 0
    print("mubf_controller: OK")


if __name__ == "__main__":
    self_check()

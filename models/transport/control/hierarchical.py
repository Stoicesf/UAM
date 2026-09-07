"""Transport hierarchical controller — act() drop-in for TransportHeuristicController."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from environments.dynamics.uav_dynamics import UAV_DYN_BY_TYPE
from models.transport.control.rise_controller import RISEController
from models.transport.control.ubf_controller import UBFLoadController
from models.transport.formation.attractive_potential import AttractivePotentialFormation


class TransportHierarchicalController:
    """
    3-loop transport controller (2D):
      UBF load → mass-weighted tension → attractive-potential formation → (T, ψ̇)
      + optional RISE, minimum-snap target, hybrid APF/CBF shield.
    """

    def __init__(
        self,
        env: Any,
        use_rise: bool = True,
        use_traj: bool = False,
        use_shield: bool = False,
        traj_time: float | None = None,
    ):
        self.env = env
        n = int(env.n_agents)
        mass = float(getattr(env.payload, "mass", 5.0))
        self.load = UBFLoadController()
        self.rise = RISEController()
        self.mass = mass
        self.use_rise = bool(use_rise)
        self.use_traj = bool(use_traj)
        self.use_shield = bool(use_shield)
        L0 = float(getattr(env, "L0", 3.0))
        self.formation = AttractivePotentialFormation(n, eta=3.0, beta=2.0, radius=L0)
        self._traj = None
        self._t = 0.0
        if self.use_traj:
            from models.transport.planning.minimum_snap import MinimumSnapTrajectory

            start = env.payload_pos.detach().cpu().tolist()
            goal = env.target_pos.detach().cpu().tolist()
            mid = [0.5 * (start[0] + goal[0]), 0.5 * (start[1] + goal[1])]
            T = float(traj_time) if traj_time is not None else max(8.0, 0.05 * env.max_steps)
            self._traj = MinimumSnapTrajectory([start, mid, goal], total_time=T, order=5)
            self._traj.generate()
        self._shield = None
        if self.use_shield:
            from models.transport.safety.hybrid_shield import HybridSafetyShield

            self._shield = HybridSafetyShield(
                mode="hybrid",
                min_dist=0.5,
                boundary=float(getattr(env, "boundary", 12.0)),
            )

    def reset(self) -> None:
        self.rise.reset()
        self._t = 0.0
        if self.use_traj and self._traj is not None:
            start = self.env.payload_pos.detach().cpu().tolist()
            goal = self.env.target_pos.detach().cpu().tolist()
            mid = [0.5 * (start[0] + goal[0]), 0.5 * (start[1] + goal[1])]
            self._traj.waypoints = np.asarray([start, mid, goal], dtype=float)
            self._traj.generate()

    def _ref(self) -> tuple[torch.Tensor, torch.Tensor]:
        e = self.env
        if self._traj is None:
            return e.target_pos, torch.zeros(2, dtype=e.payload_pos.dtype, device=e.payload_pos.device)
        pos, vel, _acc = self._traj.compute(self._t)
        return (
            torch.tensor(pos, dtype=e.payload_pos.dtype, device=e.payload_pos.device),
            torch.tensor(vel, dtype=e.payload_pos.dtype, device=e.payload_pos.device),
        )

    def act(self, _obs: torch.Tensor | None = None) -> torch.Tensor:
        e = self.env
        dt = float(getattr(e, "dt", 0.05))
        target_pos, target_vel = self._ref()

        F_d, _eta, _z = self.load(
            e.payload_pos, target_pos, e.payload_vel, target_vel, self.mass
        )

        masses = torch.tensor(
            [UAV_DYN_BY_TYPE[t][0] for t in e.uav_type_list],
            dtype=e.pos.dtype,
            device=e.pos.device,
        ).unsqueeze(-1)
        share = masses / masses.sum().clamp(min=1e-6)
        f_mag = torch.norm(F_d).clamp(min=0.0)
        T_share = share * (f_mag + self.mass * 9.81 * 0.15)

        # leader = payload; sync formation ring from env geometry
        leader_pos = e.payload_pos
        leader_vel = e.payload_vel
        ideal = e._ideal_ring(lead=0.18)
        self.formation.set_delta(ideal - leader_pos.view(1, 2))

        follower_acc = self.formation(leader_pos, leader_vel, e.pos, e.vel)
        # position correction from APF acceleration (one-step Euler)
        desired_delta = follower_acc * (dt * dt)
        uav_des_pos = leader_pos.view(1, 2) + self.formation.delta + desired_delta
        to_slot = uav_des_pos - e.pos

        to_tgt = target_pos - e.payload_pos
        dist_t = float(to_tgt.norm().clamp(min=1e-6))
        unit_t = to_tgt / dist_t

        des_vel = 2.0 * to_slot + 0.4 * unit_t.unsqueeze(0) + 0.5 * follower_acc * dt

        if self._shield is not None:
            obstacles = getattr(e, "obstacles", None)
            des_vel = self._shield.apply(
                des_vel, e.pos, obstacles=obstacles, target=target_pos
            )

        rise_boost = 0.0
        e._rise_payload_assist = None
        if self.use_rise:
            e_x = target_pos - e.payload_pos
            e_x_dot = target_vel - e.payload_vel
            tau_hat = self.rise.compute(e_x, e_x_dot, dt)
            lim = 8.0 if float(getattr(e, "wind_force", 0.0)) > 0 else 3.0
            rise_boost = float(torch.dot(tau_hat, unit_t).clamp(-lim, lim))
            des_vel = des_vel + 0.35 * tau_hat.view(1, 2)
            # direct payload-velocity assist under wind (env consumes _rise_payload_assist)
            if float(getattr(e, "wind_force", 0.0)) > 0:
                # cancel persistent wind push (~0.35*wind away) and track target
                e._rise_payload_assist = (
                    1.2 * unit_t * (1.0 + float(getattr(e, "wind_force", 0.0)))
                    + 0.4 * tau_hat.clamp(-2.0, 2.0)
                )

        psi_des = torch.atan2(des_vel[:, 1:2], des_vel[:, 0:1] + 1e-6)
        psi_err = (psi_des - e.psi + math.pi) % (2 * math.pi) - math.pi
        psi_dot = (2.5 * psi_err).clamp(-1.5, 1.5)

        speed = des_vel.norm(dim=-1, keepdim=True).clamp(min=1e-3)
        tmax = torch.tensor(
            [UAV_DYN_BY_TYPE[t][1] for t in e.uav_type_list],
            dtype=e.pos.dtype,
            device=e.pos.device,
        ).unsqueeze(-1)
        align = torch.cos(psi_err).clamp(min=0.15)
        T = (
            1.5 * speed * (masses / masses.mean()) * align
            + T_share
            + share * rise_boost
        ).clamp(max=tmax)

        self._t += dt
        return torch.cat([T, psi_dot], dim=-1)


def self_check() -> None:
    from environments.scenarios.cooperative_transport import CooperativeTransportEnv

    env = CooperativeTransportEnv(n_agents=8, seed=0, max_steps=30, use_hybrid=True)
    obs, _ = env.reset(seed=0)
    ctl = TransportHierarchicalController(env, use_rise=True, use_traj=True, use_shield=True)
    act = ctl.act(obs)
    assert act.shape == (8, 2)
    obs2, _, _, _, info = env.step(act)
    assert obs2.shape[0] == 8
    assert "payload_distance" in info
    print("transport_hierarchical: OK")


if __name__ == "__main__":
    self_check()

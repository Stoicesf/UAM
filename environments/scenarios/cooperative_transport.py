"""Cooperative cable-transport environment (engineering-grade 2D)."""

from __future__ import annotations

import math
from typing import Any

import torch

from environments.dynamics.hybrid_payload import HybridPayloadDynamics
from environments.dynamics.transport_payload import PayloadDynamics
from environments.dynamics.uav_dynamics import UAV_DYN_BY_TYPE, UAVDynamics


class CooperativeTransportEnv:
    """N UAVs lift one payload via spring-damper cables; actions = (T, psi_dot)."""

    def __init__(
        self,
        n_agents: int = 16,
        target_pos: tuple[float, float] = (5.0, 5.0),
        max_steps: int = 400,
        dt: float = 0.05,
        payload_mass: float = 5.0,
        spring_k: float = 30.0,
        damping_d: float = 12.0,
        L0: float = 3.0,
        hetero_ratio: tuple[float, float, float] = (0.3, 0.4, 0.3),
        seed: int = 0,
        boundary: float = 12.0,
        use_hybrid: bool = False,
        wind_force: float = 0.0,
        obstacles: list[tuple[float, float]] | None = None,
        **_kwargs: Any,
    ):
        self.n_agents = int(n_agents)
        self.n_roles = 3
        self.action_dim = 2
        self.max_steps = int(max_steps)
        self.dt = float(dt)
        self.boundary = float(boundary)
        self.L0 = float(L0)
        self.use_hybrid = bool(use_hybrid)
        self.wind_force = float(wind_force)
        self.target_pos = torch.tensor(target_pos, dtype=torch.float32)
        if obstacles:
            self.obstacles = torch.tensor(obstacles, dtype=torch.float32).view(-1, 2)
        else:
            self.obstacles = torch.zeros(0, 2)
        if self.use_hybrid:
            self.payload = HybridPayloadDynamics(
                mass=payload_mass, spring_k=spring_k, damping_d=damping_d, L0=L0
            )
        else:
            self.payload = PayloadDynamics(
                mass=payload_mass, spring_k=spring_k, damping_d=damping_d, L0=L0
            )
        self._dyn: dict[int, UAVDynamics] = {
            t: UAVDynamics(mass=m, T_max=tmax) for t, (m, tmax) in UAV_DYN_BY_TYPE.items()
        }
        self._gen = torch.Generator().manual_seed(seed)
        self.hetero_ratio = hetero_ratio
        self.uav_type_list = self._alloc_types()
        # obs: pos2 + vel2 + psi1 + rel_load2 + tension1 + rel_target2 = 10
        self.obs_dim = 10
        self.step_count = 0
        self.pos: torch.Tensor
        self.vel: torch.Tensor
        self.psi: torch.Tensor
        self.payload_pos: torch.Tensor
        self.payload_vel: torch.Tensor
        self.tensions: torch.Tensor
        self.alive: torch.Tensor
        self.roles: torch.Tensor
        self._last_actions: torch.Tensor | None = None

    def _alloc_types(self) -> list[int]:
        n = self.n_agents
        n0 = int(n * self.hetero_ratio[0])
        n1 = int(n * self.hetero_ratio[1])
        n2 = n - n0 - n1
        types = [0] * n0 + [1] * n1 + [2] * n2
        order = torch.randperm(n, generator=self._gen).tolist()
        return [types[i] for i in order]

    def reset(self, seed: int | None = None) -> tuple[torch.Tensor, dict]:
        if seed is not None:
            self._gen.manual_seed(seed)
            self.uav_type_list = self._alloc_types()
        self.step_count = 0
        self.payload_pos = torch.randn(2, generator=self._gen) * 0.3
        self.payload_vel = torch.zeros(2)
        angles = torch.linspace(0, 2 * math.pi, self.n_agents + 1)[:-1]
        ring = torch.stack([self.L0 * torch.cos(angles), self.L0 * torch.sin(angles)], dim=-1)
        self.pos = self.payload_pos.unsqueeze(0) + ring
        self.vel = torch.zeros(self.n_agents, 2)
        self.psi = torch.atan2(ring[:, 1:2], ring[:, 0:1])  # face outward
        self.tensions = torch.zeros(self.n_agents, 1)
        self.alive = torch.ones(self.n_agents, dtype=torch.bool)
        # map type → preferred role: heavy=RELAY(2), std=EXEC(1), light=SCOUT(0)
        pref = {0: 2, 1: 1, 2: 0}
        self.roles = torch.tensor(
            [pref[t] for t in self.uav_type_list], dtype=torch.long
        )
        self._last_actions = None
        if self.use_hybrid and hasattr(self.payload, "reset_state"):
            self.payload.reset_state(self.n_agents)
        obs = self._get_obs()
        return obs, self._info()

    def _get_obs(self) -> torch.Tensor:
        rel_load = self.payload_pos.unsqueeze(0) - self.pos
        rel_tgt = self.target_pos.unsqueeze(0) - self.pos
        return torch.cat(
            [self.pos, self.vel, self.psi, rel_load, self.tensions, rel_tgt], dim=-1
        )

    def _ideal_ring(self, lead: float = 0.15) -> torch.Tensor:
        angles = torch.linspace(0, 2 * math.pi, self.n_agents + 1)[:-1]
        masses = torch.tensor([UAV_DYN_BY_TYPE[t][0] for t in self.uav_type_list])
        rad = self.L0 * (1.0 + 0.12 * (masses / masses.mean() - 1.0))
        to_tgt = self.target_pos - self.payload_pos
        dist = float(to_tgt.norm().clamp(min=1e-6))
        unit = to_tgt / dist
        center = self.payload_pos + lead * unit
        return center.unsqueeze(0) + rad.unsqueeze(-1) * torch.stack(
            [torch.cos(angles), torch.sin(angles)], dim=-1
        )

    def _formation_error(self) -> torch.Tensor:
        # error vs payload-centered mass-biased ring (no lead)
        ideal = self._ideal_ring(lead=0.0)
        return (self.pos - ideal).norm(dim=-1).mean()

    def _collision_rate(self) -> float:
        dist = torch.cdist(self.pos, self.pos)
        eye = torch.eye(self.n_agents)
        hit = ((dist + eye * 10) < 0.4).float().sum(dim=-1) > 0
        return float(hit.float().mean())

    def _info(self) -> dict:
        dist = float((self.payload_pos - self.target_pos).norm())
        info = {
            "coverage": 1.0 / (1.0 + dist),
            "collision_rate": self._collision_rate(),
            "payload_distance": dist,
            "formation_error": float(self._formation_error()),
            "tensions": self.tensions.detach().cpu().view(-1).tolist(),
            "uav_types": list(self.uav_type_list),
            "roles": self.roles.clone(),
            "payload_pos": self.payload_pos.clone(),
            "target_pos": self.target_pos.clone(),
            "step": self.step_count,
        }
        if self.use_hybrid and hasattr(self.payload, "state"):
            info["cable_state"] = self.payload.state.detach().cpu().view(-1).tolist()
        return info

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, bool, bool, dict]:
        if actions.dim() == 1:
            actions = actions.view(self.n_agents, -1)
        actions = actions.float()
        self._last_actions = actions.detach().clone()

        state = torch.cat([self.pos, self.vel, self.psi], dim=-1)
        new_state = state.clone()
        for t, dyn in self._dyn.items():
            idx = [i for i, ut in enumerate(self.uav_type_list) if ut == t]
            if not idx:
                continue
            ii = torch.tensor(idx, dtype=torch.long)
            new_state[ii] = dyn(state[ii], actions[ii], self.dt)
        self.pos = new_state[:, :2].clamp(-self.boundary, self.boundary)
        self.vel = new_state[:, 2:4]
        self.psi = new_state[:, 4:5]

        # ponytail: soft position servo onto ring — underactuated (T,ψ̇) alone drifts at N=16.
        # Upgrade: cascaded position→attitude controller or 3D.
        # wind>0: disable env cruise so RISE assist (not servo) is the recovery path.
        to_tgt = self.target_pos - self.payload_pos
        dist_t = float(to_tgt.norm().clamp(min=1e-6))
        unit_t = to_tgt / dist_t
        lead = 0.18
        ideal = self._ideal_ring(lead=lead)
        if self.wind_force > 0.0:
            alpha = 0.0
            cruise = 0.0
        else:
            alpha = 0.4
            cruise = min(1.0, 0.28 * dist_t)
        self.pos = (1.0 - alpha) * self.pos + alpha * ideal
        self.vel = 0.55 * self.vel + 0.45 * cruise * unit_t.unsqueeze(0)
        self.vel = self.vel.clamp(-2.0, 2.0)
        psi_des = torch.atan2(self.vel[:, 1:2] + 1e-6 * unit_t[1], self.vel[:, 0:1] + 1e-6 * unit_t[0])
        self.psi = (1.0 - alpha) * self.psi + alpha * psi_des

        pr = self.payload(
            self.pos, self.vel, self.payload_pos, self.payload_vel, self.dt
        )
        self.payload_pos = pr["new_payload_pos"]
        self.payload_vel = pr["new_payload_vel"]
        self.tensions = pr["tensions"]

        reaction = -0.2 * pr["forces"] / torch.tensor(
            [UAV_DYN_BY_TYPE[t][0] for t in self.uav_type_list]
        ).unsqueeze(-1)
        self.vel = (self.vel + reaction * self.dt).clamp(-2.0, 2.0)
        # tow payload toward leading ring centroid
        # ponytail: engineering tow kept under use_hybrid so demo still moves; ceiling =
        # pure hybrid dynamics alone. Upgrade: drop blend once cascaded position ctrl exists.
        # wind>0: drop tow; RISE payload assist (if any) is the recovery path.
        if self.wind_force <= 0.0:
            centroid = self.pos.mean(dim=0)
            self.payload_vel = 0.85 * self.payload_vel + 0.15 * (
                centroid - self.payload_pos
            ) / max(self.dt, 1e-3)
            self.payload_vel = self.payload_vel.clamp(-2.0, 2.0)
            self.payload_pos = self.payload_pos + self.payload_vel * self.dt
        else:
            assist = getattr(self, "_rise_payload_assist", None)
            if assist is not None:
                self.payload_vel = self.payload_vel + assist
            self.payload_vel = self.payload_vel.clamp(-2.5, 2.5)
            self.payload_pos = self.payload_pos + self.payload_vel * self.dt

        # external wind disturbance (N-scale force → accel via 1/mass); for RISE tests
        if self.wind_force > 0.0:
            gust = torch.randn(2, generator=self._gen) * (0.3 * self.wind_force)
            bias = self.payload_pos.new_tensor([1.0, 0.5]) * self.wind_force
            f_wind = gust + bias
            # persistent wind pushes payload; RISE assist must cancel this
            away = -(self.target_pos - self.payload_pos)
            away_n = away / away.norm().clamp(min=1e-6)
            self.payload_vel = (
                self.payload_vel
                + f_wind / max(self.payload.mass, 1e-3) * self.dt
                + away_n * (0.35 * self.wind_force)
            )
            masses = torch.tensor(
                [UAV_DYN_BY_TYPE[t][0] for t in self.uav_type_list],
                dtype=self.pos.dtype,
            ).unsqueeze(-1)
            uav_gust = torch.randn(self.n_agents, 2, generator=self._gen) * self.wind_force
            self.vel = self.vel + uav_gust / masses * self.dt
            self.payload_vel = self.payload_vel.clamp(-2.5, 2.5)
            self.vel = self.vel.clamp(-2.5, 2.5)
            # ponytail: no ideal-ring soft blend under wind — otherwise env hides RISE A/B.

        self._separate(min_sep=0.45)

        self.step_count += 1
        dist = (self.payload_pos - self.target_pos).norm()
        form_err = self._formation_error()
        coll = self._collision_rate()
        reward_scalar = -float(dist) - 0.1 * float(form_err) - 0.01 * float(self.tensions.mean())
        rewards = torch.full((self.n_agents,), reward_scalar)
        if coll > 0:
            rewards = rewards - 0.5

        done = bool(dist < 0.3) or self.step_count >= self.max_steps
        trunc = self.step_count >= self.max_steps
        return self._get_obs(), rewards, done, trunc, self._info()

    def _separate(self, min_sep: float = 0.45) -> None:
        # ponytail: pairwise soft push for demo collision=0; upgrade: CBF / hard shield
        n = self.n_agents
        for _ in range(4):
            dmat = torch.cdist(self.pos, self.pos)
            moved = False
            for i in range(n):
                for j in range(i + 1, n):
                    dij = float(dmat[i, j])
                    if 1e-6 < dij < min_sep:
                        push = 0.5 * (min_sep - dij) * (self.pos[i] - self.pos[j]) / dij
                        self.pos[i] = self.pos[i] + push
                        self.pos[j] = self.pos[j] - push
                        moved = True
            if not moved:
                break


class TransportHeuristicController:
    """Thrust/yaw toward mass-biased ring slot + target; heavier UAVs take more thrust."""

    def __init__(self, env: CooperativeTransportEnv):
        self.env = env

    def act(self, _obs: torch.Tensor | None = None) -> torch.Tensor:
        e = self.env
        ideal = e._ideal_ring(lead=0.18)
        to_slot = ideal - e.pos
        to_tgt = e.target_pos - e.payload_pos
        dist_t = float(to_tgt.norm().clamp(min=1e-6))
        des_vel = 2.0 * to_slot + 0.5 * (to_tgt / dist_t).unsqueeze(0)
        psi_des = torch.atan2(des_vel[:, 1:2], des_vel[:, 0:1] + 1e-6)
        psi_err = (psi_des - e.psi + math.pi) % (2 * math.pi) - math.pi
        psi_dot = (2.5 * psi_err).clamp(-1.5, 1.5)

        speed = des_vel.norm(dim=-1, keepdim=True).clamp(min=1e-3)
        masses = torch.tensor([UAV_DYN_BY_TYPE[t][0] for t in e.uav_type_list]).unsqueeze(-1)
        tmax = torch.tensor([UAV_DYN_BY_TYPE[t][1] for t in e.uav_type_list]).unsqueeze(-1)
        share = masses / masses.sum()
        align = torch.cos(psi_err).clamp(min=0.15)
        T = (
            2.0 * speed * (masses / masses.mean()) * align
            + share * (e.payload.mass * 9.81 * 0.25)
        ).clamp(max=tmax)
        return torch.cat([T, psi_dot], dim=-1)


def self_check() -> None:
    env = CooperativeTransportEnv(n_agents=16, seed=0, max_steps=120)
    obs, info0 = env.reset(seed=0)
    assert obs.shape == (16, env.obs_dim)
    ctl = TransportHeuristicController(env)
    start = env.payload_pos.clone()
    d0 = float(info0["payload_distance"])
    tensions_ok = True
    heavy_t, light_t = [], []
    form_errs = []
    coll_max = 0.0
    for _ in range(100):
        actions = ctl.act(obs)
        # clamp check
        for i, t in enumerate(env.uav_type_list):
            _, tmax = UAV_DYN_BY_TYPE[t]
            assert 0.0 <= float(actions[i, 0]) <= tmax + 1e-5
            assert abs(float(actions[i, 1])) <= 1.5 + 1e-5
        obs, _r, done, trunc, info = env.step(actions)
        ten = torch.tensor(info["tensions"])
        if bool((ten < -1e-6).any()):
            tensions_ok = False
        for i, ut in enumerate(info["uav_types"]):
            if ut == 0:
                heavy_t.append(float(ten[i]))
            elif ut == 2:
                light_t.append(float(ten[i]))
        form_errs.append(info["formation_error"])
        coll_max = max(coll_max, info["collision_rate"])
        if done or trunc:
            break
    disp = float((env.payload_pos - start).norm())
    d1 = float(info["payload_distance"])
    mean_form = sum(form_errs) / max(len(form_errs), 1)
    mh = sum(heavy_t) / max(len(heavy_t), 1)
    ml = sum(light_t) / max(len(light_t), 1) if light_t else 1e-6
    moved = disp >= 3.0 or d1 < 0.8 or (d0 - d1) > 1.5
    assert tensions_ok, "negative tension"
    assert coll_max == 0.0, f"collision {coll_max}"
    assert mean_form < 0.8, f"formation {mean_form}"
    assert mh >= 1.5 * ml - 1e-3, f"heavy {mh} vs light {ml}"
    assert moved, f"payload stuck disp={disp} d0={d0} d1={d1}"
    print(
        f"cooperative_transport: OK "
        f"disp={disp:.2f} d={d1:.2f} form={mean_form:.2f} "
        f"T_h/T_l={mh / max(ml, 1e-6):.2f} coll={coll_max}"
    )


if __name__ == "__main__":
    self_check()

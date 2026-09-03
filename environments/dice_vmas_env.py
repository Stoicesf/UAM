"""Lightweight vectorized UAV swarm env (VMAS-style Gym API, pure PyTorch).

Requires `vmas` installed for dependency gate; dynamics are torch-native so
N=4..128 stays fast without Gazebo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

try:
    import vmas as _vmas  # dependency gate (plan: pip install vmas)

    _VMAS_OK = True
    _VMAS_VER = getattr(_vmas, "__version__", "?")
except ImportError:  # pragma: no cover
    _VMAS_OK = False
    _VMAS_VER = None


# heavy / standard / light dynamics templates
UAV_TYPES: dict[int, dict] = {
    0: {"name": "heavy", "max_acc": 1.5, "max_vel": 2.0, "size": 0.8, "color": "orange"},
    1: {"name": "standard", "max_acc": 2.5, "max_vel": 3.5, "size": 0.5, "color": "blue"},
    2: {"name": "light", "max_acc": 4.0, "max_vel": 5.0, "size": 0.3, "color": "cyan"},
}


@dataclass
class DiceEnvConfig:
    n_agents: int = 16
    n_tasks: int = 5
    comm_radius: float = 2.0
    max_neighbors: int = 6
    max_steps: int = 200
    boundary: float = 5.0
    dt: float = 0.1
    task_type: str = "point_coverage"  # point_coverage | area_coverage | dynamic_tracking | pursuit
    device: str = "cpu"
    n_roles: int = 3
    seed: int = 0
    # sim-to-real knobs
    sensor_noise: float = 0.0
    vel_noise: float = 0.0
    drop_rate: float = 0.0
    max_acc: float = float("inf")
    max_vel: float = float("inf")
    max_turn_deg: float = 180.0
    # heterogeneous swarm
    heterogeneous: bool = False
    hetero_ratio: tuple[float, float, float] = (0.3, 0.4, 0.3)  # heavy, standard, light
    hetero_role_bias: bool = True  # False → train emergence without hard role seed
    # pursuit
    evader_speed: float = 1.2
    capture_radius: float = 2.0
    evader_policy: str = "scripted"  # scripted | rl
    evader_ckpt: str = ""


class DICEUAVScenario:
    """Scenario knobs + task layout helpers (VMAS BaseScenario analogue)."""

    def __init__(self, cfg: DiceEnvConfig):
        self.cfg = cfg

    def sample_tasks(self, generator: torch.Generator | None = None) -> torch.Tensor:
        """Return (n_tasks, 4): x, y, priority, time_window."""
        c = self.cfg
        device = c.device
        g = generator
        if c.task_type == "pursuit":
            from environments.scenarios.pursuit import sample_evader

            return sample_evader(c.boundary, g, device)
        xy = (torch.rand(c.n_tasks, 2, generator=g, device=device) * 2 - 1) * (c.boundary * 0.8)
        if c.task_type == "area_coverage":
            xy = xy.abs() * 0.5 + c.boundary * 0.2
        pri = torch.rand(c.n_tasks, 1, generator=g, device=device) * 0.5 + 0.5
        tw = torch.full((c.n_tasks, 1), float(c.max_steps), device=device)
        return torch.cat([xy, pri, tw], dim=-1)

    def move_tasks(self, tasks: torch.Tensor, step: int) -> torch.Tensor:
        if self.cfg.task_type == "pursuit":
            return tasks  # moved in env.step via pursuit helper
        if self.cfg.task_type != "dynamic_tracking":
            return tasks
        t = tasks.clone()
        ang = 0.02 * step * float(getattr(self.cfg, "track_speed", 1.0) or 1.0)
        t[:, 0] = t[:, 0] + 0.05 * torch.cos(torch.tensor(ang, device=tasks.device))
        t[:, 1] = t[:, 1] + 0.05 * torch.sin(torch.tensor(ang, device=tasks.device))
        return t


class DICEVMASEnv:
    """Gymnasium-like multi-agent env: obs per agent = self + neighbors."""

    def __init__(self, **kwargs: Any):
        if not _VMAS_OK:
            raise ImportError("vmas is required: pip install vmas")
        cfg_keys = {f.name for f in DiceEnvConfig.__dataclass_fields__.values()}  # type: ignore
        self.cfg = DiceEnvConfig(**{k: v for k, v in kwargs.items() if k in cfg_keys})
        self.device = torch.device(self.cfg.device)
        self.scenario = DICEUAVScenario(self.cfg)
        self._gen = torch.Generator(device="cpu")
        self._gen.manual_seed(self.cfg.seed)

        self.n_agents = self.cfg.n_agents
        self.n_tasks = self.cfg.n_tasks
        self.n_roles = self.cfg.n_roles
        self.self_dim = 2 + 2 + 1 + 1 + self.n_roles
        self.neigh_dim = 2 + 2 + self.n_roles
        self.obs_dim = self.self_dim + self.cfg.max_neighbors * self.neigh_dim
        self.action_dim = 2 + self.n_roles

        self.UAV_TYPES = UAV_TYPES
        self.uav_type_list: list[int] = [1] * self.n_agents
        self.pos: torch.Tensor
        self.vel: torch.Tensor
        self.yaw: torch.Tensor
        self.batt: torch.Tensor
        self.roles: torch.Tensor
        self.prev_roles: torch.Tensor
        self.tasks: torch.Tensor
        self.task_done: torch.Tensor
        self.alive: torch.Tensor
        self.step_count = 0
        self.assignment: torch.Tensor
        self.trap_center: torch.Tensor
        self.captured: bool = False
        self._last_collision_rate: float = 0.0
        self.evader_vel: torch.Tensor
        self._evader_ppo = None
        if self.cfg.task_type == "pursuit" and self.cfg.evader_policy == "rl":
            from environments.scenarios.adversarial_pursuit import load_evader

            self._evader_ppo = load_evader(self.cfg.n_agents, self.cfg.evader_ckpt or None)

    def _alloc_uav_types(self) -> list[int]:
        c = self.cfg
        if not c.heterogeneous:
            return [1] * c.n_agents
        n_heavy = int(c.n_agents * c.hetero_ratio[0])
        n_standard = int(c.n_agents * c.hetero_ratio[1])
        n_light = c.n_agents - n_heavy - n_standard
        types = [0] * n_heavy + [1] * n_standard + [2] * n_light
        # ponytail: Fisher-Yates via torch generator so seed is reproducible
        order = torch.randperm(c.n_agents, generator=self._gen).tolist()
        return [types[i] for i in order]

    def reset(self, seed: int | None = None) -> tuple[torch.Tensor, dict]:
        if seed is not None:
            self._gen.manual_seed(seed)
        c = self.cfg
        self.uav_type_list = self._alloc_uav_types()
        b = c.boundary * 0.7
        self.pos = (torch.rand(c.n_agents, 2, generator=self._gen) * 2 - 1) * b
        self.pos = self.pos.to(self.device)
        self.vel = torch.zeros(c.n_agents, 2, device=self.device)
        self.yaw = torch.zeros(c.n_agents, 1, device=self.device)
        self.batt = torch.ones(c.n_agents, device=self.device)
        # hetero: optional soft-match roles to airframe (disable for emergence training)
        if c.heterogeneous and c.hetero_role_bias:
            pref = {0: min(2, c.n_roles - 1), 1: min(1, c.n_roles - 1), 2: 0}
            self.roles = torch.tensor(
                [pref[t] for t in self.uav_type_list], dtype=torch.long, device=self.device
            )
        else:
            self.roles = torch.zeros(c.n_agents, dtype=torch.long, device=self.device)
        self.prev_roles = self.roles.clone()
        self.tasks = self.scenario.sample_tasks(self._gen).to(self.device)
        self.task_done = torch.zeros(c.n_tasks, dtype=torch.bool, device=self.device)
        self.alive = torch.ones(c.n_agents, dtype=torch.bool, device=self.device)
        self.assignment = torch.full((c.n_agents,), -1, dtype=torch.long, device=self.device)
        self.trap_center = torch.zeros(2, device=self.device)
        self.captured = False
        self._last_collision_rate = 0.0
        self.evader_vel = torch.zeros(2, device=self.device)
        self.step_count = 0
        obs = self._build_obs()
        return obs, self._info()

    def neighbor_mask(self) -> torch.Tensor:
        dist = torch.cdist(self.pos, self.pos)
        mask = (dist < self.cfg.comm_radius) & (dist > 0)
        mask = mask & self.alive.unsqueeze(0) & self.alive.unsqueeze(1)
        return mask

    def _role_oh(self, roles: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(roles.clamp(0, self.n_roles - 1), self.n_roles).float()

    def _build_obs(self) -> torch.Tensor:
        c = self.cfg
        pos, vel = self.pos, self.vel
        if c.sensor_noise > 0:
            pos = pos + torch.randn_like(pos) * c.sensor_noise
        if c.vel_noise > 0:
            vel = vel + torch.randn_like(vel) * c.vel_noise
        self_feat = torch.cat(
            [pos, vel, self.yaw, self.batt.unsqueeze(-1), self._role_oh(self.roles)],
            dim=-1,
        )
        dist = torch.cdist(self.pos, self.pos)
        dist = dist + torch.eye(c.n_agents, device=self.device) * 1e6
        knn = []
        for i in range(c.n_agents):
            d = dist[i].clone()
            d[~self.alive] = 1e6
            d[i] = 1e6
            vals, idx = torch.topk(-d, k=min(c.max_neighbors, c.n_agents - 1))
            idx = idx[(-vals) < c.comm_radius]
            feats = []
            for j in idx.tolist():
                pj = self.pos[j] + (torch.randn_like(self.pos[j]) * c.sensor_noise if c.sensor_noise > 0 else 0)
                vj = self.vel[j] + (torch.randn_like(self.vel[j]) * c.vel_noise if c.vel_noise > 0 else 0)
                feats.append(torch.cat([pj - pos[i], vj, self._role_oh(self.roles[j : j + 1])[0]]))
            while len(feats) < c.max_neighbors:
                feats.append(torch.zeros(self.neigh_dim, device=self.device))
            knn.append(torch.stack(feats[: c.max_neighbors]))
        neigh = torch.stack(knn).reshape(c.n_agents, -1)
        obs = torch.cat([self_feat, neigh], dim=-1)
        if c.drop_rate > 0:
            drop = torch.rand(c.n_agents, device=self.device) < c.drop_rate
            obs = obs.clone()
            obs[drop] = 0
        return obs

    def _apply_dynamics(self, dxdy: torch.Tensor) -> None:
        import math

        c = self.cfg
        if c.max_turn_deg < 180:
            max_rad = c.max_turn_deg * math.pi / 180.0
            cur = torch.atan2(self.vel[:, 1], self.vel[:, 0] + 1e-8)
            des = torch.atan2(dxdy[:, 1], dxdy[:, 0] + 1e-8)
            delta = (des - cur + math.pi) % (2 * math.pi) - math.pi
            delta = delta.clamp(-max_rad, max_rad)
            mag = dxdy.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            ang = cur + delta
            dxdy = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1) * mag

        if c.heterogeneous:
            max_vel = torch.tensor(
                [float(UAV_TYPES[t]["max_vel"]) for t in self.uav_type_list],
                device=self.device,
            ).unsqueeze(-1)
            max_acc = torch.tensor(
                [float(UAV_TYPES[t]["max_acc"]) for t in self.uav_type_list],
                device=self.device,
            ).unsqueeze(-1)
            v_des = dxdy.clamp(-1, 1) * max_vel
            acc = (v_des - self.vel) / max(c.dt, 1e-3)
            acc = acc.clamp(-max_acc, max_acc)
            self.vel = (self.vel + acc * c.dt).clamp(-max_vel, max_vel)
        else:
            speed_cap = 3.0 if not math.isfinite(c.max_vel) else float(c.max_vel)
            v_des = dxdy.clamp(-1, 1) * speed_cap
            acc = (v_des - self.vel) / max(c.dt, 1e-3)
            if math.isfinite(c.max_acc):
                acc = acc.clamp(-c.max_acc, c.max_acc)
            self.vel = self.vel + acc * c.dt
            if math.isfinite(c.max_vel):
                self.vel = self.vel.clamp(-c.max_vel, c.max_vel)
        self.vel = self.vel * self.alive.unsqueeze(-1).float()
        self.pos = self.pos + self.vel * c.dt
        self.yaw = torch.atan2(self.vel[:, 1:2] + 1e-6, self.vel[:, 0:1] + 1e-6)

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool, bool, dict]:
        """actions: (n_agents, 2+n_roles) — dx,dy + role scores."""
        c = self.cfg
        if actions.dim() == 1:
            actions = actions.view(c.n_agents, -1)
        actions = actions.to(self.device)
        dxdy = actions[:, :2].clamp(-1, 1)
        role_scores = actions[:, 2 : 2 + self.n_roles]
        self.prev_roles = self.roles.clone()
        new_roles = role_scores.argmax(dim=-1)
        conf = role_scores.softmax(-1).max(dim=-1).values
        switch = conf > 0.4
        self.roles = torch.where(switch & self.alive, new_roles, self.roles)

        self._apply_dynamics(dxdy)
        self.batt = (self.batt - 0.001 * self.alive.float()).clamp(0, 1)

        rewards = torch.zeros(c.n_agents, device=self.device)
        if c.task_type == "pursuit":
            from environments.scenarios.pursuit import pursuit_rewards, step_evader

            e_xy = self.tasks[0, :2]
            if c.evader_policy == "rl" and self._evader_ppo is not None:
                from environments.scenarios.adversarial_pursuit import rl_evader_step

                e_xy, self.evader_vel, _, _, _ = rl_evader_step(
                    self._evader_ppo,
                    e_xy,
                    self.evader_vel,
                    self.pos,
                    boundary=c.boundary,
                    dt=c.dt,
                    max_speed=c.evader_speed * 1.15,
                    deterministic=True,
                )
            else:
                e_xy = step_evader(
                    e_xy,
                    self.pos,
                    self.alive,
                    speed=c.evader_speed,
                    boundary=c.boundary,
                    dt=c.dt,
                )
                self.evader_vel = (e_xy - self.tasks[0, :2]) / max(c.dt, 1e-3)
            self.tasks = self.tasks.clone()
            self.tasks[0, :2] = e_xy
            pr, captured, meta = pursuit_rewards(
                self.pos, self.alive, e_xy, self.trap_center, capture_radius=c.capture_radius
            )
            rewards = rewards + pr
            self.captured = captured
            if captured:
                self.task_done[:] = True
        else:
            self.tasks = self.scenario.move_tasks(self.tasks, self.step_count)
            if (~self.task_done).any():
                d = torch.cdist(self.pos, self.tasks[:, :2])
                for ti in range(c.n_tasks):
                    if self.task_done[ti]:
                        continue
                    near = (d[:, ti] < 0.4) & self.alive
                    if near.any():
                        self.task_done[ti] = True
                        rewards[near] += 1.0 * float(self.tasks[ti, 2])

        dist = torch.cdist(self.pos, self.pos)
        eye = torch.eye(c.n_agents, device=self.device)
        coll = ((dist + eye * 10) < 0.3).float().sum(dim=-1)
        self._last_collision_rate = float((coll > 0).float().mean())
        rewards = rewards - 0.2 * coll
        n_edges = float(self.neighbor_mask().float().sum())
        rewards = rewards - 0.001 * n_edges / max(c.n_agents, 1)

        self.step_count += 1
        done = bool(self.step_count >= c.max_steps or bool(self.task_done.all()) or self.captured)
        trunc = bool(self.step_count >= c.max_steps)
        obs = self._build_obs()
        return obs, rewards, done, trunc, self._info()

    def _info(self) -> dict:
        return {
            "roles": self.roles.clone(),
            "prev_roles": self.prev_roles.clone(),
            "task_types": torch.zeros(self.n_tasks, dtype=torch.long, device=self.device),
            "task_done": self.task_done.clone(),
            "tasks": self.tasks.clone(),
            "coverage": float(self.task_done.float().mean()) if self.cfg.task_type != "pursuit" else float(self.captured),
            "alive": self.alive.clone(),
            "step": self.step_count,
            "vmas_version": _VMAS_VER,
            "assignment": self.assignment.clone(),
            "collision_rate": self._last_collision_rate,
            "captured": self.captured,
            "uav_types": list(self.uav_type_list),
        }

    def random_actions(self) -> torch.Tensor:
        a = torch.randn(self.n_agents, self.action_dim, generator=self._gen)
        return a.to(self.device)


def self_check() -> None:
    env = DICEVMASEnv(n_agents=8, n_tasks=3, max_steps=20, device="cpu", sensor_noise=0.05, max_acc=2.0, max_vel=3.0)
    obs, info = env.reset(seed=0)
    assert obs.shape == (8, env.obs_dim)
    for _ in range(10):
        obs, r, done, trunc, info = env.step(env.random_actions())
        if done:
            break
    env2 = DICEVMASEnv(n_agents=6, n_tasks=1, max_steps=30, task_type="pursuit", evader_speed=1.0)
    obs, _ = env2.reset(0)
    for _ in range(15):
        obs, r, done, trunc, info = env2.step(env2.random_actions())
        if done:
            break
    assert "coverage" in info
    env3 = DICEVMASEnv(n_agents=10, n_tasks=2, max_steps=15, heterogeneous=True, hetero_ratio=(0.3, 0.4, 0.3))
    obs, info = env3.reset(seed=1)
    assert len(info["uav_types"]) == 10
    assert set(info["uav_types"]) <= {0, 1, 2}
    assert info["uav_types"].count(0) == 3
    for _ in range(5):
        obs, r, done, trunc, info = env3.step(env3.random_actions())
    print(f"dice_vmas_env: OK (vmas={_VMAS_VER}, obs={env.obs_dim}, pursuit_ok, hetero_ok)")


if __name__ == "__main__":
    self_check()

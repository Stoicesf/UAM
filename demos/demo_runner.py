"""Environment + CompleteController step loop for demos."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from dice.failure_injector import FailureInjector
from dice.local_obs import neighbor_mask
from dice.safety_shield import SafetyShield
from demos.scene_library import DemoScene, env_kwargs
from environments.dice_vmas_env import DICEVMASEnv
from models.complete_controller import CompleteController
from visualization.plot_utils import LEVEL_BYTES

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class DemoFrame:
    step: int
    pos: torch.Tensor
    vel: torch.Tensor
    roles: torch.Tensor
    alive: torch.Tensor
    tasks: torch.Tensor
    task_done: torch.Tensor
    blockers: torch.Tensor
    send: torch.Tensor
    levels: torch.Tensor
    links: list[dict[str, Any]]
    reward: float
    coverage: float
    bandwidth_hz: float
    bandwidth_max: float
    bytes_step: float
    bytes_cum: float
    level_counts: dict[int, int]
    gate_trace: list[dict[str, Any]]
    correction: torch.Tensor | None = None
    done: bool = False
    uav_types: list[int] = field(default_factory=list)
    uav_sizes: list[float] = field(default_factory=list)
    payload_pos: torch.Tensor | None = None
    target_pos: torch.Tensor | None = None
    payload_distance: float = 0.0


@dataclass
class DemoRunner:
    scene: DemoScene
    n_agents: int = 16
    seed: int = 42
    load_encoder_ckpt: bool = True
    use_projection_shield: bool = True
    shield_type: str | None = None  # None→hard if use_projection_shield; hard|cbf|none
    heterogeneous: bool = False
    hetero_ratio: tuple[float, float, float] = (0.3, 0.4, 0.3)
    match_bias: bool = True  # False = --no_match_bias (test pure emergence)
    ctrl: str = "heuristic"  # transport: heuristic | theory
    use_hybrid: bool = False  # transport: HybridPayloadDynamics
    wind_force: float = 0.0  # transport: external wind magnitude
    use_rise: bool = True
    use_traj: bool = False
    use_shield: bool = False

    env: DICEVMASEnv = field(init=False)
    ctl: CompleteController = field(init=False)
    obs: torch.Tensor = field(init=False)
    blockers: torch.Tensor = field(init=False)
    bytes_cum: float = 0.0
    level_hist: dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0, 2: 0, 3: 0})
    reward_acc: float = 0.0
    injected: bool = False
    fi: FailureInjector = field(init=False)
    link_topk: int = 0  # 0 → auto
    history_len: int = 50
    history: dict[str, list] = field(default_factory=dict)
    _transport_ctl: Any = field(init=False, default=None)
    _transport_mode: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.reset(self.seed)

    def _clear_history(self) -> None:
        self.history = {
            "step": [],
            "pos": [],
            "role": [],
            "level": [],
            "send": [],
            "gain": [],
            "cost": [],
            "alive": [],
        }

    def _push_history(self, frame: "DemoFrame") -> None:
        h = self.history
        h["step"].append(frame.step)
        h["pos"].append(frame.pos.detach().cpu().clone())
        h["role"].append(frame.roles.detach().cpu().clone())
        h["level"].append(frame.levels.detach().cpu().clone())
        h["send"].append(frame.send.detach().cpu().clone())
        gains = torch.zeros(self.n_agents)
        costs = torch.zeros(self.n_agents)
        for row in frame.gate_trace:
            i = int(row.get("src_id", 0))
            if 0 <= i < self.n_agents:
                gains[i] = float(row.get("gain", 0.0))
                costs[i] = float(row.get("cost", 0.0))
        h["gain"].append(gains)
        h["cost"].append(costs)
        h["alive"].append(frame.alive.detach().cpu().clone())
        while len(h["step"]) > self.history_len:
            for k in h:
                h[k].pop(0)

    def agent_timeline(self, agent_id: int, last_n: int = 10) -> dict[str, Any]:
        """Slice recent history for one UAV (for explain panel)."""
        h = self.history
        if not h.get("step"):
            return {"agent_id": agent_id, "steps": [], "roles": [], "levels": [], "pos": [], "gains": [], "costs": []}
        n = min(last_n, len(h["step"]))
        sl = slice(-n, None)
        pos = [p[agent_id].tolist() for p in h["pos"][sl]]
        return {
            "agent_id": agent_id,
            "steps": h["step"][sl],
            "roles": [int(r[agent_id]) for r in h["role"][sl]],
            "levels": [int(lv[agent_id]) for lv in h["level"][sl]],
            "pos": pos,
            "gains": [float(g[agent_id]) for g in h["gain"][sl]],
            "costs": [float(c[agent_id]) for c in h["cost"][sl]],
            "sends": [bool(s[agent_id]) for s in h["send"][sl]],
        }

    def reset(self, seed: int | None = None) -> DemoFrame:
        if seed is not None:
            self.seed = seed
        torch.manual_seed(self.seed)
        self._transport_mode = self.scene.name == "transport" or self.scene.task_type == "transport"
        if self._transport_mode:
            from environments.scenarios.cooperative_transport import (
                CooperativeTransportEnv,
                TransportHeuristicController,
            )

            self.env = CooperativeTransportEnv(  # type: ignore[assignment]
                n_agents=self.n_agents,
                target_pos=(5.0, 5.0),
                max_steps=self.scene.max_steps,
                seed=self.seed,
                hetero_ratio=tuple(
                    getattr(self.scene, "hetero_ratio", None) or self.hetero_ratio
                ),
                use_hybrid=bool(self.use_hybrid),
                wind_force=float(self.wind_force),
            )
            self.obs, info = self.env.reset(self.seed)
            if self.ctrl == "theory":
                from models.transport.control.hierarchical import (
                    TransportHierarchicalController,
                )

                self._transport_ctl = TransportHierarchicalController(
                    self.env,
                    use_rise=bool(self.use_rise),
                    use_traj=bool(self.use_traj),
                    use_shield=bool(self.use_shield),
                )
            else:
                self._transport_ctl = TransportHeuristicController(self.env)
            self.ctl = None  # type: ignore[assignment]
            self.blockers = torch.zeros(0, 2)
            self.fi = FailureInjector(self.n_agents)
            self.bytes_cum = 0.0
            self.level_hist = {0: 0, 1: 0, 2: 0, 3: 0}
            self.reward_acc = 0.0
            self.injected = False
            self._clear_history()
            fr = self._frame(
                step=0,
                send=torch.zeros(self.n_agents, dtype=torch.bool),
                levels=torch.zeros(self.n_agents, dtype=torch.long),
                links=[],
                reward=0.0,
                bytes_step=0.0,
                gate_trace=[],
                done=False,
                correction=None,
                coverage=float(info.get("coverage", 0.0)),
                payload_pos=info.get("payload_pos"),
                target_pos=info.get("target_pos"),
                payload_distance=float(info.get("payload_distance", 0.0)),
            )
            self._push_history(fr)
            return fr

        kw = env_kwargs(self.scene, self.n_agents)
        kw["heterogeneous"] = self.heterogeneous or bool(getattr(self.scene, "heterogeneous", False))
        ratio = getattr(self.scene, "hetero_ratio", None) or self.hetero_ratio
        kw["hetero_ratio"] = tuple(ratio)
        kw["hetero_role_bias"] = bool(self.match_bias)
        self.env = DICEVMASEnv(**kw, seed=self.seed)
        self.obs, info = self.env.reset(self.seed)
        st = self.shield_type
        if st is None:
            st = "hard" if self.use_projection_shield else "none"
        self.ctl = CompleteController(
            self.n_agents,
            self.scene.n_tasks,
            self.env.obs_dim,
            n_roles=self.env.n_roles,
            use_semantic=True,
            use_icps_resource=True,
            use_dice=True,
            use_projection_shield=self.use_projection_shield,
            shield_type=st,
            comm_delay=int(self.scene.comm_delay),
        )
        self.ctl.resource.bandwidth_hz *= self.scene.bandwidth_scale
        if self.load_encoder_ckpt:
            ckpt = ROOT / "experiment_results" / "semantic" / "semantic_encoder.pt"
            if ckpt.exists():
                payload = torch.load(ckpt, map_location="cpu", weights_only=False)
                if payload.get("obs_dim") == self.env.obs_dim:
                    self.ctl.encoder.load_state_dict(payload["state_dict"])
            bias_ckpt = ROOT / "experiment_results" / "coevolution" / "role_bias.pt"
            if bias_ckpt.exists():
                b = torch.load(bias_ckpt, map_location="cpu", weights_only=False)
                if "role_level_bias" in b and b["role_level_bias"].shape == self.ctl.gate.role_level_bias.shape:
                    self.ctl.gate.role_level_bias.data.copy_(b["role_level_bias"])
        # static blockers from obstacle_density
        n_obs = int(round(self.scene.obstacle_density * 10))
        if n_obs > 0:
            g = torch.Generator().manual_seed(self.seed + 7)
            self.blockers = (torch.rand(n_obs, 2, generator=g) * 2 - 1) * self.env.cfg.boundary * 0.6
        else:
            self.blockers = torch.zeros(0, 2)
        self.fi = FailureInjector(self.n_agents)
        self.bytes_cum = 0.0
        self.level_hist = {0: 0, 1: 0, 2: 0, 3: 0}
        self.reward_acc = 0.0
        self.injected = False
        self._clear_history()
        fr = self._frame(
            step=0,
            send=torch.zeros(self.n_agents, dtype=torch.bool),
            levels=torch.zeros(self.n_agents, dtype=torch.long),
            links=[],
            reward=0.0,
            bytes_step=0.0,
            gate_trace=[],
            done=False,
            correction=None,
            coverage=float(info.get("coverage", 0.0)),
        )
        self._push_history(fr)
        return fr

    def _build_links(
        self,
        send: torch.Tensor,
        levels: torch.Tensor,
        *,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Broadcast links: each sending agent → neighbors in comm radius."""
        pos = self.env.pos
        mask = neighbor_mask(pos, self.env.cfg.comm_radius, self.env.alive)
        links: list[dict[str, Any]] = []
        n = self.n_agents
        for i in range(n):
            if not bool(send[i]) or not bool(self.env.alive[i]):
                continue
            lv = int(levels[i])
            if lv <= 0:
                continue
            # optional packet drop
            if self.scene.comm_loss > 0 and torch.rand(()) < self.scene.comm_loss:
                continue
            nbrs = mask[i].nonzero(as_tuple=False).view(-1).tolist()
            for j in nbrs:
                dist = float((pos[i] - pos[j]).norm())
                links.append(
                    {
                        "src": i,
                        "dst": j,
                        "level": lv,
                        "bytes": LEVEL_BYTES.get(lv, 0),
                        "dist": dist,
                    }
                )
        if top_k is None:
            top_k = self.link_topk if self.link_topk > 0 else min(n * 2, 64)
        if len(links) > top_k:
            links.sort(key=lambda x: x["bytes"], reverse=True)
            links = links[:top_k]
        return links

    def step(self) -> DemoFrame:
        env = self.env
        if self._transport_mode:
            act = self._transport_ctl.act(self.obs)
            self.obs, r, done, trunc, info = env.step(act)
            reward = float(r.mean()) if torch.is_tensor(r) else float(r)
            self.reward_acc += reward
            # visual cables: UAV → payload (encoded as links to self with level 2)
            links: list[dict[str, Any]] = []
            for i in range(self.n_agents):
                dist = float((env.pos[i] - env.payload_pos).norm())
                links.append(
                    {"src": i, "dst": i, "level": 2, "bytes": 0, "dist": dist, "to_payload": True}
                )
            fr = self._frame(
                step=env.step_count,
                send=torch.zeros(self.n_agents, dtype=torch.bool),
                levels=torch.zeros(self.n_agents, dtype=torch.long),
                links=links,
                reward=self.reward_acc,
                bytes_step=0.0,
                gate_trace=[],
                done=bool(done or trunc or env.step_count >= self.scene.max_steps),
                correction=None,
                coverage=float(info.get("coverage", 0.0)),
                payload_pos=info.get("payload_pos"),
                target_pos=info.get("target_pos"),
                payload_distance=float(info.get("payload_distance", 0.0)),
            )
            self._push_history(fr)
            return fr

        # adversarial / mixed failure injection
        if (
            self.scene.failure_ratio > 0
            and not self.injected
            and env.step_count >= int(self.scene.max_steps * self.scene.kill_at_frac)
        ):
            env.alive = self.fi.inject("node_kill", self.scene.failure_ratio, env.alive)
            self.injected = True

        gate_trace: list[dict[str, Any]] = []
        out = self.ctl.step(
            self.obs,
            env.pos,
            env.vel,
            env.tasks,
            env.roles,
            env.alive,
            env.task_done,
            env.assignment,
            snr_db=self.scene.snr_db,
            gate_trace=gate_trace,
        )
        send, levels = out.send, out.semantic_level
        # stamp agent ids onto per-agent gate rows
        for i, row in enumerate(gate_trace):
            row["src_id"] = i
            row["selected_level"] = int(levels[i]) if i < levels.numel() else row.get("selected_level", 0)

        links = self._build_links(send, levels)
        bytes_step = float(sum(L["bytes"] for L in links))
        self.bytes_cum += bytes_step
        for lv in levels.tolist():
            self.level_hist[int(lv)] = self.level_hist.get(int(lv), 0) + 1

        act = torch.zeros(self.n_agents, env.action_dim)
        dxdy = out.dxdy
        if self.scene.task_type == "pursuit" and env.tasks.numel() and self.scene.evader_policy != "rl":
            import math

            e = env.tasks[0, :2]
            ring_w = 0.65
            ring = []
            for i in range(self.n_agents):
                ang = 2 * math.pi * i / max(self.n_agents, 1)
                goal = e + 1.1 * torch.tensor([math.cos(ang), math.sin(ang)], device=e.device)
                ring.append((goal - env.pos[i]).clamp(-1, 1))
            ring_t = torch.stack(ring)
            dxdy = ((1.0 - ring_w) * dxdy + ring_w * ring_t).clamp(-1, 1)
            scout = out.roles == 0
            if scout.any():
                dxdy[scout] = (0.25 * out.dxdy[scout] + 0.75 * (e - env.pos[scout]).clamp(-1, 1)).clamp(-1, 1)
        elif self.scene.task_type == "pursuit" and env.tasks.numel() and self.scene.evader_policy == "rl":
            # light chase only — no perfect ring (lets RL evader stress the swarm)
            e = env.tasks[0, :2]
            dxdy = (0.55 * dxdy + 0.45 * (e - env.pos).clamp(-1, 1)).clamp(-1, 1)
        act[:, :2] = dxdy
        # hetero: optional hard bias toward airframe-matched roles
        if self.env.cfg.heterogeneous and self.match_bias:
            pref = {0: min(2, env.n_roles - 1), 1: min(1, env.n_roles - 1), 2: 0}
            for i, t in enumerate(self.env.uav_type_list):
                act[i, 2 + pref[t]] = 2.0
        else:
            act[torch.arange(self.n_agents), 2 + out.roles.clamp(0, env.n_roles - 1)] = 2.0
        # soft repulsion from blockers
        if self.blockers.numel():
            d = (env.pos.unsqueeze(1) - self.blockers.unsqueeze(0)).norm(dim=-1)
            for i in range(self.n_agents):
                for j in range(self.blockers.shape[0]):
                    if float(d[i, j]) < 0.4 and float(d[i, j]) > 1e-6:
                        act[i, :2] += 0.25 * (env.pos[i] - self.blockers[j]) / d[i, j]

        corr = None
        st = getattr(self.ctl, "shield_type", "hard" if self.use_projection_shield else "none")
        if st == "cbf":
            from dice.cbf_projection import project_dxdy_cbf

            _, corr = project_dxdy_cbf(dxdy, env.pos, env.vel, return_correction=True)
        elif st == "hard":
            from dice.safety_shield_projection import project_action

            _, corr = project_action(dxdy, env.pos, return_correction=True)

        env.pos, env.vel = SafetyShield.apply(env.pos, env.vel)
        self.obs, r, done, trunc, info = env.step(act)
        reward = float(r.mean()) if torch.is_tensor(r) else float(r)
        if self.env.cfg.heterogeneous and self.match_bias:
            from dice.role_reward import RoleReward

            rr = RoleReward(env.n_roles)
            reward += float(rr.capability_match_bonus(env.roles, env.uav_type_list).mean())
        self.reward_acc += reward
        fr = self._frame(
            step=env.step_count,
            send=send,
            levels=levels,
            links=links,
            reward=self.reward_acc,
            bytes_step=bytes_step,
            gate_trace=gate_trace,
            done=bool(done or trunc or env.step_count >= self.scene.max_steps),
            correction=corr,
            coverage=float(info.get("coverage", 0.0)),
        )
        self._push_history(fr)
        return fr

    def _frame(
        self,
        *,
        step: int,
        send: torch.Tensor,
        levels: torch.Tensor,
        links: list[dict[str, Any]],
        reward: float,
        bytes_step: float,
        gate_trace: list[dict[str, Any]],
        done: bool,
        correction: torch.Tensor | None,
        coverage: float,
        payload_pos: torch.Tensor | None = None,
        target_pos: torch.Tensor | None = None,
        payload_distance: float = 0.0,
    ) -> DemoFrame:
        types = list(getattr(self.env, "uav_type_list", []) or [])
        from environments.dice_vmas_env import UAV_TYPES

        hetero = bool(getattr(getattr(self.env, "cfg", None), "heterogeneous", False)) or self._transport_mode
        if hetero and types:
            sizes = [float(UAV_TYPES[t]["size"]) * 80 for t in types]
        else:
            types, sizes = [], []

        if self._transport_mode:
            tgt = getattr(self.env, "target_pos", None)
            tasks = tgt.detach().clone().view(1, -1) if tgt is not None else torch.zeros(0, 2)
            task_done = torch.zeros(tasks.shape[0], dtype=torch.bool)
            roles = self.env.roles.detach().clone()
            alive = self.env.alive.detach().clone()
            bw_hz, bw_max = 1.0, 1.0
            if payload_pos is None:
                payload_pos = getattr(self.env, "payload_pos", None)
                if payload_pos is not None:
                    payload_pos = payload_pos.detach().clone()
            if target_pos is None and tgt is not None:
                target_pos = tgt.detach().clone()
            if payload_distance == 0.0 and payload_pos is not None and target_pos is not None:
                payload_distance = float((payload_pos - target_pos).norm())
        else:
            tasks = self.env.tasks.detach().clone()
            task_done = self.env.task_done.detach().clone()
            roles = self.env.roles.detach().clone()
            alive = self.env.alive.detach().clone()
            bw_hz = float(self.ctl.resource.bandwidth_hz)
            bw_max = float(self.ctl.limits.bandwidth_max)

        return DemoFrame(
            step=step,
            pos=self.env.pos.detach().clone(),
            vel=self.env.vel.detach().clone(),
            roles=roles,
            alive=alive,
            tasks=tasks,
            task_done=task_done,
            blockers=self.blockers.detach().clone(),
            send=send.detach().clone(),
            levels=levels.detach().clone(),
            links=links,
            reward=reward,
            coverage=coverage,
            bandwidth_hz=bw_hz,
            bandwidth_max=bw_max,
            bytes_step=bytes_step,
            bytes_cum=self.bytes_cum,
            level_counts=dict(self.level_hist),
            gate_trace=gate_trace,
            correction=None if correction is None else correction.detach().clone(),
            done=done,
            uav_types=types,
            uav_sizes=sizes,
            payload_pos=None if payload_pos is None else payload_pos.detach().clone(),
            target_pos=None if target_pos is None else target_pos.detach().clone(),
            payload_distance=payload_distance,
        )

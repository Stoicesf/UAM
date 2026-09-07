#!/usr/bin/env python3
"""AAS Docker bridge driven by TransportHierarchicalController.

Host brain: theory controller + virtual cable payload → ENU set_reposition
into AAS aircraft containers. Does NOT use host MAVROS / AttitudeTarget
(see RELEASE_NOTES / aas_transport_bridge).

Usage (WSL, sim up):
  python3 ros_nodes/transport_theory_bridge.py --uav_count 4 --target_x 5 --target_y 0

Dry-run:
  python3 ros_nodes/transport_theory_bridge.py --dry_run --steps 80 --hybrid
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ros_nodes"))

from aas_cmd_bridge import (  # noqa: E402
    set_reposition,
    takeoff,
    wait_containers,
    wait_preflight,
)
from environments.dynamics.hybrid_payload import HybridPayloadDynamics  # noqa: E402
from environments.dynamics.transport_payload import PayloadDynamics  # noqa: E402
from environments.dynamics.uav_dynamics import UAV_DYN_BY_TYPE  # noqa: E402
from models.transport.control.hierarchical import (  # noqa: E402
    TransportHierarchicalController,
)


class BridgeEnv:
    """Minimal env duck-type for TransportHierarchicalController (reads live tensors)."""

    def __init__(
        self,
        n_agents: int,
        target: torch.Tensor,
        *,
        L0: float = 3.0,
        payload_mass: float = 5.0,
        use_hybrid: bool = True,
        wind_force: float = 0.0,
        dt: float = 0.05,
        max_steps: int = 200,
    ):
        self.n_agents = int(n_agents)
        self.L0 = float(L0)
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.boundary = 50.0
        self.wind_force = float(wind_force)
        self.use_hybrid = bool(use_hybrid)
        self.target_pos = target.float()
        self.uav_type_list = [1] * self.n_agents  # standard airframe
        if use_hybrid:
            self.payload: Any = HybridPayloadDynamics(
                mass=payload_mass, spring_k=30.0, damping_d=12.0, L0=L0, g=0.0
            )
            self.payload.reset_state(self.n_agents)
        else:
            self.payload = PayloadDynamics(
                mass=payload_mass, spring_k=30.0, damping_d=12.0, L0=L0
            )
        self.pos = torch.zeros(self.n_agents, 2)
        self.vel = torch.zeros(self.n_agents, 2)
        self.psi = torch.zeros(self.n_agents, 1)
        self.payload_pos = torch.zeros(2)
        self.payload_vel = torch.zeros(2)
        self.obstacles = torch.zeros(0, 2)
        self._rise_payload_assist: torch.Tensor | None = None

    def _ideal_ring(self, lead: float = 0.15) -> torch.Tensor:
        angles = torch.linspace(0, 2 * math.pi, self.n_agents + 1)[:-1]
        masses = torch.tensor(
            [UAV_DYN_BY_TYPE[t][0] for t in self.uav_type_list], dtype=torch.float32
        )
        rad = self.L0 * (1.0 + 0.12 * (masses / masses.mean() - 1.0))
        to_tgt = self.target_pos - self.payload_pos
        dist = float(to_tgt.norm().clamp(min=1e-6))
        unit = to_tgt / dist
        center = self.payload_pos + lead * unit
        return center.unsqueeze(0) + rad.unsqueeze(-1) * torch.stack(
            [torch.cos(angles), torch.sin(angles)], dim=-1
        )


def _formation_error(env: BridgeEnv) -> float:
    ideal = env._ideal_ring(lead=0.0)
    return float((env.pos - ideal).norm(dim=-1).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description="AAS theory-controller transport bridge")
    ap.add_argument("--uav_count", type=int, default=4)
    ap.add_argument("--instance", type=int, default=0)
    ap.add_argument("--altitude", type=float, default=12.0)
    ap.add_argument("--rate_hz", type=float, default=0.5, help="reposition rate (keep ≤1 Hz)")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--skip_takeoff", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--target_x", type=float, default=5.0)
    ap.add_argument("--target_y", type=float, default=0.0)
    ap.add_argument("--L0", type=float, default=3.0)
    ap.add_argument("--payload_mass", type=float, default=5.0)
    ap.add_argument("--hybrid", action="store_true", default=True)
    ap.add_argument("--no_hybrid", action="store_true")
    ap.add_argument("--rise", action="store_true", default=True)
    ap.add_argument("--no_rise", action="store_true")
    ap.add_argument("--traj", action="store_true")
    ap.add_argument("--apf", action="store_true")
    ap.add_argument("--wind", type=float, default=0.0)
    ap.add_argument("--accept_dist", type=float, default=1.0)
    ap.add_argument("--no_accept", action="store_true")
    ap.add_argument("--output", type=str, default="")
    args = ap.parse_args()

    use_hybrid = bool(args.hybrid) and not bool(args.no_hybrid)
    use_rise = bool(args.rise) and not bool(args.no_rise)
    n = args.uav_count
    cmd_dt = 1.0 / max(args.rate_hz, 0.1)
    physics_dt = 0.05
    n_sub = max(1, int(round(cmd_dt / physics_dt)))

    target = torch.tensor([args.target_x, args.target_y], dtype=torch.float32)
    env = BridgeEnv(
        n,
        target,
        L0=args.L0,
        payload_mass=args.payload_mass,
        use_hybrid=use_hybrid,
        wind_force=float(args.wind),
        dt=physics_dt,
        max_steps=args.steps * n_sub,
    )

    # init ring around origin
    angles = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
    env.payload_pos = torch.zeros(2)
    env.payload_vel = torch.zeros(2)
    env.pos = env.payload_pos.unsqueeze(0) + args.L0 * torch.stack(
        [torch.cos(angles), torch.sin(angles)], dim=-1
    )
    env.vel = torch.zeros(n, 2)
    env.psi = torch.atan2(env.pos[:, 1:2], env.pos[:, 0:1] + 1e-6)

    ctl = TransportHierarchicalController(
        env,
        use_rise=use_rise,
        use_traj=bool(args.traj),
        use_shield=bool(args.apf),
    )

    if not args.dry_run:
        wait_containers(n, args.instance)
        print(f"found {n} aircraft containers")
        if not args.skip_takeoff:
            for i in range(1, n + 1):
                wait_preflight(i, args.instance)
                takeoff(i, args.altitude, args.instance)
            print("waiting 15s for hover…")
            time.sleep(15.0)
    else:
        print("[dry_run] skipping Docker / takeoff")

    d0 = float((env.payload_pos - env.target_pos).norm())
    form = 0.0
    cable_state = None
    steps_run = 0
    print(
        f"theory transport start d0={d0:.2f} hybrid={use_hybrid} rise={use_rise} "
        f"traj={args.traj} apf={args.apf} wind={args.wind} dry_run={args.dry_run}"
    )

    for step in range(args.steps):
        steps_run = step + 1
        for _ in range(n_sub):
            actions = ctl.act(None)  # [N,2] T, psi_dot
            psi_dot = actions[:, 1:2]
            env.psi = env.psi + psi_dot * physics_dt

            # formation desired XY from controller ring (synced inside act)
            des = env.payload_pos.unsqueeze(0) + ctl.formation.delta
            # slew host estimate toward desired (open-loop setpoint model)
            new_pos = 0.55 * env.pos + 0.45 * des
            env.vel = ((new_pos - env.pos) / max(physics_dt, 1e-3)).clamp(-2.0, 2.0)
            env.pos = new_pos

            pr = env.payload(
                env.pos, env.vel, env.payload_pos, env.payload_vel, physics_dt
            )
            env.payload_pos = pr["new_payload_pos"]
            env.payload_vel = pr["new_payload_vel"]
            if "state" in pr:
                cable_state = pr["state"].detach().cpu().tolist()

            # tow + optional RISE assist / wind (mirror sim policy lightly)
            if env.wind_force <= 0.0:
                centroid = env.pos.mean(dim=0)
                env.payload_vel = 0.85 * env.payload_vel + 0.15 * (
                    centroid - env.payload_pos
                ) / max(physics_dt, 1e-3)
            else:
                assist = getattr(env, "_rise_payload_assist", None)
                if assist is not None:
                    env.payload_vel = env.payload_vel + assist
                away = -(env.target_pos - env.payload_pos)
                away_n = away / away.norm().clamp(min=1e-6)
                env.payload_vel = env.payload_vel + away_n * (0.35 * env.wind_force)
                env.payload_vel = env.payload_vel + torch.tensor(
                    [env.wind_force, 0.0]
                ) / max(float(env.payload.mass), 1e-3) * physics_dt

            env.payload_vel = env.payload_vel.clamp(-2.5, 2.5)
            env.payload_pos = env.payload_pos + env.payload_vel * physics_dt

        # command ENU once per outer step
        for i in range(n):
            east, north = float(env.pos[i, 0]), float(env.pos[i, 1])
            if args.dry_run:
                continue
            try:
                set_reposition(i + 1, east, north, args.altitude, args.instance)
            except Exception as e:
                print(f"[warn] reposition Drone{i+1}: {e}")

        dist = float((env.payload_pos - env.target_pos).norm())
        form = _formation_error(env)
        cs = ""
        if cable_state is not None:
            cs = f" cable={cable_state}"
        print(
            f"step={step} payload_d={dist:.2f} form={form:.2f} "
            f"pos0=({float(env.pos[0,0]):.1f},{float(env.pos[0,1]):.1f}){cs}"
        )
        if dist < 0.3:
            print("reached target")
            break
        if not args.dry_run:
            time.sleep(cmd_dt)

    d1 = float((env.payload_pos - env.target_pos).norm())
    moved = (d0 - d1) > 1.5 or d1 < args.accept_dist
    print(f"transport_theory_bridge: done d0={d0:.2f} d1={d1:.2f} form={form:.2f} moved={moved}")
    if args.no_accept:
        accept = "skip"
        print("ACCEPT SKIP (--no_accept)")
        rc = 0
    elif not moved or d1 >= args.accept_dist:
        accept = "fail"
        print(
            f"ACCEPT FAIL: need moved and d1 < {args.accept_dist} (got d1={d1:.2f})",
            file=sys.stderr,
        )
        rc = 1
    else:
        accept = "ok"
        print("ACCEPT OK")
        rc = 0

    if args.output:
        out = {
            "d0": d0,
            "d1": d1,
            "moved": moved,
            "form": form,
            "uav_count": n,
            "hybrid": use_hybrid,
            "rise": use_rise,
            "traj": bool(args.traj),
            "apf": bool(args.apf),
            "wind": args.wind,
            "steps": steps_run,
            "accept": accept,
            "cable_state": cable_state,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

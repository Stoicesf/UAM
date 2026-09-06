#!/usr/bin/env python3
"""AAS Docker transport bridge: virtual cable load + ideal ring → set_reposition.

Host brain (UAM) computes ring formation / payload dynamics; injects ENU
waypoints via docker exec into AAS aircraft containers (ROS2 Humble,
ROS_DOMAIN_ID=drone_id). Does NOT use host MAVROS / AttitudeTarget.

ponytail: setpoint-open-loop (host tracks last commanded XY); upgrade: echo
vehicle_local_position. Rope is virtual — no Gazebo cable plugin.

Usage (WSL, while sim_run.sh is up):
  python3 ros_nodes/aas_transport_bridge.py --uav_count 4 --target_x 5 --target_y 0

Dry-run (no Docker; validates payload_distance drop):
  python3 ros_nodes/aas_transport_bridge.py --dry_run --steps 80
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ros_nodes"))

from aas_cmd_bridge import (  # noqa: E402
    container_name,
    set_reposition,
    takeoff,
    wait_containers,
    wait_preflight,
)


def _ideal_ring_np(
    n: int,
    payload: np.ndarray,
    target: np.ndarray,
    L0: float,
    lead: float,
    masses: np.ndarray,
    broken: set[int] | None = None,
) -> np.ndarray:
    """Mass-biased ring around payload+lead toward target. broken UAVs keep last slot skipped."""
    angles = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    rad = L0 * (1.0 + 0.12 * (masses / masses.mean() - 1.0))
    to_tgt = target - payload
    dist = float(np.linalg.norm(to_tgt)) + 1e-6
    unit = to_tgt / dist
    center = payload + lead * unit
    ring = center + rad[:, None] * np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    if broken:
        # leave broken agents where they are (caller blends); mark by returning NaN slot
        for i in broken:
            ring[i] = np.nan
    return ring.astype(np.float32)


def _payload_step_np(
    uav_pos: np.ndarray,
    uav_vel: np.ndarray,
    payload_pos: np.ndarray,
    payload_vel: np.ndarray,
    *,
    mass: float,
    k: float,
    d: float,
    L0: float,
    dt: float,
    wind: np.ndarray,
    broken: set[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spring-damper payload update (numpy). Broken cables → zero tension."""
    n = uav_pos.shape[0]
    tensions = np.zeros((n, 1), dtype=np.float32)
    forces = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        if i in broken:
            continue
        diff = uav_pos[i] - payload_pos
        dist = float(np.linalg.norm(diff)) + 1e-6
        direction = diff / dist
        rel_v = uav_vel[i] - payload_vel
        delta_L = dist - L0
        delta_v = float(np.dot(rel_v, direction))
        T = max(0.0, k * delta_L + d * delta_v)
        tensions[i, 0] = T
        forces[i] = T * direction
    total = forces.sum(axis=0) + wind
    drag = -0.5 * payload_vel
    acc = (total + drag) / max(mass, 1e-3)
    new_vel = payload_vel + acc * dt
    new_pos = payload_pos + new_vel * dt
    return new_pos.astype(np.float32), new_vel.astype(np.float32), tensions


def _formation_error_np(pos: np.ndarray, payload: np.ndarray, L0: float, masses: np.ndarray) -> float:
    ideal = _ideal_ring_np(len(pos), payload, payload + np.array([1.0, 0.0]), L0, 0.0, masses)
    # when target==payload+e1 lead0, center=payload
    return float(np.linalg.norm(pos - ideal, axis=1).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description="AAS transport bridge (virtual cable + reposition)")
    ap.add_argument("--uav_count", type=int, default=4)
    ap.add_argument("--instance", type=int, default=0)
    ap.add_argument("--altitude", type=float, default=12.0)
    ap.add_argument("--rate_hz", type=float, default=0.5, help="reposition rate (keep ≤1 Hz)")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--skip_takeoff", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="no Docker; host-side physics only")
    ap.add_argument("--target_x", type=float, default=5.0)
    ap.add_argument("--target_y", type=float, default=0.0)
    ap.add_argument("--L0", type=float, default=3.0)
    ap.add_argument("--payload_mass", type=float, default=5.0)
    ap.add_argument("--payload_mass_scale", type=float, default=1.0)
    ap.add_argument(
        "--cable_break",
        type=int,
        default=-1,
        help="0-based UAV index with broken cable (-1=none)",
    )
    ap.add_argument("--wind", type=float, default=0.0, help="virtual wind force on payload (N, +east)")
    ap.add_argument("--lead", type=float, default=0.18)
    ap.add_argument("--accept_dist", type=float, default=2.0, help="pass if final dist < this")
    ap.add_argument(
        "--no_accept",
        action="store_true",
        help="skip distance gate (for stress: cable_break / wind)",
    )
    args = ap.parse_args()

    n = args.uav_count
    dt = 1.0 / max(args.rate_hz, 0.1)
    target = np.array([args.target_x, args.target_y], dtype=np.float32)
    mass = float(args.payload_mass * args.payload_mass_scale)
    broken: set[int] = set()
    if 0 <= args.cable_break < n:
        broken.add(int(args.cable_break))
    wind = np.array([args.wind, 0.0], dtype=np.float32)
    # equal masses for ring bias when hetero not specified
    masses = np.ones(n, dtype=np.float32)

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

    # init: ring around origin payload
    payload_pos = np.zeros(2, dtype=np.float32)
    payload_vel = np.zeros(2, dtype=np.float32)
    angles = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    pos = (payload_pos + args.L0 * np.stack([np.cos(angles), np.sin(angles)], axis=-1)).astype(
        np.float32
    )
    vel = np.zeros((n, 2), dtype=np.float32)

    d0 = float(np.linalg.norm(payload_pos - target))
    print(
        f"transport start d0={d0:.2f} mass={mass:.1f} wind={args.wind} "
        f"broken={sorted(broken) or 'none'} dry_run={args.dry_run}"
    )

    for step in range(args.steps):
        ideal = _ideal_ring_np(n, payload_pos, target, args.L0, args.lead, masses, broken)
        new_pos = pos.copy()
        for i in range(n):
            if i in broken or not np.isfinite(ideal[i]).all():
                continue
            # slew host estimate toward ideal (open-loop setpoint model)
            new_pos[i] = 0.55 * pos[i] + 0.45 * ideal[i]
        vel = np.clip((new_pos - pos) / max(dt, 1e-3), -2.0, 2.0)
        pos = new_pos

        payload_pos, payload_vel, tensions = _payload_step_np(
            pos,
            vel,
            payload_pos,
            payload_vel,
            mass=mass,
            k=30.0,
            d=12.0,
            L0=args.L0,
            dt=min(dt, 0.1),
            wind=wind,
            broken=broken,
        )
        # extra tow toward active ring centroid (virtual cable + lead)
        active = [i for i in range(n) if i not in broken]
        if active:
            centroid = pos[active].mean(axis=0)
            payload_vel = 0.85 * payload_vel + 0.15 * (centroid - payload_pos) / max(dt, 1e-3)
            payload_vel = np.clip(payload_vel, -2.0, 2.0)
            payload_pos = payload_pos + payload_vel * min(dt, 0.1) * 0.5

        # command ENU (east, north) = (x, y)
        for i in range(n):
            east, north = float(pos[i, 0]), float(pos[i, 1])
            if args.dry_run:
                continue
            try:
                set_reposition(i + 1, east, north, args.altitude, args.instance)
            except Exception as e:
                print(f"[warn] reposition Drone{i+1}: {e}")

        dist = float(np.linalg.norm(payload_pos - target))
        form = _formation_error_np(pos, payload_pos, args.L0, masses)
        tmean = float(tensions.mean())
        print(
            f"step={step} payload_d={dist:.2f} form={form:.2f} "
            f"T_mean={tmean:.2f} pos0=({pos[0,0]:.1f},{pos[0,1]:.1f})"
        )
        if dist < 0.3:
            print("reached target")
            break
        if not args.dry_run:
            time.sleep(dt)

    d1 = float(np.linalg.norm(payload_pos - target))
    moved = (d0 - d1) > 1.5 or d1 < args.accept_dist
    print(f"aas_transport_bridge: done d0={d0:.2f} d1={d1:.2f} moved={moved}")
    if args.no_accept:
        print("ACCEPT SKIP (--no_accept)")
        return 0
    if not moved:
        print("ACCEPT FAIL: payload_distance did not drop enough", file=sys.stderr)
        return 1
    print("ACCEPT OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

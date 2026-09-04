#!/usr/bin/env python3
"""AAS Docker bridge: host brain → takeoff + set_reposition via docker exec.

AAS SITL puts each drone on ROS_DOMAIN_ID=DRONE_ID inside docker networks.
Host ROS2 cannot see those topics; this injects goals with `docker exec`.

Default path is numpy-only (no torch / CompleteController) so WSL works
without a GPU torch env. Optional --brain loads CompleteController if torch
is importable.

Usage (WSL, while sim_run.sh is up):
  python3 ros_nodes/aas_cmd_bridge.py --uav_count 4 --altitude 12
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _sh(cmd: str, timeout: float = 30.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-lc", cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def docker_exec(container: str, inner: str, timeout: float = 60.0, *, domain_id: int | None = None) -> str:
    # AAS aircraft-image ships ROS2 Humble; each drone uses ROS_DOMAIN_ID=DRONE_ID.
    env_prefix = f"export ROS_DOMAIN_ID={domain_id}; " if domain_id is not None else ""
    wrapped = (
        f"{env_prefix}"
        "source /opt/ros/humble/setup.bash && "
        "source /aas/aircraft_ws/install/setup.bash && "
        f"{inner}"
    )
    r = subprocess.run(
        ["docker", "exec", container, "bash", "-lc", wrapped],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if r.returncode != 0:
        raise RuntimeError(f"docker exec {container} failed:\n{r.stderr or r.stdout}")
    return r.stdout


def container_name(i: int, instance: int = 0) -> str:
    return f"aircraft-container-inst{instance}_{i}"


def wait_containers(n: int, instance: int = 0, timeout: float = 120.0) -> None:
    t0 = time.time()
    need = {container_name(i, instance) for i in range(1, n + 1)}
    while time.time() - t0 < timeout:
        r = _sh("docker ps --format '{{.Names}}'")
        names = set(r.stdout.split())
        if need.issubset(names):
            return
        time.sleep(2)
    raise SystemExit(f"timeout waiting for aircraft containers: {need - names}")


def wait_preflight(i: int, instance: int = 0, timeout: float = 180.0) -> None:
    """PX4 rejects takeoff until vehicle_status.pre_flight_checks_pass."""
    c = container_name(i, instance)
    print(f"[preflight] waiting Drone{i}…")
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            out = docker_exec(
                c,
                "ros2 daemon stop >/dev/null 2>&1 || true; "
                f"timeout 5 ros2 topic echo /Drone{i}/fmu/out/vehicle_status_v1 --once "
                f"--qos-reliability best_effort --qos-durability transient_local",
                timeout=20.0,
                domain_id=i,
            )
        except RuntimeError as e:
            print(f"[preflight] Drone{i} echo retry: {e}")
            time.sleep(2.0)
            continue
        if "pre_flight_checks_pass: true" in out:
            print(f"[preflight] Drone{i} OK")
            return
        time.sleep(2.0)
    raise SystemExit(f"timeout: Drone{i} pre_flight_checks_pass still false")


def takeoff(i: int, altitude: float, instance: int = 0) -> None:
    c = container_name(i, instance)
    cmd = (
        f"ros2 action send_goal /Drone{i}/takeoff_action "
        f"autopilot_interface_msgs/action/Takeoff "
        f"'{{takeoff_altitude: {altitude:.1f}}}' --feedback"
    )
    print(f"[takeoff] Drone{i} → {altitude:.1f} m")
    out = docker_exec(c, cmd, timeout=180.0, domain_id=i)
    if "Goal was rejected" in out:
        # CLI only prints "Goal was rejected"; common case = already airborne
        print(f"[takeoff] Drone{i} goal rejected (likely already airborne) — continue")
        return
    if "success: false" in out and "SUCCEEDED" not in out:
        raise RuntimeError(f"takeoff failed:\n{out[-600:]}")
    print(out[-400:] if len(out) > 400 else out)


def set_reposition(i: int, east: float, north: float, altitude: float, instance: int = 0) -> None:
    c = container_name(i, instance)
    cmd = (
        f"ros2 service call /Drone{i}/set_reposition "
        f"autopilot_interface_msgs/srv/SetReposition "
        f"'{{east: {east:.2f}, north: {north:.2f}, altitude: {altitude:.2f}}}'"
    )
    docker_exec(c, cmd, timeout=20.0, domain_id=i)


def lissajous(n: int, t: float, scale: float = 8.0) -> np.ndarray:
    rows = []
    for i in range(n):
        phase = 2 * math.pi * i / max(n, 1)
        r = scale + 2.0 * math.sin(t * 0.15 + phase)
        east = r * math.cos(t * 0.25 + phase)
        north = r * math.sin(t * 0.35 + phase)
        rows.append([east, north])
    return np.asarray(rows, dtype=np.float32)


def project_dxdy_cbf_np(
    dxdy: np.ndarray,
    pos: np.ndarray,
    vel: np.ndarray,
    *,
    dt: float = 0.1,
    speed_cap: float = 3.0,
    min_distance: float = 0.5,
    alpha: float = 1.0,
    max_iter: int = 10,
) -> np.ndarray:
    """Numpy port of dice.cbf_projection.project_dxdy_cbf (no torch)."""
    v_des = np.clip(dxdy, -1.0, 1.0) * speed_cap
    u = (v_des - vel) / max(dt, 1e-3)
    d_min_sq = min_distance**2
    n = u.shape[0]
    for _ in range(max_iter):
        max_violation = 0.0
        worst = None
        for i in range(n):
            for j in range(i + 1, n):
                dp = pos[i] - pos[j]
                dv = vel[i] - vel[j]
                h = float(np.dot(dp, dp) - d_min_sq)
                if h > 0.1:
                    continue
                dh_dt = 2.0 * float(np.dot(dp, dv))
                g_i = -2.0 * dp
                g_j = 2.0 * dp
                b = alpha * h + dh_dt
                current = float(np.dot(g_i, u[i]) + np.dot(g_j, u[j]))
                viol = current - b
                if viol > max_violation:
                    max_violation = viol
                    worst = (i, j, g_i, g_j, b)
        if worst is None or max_violation <= 1e-6:
            break
        i, j, g_i, g_j, b = worst
        current = float(np.dot(g_i, u[i]) + np.dot(g_j, u[j]))
        excess = current - b
        denom = float(np.dot(g_i, g_i) + np.dot(g_j, g_j))
        if denom > 1e-10 and excess > 0:
            delta = excess / denom
            u[i] = u[i] - delta * g_i
            u[j] = u[j] - delta * g_j
        u = np.clip(u, -4.0, 4.0)
    v_safe = vel + u * dt
    return np.clip(v_safe / max(speed_cap, 1e-3), -1.0, 1.0)


def main() -> int:
    ap = argparse.ArgumentParser(description="AAS Docker cmd bridge (takeoff + reposition)")
    ap.add_argument("--uav_count", type=int, default=4)
    ap.add_argument("--instance", type=int, default=0)
    ap.add_argument("--altitude", type=float, default=12.0, help="AGL relative to home")
    ap.add_argument("--rate_hz", type=float, default=0.5, help="reposition rate (keep ≤1 Hz)")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--shield", type=str, default="cbf", choices=["cbf", "hard", "none"])
    ap.add_argument("--skip_takeoff", action="store_true")
    ap.add_argument(
        "--brain",
        action="store_true",
        help="use CompleteController if torch is available (else kinematic)",
    )
    ap.add_argument("--obs_dim", type=int, default=48)
    args = ap.parse_args()

    n = args.uav_count
    wait_containers(n, args.instance)
    print(f"found {n} aircraft containers")

    if not args.skip_takeoff:
        for i in range(1, n + 1):
            wait_preflight(i, args.instance)
            takeoff(i, args.altitude, args.instance)
        print("waiting 15s for hover…")
        time.sleep(15.0)

    ctl = None
    if args.brain:
        try:
            import torch
            from models.complete_controller import CompleteController

            ctl = CompleteController(
                n,
                n,
                args.obs_dim,
                n_roles=3,
                use_semantic=False,
                use_icps_resource=False,
                use_dice=True,
                shield_type=args.shield if args.shield != "none" else "none",
            )
            ctl.eval()
            print("brain: CompleteController")
        except Exception as e:
            print(f"[warn] --brain unavailable ({e}); using kinematic tracker")

    pos = np.zeros((n, 2), dtype=np.float32)
    vel = np.zeros((n, 2), dtype=np.float32)
    dt = 1.0 / max(args.rate_hz, 0.1)
    t0 = time.time()

    for step in range(args.steps):
        t = time.time() - t0
        goals = lissajous(n, t, scale=8.0)

        if ctl is not None:
            import torch

            roles = torch.zeros(n, dtype=torch.long)
            alive = torch.ones(n, dtype=torch.bool)
            tasks = torch.zeros(n, 4)
            tasks[:, :2] = torch.from_numpy(goals)
            tasks[:, 2] = 1.0
            done = torch.zeros(n, dtype=torch.bool)
            asn = torch.arange(n, dtype=torch.long)
            obs = torch.zeros(n, args.obs_dim)
            obs[:, 0:2] = torch.from_numpy(pos)
            obs[:, 2:4] = torch.from_numpy(vel)
            with torch.no_grad():
                out = ctl.step(
                    obs,
                    torch.from_numpy(pos),
                    torch.from_numpy(vel),
                    tasks,
                    roles,
                    alive,
                    done,
                    asn,
                    snr_db=20.0,
                )
            dxdy = (
                0.4 * out.dxdy.numpy()
                + 0.6 * np.clip(goals - pos, -1.0, 1.0)
            ).clip(-1.0, 1.0)
        else:
            # kinematic: chase Lissajous goals
            dxdy = np.clip(goals - pos, -1.0, 1.0)

        if args.shield == "cbf":
            dxdy = project_dxdy_cbf_np(dxdy, pos, vel, dt=dt)
        elif args.shield == "hard":
            # crude hard clamp on pairwise approach
            for i in range(n):
                for j in range(i + 1, n):
                    dp = pos[i] - pos[j]
                    d = float(np.linalg.norm(dp))
                    if d < 1.0 and d > 1e-6:
                        push = dp / d
                        dxdy[i] += 0.3 * push
                        dxdy[j] -= 0.3 * push
            dxdy = np.clip(dxdy, -1.0, 1.0)

        vel = dxdy * 2.0
        pos = pos + vel * dt

        dists = []
        for i in range(n):
            east = float(pos[i, 0])
            north = float(pos[i, 1])
            try:
                set_reposition(i + 1, east, north, args.altitude, args.instance)
            except Exception as e:
                print(f"[warn] reposition Drone{i+1}: {e}")
            dists.append(float(np.linalg.norm(goals[i] - pos[i])))

        min_sep = float("inf")
        for i in range(n):
            for j in range(i + 1, n):
                min_sep = min(min_sep, float(np.linalg.norm(pos[i] - pos[j])))
        d_str = ",".join(f"{d:.1f}" for d in dists)
        print(f"step={step} Dist→goal=[{d_str}] min_sep≈{min_sep:.1f}m")
        time.sleep(dt)

    print("aas_cmd_bridge: done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

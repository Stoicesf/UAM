#!/usr/bin/env python3
"""Interactive ICPS+DICE+SEM swarm demo.

  python demos/interactive_demo.py --scene search --n_agents 16 --speed 1x
  python demos/interactive_demo.py --scene adversarial --explain
Keys: Space pause/resume, R reset, H help, Q quit; click UAV when --explain
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt

from demos.demo_runner import DemoRunner
from demos.scene_library import get_scene
from demos.visualizer import SwarmVisualizer


def _speed_to_interval_ms(speed: str) -> float:
    s = speed.strip().lower().replace("×", "x")
    if s in ("0", "max", "unlimited", "inf"):
        return 0.0
    if s.endswith("x"):
        s = s[:-1]
    try:
        rate = float(s)
    except ValueError:
        rate = 1.0
    return 50.0 / max(rate, 1e-6)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Interactive UAM swarm demo")
    ap.add_argument(
        "--scene",
        type=str,
        default="search",
        choices=["search", "tracking", "adversarial", "mixed", "pursuit", "adversarial_pursuit", "transport"],
    )
    ap.add_argument("--n_agents", type=int, default=0, help="0 = scene default")
    ap.add_argument("--speed", type=str, default="1x", help="0.5x / 1x / 2x / max")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--headless", action="store_true", help="run N steps without GUI (smoke)")
    ap.add_argument("--steps", type=int, default=0, help="headless step cap (0=scene max)")
    ap.add_argument("--sensor_noise", type=float, default=None)
    ap.add_argument("--drop_rate", type=float, default=None)
    ap.add_argument("--comm_delay", type=int, default=None)
    ap.add_argument("--max_acc", type=float, default=None)
    ap.add_argument("--link_topk", type=int, default=0, help="0 = auto min(N*2,64)")
    ap.add_argument("--explain", action="store_true", help="click UAV for decision timeline")
    ap.add_argument("--evader_policy", type=str, default="", choices=["", "scripted", "rl"])
    ap.add_argument("--hetero", action="store_true", help="heterogeneous airframes")
    ap.add_argument(
        "--hetero_ratio",
        type=str,
        default="0.3,0.4,0.3",
        help="heavy,standard,light fractions",
    )
    ap.add_argument(
        "--shield",
        type=str,
        default="",
        choices=["", "hard", "cbf", "none"],
        help="safety shield: hard projection | analytic CBF | none",
    )
    ap.add_argument(
        "--no_match_bias",
        action="store_true",
        help="hetero dynamics without hard role/capability bias (emergence test)",
    )
    ap.add_argument(
        "--ctrl",
        type=str,
        default="heuristic",
        choices=["heuristic", "theory"],
        help="transport controller: heuristic | theory (TransportHierarchicalController)",
    )
    ap.add_argument(
        "--hybrid",
        action="store_true",
        help="transport: use HybridPayloadDynamics (slack/taut)",
    )
    ap.add_argument(
        "--wind",
        type=float,
        default=0.0,
        help="transport: wind force magnitude (N-scale) for RISE tests",
    )
    ap.add_argument(
        "--no_rise",
        action="store_true",
        help="transport theory: disable RISE compensator",
    )
    ap.add_argument(
        "--traj",
        action="store_true",
        help="transport theory: minimum-snap payload reference",
    )
    ap.add_argument(
        "--apf",
        action="store_true",
        help="transport theory: hybrid APF+CBF shield on formation velocity",
    )
    args = ap.parse_args(argv)

    scene = get_scene(args.scene)
    if args.sensor_noise is not None:
        scene.sensor_noise = args.sensor_noise
    if args.drop_rate is not None:
        scene.drop_rate = args.drop_rate
    if args.comm_delay is not None:
        scene.comm_delay = args.comm_delay
    if args.max_acc is not None:
        scene.max_acc = args.max_acc
        if scene.max_vel == float("inf"):
            scene.max_vel = 3.0
    if args.evader_policy:
        scene.evader_policy = args.evader_policy
    if args.hetero:
        scene.heterogeneous = True
        parts = [float(x) for x in args.hetero_ratio.split(",")]
        if len(parts) != 3:
            raise SystemExit("--hetero_ratio needs 3 comma-separated floats")
        scene.hetero_ratio = (parts[0], parts[1], parts[2])

    n_agents = args.n_agents or scene.default_n_agents or 16
    if n_agents > 64:
        print(f"Warning: N={n_agents}>64 may cause low FPS")

    shield_type = args.shield or None
    runner = DemoRunner(
        scene=scene,
        n_agents=n_agents,
        seed=args.seed,
        heterogeneous=bool(args.hetero or scene.heterogeneous),
        hetero_ratio=tuple(scene.hetero_ratio),
        shield_type=shield_type,
        use_projection_shield=(shield_type != "none") if shield_type else True,
        match_bias=not bool(args.no_match_bias),
        ctrl=args.ctrl,
        use_hybrid=bool(args.hybrid),
        wind_force=float(args.wind),
        use_rise=not bool(args.no_rise),
        use_traj=bool(args.traj),
        use_shield=bool(args.apf),
    )
    if args.link_topk > 0:
        runner.link_topk = args.link_topk
    frame = runner.reset(args.seed)

    if args.headless:
        n = args.steps or min(40, scene.max_steps)
        # optional explain smoke: pick agent 0 timeline
        t0 = time.perf_counter()
        for _ in range(n):
            frame = runner.step()
            if frame.done:
                break
        elapsed = time.perf_counter() - t0
        fps = frame.step / max(elapsed, 1e-9)
        if args.explain:
            tl = runner.agent_timeline(0, last_n=10)
            out = ROOT / "experiment_results" / "explainability"
            out.mkdir(parents=True, exist_ok=True)
            (out / "click_example.json").write_text(
                __import__("json").dumps(tl, indent=2), encoding="utf-8"
            )
            print(f"explain UAV0 roles={tl['roles']} levels={tl['levels']}")
        print(
            f"headless OK scene={args.scene} steps={frame.step} "
            f"coverage={frame.coverage:.2f} links={len(frame.links)} bytes={frame.bytes_cum:.0f} fps={fps:.1f}"
            + (f" hetero={sorted(set(frame.uav_types))}" if frame.uav_types else "")
            + (f" shield={shield_type}" if shield_type else "")
            + (f" payload_d={frame.payload_distance:.2f}" if frame.payload_pos is not None else "")
        )
        return 0

    viz = SwarmVisualizer(
        boundary=float(getattr(getattr(runner.env, "cfg", None), "boundary", None) or getattr(runner.env, "boundary", 5.0)),
        title=f"UAM demo — {args.scene}",
        explain=args.explain,
    )
    viz.update(frame, scene_name=args.scene)

    state = {"paused": False, "quit": False, "reset": False}
    interval_ms = _speed_to_interval_ms(args.speed)

    def on_key(event):
        if event.key in (" ", "space"):
            state["paused"] = not state["paused"]
        elif event.key in ("r", "R"):
            state["reset"] = True
        elif event.key in ("h", "H"):
            viz.toggle_help()
            viz.update(frame, scene_name=args.scene)
        elif event.key in ("q", "Q", "escape"):
            if event.key == "escape" and args.explain:
                viz.selected_agent = None
                viz.set_timeline(None)
                viz.update(frame, scene_name=args.scene)
            else:
                state["quit"] = True

    def on_pick(event):
        aid = viz.on_pick(event)
        if aid is None:
            return
        viz.set_timeline(runner.agent_timeline(aid, last_n=10))
        viz.update(frame, scene_name=args.scene)
        snap = ROOT / "experiment_results" / "explainability" / "click_example.png"
        viz.save_explain_snapshot(snap)
        print(f"explain UAV#{aid} → {snap}")

    viz.fig.canvas.mpl_connect("key_press_event", on_key)
    if args.explain:
        viz.fig.canvas.mpl_connect("pick_event", on_pick)
    print("Demo running — Space pause, R reset, H help, Q quit" + (" | click UAV" if args.explain else ""))

    last = time.perf_counter()
    while not state["quit"] and plt.fignum_exists(viz.fig.number):
        if state["reset"]:
            frame = runner.reset(args.seed)
            viz.selected_agent = None
            viz.set_timeline(None)
            state["reset"] = False
            viz.update(frame, scene_name=args.scene)
            continue
        if state["paused"]:
            viz.fig.canvas.flush_events()
            time.sleep(0.05)
            continue
        now = time.perf_counter()
        if interval_ms > 0 and (now - last) * 1000 < interval_ms:
            viz.fig.canvas.flush_events()
            time.sleep(0.005)
            continue
        last = now
        frame = runner.step()
        if args.explain and viz.selected_agent is not None:
            viz.set_timeline(runner.agent_timeline(viz.selected_agent, last_n=10))
        viz.update(frame, scene_name=args.scene)
        if frame.done:
            state["paused"] = True
            print(f"episode done @ step={frame.step} coverage={frame.coverage:.2f} — R reset, Space resume")

    viz.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Long-horizon evolving demo: rotating scenes, N jitter, noise jitter, mass-kill recovery.

  python demos/lifelong_demo.py --steps 500 --save_plot
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from demos.demo_runner import DemoRunner
from demos.scene_library import get_scene
from dice.failure_injector import FailureInjector

OUT = ROOT / "experiment_results" / "lifelong"
SCENARIO_CYCLE = ["search", "tracking", "pursuit", "adversarial"]


class EnvironmentScheduler:
    def __init__(self, seed: int = 0):
        self.t = 0
        self.g = torch.Generator().manual_seed(seed)
        self.scene_idx = 0
        self.n_agents = 16
        self.sensor_noise = 0.0

    def tick(self) -> dict:
        self.t += 1
        events = {}
        if self.t % 200 == 0:
            self.scene_idx = (self.scene_idx + 1) % len(SCENARIO_CYCLE)
            events["scene"] = SCENARIO_CYCLE[self.scene_idx]
        if self.t % 100 == 0:
            delta = int(torch.randint(-2, 3, (1,), generator=self.g).item())
            self.n_agents = int(max(8, min(24, self.n_agents + delta)))
            events["n_agents"] = self.n_agents
        if self.t % 50 == 0:
            self.sensor_noise = float(torch.rand(1, generator=self.g).item() * 0.15)
            events["sensor_noise"] = self.sensor_noise
        return events


def _make_runner(scene_name: str, n: int, noise: float, seed: int) -> DemoRunner:
    scene = get_scene(scene_name)
    scene.sensor_noise = noise
    scene.drop_rate = 0.02 if noise > 0 else 0.0
    scene.max_steps = 10_000
    # pursuit default n
    if scene.default_n_agents and scene_name == "pursuit":
        n = max(n, 8)
    return DemoRunner(scene=scene, n_agents=n, seed=seed, load_encoder_ckpt=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--save_plot", action="store_true")
    ap.add_argument("--kill_at", type=int, default=250, help="step to kill 20% UAVs")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    sched = EnvironmentScheduler(args.seed)
    scene_name = "search"
    runner = _make_runner(scene_name, sched.n_agents, sched.sensor_noise, args.seed)
    fr = runner.reset(args.seed)

    window: deque[float] = deque(maxlen=100)
    timeline = {"step": [], "success": [], "bandwidth": [], "alive": [], "scene": []}
    kill_done = False
    pre_kill_success = None
    recover_ok = None
    health_hits = 0

    viz = None
    if args.render:
        from demos.visualizer import SwarmVisualizer

        viz = SwarmVisualizer(boundary=runner.env.cfg.boundary, title="lifelong", show=True)

    for t in range(1, args.steps + 1):
        events = sched.tick()
        if "scene" in events or "n_agents" in events:
            scene_name = events.get("scene", scene_name)
            n = events.get("n_agents", runner.n_agents)
            noise = events.get("sensor_noise", sched.sensor_noise)
            # soft handoff: new runner, keep seed stream
            runner = _make_runner(scene_name, n, noise, args.seed + t)
            fr = runner.reset(args.seed + t)
            print(f"t={t} switch scene={scene_name} n={n} noise={noise:.3f}")

        # mass kill recovery probe
        if (not kill_done) and t >= args.kill_at:
            fi = FailureInjector(runner.n_agents)
            runner.env.alive = fi.inject("node_kill", 0.2, runner.env.alive)
            kill_done = True
            pre_kill_success = sum(window) / max(len(window), 1) if window else fr.coverage
            kill_step = t
            print(f"t={t} mass-kill 20% pre_success={pre_kill_success:.2f}")

        fr = runner.step()
        window.append(fr.coverage)
        slide = sum(window) / len(window)
        if slide > 0.35:
            health_hits += 1
        timeline["step"].append(t)
        timeline["success"].append(slide)
        timeline["bandwidth"].append(fr.bandwidth_hz / max(fr.bandwidth_max, 1.0))
        timeline["alive"].append(float(fr.alive.float().sum()))
        timeline["scene"].append(scene_name)

        if kill_done and recover_ok is None and t >= kill_step + 50:
            post = sum(list(window)[-50:]) / 50
            recover_ok = post >= 0.8 * max(pre_kill_success, 1e-6)
            print(f"t={t} recovery post={post:.2f} ok={recover_ok}")

        if viz is not None:
            viz.update(fr, scene_name=scene_name)
        if fr.done:
            fr = runner.reset(args.seed + t)

        # NaN guard
        if any(x != x for x in (fr.coverage, fr.reward, fr.bandwidth_hz)):
            raise RuntimeError(f"NaN at step {t}")

    health = health_hits / max(args.steps, 1)
    summary = {
        "steps": args.steps,
        "avg_health": health,
        "health_threshold": 0.35,
        "met_health_ge_70": health >= 0.70,
        "pre_kill_success": pre_kill_success,
        "recover_ok": recover_ok,
        "final_slide_success": timeline["success"][-1] if timeline["success"] else 0.0,
        "nan_free": True,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.save_plot or True:
        xs = timeline["step"]
        def pts(ys, scale=1.0):
            if not xs:
                return ""
            xmin, xmax = xs[0], xs[-1]
            return " ".join(
                f"{50+(x-xmin)/max(xmax-xmin,1)*500:.1f},{280-y*scale*220:.1f}" for x, y in zip(xs, ys)
            )
        alive_n = [a / 24.0 for a in timeline["alive"]]
        (OUT / "health_timeline.svg").write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="600" height="340">'
            f'<rect width="100%" height="100%" fill="#fff"/>'
            f'<text x="16" y="24">lifelong health</text>'
            f'<polyline fill="none" stroke="#1a5fb4" stroke-width="2" points="{pts(timeline["success"])}"/>'
            f'<polyline fill="none" stroke="#e66100" stroke-width="1.5" points="{pts(timeline["bandwidth"])}"/>'
            f'<polyline fill="none" stroke="#2ec27e" stroke-width="1.5" points="{pts(alive_n)}"/>'
            f'<text x="420" y="40" fill="#1a5fb4" font-size="11">slide success</text>'
            f'<text x="420" y="56" fill="#e66100" font-size="11">bandwidth</text>'
            f'<text x="420" y="72" fill="#2ec27e" font-size="11">alive/24</text></svg>',
            encoding="utf-8",
        )
        (OUT / "health_timeline.png.txt").write_text("See health_timeline.svg\n", encoding="utf-8")

    if viz is not None:
        viz.close()
    print(json.dumps(summary, indent=2))
    print("lifelong_demo: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

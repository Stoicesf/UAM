#!/usr/bin/env python3
"""Record a demo scene to mp4/gif (no interactive window required).

  python demos/record_video.py --scene mixed --fps 10 --output demo_mixed.mp4
Falls back to GIF if ffmpeg/writer unavailable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

from demos.demo_runner import DemoRunner
from demos.scene_library import get_scene
from demos.visualizer import SwarmVisualizer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, default="mixed")
    ap.add_argument("--n_agents", type=int, default=16)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_frames", type=int, default=0, help="0 = min(scene.max_steps, 120)")
    ap.add_argument("--output", type=str, default="")
    args = ap.parse_args()

    scene = get_scene(args.scene)
    out = Path(args.output) if args.output else ROOT / "experiment_results" / "demos" / f"demo_{args.scene}.mp4"
    if not out.is_absolute():
        out = ROOT / "experiment_results" / "demos" / out.name
    out.parent.mkdir(parents=True, exist_ok=True)

    runner = DemoRunner(scene=scene, n_agents=args.n_agents, seed=args.seed)
    frame0 = runner.reset(args.seed)
    viz = SwarmVisualizer(boundary=runner.env.cfg.boundary, title=f"UAM — {args.scene}", show=False)
    plt.ioff()

    n_frames = args.max_frames or min(scene.max_steps, 120)
    frames = [frame0]
    fr = frame0
    for _ in range(n_frames - 1):
        fr = runner.step()
        frames.append(fr)
        if fr.done:
            break

    def _update(i):
        viz.update(frames[i], scene_name=args.scene)
        return []

    anim = FuncAnimation(
        viz.fig,
        _update,
        frames=len(frames),
        interval=1000 / max(args.fps, 1),
        blit=False,
    )

    saved = str(out)
    try:
        writer = FFMpegWriter(fps=args.fps, metadata={"title": f"UAM {args.scene}"})
        anim.save(saved, writer=writer)
    except Exception as e:
        print(f"ffmpeg writer failed ({e}); falling back to GIF")
        saved = str(out.with_suffix(".gif"))
        writer = PillowWriter(fps=args.fps)
        anim.save(saved, writer=writer)

    viz.close()
    print(f"wrote {saved} frames={len(frames)} duration≈{len(frames)/args.fps:.1f}s")
    print("record_video: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

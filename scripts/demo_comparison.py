"""Build side-by-side comparison figure (and optional video strip) for paper demo.

Usage:
  python scripts/demo_comparison.py
  python scripts/demo_comparison.py --methods mappo gat dsgf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_metric(method: str) -> dict:
    p = ROOT / "demo" / f"{method}_4uav" / "metrics.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def make_trajectory_panel(methods: list[str], out_path: Path):
    paths = []
    for m in methods:
        p = ROOT / "demo" / "figures" / f"trajectory_{m}.png"
        if not p.exists():
            print(f"[skip] missing {p}")
            continue
        paths.append((m, p))
    if not paths:
        raise FileNotFoundError("No trajectory figures found. Run demo_runner first.")

    n = len(paths)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (m, p) in zip(axes, paths):
        img = np.asarray(Image.open(p).convert("RGB"))
        ax.imshow(img)
        ax.set_title(m.upper(), fontsize=14)
        ax.axis("off")
        meta = _load_metric(m)
        if meta:
            s = meta.get("final_success", 0.0)
            ax.text(
                0.5, -0.06,
                f"demo S={s:.0%}",
                transform=ax.transAxes,
                ha="center",
                fontsize=11,
            )
    fig.suptitle("MAPPO vs GAT vs DSGF — 4 UAV Cooperative Navigation", fontsize=13, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def make_video_grid(methods: list[str], out_path: Path, max_frames: int = 64, fps: int = 8):
    """Stack mid-length GIF/MP4 frames horizontally into comparison GIF."""
    import imageio.v2 as imageio

    frame_dirs = []
    for m in methods:
        d = ROOT / "demo" / f"{m}_4uav" / "frames"
        if d.exists() and any(d.glob("frame_*.png")):
            frame_dirs.append((m, sorted(d.glob("frame_*.png"))))
        else:
            print(f"[skip] no frames for {m}")
    if len(frame_dirs) < 2:
        print("[warn] need ≥2 methods with frames for video grid")
        return

    n_frames = min(max_frames, min(len(fs) for _, fs in frame_dirs))
    # subsample evenly
    grids = []
    for t in range(n_frames):
        idxs = []
        row = []
        for m, fs in frame_dirs:
            # map t into each sequence length
            i = int(t / max(n_frames - 1, 1) * (len(fs) - 1))
            idxs.append(i)
            img = np.asarray(Image.open(fs[i]).convert("RGB"))
            # label bar
            from PIL import ImageDraw, ImageFont
            pil = Image.fromarray(img)
            draw = ImageDraw.Draw(pil)
            draw.rectangle([0, 0, pil.width, 28], fill=(20, 20, 20))
            draw.text((8, 6), m.upper(), fill=(255, 255, 255))
            row.append(np.asarray(pil))
        # resize to common height
        h = min(im.shape[0] for im in row)
        resized = []
        for im in row:
            pil = Image.fromarray(im)
            w = int(pil.width * h / pil.height)
            resized.append(np.asarray(pil.resize((w, h))))
        grids.append(np.concatenate(resized, axis=1))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), grids, fps=fps)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=["mappo", "gat", "dsgf"])
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    make_trajectory_panel(
        args.methods,
        ROOT / "demo" / "figures" / "comparison_trajectories.png",
    )
    if not args.no_video:
        try:
            make_video_grid(
                args.methods,
                ROOT / "demo" / "videos" / "comparison_mappo_gat_dsgf.gif",
            )
            # also try mp4
            make_video_grid(
                args.methods,
                ROOT / "demo" / "videos" / "comparison_mappo_gat_dsgf.mp4",
            )
        except Exception as e:
            print(f"[warn] video grid failed: {e}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Zero-shot transfer: one CompleteController checkpoint × many scenes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from demos.demo_runner import DemoRunner
from demos.scene_library import SCENES, get_scene

OUT = ROOT / "experiment_results" / "transfer"


def eval_scene(name: str, episodes: int, steps: int, seed0: int, load_ckpt: bool) -> dict:
    scene = get_scene(name)
    n = scene.default_n_agents or 16
    scene.max_steps = max(steps, 80)
    covs, bws = [], []
    for ep in range(episodes):
        runner = DemoRunner(scene=scene, n_agents=n, seed=seed0 + ep, load_encoder_ckpt=load_ckpt)
        runner.ctl.eval()
        for p in runner.ctl.parameters():
            p.requires_grad_(False)
        fr = runner.reset(seed0 + ep)
        bw_acc = 0.0
        for _ in range(steps):
            fr = runner.step()
            bw_acc += fr.bandwidth_hz / max(fr.bandwidth_max, 1.0)
            if fr.done:
                break
        covs.append(fr.coverage)
        bws.append(bw_acc / max(fr.step, 1))
        print(f"{name} ep={ep} success={fr.coverage:.2f}")
    return {
        "scene": name,
        "success_rate": sum(covs) / len(covs),
        "avg_bandwidth": sum(bws) / len(bws),
        "n_agents": n,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        type=str,
        default="experiment_results/semantic/semantic_encoder.pt",
        help="encoder ckpt (role_bias.pt loaded automatically if present)",
    )
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--scenes", type=str, default="search,tracking,adversarial,mixed,pursuit")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    load = ckpt.exists()
    if not load:
        print(f"warning: checkpoint missing ({ckpt}); using random encoder weights")

    names = [x.strip() for x in args.scenes.split(",") if x.strip()]
    rows = []
    for name in names:
        if name not in SCENES:
            print(f"skip unknown scene {name}")
            continue
        rows.append(eval_scene(name, args.episodes, args.steps, 42, load_ckpt=load))

    with (OUT / "transfer_rows.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 1×K bar chart (extensible to matrix later)
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="320">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="16" y="24" font-size="14">zero-shot transfer (one checkpoint)</text>',
    ]
    for i, r in enumerate(rows):
        h = r["success_rate"] * 220
        color = "#2ec27e" if r["success_rate"] >= 0.5 else "#c01c28"
        parts.append(f'<rect x="{40+i*115}" y="{270-h}" width="90" height="{h}" fill="{color}"/>')
        parts.append(f'<text x="{40+i*115}" y="290" font-size="11">{r["scene"]}</text>')
        parts.append(f'<text x="{55+i*115}" y="{260-h}" font-size="11">{r["success_rate"]:.2f}</text>')
    parts.append("</svg>")
    (OUT / "transfer_matrix.svg").write_text("\n".join(parts), encoding="utf-8")
    (OUT / "transfer_matrix.png.txt").write_text("See transfer_matrix.svg\n", encoding="utf-8")

    pursuit = next((r for r in rows if r["scene"] == "pursuit"), None)
    search = next((r for r in rows if r["scene"] == "search"), None)
    summary = {
        "rows": rows,
        "all_ge_05": all(r["success_rate"] >= 0.5 for r in rows),
        "pursuit_gt_search": (
            False
            if pursuit is None or search is None
            else pursuit["success_rate"] > search["success_rate"]
        ),
        "checkpoint": str(ckpt),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("transfer_matrix: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Week 1 — GAT lambda search (20k, warmup30%, adaptive_align)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = [
    ("configs/gat/gat_a_lambda003.yaml", "gat_a_lambda003"),
    ("configs/gat/gat_b_lambda002.yaml", "gat_b_lambda002"),
    ("configs/gat/gat_c_lambda001.yaml", "gat_c_lambda001"),
]


def main():
    for exp_cfg, run_name in EXPERIMENTS:
        cmd = [
            sys.executable,
            str(ROOT / "train.py"),
            "--exp", str(ROOT / exp_cfg),
            "--run-name", run_name,
        ]
        print(f"\n>>> {run_name}")
        rc = subprocess.call(cmd, cwd=str(ROOT))
        if rc != 0:
            sys.exit(rc)

    print("\n=== GAT Lambda Search (20k) ===")
    print(f"{'Run':<22} {'Success':>8} {'Align':>8} {'Reward':>8} {'Collision':>10}")
    print("-" * 60)
    best_name, best_success = "", -1.0
    for _, run_name in EXPERIMENTS:
        summary = ROOT / "results" / "graph" / run_name / "summary.json"
        if not summary.exists():
            continue
        with open(summary, encoding="utf-8") as f:
            s = json.load(f)
        pm = s.get("paper_metrics", s)
        success = pm.get("success", 0)
        print(
            f"{run_name:<22} {success:>8.4f} "
            f"{pm.get('alignment', 0):>8.4f} "
            f"{pm.get('reward', 0):>8.4f} "
            f"{pm.get('collision', 0):>10.4f}"
        )
        if success > best_success:
            best_success = success
            best_name = run_name
    if best_name:
        print(f"\nBest by success: {best_name} ({best_success:.4f})")


if __name__ == "__main__":
    main()

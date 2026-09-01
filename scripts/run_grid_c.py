"""Run C1–C4 grid experiments (Gate 3 / 256k for C4)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = [
    ("configs/experiments/ablation_c1_warmup05.yaml", "grid_c1", "3"),
    ("configs/experiments/ablation_c2_warmup03.yaml", "grid_c2", "3"),
    ("configs/experiments/ablation_c3_adaptive.yaml", "grid_c3", "3"),
    ("configs/experiments/ablation_c4_adaptive_256k.yaml", "grid_c4", "3"),
]


def main():
    for exp_cfg, run_name, gate in EXPERIMENTS:
        cmd = [
            sys.executable,
            str(ROOT / "train.py"),
            "--exp", str(ROOT / exp_cfg),
            "--gate", gate,
            "--run-name", run_name,
        ]
        print(f"\n>>> {run_name}")
        rc = subprocess.call(cmd, cwd=str(ROOT))
        if rc != 0:
            sys.exit(rc)

    print("\n=== Grid C summary ===")
    for _, run_name, _ in EXPERIMENTS:
        summary = ROOT / "results" / "guide" / run_name / "summary.json"
        if summary.exists():
            with open(summary, encoding="utf-8") as f:
                s = json.load(f)
            pm = s.get("paper_metrics", s)
            print(
                f"{run_name}: success={pm.get('success', 0):.4f} "
                f"align={pm.get('alignment', 0):.4f} "
                f"reward={pm.get('reward', 0):.4f} "
                f"schedule={s.get('guidance_schedule', '?')}"
            )


if __name__ == "__main__":
    main()

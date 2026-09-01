"""Step 2 — 10k quick validation: static warmup vs adaptive lambda."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = [
    ("configs/experiments/ablation_c_warmup.yaml", "adaptive_10k_c_static"),
    ("configs/experiments/ablation_c3_adaptive.yaml", "adaptive_10k_c3_decay"),
    ("configs/experiments/ablation_c3_adaptive_align.yaml", "adaptive_10k_c3_align"),
]


def main():
    for exp_cfg, run_name in EXPERIMENTS:
        cmd = [
            sys.executable,
            str(ROOT / "train.py"),
            "--exp", str(ROOT / exp_cfg),
            "--gate", "2",
            "--run-name", run_name,
        ]
        print(f"\n>>> {run_name}")
        rc = subprocess.call(cmd, cwd=str(ROOT))
        if rc != 0:
            sys.exit(rc)

    print("\n=== 10k comparison ===")
    for _, run_name in EXPERIMENTS:
        summary = ROOT / "results" / "guide" / run_name / "summary.json"
        if summary.exists():
            import json

            with open(summary, encoding="utf-8") as f:
                s = json.load(f)
            pm = s.get("paper_metrics", s)
            print(
                f"{run_name}: success={pm.get('success', 0):.4f} "
                f"align={pm.get('alignment', 0):.4f} reward={pm.get('reward', 0):.4f}"
            )


if __name__ == "__main__":
    main()

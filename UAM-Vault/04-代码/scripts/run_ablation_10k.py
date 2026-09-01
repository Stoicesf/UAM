"""Run Stage 1 lambda ablations (10k frames each)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = [
    ("configs/experiments/ablation_a_lambda1.yaml", "ablation_a_lambda1"),
    ("configs/experiments/ablation_b_lambda01.yaml", "ablation_b_lambda01"),
    ("configs/experiments/ablation_c_warmup.yaml", "ablation_c_warmup"),
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

    print("\n>>> Analyzing results...")
    for _, run_name in EXPERIMENTS:
        metrics = ROOT / "results" / "guide" / run_name / "metrics.csv"
        if metrics.exists():
            subprocess.call([
                sys.executable,
                str(ROOT / "scripts" / "analyze_reward_conflict.py"),
                "--run", str(metrics),
            ])


if __name__ == "__main__":
    main()

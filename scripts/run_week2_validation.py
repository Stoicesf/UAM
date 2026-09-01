"""Week 2 paper validation: DSGF v2 (3 seeds) + 3 ablations @ 102k."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 3407, 2026]

EXPERIMENTS = [
    ("configs/dsgf/dsgf_v2_frozen.yaml", "dsgf_v2_full", True),
    ("configs/dsgf/ablation_wo_residual.yaml", "ablation_wo_residual", False),
    ("configs/dsgf/ablation_wo_dynamic_graph.yaml", "ablation_wo_dg", False),
    ("configs/dsgf/ablation_wo_temporal.yaml", "ablation_wo_temporal", False),
]


def main():
    for exp_cfg, base_name, multi_seed in EXPERIMENTS:
        seeds = SEEDS if multi_seed else [42]
        for seed in seeds:
            run_name = f"{base_name}_s{seed}" if multi_seed else f"{base_name}_102k"
            cmd = [
                sys.executable,
                str(ROOT / "train.py"),
                "--exp", str(ROOT / exp_cfg),
                "--gate", "3",
                "--seed", str(seed),
                "--run-name", run_name,
            ]
            print(f"\n>>> {run_name}")
            rc = subprocess.call(cmd, cwd=str(ROOT))
            if rc != 0:
                sys.exit(rc)

    print("\n=== Week 2 Validation Summary ===")
    print(f"{'Run':<28} {'Success':>8} {'Align':>8} {'Reward':>8}")
    print("-" * 56)
    for _, base_name, multi_seed in EXPERIMENTS:
        seeds = SEEDS if multi_seed else [42]
        for seed in seeds:
            run_name = f"{base_name}_s{seed}" if multi_seed else f"{base_name}_102k"
            summary = ROOT / "results" / "dsgf" / run_name / "summary.json"
            if summary.exists():
                with open(summary, encoding="utf-8") as f:
                    s = json.load(f)
                pm = s.get("paper_metrics", s)
                print(
                    f"{run_name:<28} {pm.get('success', 0):>8.4f} "
                    f"{pm.get('alignment', 0):>8.4f} {pm.get('reward', 0):>8.4f}"
                )


if __name__ == "__main__":
    main()

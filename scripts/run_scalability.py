"""Phase 1 Week 1 — DSGF v2 scalability: 4/8/16/32 UAV @ 102k seed=42."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = [
    ("configs/scalability/uav4.yaml", "uav4"),
    ("configs/scalability/uav8.yaml", "uav8"),
    ("configs/scalability/uav16.yaml", "uav16"),
    ("configs/scalability/uav32.yaml", "uav32"),
]


def main():
    for exp_cfg, run_name in EXPERIMENTS:
        cmd = [
            sys.executable,
            str(ROOT / "train.py"),
            "--exp", str(ROOT / exp_cfg),
            "--gate", "3",
            "--run-name", run_name,
        ]
        print(f"\n>>> scalability/{run_name}")
        rc = subprocess.call(cmd, cwd=str(ROOT))
        if rc != 0:
            sys.exit(rc)

    print("\n=== Scalability Summary (DSGF v2, 102k, seed=42) ===")
    print(f"{'UAV':<6} {'Success':>8} {'Collision':>10} {'Comm':>8} {'PathLen':>8}")
    print("-" * 48)
    rows = []
    for _, run_name in EXPERIMENTS:
        summary = ROOT / "results" / "scalability" / run_name / "summary.json"
        if summary.exists():
            with open(summary, encoding="utf-8") as f:
                s = json.load(f)
            pm = s.get("paper_metrics", s)
            n = int(run_name.replace("uav", ""))
            comm = pm.get("communication_cost", s.get("communication_cost_mean", 0))
            rows.append({
                "uav": n,
                "success": pm.get("success", 0),
                "collision": pm.get("collision", 0),
                "reward": pm.get("reward", 0),
                "path_length": pm.get("path_length", 0),
                "communication_cost": comm,
            })
            print(
                f"{n:<6} {pm.get('success', 0):>8.4f} "
                f"{pm.get('collision', 0):>10.4f} "
                f"{comm:>8.2f} "
                f"{pm.get('path_length', 0):>8.2f}"
            )
    out = ROOT / "results" / "scalability" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()

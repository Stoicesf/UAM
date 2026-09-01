"""Week 2 — Baseline comparison @ 16 UAV, 102k, seed=42.

Day 1: mappo + gat
Day 2: transformer + dsgf
All:   python scripts/run_baseline16.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = [
    ("configs/baseline16/mappo.yaml", "mappo"),
    ("configs/baseline16/gat_mappo.yaml", "gat"),
    ("configs/baseline16/full_attention.yaml", "transformer"),
    ("configs/baseline16/dsgf_v2.yaml", "dsgf"),
]

DAY1 = {"mappo", "gat"}
DAY2 = {"transformer", "dsgf"}


def print_table():
    print("\n=== Table I — 16 UAV Comparison (seed=42, 102k) ===")
    print(f"{'Method':<14} {'Success':>8} {'Collision':>10} {'Reward':>10} {'Episode':>8}")
    print("-" * 54)
    rows = []
    for _, run_name in EXPERIMENTS:
        summary = ROOT / "results" / "baseline16" / run_name / "summary.json"
        if not summary.exists():
            print(f"{run_name:<14} {'—':>8} {'—':>10} {'—':>10} {'—':>8}")
            continue
        with open(summary, encoding="utf-8") as f:
            s = json.load(f)
        pm = s.get("paper_metrics", s)
        rows.append({"method": run_name, **pm})
        print(
            f"{run_name:<14} {pm.get('success', 0):>8.4f} "
            f"{pm.get('collision', 0):>10.4f} "
            f"{pm.get('reward', 0):>10.4f} "
            f"{pm.get('episode_length', 0):>8.1f}"
        )
    out = ROOT / "paper" / "tables" / "table1_baseline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", type=int, choices=[1, 2], default=None)
    parser.add_argument("--only", nargs="+", choices=[n for _, n in EXPERIMENTS])
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    if args.summary:
        print_table()
        return

    if args.only:
        targets = set(args.only)
    elif args.day == 1:
        targets = DAY1
    elif args.day == 2:
        targets = DAY2
    else:
        targets = {n for _, n in EXPERIMENTS}

    for exp_cfg, run_name in EXPERIMENTS:
        if run_name not in targets:
            continue
        cmd = [
            sys.executable,
            str(ROOT / "train.py"),
            "--exp", str(ROOT / exp_cfg),
            "--gate", "3",
            "--run-name", run_name,
        ]
        print(f"\n>>> baseline16/{run_name}")
        rc = subprocess.call(cmd, cwd=str(ROOT))
        if rc != 0:
            sys.exit(rc)

    print_table()


if __name__ == "__main__":
    main()

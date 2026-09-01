"""Run Gate 4 — same experiment with multiple seeds."""

from __future__ import annotations

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Multi-seed runs for paper stats")
    parser.add_argument("--exp", default="configs/experiments/exp1_guide.yaml")
    parser.add_argument("--gate", type=int, default=3, choices=[2, 3])
    parser.add_argument("--seeds", default="42,3407,2026")
    parser.add_argument("--prefix", default="guide")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    for seed in seeds:
        run_name = f"{args.prefix}_gate{args.gate}_seed{seed}"
        cmd = [
            sys.executable,
            "train.py",
            "--exp",
            args.exp,
            "--gate",
            str(args.gate),
            "--seed",
            str(seed),
            "--run-name",
            run_name,
        ]
        print(f"\n>>> Running seed={seed} -> {run_name}")
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"[FAIL] seed {seed} exited with {rc}")
            sys.exit(rc)

    print(f"\n[OK] All {len(seeds)} seeds finished. Run: python scripts/plot_results.py")


if __name__ == "__main__":
    main()

"""P1-3 — Batch train communication radius sweep.

Usage:
  python scripts/run_communication_train.py --profile fast
  python scripts/run_communication_train.py --profile full
  python scripts/run_communication_train.py --profile smoke --gate 2
  python scripts/run_communication_train.py --run gat_r1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.communication_config import list_runs, write_generated_config


def main():
    parser = argparse.ArgumentParser(description="P1-3 communication radius training")
    parser.add_argument("--profile", default="fast", choices=["full", "fast", "smoke"])
    parser.add_argument("--gate", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--run", default=None, help="Single run_name from manifest (e.g. gat_r1)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if summary.json exists")
    args = parser.parse_args()

    if args.run:
        runs = [r for r in list_runs("full") + list_runs("fast") + list_runs("smoke")
                if r["run_name"] == args.run]
        if not runs:
            print(f"Unknown run_name: {args.run}")
            sys.exit(1)
    else:
        runs = list_runs(args.profile)

    print(f"Profile={args.profile or args.run}, gate={args.gate}, runs={len(runs)}")

    for spec in runs:
        run_name = spec["run_name"]
        summary = ROOT / "results" / "communication" / run_name / "summary.json"
        if args.skip_existing and summary.exists():
            print(f"[skip] {run_name} (summary exists)")
            continue

        cfg_path = write_generated_config(spec["method"], spec["radius"], run_name)
        cmd = [
            sys.executable,
            str(ROOT / "train.py"),
            "--exp", str(cfg_path),
            "--gate", str(args.gate),
            "--run-name", run_name,
        ]
        print(f"\n>>> communication/{run_name}  (R={spec['radius']}, {spec['method']})")
        rc = subprocess.call(cmd, cwd=str(ROOT))
        if rc != 0:
            sys.exit(rc)

    print("\n[OK] Communication training done. Next:")
    print("  python scripts/eval_communication.py --plot")


if __name__ == "__main__":
    main()

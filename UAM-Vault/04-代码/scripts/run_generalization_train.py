"""Train MAPPO / GAT / DSGF once on obstacle=4 for generalization (Step 3)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RUNS = [
    ("configs/generalization/methods/mappo_obs4.yaml", "mappo_obs4"),
    ("configs/generalization/methods/gat_obs4.yaml", "gat_obs4"),
    ("configs/generalization/methods/dsgf_obs4.yaml", "dsgf_obs4"),
]


def main():
    for exp_cfg, run_name in RUNS:
        cmd = [
            sys.executable,
            str(ROOT / "train.py"),
            "--exp", str(ROOT / exp_cfg),
            "--gate", "3",
            "--run-name", run_name,
        ]
        print(f"\n>>> generalization/{run_name}")
        rc = subprocess.call(cmd, cwd=str(ROOT))
        if rc != 0:
            sys.exit(rc)

    print("\n[OK] Generalization training done. Next:")
    print("  python scripts/eval_generalization.py --plot")


if __name__ == "__main__":
    main()

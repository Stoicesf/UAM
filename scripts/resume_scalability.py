"""Resume scalability from uav8 (uav4 already complete)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REMAINING = [
    ("configs/scalability/uav8.yaml", "uav8"),
    ("configs/scalability/uav16.yaml", "uav16"),
    ("configs/scalability/uav32.yaml", "uav32"),
]

ALL = [
    ("configs/scalability/uav4.yaml", "uav4"),
    *REMAINING,
]


def print_summary():
    print("\n=== Scalability Summary (DSGF v2, 102k, seed=42) ===")
    print(f"{'UAV':<6} {'Success':>8} {'Collision':>10} {'PathLen':>8}")
    print("-" * 40)
    rows = []
    for _, run_name in ALL:
        p = ROOT / "results" / "scalability" / run_name / "summary.json"
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            s = json.load(f)
        pm = s.get("paper_metrics", s)
        n = int(run_name.replace("uav", ""))
        rows.append({"uav": n, **{k: pm.get(k) for k in ("success", "collision", "path_length", "reward")}})
        print(
            f"{n:<6} {pm.get('success', 0):>8.4f} "
            f"{pm.get('collision', 0):>10.4f} "
            f"{pm.get('path_length', 0):>8.2f}"
        )
    out = ROOT / "results" / "scalability" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def main():
    for exp_cfg, run_name in REMAINING:
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
    print_summary()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        print_summary()
    else:
        main()

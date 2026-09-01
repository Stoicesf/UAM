"""Phase 1-1 — AC-DSGF @ 16 UAV × 5 seeds (102k), aligned with baseline16.

Seeds match Table I: [42, 3407, 2026, 1234, 8888]
(8888 used instead of 9999 for fair Table I++ pairing.)

Usage:
  python scripts/run_5seed_ac_dsgf16.py
  python scripts/run_5seed_ac_dsgf16.py --seeds 42
  python scripts/run_5seed_ac_dsgf16.py --skip-existing
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SEEDS = [42, 3407, 2026, 1234, 8888]
EXP = ROOT / "configs" / "ac_dsgf" / "ac_dsgf_16uav.yaml"


def run_dir(seed: int) -> Path:
    return ROOT / "results" / "ac_dsgf" / "uav16" / f"s{seed}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=None, help="Comma list, default all five")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--gate", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else list(SEEDS)
    print(f"AC-DSGF 16UAV seeds={seeds} gate={args.gate}")

    for seed in seeds:
        out = run_dir(seed)
        summary = out / "summary.json"
        if args.skip_existing and summary.exists():
            print(f"[skip] s{seed} exists → {summary}")
            continue

        out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(ROOT / "train.py"),
            "--exp",
            str(EXP),
            "--gate",
            str(args.gate),
            "--seed",
            str(seed),
            "--run-name",
            f"uav16_s{seed}",
            "--output-root",
            str(out),
        ]
        print(f"\n>>> AC-DSGF 16UAV seed={seed}")
        print(" ", " ".join(cmd))
        if args.dry_run:
            continue
        rc = subprocess.call(cmd, cwd=str(ROOT))
        if rc != 0:
            print(f"[FAIL] seed={seed} rc={rc}", file=sys.stderr)
            sys.exit(rc)
        print(f"[ok] seed={seed} → {out}")

    print("\n[OK] AC-DSGF 16UAV seed runs finished.")
    print("Next: python scripts/analyze_ac_dsgf16_table.py")


if __name__ == "__main__":
    main()

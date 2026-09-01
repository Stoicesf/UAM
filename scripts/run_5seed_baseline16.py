"""Day 4-5 — Five-seed baseline16 runs (4 methods × 5 seeds = 20 experiments).

Usage:
  python scripts/run_5seed_baseline16.py              # all missing
  python scripts/run_5seed_baseline16.py --dry-run
  python scripts/run_5seed_baseline16.py --only dsgf --seeds 3407,2026
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SEEDS = [42, 3407, 2026, 1234, 8888]

EXPERIMENTS = [
    ("configs/baseline16/mappo.yaml", "mappo"),
    ("configs/baseline16/gat_mappo.yaml", "gat"),
    ("configs/baseline16/full_attention.yaml", "transformer"),
    ("configs/baseline16/dsgf_v2.yaml", "dsgf"),
]


def seed_dir(method: str, seed: int) -> Path:
    return ROOT / "results" / "baseline16_seeds" / method / f"s{seed}"


def legacy_seed42_dir(method: str) -> Path:
    return ROOT / "results" / "baseline16" / method


def import_seed42(method: str) -> bool:
    """Copy completed seed=42 baseline16 run into seeds layout."""
    dst = seed_dir(method, 42)
    if (dst / "summary.json").exists():
        return True
    src = legacy_seed42_dir(method)
    if not (src / "summary.json").exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"[import] seed42 {method} <- {src}")
    return True


def run_one(exp_cfg: str, method: str, seed: int, dry_run: bool) -> int:
    out = seed_dir(method, seed)
    if (out / "summary.json").exists():
        print(f"[skip] {method} seed={seed} (summary exists)")
        return 0

    if seed == 42:
        if import_seed42(method):
            print(f"[skip] {method} seed=42 (imported from baseline16)")
            return 0

    cmd = [
        sys.executable,
        str(ROOT / "train.py"),
        "--exp", str(ROOT / exp_cfg),
        "--gate", "3",
        "--seed", str(seed),
        "--run-name", f"s{seed}",
        "--output-root", str(out),
    ]
    print(f"\n>>> {method} seed={seed} -> {out}")
    if dry_run:
        print(" ".join(cmd))
        return 0

    return subprocess.call(cmd, cwd=str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--only", nargs="+", choices=[m for _, m in EXPERIMENTS])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    methods = {m for _, m in EXPERIMENTS}
    if args.only:
        methods = set(args.only)

    for seed in seeds:
        for exp_cfg, method in EXPERIMENTS:
            if method not in methods:
                continue
            rc = run_one(exp_cfg, method, seed, args.dry_run)
            if rc != 0:
                sys.exit(rc)

    print("\n[OK] 5-seed batch finished.")
    rc = subprocess.call(
        [sys.executable, str(ROOT / "scripts" / "pipeline_after_5seed.py")],
        cwd=str(ROOT),
    )
    if rc != 0:
        print("[WARN] Pipeline not complete yet — watcher will retry when summaries ready.")
        sys.exit(rc)


if __name__ == "__main__":
    main()

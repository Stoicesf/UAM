#!/usr/bin/env python
"""SECDO multi-regime / multi-method / multi-seed experiment entry.

Usage:
  python scripts/run_all_secdo.py --exp fast_drift --methods secdo,reactive,oracle --seeds 0 1 2
  python -m scripts.run_all_secdo --exp all --methods secdo,reactive,oracle,dsgf,ac_dsgf --seeds 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from secdo.experiments.runner import run_experiment


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run SECDO paper experiments")
    p.add_argument("--exp", type=str, default="fast_drift", help="slow_drift|fast_drift|stress_test|all")
    p.add_argument("--methods", type=str, default="secdo,reactive,oracle")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--horizon", type=int, default=48)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--ckpt", type=str, default="checkpoints/secdo_platform/pretrain/best.pt")
    p.add_argument("--out", type=str, default="results/secdo")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args(argv)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    exps = ["slow_drift", "fast_drift", "stress_test"] if args.exp == "all" else [args.exp]

    # ensure a predictor ckpt exists (quick pretrain if missing)
    ckpt = Path(args.ckpt)
    if "secdo" in methods and not ckpt.is_file():
        print("Missing predictor ckpt — running short pretrain...", flush=True)
        from secdo.training.trainer import Trainer, load_train_config

        cfg = load_train_config(ROOT / "secdo/configs/train/pretrain.yaml", mode="pretrain")
        cfg.epochs = 3
        cfg.n_train_traj = 32
        cfg.n_val_traj = 8
        cfg.traj_length = 24
        cfg.batch = 8
        cfg.ckpt_dir = str(ckpt.parent)
        cfg.log_dir = "runs/secdo_platform/exp_pretrain"
        Trainer(cfg).run()

    for exp in exps:
        out = Path(args.out) / exp
        run_experiment(
            exp=exp,
            methods=methods,
            seeds=args.seeds,
            horizon=args.horizon,
            batch=args.batch,
            ckpt=str(ckpt),
            out_dir=out,
            device=args.device,
            plot=not args.no_plot,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

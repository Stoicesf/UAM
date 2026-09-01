"""R4 acceptance: solvers + results.json + figures.

Usage:
  python -m secdo.experiments.accept_r4
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from secdo.baselines import get_solver, list_solvers
from secdo.experiments.runner import run_experiment
from secdo.training.trainer import Trainer, load_train_config


def main() -> int:
    assert "secdo" in list_solvers() and "oracle" in list_solvers()
    ckpt_dir = Path("checkpoints/secdo_platform/r4_accept/pretrain")
    ckpt = ckpt_dir / "best.pt"
    if not ckpt.is_file():
        cfg = load_train_config(ROOT / "secdo/configs/train/pretrain.yaml", mode="pretrain")
        cfg.epochs = 2
        cfg.n_train_traj = 16
        cfg.n_val_traj = 4
        cfg.traj_length = 16
        cfg.batch = 4
        cfg.ckpt_dir = str(ckpt_dir)
        cfg.log_dir = "runs/secdo_platform/r4_accept"
        Trainer(cfg).run()

    out = Path("results/secdo/r4_accept")
    path = run_experiment(
        exp="fast_drift",
        methods=["secdo", "reactive", "oracle", "dsgf", "ac_dsgf"],
        seeds=[0],
        horizon=24,
        batch=4,
        ckpt=str(ckpt),
        out_dir=out,
        device="cuda:0",
        plot=True,
    )
    assert path.is_file()
    assert (out / "figures" / "fig1_architecture.pdf").is_file()
    assert (out / "figures" / "fig5_ablation.pdf").is_file()
    # interface: get_solver only
    _ = get_solver("secdo", ckpt=str(ckpt), device="cuda:0")
    print("PASS accept_r4", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""R3 acceptance: unified trainer pretrain → joint → online.

Usage:
  python -m secdo.training.accept_r3
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from secdo.training.trainer import Trainer, load_train_config


def main() -> int:
    # pretrain
    cfg = load_train_config(ROOT / "secdo/configs/train/pretrain.yaml", mode="pretrain")
    cfg.epochs = 2
    cfg.n_train_traj = 16
    cfg.n_val_traj = 4
    cfg.traj_length = 16
    cfg.batch = 4
    cfg.ckpt_dir = "checkpoints/secdo_platform/r3_accept/pretrain"
    cfg.log_dir = "runs/secdo_platform/r3_accept/pretrain"
    logs_p = Trainer(cfg).run()
    assert (Path(cfg.ckpt_dir) / "best.pt").is_file()
    assert logs_p[-1]["delta"] >= 0
    d0, d1 = logs_p[0]["delta"], logs_p[-1]["delta"]
    print(f"pretrain delta {d0:.6f} → {d1:.6f}", flush=True)

    # joint (load platform pretrain)
    cfg_j = load_train_config(ROOT / "secdo/configs/train/joint.yaml", mode="joint")
    cfg_j.load_predictor = str(Path(cfg.ckpt_dir) / "best.pt")
    cfg_j.frozen_epochs = 1
    cfg_j.head_only_epochs = 1
    cfg_j.joint_epochs = 1
    cfg_j.steps_per_epoch = 2
    cfg_j.batch = 4
    cfg_j.horizon = 16
    cfg_j.ckpt_dir = "checkpoints/secdo_platform/r3_accept/joint"
    cfg_j.log_dir = "runs/secdo_platform/r3_accept/joint"
    logs_j = Trainer(cfg_j).run()
    assert (Path(cfg_j.ckpt_dir) / "best.pt").is_file()
    print(f"joint last violation={logs_j[-1]['violation']:.6f} gap={logs_j[-1]['gap']:.6f}", flush=True)

    # online
    cfg_o = load_train_config(ROOT / "secdo/configs/train/online.yaml", mode="online")
    cfg_o.load_predictor = str(Path(cfg_j.ckpt_dir) / "best.pt")
    cfg_o.online_steps = 10
    cfg_o.batch = 2
    cfg_o.horizon = 16
    cfg_o.ckpt_dir = "checkpoints/secdo_platform/r3_accept/online"
    logs_o = Trainer(cfg_o).run()
    assert logs_o and "L_c" in logs_o[-1]
    print(f"online last L_c={logs_o[-1]['L_c']:.6f}", flush=True)

    print("PASS accept_r3", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

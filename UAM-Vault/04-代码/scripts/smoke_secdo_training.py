#!/usr/bin/env python
"""Smoke against frozen training entrypoints (cuda:0)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from experiments.secdo_uav.env_uav_bandwidth import UAVBandwidthConfig
from secdo.training.online_adapt import OnlineAdaptConfig, adapt_step_residual
from secdo.training.pretrain_predictor import PretrainConfig, pretrain_predictor
from secdo.training.train_secdo import SecdoTrainConfig, train_secdo
from secdo.utils.device import build_device_context, print_training_banner


def main() -> int:
    ctx = build_device_context(prefer="cuda:0", amp=True)
    print_training_banner(ctx, projection="anticipatory", projection_detach=True)

    env_cfg = UAVBandwidthConfig(regime="fast", horizon=24, seed=0)
    pcfg = PretrainConfig(
        epochs=1,
        batch=4,
        n_train_traj=8,
        n_val_traj=2,
        device="cuda:0",
        amp=True,
        lr=1e-3,
        ckpt_dir="checkpoints/secdo_smoke_predictor",
        log_dir="runs/secdo_smoke_pretrain",
    )
    modules, plogs = pretrain_predictor(env_cfg, pcfg)
    print("Stage I last:", plogs[-1], flush=True)

    scfg = SecdoTrainConfig(
        skip_pretrain=True,
        head_only_epochs=1,
        joint_epochs=1,
        steps_per_epoch=1,
        batch=4,
        device="cuda:0",
        amp=True,
        log_dir="runs/secdo_smoke_joint",
        ckpt_dir="checkpoints/secdo_smoke_joint",
    )
    env_cfg = UAVBandwidthConfig(regime="fast", horizon=12, seed=1)
    modules, logs = train_secdo(env_cfg, scfg, modules=modules)
    print("Stage II last stage:", logs[-1].get("stage"), flush=True)

    head = modules["head"].to(ctx.device)
    loss = adapt_step_residual(
        head,
        torch.randn(4, 32, device=ctx.device),
        torch.ones(4, 1, device=ctx.device),
        torch.ones(4, 1, device=ctx.device) * 1.1,
        OnlineAdaptConfig(lr=1e-2, zeta=0.5, device="cuda:0"),
    )
    print("Online adapt L_c:", loss, flush=True)
    print("PASS smoke_secdo_training", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

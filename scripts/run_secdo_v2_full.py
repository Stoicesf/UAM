#!/usr/bin/env python
"""SECDO v2 full GPU pipeline — no interactive prompts.

Usage:
  E:\\ANACONDA\\envs\\pytorch12\\python.exe -u scripts/run_secdo_v2_full.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PY_OK = torch.cuda.is_available()
DEVICE = "cuda:0" if PY_OK else "cpu"


def banner(msg: str) -> None:
    print("\n" + "=" * 60 + f"\n{msg}\n" + "=" * 60, flush=True)
    if PY_OK:
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)


def step_pretrain() -> Path:
    banner("Stage I — Predictor pretrain")
    from secdo.training.trainer import Trainer, load_train_config

    cfg = load_train_config(ROOT / "secdo/configs/train/pretrain.yaml", mode="pretrain")
    cfg.epochs = 30
    cfg.n_train_traj = 512
    cfg.n_val_traj = 64
    cfg.traj_length = 64
    cfg.batch = 32
    cfg.amp = True
    cfg.ckpt_dir = "checkpoints/secdo_v2/pretrain"
    cfg.log_dir = "runs/secdo_v2/pretrain"
    cfg.data_dir = ""  # regenerate cache if needed
    Trainer(cfg).run()
    return Path(cfg.ckpt_dir) / "best.pt"


def step_joint(ckpt: Path) -> Path:
    banner("Stage II — Joint SECDO")
    from secdo.training.trainer import Trainer, load_train_config

    cfg = load_train_config(ROOT / "secdo/configs/train/joint.yaml", mode="joint")
    cfg.load_predictor = str(ckpt)
    cfg.frozen_epochs = 3
    cfg.head_only_epochs = 10
    cfg.joint_epochs = 15
    cfg.steps_per_epoch = 12
    cfg.batch = 16
    cfg.horizon = 64
    cfg.amp = True
    cfg.ckpt_dir = "checkpoints/secdo_v2/joint"
    cfg.log_dir = "runs/secdo_v2/joint"
    Trainer(cfg).run()
    return Path(cfg.ckpt_dir) / "best.pt"


def step_online(ckpt: Path) -> None:
    banner("Stage III — Online adapt")
    from secdo.training.trainer import Trainer, load_train_config

    cfg = load_train_config(ROOT / "secdo/configs/train/online.yaml", mode="online")
    cfg.load_predictor = str(ckpt)
    cfg.online_steps = 200
    cfg.batch = 8
    cfg.horizon = 128
    cfg.ckpt_dir = "checkpoints/secdo_v2/online"
    cfg.log_dir = "runs/secdo_v2/online"
    Trainer(cfg).run()


def step_uav(ckpt: Path) -> None:
    banner("UAV multi-regime × multi-seed")
    from secdo.experiments.runner import run_experiment
    from secdo.evaluation.plots import render_all

    methods = ["secdo", "reactive", "oracle", "dsgf", "ac_dsgf"]
    seeds = [0, 1, 2, 3, 4]
    for exp in ("slow_drift", "fast_drift", "stress_test"):
        out = Path(f"results/secdo_v2/uav/{exp}")
        # resume: skip finished regimes
        if (out / "summary.json").is_file() and exp == "slow_drift":
            print("skip finished", exp, flush=True)
            try:
                render_all(out / "results.json", out / "figures")
            except Exception as e:
                print("plot retry:", e, flush=True)
            continue
        run_experiment(
            exp=exp,
            methods=methods,
            seeds=seeds,
            horizon=96,
            batch=16,
            ckpt=str(ckpt),
            out_dir=str(out),
            device=DEVICE,
            plot=True,
        )


def step_theory_exps() -> None:
    banner("Synthetic / PI boundary / Crash recovery")
    from secdo.experiments.synthetic_convex.run import main as synth_main
    from secdo.experiments.pi_boundary.run import main as pi_main
    from secdo.experiments.crash_recovery.run import main as crash_main

    # force cuda for synth tensors where possible
    import secdo.experiments.synthetic_convex.env as env_mod

    # monkey-patch default device via configs inside runners — patch SynthConfig usage
    _orig = env_mod.SynthConfig

    def SynthConfigGPU(**kwargs):
        kwargs.setdefault("device", DEVICE)
        return _orig(**kwargs)

    env_mod.SynthConfig = SynthConfigGPU  # type: ignore
    synth_main()
    pi_main()
    crash_main()


def step_dataset_manifest() -> None:
    banner("Offline UAV datasets (paper-scale manifests)")
    from secdo.datasets.uav.generator import GenerateConfig, generate_dataset

    for regime, name, n in (
        ("slow_drift", "uav_slow_v1", 1000),
        ("fast_drift", "uav_fast_v1", 1000),
        ("random_waypoint", "uav_stress_v1", 1000),
    ):  # paper-scale offline corpora
        out = Path(f"data/secdo/{name}")
        if (out / "manifest.json").is_file():
            print("exists", out, flush=True)
            continue
        cfg = GenerateConfig(
            version="v1",
            trajectories=n,
            length=200,
            seed=42,
            regime=regime,
            n_agents=8,
            device="cpu",  # generation is CPU-bound
            stress_eps=0.35 if "stress" in name else 0.0,
            stress_delta=0.25 if "stress" in name else 0.0,
        )
        generate_dataset(out, cfg)
        print("wrote", out, flush=True)


def write_master_summary() -> None:
    banner("Master summary")
    root = Path("results/secdo_v2")
    root.mkdir(parents=True, exist_ok=True)
    summary = {
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if PY_OK else None,
        "checkpoints": {
            "pretrain": str(Path("checkpoints/secdo_v2/pretrain/best.pt")),
            "joint": str(Path("checkpoints/secdo_v2/joint/best.pt")),
            "online": str(Path("checkpoints/secdo_v2/online/last.pt")),
        },
        "uav_results": {},
    }
    for exp in ("slow_drift", "fast_drift", "stress_test"):
        p = Path(f"results/secdo_v2/uav/{exp}/summary.json")
        if p.is_file():
            summary["uav_results"][exp] = json.loads(p.read_text(encoding="utf-8"))
    (root / "MASTER_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote", root / "MASTER_SUMMARY.json", flush=True)


def main() -> int:
    assert PY_OK, "CUDA required — activate pytorch12 with GPU"
    torch.cuda.set_device(0)
    banner(f"SECDO v2 FULL PIPELINE on {DEVICE}")
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--skip-data", action="store_true")
    p.add_argument("--from", dest="from_stage", default="data",
                   choices=["data", "pretrain", "joint", "online", "uav", "theory"])
    args = p.parse_args()

    stages = ["data", "pretrain", "joint", "online", "uav", "theory"]
    start = stages.index(args.from_stage)
    ckpt = Path("checkpoints/secdo_v2/pretrain/best.pt")
    joint = Path("checkpoints/secdo_v2/joint/best.pt")

    if start <= 0 and not args.skip_data:
        step_dataset_manifest()
    if start <= 1:
        ckpt = step_pretrain()
    if start <= 2:
        joint = step_joint(ckpt if ckpt.is_file() else Path("checkpoints/secdo_v2/pretrain/best.pt"))
    if start <= 3:
        step_online(joint if joint.is_file() else ckpt)
    if start <= 4:
        step_uav(joint if joint.is_file() else ckpt)
    if start <= 5:
        step_theory_exps()
    write_master_summary()
    banner("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

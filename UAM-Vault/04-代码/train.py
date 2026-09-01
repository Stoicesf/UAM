"""Unified training entry — experiment-driven with Gate workflow."""

from __future__ import annotations

import argparse
import sys

from utils.experiment import ExperimentRun, load_experiment_config, normalize_train_frames
from utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Staged thesis experiments with Gates")
    parser.add_argument("--exp", default="configs/experiments/exp1_guide.yaml")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--smoke", action="store_true", help="2048 frames quick check")
    parser.add_argument(
        "--gate",
        type=int,
        default=None,
        choices=[1, 2, 3, 4],
        help="Gate 1=forward only, 2=10k, 3=102k, 4=multi-seed (use run_seeds.py)",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--output-root", default=None, help="Override results output directory")
    args = parser.parse_args()

    if args.gate == 1:
        import subprocess
        rc = subprocess.call([sys.executable, "scripts/gate1_forward_check.py", "--exp", args.exp])
        sys.exit(rc)

    if args.gate == 4:
        import subprocess
        rc = subprocess.call([
            sys.executable, "scripts/run_seeds.py",
            "--exp", args.exp, "--gate", "3",
        ])
        sys.exit(rc)

    exp_cfg = load_experiment_config(args.exp)
    train_cfg = exp_cfg["train"]
    env_cfg = exp_cfg["env"]
    guidance_cfg = exp_cfg.get("guidance", {"mode": "none"})
    meta = exp_cfg.get("experiment", {})

    if args.seed is not None:
        train_cfg["seed"] = args.seed

    normalize_train_frames(train_cfg, smoke=args.smoke, gate=args.gate)
    set_seed(train_cfg.get("seed", 42))

    exp_run = ExperimentRun(
        exp_cfg, run_name=args.run_name, config_path=args.exp, output_root=args.output_root
    )

    stage = meta.get("stage", 0)
    mode = guidance_cfg.get("mode", "none")

    print(f"Experiment: {meta.get('id')} (Stage {stage})")
    print(f"Gate: {args.gate or 'default'}, Seed: {train_cfg.get('seed')}")
    print(f"Guidance: {mode}, Frames: {train_cfg['total_frames']}")
    print(f"Output: {exp_run.root}")

    if mode == "none" or stage == 0:
        from algorithms.baseline.runner import train_mappo_with_experiment
        curve = train_mappo_with_experiment(train_cfg, env_cfg, exp_run, resume=args.resume)
    else:
        from algorithms.guided.runner import train_guided_mappo
        curve = train_guided_mappo(train_cfg, env_cfg, guidance_cfg, exp_run)

    print(f"Done. Final reward: {curve[-1]:.4f}" if curve else "Done.")
    print(f"Metrics:       {exp_run.metrics_path}")
    print(f"Alignment CSV: {exp_run.action_alignment_path}")
    print(f"Comm CSV:      {exp_run.communication_path}")
    print(f"TensorBoard:   tensorboard --logdir {exp_run.log_dir}")


if __name__ == "__main__":
    main()

"""Re-evaluate a finished baseline16 run and refresh summary.json."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

from algorithms.baseline.mappo import build_mappo
from utils.eval_rollout import evaluate_policy
from utils.experiment import load_experiment_config
from utils.training_helpers import build_paper_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    root = Path("results/baseline16") / args.run_name
    cfg = load_experiment_config(args.exp)
    train_cfg = cfg["train"]
    env_cfg = cfg["env"]

    components = build_mappo(train_cfg, env_cfg)
    ckpt_path = root / "checkpoints" / "final.pt"
    ckpt = torch.load(ckpt_path, map_location=components.device, weights_only=False)
    components.policy.load_state_dict(ckpt["policy"])

    eval_stats = evaluate_policy(
        components.env,
        components.policy,
        num_episodes=train_cfg.get("eval_episodes", 200),
        success_threshold=env_cfg.get("success_threshold", 0.3),
        max_steps=env_cfg.get("max_steps"),
    )

    with open(root / "summary.json", encoding="utf-8") as f:
        summary = json.load(f)

    reward_curve = summary.get("reward_curve", [])
    paper = build_paper_summary(reward_curve, [], eval_stats)
    summary["paper_metrics"] = paper
    summary.update(paper)
    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")

    with open(root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Updated {root / 'summary.json'}")
    print(paper)


if __name__ == "__main__":
    main()

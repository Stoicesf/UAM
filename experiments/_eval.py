"""Shared checkpoint evaluation — reused by experiments/*.py and evaluate.py."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from utils.eval_rollout import evaluate_policy
from utils.experiment import load_experiment_config
from utils.policy_loader import load_policy_for_eval, resolve_checkpoint
from utils.seed import set_seed


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_eval(config_path: str, *, label: str = "eval") -> dict:
    """Load checkpoint, rollout, write JSON under results_dir."""
    cfg = load_yaml(config_path)
    set_seed(cfg.get("seed", 42))

    exp_path = cfg.get("exp_config", "configs/experiments/exp0_baseline.yaml")
    exp_cfg = load_experiment_config(exp_path)
    ckpt = resolve_checkpoint(cfg.get("checkpoint"), cfg.get("fallback_checkpoint"))
    env_cfg = exp_cfg["env"]

    env, policy = load_policy_for_eval(exp_cfg, ckpt)
    stats = evaluate_policy(
        env,
        policy,
        num_episodes=cfg.get("num_episodes", 100),
        success_threshold=env_cfg.get("success_threshold", 0.3),
        max_steps=env_cfg.get("max_steps"),
    )

    results_dir = Path(cfg.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{label}.json"
    payload = {
        "label": label,
        "checkpoint": str(ckpt),
        "exp_config": exp_path,
        **stats,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"{label}: success={stats['success']:.3f} reward={stats['reward']:.3f}")
    if "communication_cost" in stats:
        print(f"  communication_cost={stats['communication_cost']:.4f}")
    print(f"  -> {out_path}")
    return stats


def run_scalability(config_path: str) -> list[dict]:
    """Sweep num_agents at eval time. ponytail: same ckpt, different env — valid smoke only."""
    cfg = load_yaml(config_path)
    set_seed(cfg.get("seed", 42))
    base_exp = load_experiment_config(cfg.get("exp_config", "configs/experiments/exp0_baseline.yaml"))
    ckpt = resolve_checkpoint(cfg.get("checkpoint"), cfg.get("fallback_checkpoint"))
    rows: list[dict] = []

    for n in cfg.get("agent_counts", [4, 8, 16, 32]):
        exp_cfg = {**base_exp, "env": {**base_exp["env"], "num_agents": n}}
        env, policy = load_policy_for_eval(exp_cfg, ckpt)
        stats = evaluate_policy(
            env,
            policy,
            num_episodes=cfg.get("num_episodes", 32),
            success_threshold=base_exp["env"].get("success_threshold", 0.3),
            max_steps=base_exp["env"].get("max_steps"),
        )
        row = {"num_agents": n, **stats}
        rows.append(row)
        print(f"  agents={n} success={stats['success']:.3f}")

    out = Path(cfg.get("results_dir", "results")) / "scalability.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"  -> {out}")
    return rows


def run_ablation(config_path: str) -> dict:
    """Run eval once; ablation flags need separate checkpoints to compare."""
    cfg = load_yaml(config_path)
    ablation = cfg.get("ablation", {})
    if ablation:
        print("Ablation flags (compare separate trained checkpoints):")
        for key, val in ablation.items():
            print(f"  {key}: {val}")
    return run_eval(config_path, label="ablation")

"""P1-2 — Zero-shot generalization evaluation (no training).

Usage:
  python scripts/eval_generalization.py --dry-run
  python scripts/eval_generalization.py --allow-fallback   # use plain-nav ckpt for smoke test
  python scripts/eval_generalization.py --plot

Requires checkpoints trained on obs=4 (see configs/generalization/manifest.json).
Does NOT call optimizer.step() or policy.train().
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml

from utils.eval_rollout import evaluate_policy
from utils.experiment import load_experiment_config
from utils.policy_loader import load_policy_for_eval, resolve_checkpoint

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "generalization" / "manifest.json"


def load_test_env_cfg(test_yaml: str) -> dict:
    with open(ROOT / test_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    defaults_env = ROOT / "configs" / "environment.yaml"
    defaults_train = ROOT / "configs" / "train.yaml"
    if defaults_train.exists():
        with open(defaults_train, encoding="utf-8") as f:
            base_train = yaml.safe_load(f) or {}
        cfg["train"] = {**base_train, **cfg.get("train", {})}
    if defaults_env.exists():
        with open(defaults_env, encoding="utf-8") as f:
            base_env = yaml.safe_load(f) or {}
        cfg["env"] = {**base_env, **cfg.get("env", {})}
    return cfg


def generalization_gap(s_train: float, s_test: float) -> float:
    if s_train <= 0:
        return 0.0
    return (s_train - s_test) / s_train


def run_eval(allow_fallback: bool, dry_run: bool) -> list[dict]:
    with open(MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)

    rows = []
    train_obs = str(manifest["train_obstacles"])

    for method, spec in manifest["methods"].items():
        ckpt = spec["checkpoint"]
        fb = spec.get("fallback_checkpoint") if allow_fallback else None
        if dry_run:
            print(f"[dry-run] {method}: exp={spec['exp_config']} ckpt={ckpt}")
            continue

        try:
            ckpt_path = resolve_checkpoint(ckpt, fb)
        except FileNotFoundError as e:
            print(f"[skip] {method}: {e}")
            continue

        if fb and Path(ckpt) != ckpt_path:
            print(f"[warn] {method}: using fallback checkpoint {ckpt_path}")

        train_exp = load_experiment_config(str(ROOT / spec["exp_config"]))
        env_cfg_base = train_exp["env"]
        train_cfg = train_exp["train"]

        for obs_label, test_yaml in manifest["test_configs"].items():
            test_cfg = load_test_env_cfg(test_yaml)
            env_cfg = {**env_cfg_base, **test_cfg["env"]}
            eval_episodes = test_cfg["train"].get("eval_episodes", 200)

            env, policy = load_policy_for_eval(
                {
                    "train": train_cfg,
                    "env": env_cfg,
                    "guidance": train_exp.get("guidance", {"mode": "none"}),
                },
                ckpt_path,
            )

            with torch.no_grad():
                stats = evaluate_policy(
                    env,
                    policy,
                    num_episodes=eval_episodes,
                    max_steps=env_cfg.get("max_steps"),
                )

            row = {
                "method": method,
                "obstacles": int(obs_label),
                "success": stats["success"],
                "collision": stats["collision"],
                "reward": stats["reward"],
                "path_length": stats["path_length"],
                "episode_length": stats["episode_length"],
                "checkpoint": str(ckpt_path),
            }
            rows.append(row)
            print(
                f"{method} obs={obs_label}: success={stats['success']:.4f} "
                f"collision={stats['collision']:.4f}"
            )

    if not dry_run and rows:
        s_train = {
            (r["method"], manifest["train_obstacles"]): r["success"]
            for r in rows
            if str(r["obstacles"]) == train_obs
        }
        for r in rows:
            key = (r["method"], r["obstacles"])
            st = s_train.get((r["method"], int(train_obs)), 0.0)
            r["gen_gap"] = round(generalization_gap(st, r["success"]), 4)

    return rows


def export_table(rows: list[dict]):
    out_dir = ROOT / "paper" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "table3_generalization.json"
    csv_path = out_dir / "table3_generalization.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    if rows:
        fields = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")


def plot_figure(rows: list[dict]):
    if not rows:
        return
    methods = sorted({r["method"] for r in rows})
    obstacles = sorted({r["obstacles"] for r in rows})
    colors = {"mappo": "#4C72B0", "gat": "#DD8452", "dsgf": "#C44E52"}

    plt.figure(figsize=(8, 5))
    for method in methods:
        ys = []
        for obs in obstacles:
            match = [r for r in rows if r["method"] == method and r["obstacles"] == obs]
            ys.append(match[0]["success"] * 100 if match else 0.0)
        plt.plot(obstacles, ys, marker="o", linewidth=2, label=method.upper(), color=colors.get(method))

    plt.xlabel("Obstacle Count (zero-shot)")
    plt.ylabel("Success Rate (%)")
    plt.title("Generalization under Unseen Obstacle Density")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = ROOT / "paper" / "figures" / "fig5_generalization.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    rows = run_eval(allow_fallback=args.allow_fallback, dry_run=args.dry_run)
    if args.dry_run:
        return

    export_table(rows)
    if args.plot and rows:
        plot_figure(rows)


if __name__ == "__main__":
    main()

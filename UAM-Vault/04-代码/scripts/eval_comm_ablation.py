"""Silence Collapse ablation — prove AC-DSGF is not trivial edge zeroing.

Variants (same 16UAV AC-DSGF checkpoint, eval-only):
  AC-full     : learned gate g_ij  (nominal)
  AC-no_budget: g = radius_mask (full local graph) — Comm should ↑↑
  AC-random   : g ~ U(0,1) on radius support — Success should ↓

Usage:
  python scripts/eval_comm_ablation.py --seed 42 --episodes 64 --plot
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchrl.envs.utils import step_mdp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithms.guided.mappo_guided import ACGuideAdapter
from utils.experiment import load_experiment_config
from utils.policy_loader import load_policy_for_eval, resolve_checkpoint
from utils.seed import set_seed

EPS = 1e-6
VARIANTS = ("full", "no_budget", "random")


def set_ablation(policy, mode: str):
    for m in policy.modules():
        if isinstance(m, ACGuideAdapter):
            m.set_ablation_mode(mode)
            m.set_budget_ratio(None)  # no hard top-k during ablation
            return m
    raise RuntimeError("ACGuideAdapter not found")


def eval_variant(env, policy, mode: str, num_episodes: int, max_steps: int, thr: float) -> dict:
    set_ablation(policy, mode)
    policy.eval()
    successes, collisions, rewards, edges = [], [], [], []

    with torch.no_grad():
        for _ in range(num_episodes):
            td = env.reset()
            ep_reward = 0.0
            ep_collision = 0.0
            ep_success = 0.0
            ep_edges = []
            for _ in range(max_steps):
                td = policy(td)
                td = env.step(td)
                obs = td.get(("next", "agents", "observation"))
                if obs.dim() == 3:
                    goal_rel = obs[0, :, 4:6]
                else:
                    goal_rel = obs[:, 4:6]
                ep_success = float((goal_rel.norm(dim=-1) < thr).float().mean())
                rew = td.get(("next", "agents", "reward"))
                ep_reward += float(rew.mean())
                info = td.get(("next", "agents", "info"), default=None)
                if info is not None and "agent_collisions" in info.keys():
                    coll = info.get("agent_collisions")
                    ep_collision = max(ep_collision, float((coll < 0).float().mean()))
                for m in policy.modules():
                    if isinstance(m, ACGuideAdapter):
                        ep_edges.append(m.last_soft_edges)
                        break
                done = td.get(("next", "done"))
                if done is not None and bool(done.any()):
                    break
                td = step_mdp(td)
            successes.append(ep_success)
            collisions.append(ep_collision)
            rewards.append(ep_reward)
            edges.append(sum(ep_edges) / max(len(ep_edges), 1))

    s = sum(successes) / len(successes)
    c = sum(edges) / len(edges)
    return {
        "variant": mode,
        "success": round(s, 4),
        "collision": round(sum(collisions) / len(collisions), 4),
        "reward": round(sum(rewards) / len(rewards), 4),
        "comm_cost": round(c, 4),
        "cei": round(s / (c + EPS), 6),
    }


def plot_ablation(rows: list[dict], out: Path):
    labels = [r["variant"] for r in rows]
    display = {"full": "AC-full", "no_budget": "AC-no budget", "random": "AC-random"}
    names = [display.get(x, x) for x in labels]
    s = [100 * r["success"] for r in rows]
    c = [r["comm_cost"] for r in rows]
    colors = ["#C44E52", "#4C72B0", "#8172B3"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(names, s, color=colors)
    axes[0].set_ylabel("Success (%)")
    axes[0].set_title("Silence Collapse Ablation — Success")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(names, c, color=colors)
    axes[1].set_ylabel("Communication Cost (soft edges)")
    axes[1].set_title("Silence Collapse Ablation — Comm")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle("AC-DSGF Communication Ablation (16 UAV)", fontsize=12)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Default: results/ac_dsgf/uav16/s{seed}/checkpoints/final.pt",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    ckpt = resolve_checkpoint(
        args.ckpt or f"results/ac_dsgf/uav16/s{args.seed}/checkpoints/final.pt"
    )
    exp_cfg = load_experiment_config(str(ROOT / "configs/ac_dsgf/ac_dsgf_16uav.yaml"))
    exp_cfg["env"]["num_envs"] = 1
    exp_cfg["env"]["device"] = "cpu"

    print(f"ckpt={ckpt} episodes={args.episodes}")
    env, policy = load_policy_for_eval(exp_cfg, ckpt)
    max_steps = int(exp_cfg["env"].get("max_steps", 128))
    thr = float(exp_cfg["env"].get("success_threshold", 0.3))

    rows = []
    for mode in VARIANTS:
        print(f"\n=== {mode} ...", flush=True)
        stats = eval_variant(env, policy, mode, args.episodes, max_steps, thr)
        rows.append(stats)
        print(
            f"  S={stats['success']:.2%} C={stats['comm_cost']:.2f} "
            f"CEI={stats['cei']:.4f} Col={stats['collision']:.2%}"
        )

    out_dir = ROOT / "results" / "ac_dsgf" / "comm_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comm_ablation.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    csv_path = out_dir / "comm_ablation.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {csv_path}")

    paper_csv = ROOT / "paper" / "tables" / "table_comm_ablation.csv"
    paper_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(paper_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {paper_csv}")

    if args.plot:
        plot_ablation(rows, ROOT / "paper" / "figures" / "fig_comm_ablation.png")
        plot_ablation(rows, out_dir / "fig_comm_ablation.png")


if __name__ == "__main__":
    main()

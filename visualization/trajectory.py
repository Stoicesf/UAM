"""Plot UAV trajectories from a guided MAPPO rollout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.guided.mappo_guided import build_guided_mappo
from utils.seed import set_seed


def _load_cfg(exp_path: str) -> tuple[dict, dict, dict]:
    with open(exp_path, encoding="utf-8") as f:
        exp = yaml.safe_load(f)
    with open(ROOT / "configs/train.yaml", encoding="utf-8") as f:
        base_train = yaml.safe_load(f) or {}
    with open(ROOT / "configs/environment.yaml", encoding="utf-8") as f:
        base_env = yaml.safe_load(f) or {}
    return (
        {**base_train, **exp.get("train", {})},
        {**base_env, **exp.get("env", {})},
        exp.get("guidance", {}),
    )


def collect_trajectory(components, max_steps: int) -> list[list[tuple[float, float]]]:
    """Return per-agent (x, y) lists."""
    env = components.env
    policy = components.policy
    n_agents = env.n_agents
    trajectories: list[list[tuple[float, float]]] = [[] for _ in range(n_agents)]

    policy.eval()
    with torch.no_grad():
        td = env.reset()
        obs = td.get(("agents", "observation"))
        for i in range(n_agents):
            trajectories[i].append((obs[0, i, 0].item(), obs[0, i, 1].item()))

        for _ in range(max_steps):
            td = policy(td)
            td = env.step(td)
            obs = td.get(("next", "agents", "observation"))
            for i in range(n_agents):
                trajectories[i].append((obs[0, i, 0].item(), obs[0, i, 1].item()))
            td = td.select(*td.keys(), strict=False)
            td.set(("agents", "observation"), obs)
            done = td.get(("next", "done"))
            if done.any():
                break
    policy.train()
    return trajectories


def plot_trajectory(
    trajectories: list[list[tuple[float, float]]],
    goal: tuple[float, float] | None,
    out_path: Path,
    title: str = "UAV Swarm Trajectory",
):
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for i, traj in enumerate(trajectories):
        xs, ys = zip(*traj)
        c = colors[i % len(colors)]
        ax.plot(xs, ys, "-", color=c, linewidth=1.5, label=f"UAV{i + 1}")
        ax.scatter(xs[0], ys[0], color=c, s=60, marker="o", zorder=5)
        ax.scatter(xs[-1], ys[-1], color=c, s=60, marker="s", zorder=5)

    if goal is not None:
        ax.scatter(goal[0], goal[1], color="gold", s=200, marker="*", label="Goal", zorder=6)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate trajectory.png from policy rollout")
    parser.add_argument("--exp", default="configs/gat/gat_a_lambda003.yaml")
    parser.add_argument("--out", default="demo/trajectory.png")
    parser.add_argument("--max-steps", type=int, default=128)
    args = parser.parse_args()

    train_cfg, env_cfg, guidance_cfg = _load_cfg(args.exp)
    set_seed(train_cfg.get("seed", 42))
    components = build_guided_mappo(train_cfg, env_cfg, guidance_cfg)

    trajectories = collect_trajectory(components, args.max_steps)
    goal = None
    if trajectories:
        last = trajectories[0][-1]
        goal = (last[0] + 0.5, last[1] + 0.5)

    out = Path(args.out)
    plot_trajectory(trajectories, goal, out)
    metrics = {
        "num_agents": len(trajectories),
        "steps": max(len(t) for t in trajectories) if trajectories else 0,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.parent / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

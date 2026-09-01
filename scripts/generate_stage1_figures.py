"""Generate Stage 1 paper figures (Figure 1-4) from experiment results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(run_dir: Path) -> list[dict]:
    path = run_dir / "metrics.csv"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_alignment(run_dir: Path) -> list[dict]:
    path = run_dir / "action_alignment.csv"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_curve(steps, values, title, ylabel, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(steps, values, linewidth=2)
    plt.xlabel("Frames")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_compare(steps, series: dict, title, ylabel, save_path):
    plt.figure(figsize=(8, 5))
    for label, vals in series.items():
        plt.plot(steps[: len(vals)], vals, label=label, linewidth=2)
    plt.xlabel("Frames")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate Stage 1 figures 1-4")
    parser.add_argument("--guide-run", default="results/guide/guide_gate3_v1")
    parser.add_argument("--baseline-run", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    guide_dir = Path(args.guide_run)
    out = Path(args.out) if args.out else guide_dir / "plots"
    out.mkdir(parents=True, exist_ok=True)

    g_metrics = load_metrics(guide_dir)
    if not g_metrics:
        print(f"No metrics in {guide_dir}")
        return

    steps = [int(r["step"]) for r in g_metrics]
    rewards = [float(r["episode_reward_mean"]) for r in g_metrics]
    success = [float(r.get("success_rate", 0)) for r in g_metrics]
    collision = [float(r.get("collision_rate", 0)) for r in g_metrics]

    align_rows = load_alignment(guide_dir)
    align_steps = [int(r["step"]) for r in align_rows]
    align_vals = [float(r["action_alignment"]) for r in align_rows]

    if args.baseline_run:
        b_metrics = load_metrics(Path(args.baseline_run))
        b_steps = [int(r["step"]) for r in b_metrics]
        b_rewards = [float(r["episode_reward_mean"]) for r in b_metrics]
        b_success = [float(r.get("success_rate", 0)) for r in b_metrics]
        b_collision = [float(r.get("collision_rate", 0)) for r in b_metrics]
        plot_compare(
            b_steps, {"MAPPO": b_rewards, "MAPPO+Guide": rewards},
            "Figure 1 — Training Reward", "Episode Reward", str(out / "figure1_reward.png"),
        )
        plot_compare(
            b_steps, {"MAPPO": b_success, "MAPPO+Guide": success},
            "Figure 2 — Success Rate", "Success Rate", str(out / "figure2_success.png"),
        )
        plot_compare(
            b_steps, {"MAPPO": b_collision, "MAPPO+Guide": collision},
            "Figure 3 — Collision Rate", "Collision Rate", str(out / "figure3_collision.png"),
        )
    else:
        plot_curve(steps, rewards, "Figure 1 — Training Reward", "Episode Reward", out / "figure1_reward.png")
        plot_curve(steps, success, "Figure 2 — Success Rate", "Success Rate", out / "figure2_success.png")
        plot_curve(steps, collision, "Figure 3 — Collision Rate", "Collision Rate", out / "figure3_collision.png")

    plot_curve(
        align_steps, align_vals,
        "Figure 4 — Action Alignment", "cos(theta_action - theta_guide)",
        out / "figure4_alignment.png",
    )

    summary_path = guide_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            s = json.load(f)
        print("Paper metrics:", json.dumps(s.get("paper_metrics", s), indent=2))

    print(f"Figures saved to {out}")


if __name__ == "__main__":
    main()

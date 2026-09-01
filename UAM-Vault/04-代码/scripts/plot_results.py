"""Generate paper figures from experiment results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.visualization import plot_learning_curve


def load_summaries(results_dir: Path) -> list[dict]:
    summaries = []
    for summary_path in sorted(results_dir.glob("*/summary.json")):
        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)
            data["_path"] = str(summary_path.parent)
            summaries.append(data)
    return summaries


def plot_reward_curves(summaries: list[dict], out_dir: Path):
    for s in summaries:
        curve = s.get("reward_curve", [])
        if not curve:
            continue
        fpb = 2048
        steps = [fpb * (i + 1) for i in range(len(curve))]
        name = s.get("experiment_id", "run")
        plot_learning_curve(
            steps,
            curve,
            title=f"Reward — {name}",
            save_path=str(out_dir / f"reward_{name}.png"),
        )


def plot_multi_seed_curves(results_dir: Path, pattern: str, out_dir: Path, title: str):
    import matplotlib.pyplot as plt
    import numpy as np

    runs = sorted(results_dir.glob(pattern))
    if not runs:
        return
    curves = []
    for run in runs:
        with open(run / "summary.json", encoding="utf-8") as f:
            data = json.load(f)
        curve = data.get("reward_curve", [])
        if curve:
            curves.append(curve)

    if not curves:
        return

    min_len = min(len(c) for c in curves)
    arr = np.array([c[:min_len] for c in curves])
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    steps = [2048 * (i + 1) for i in range(min_len)]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, mean, label="Mean")
    plt.fill_between(steps, mean - std, mean + std, alpha=0.25, label="±1 std")
    plt.xlabel("Frames")
    plt.ylabel("Episode Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{title.replace(' ', '_')}.png", dpi=150)
    plt.close()


def plot_baseline_vs_guide(results_dir: Path, out_dir: Path):
    import matplotlib.pyplot as plt

    baseline = None
    guide = None
    for p in results_dir.glob("*/summary.json"):
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("guidance_mode") == "none":
            baseline = d
        elif d.get("guidance_mode") == "mlp" and guide is None:
            guide = d

    if not baseline or not guide:
        return

    plt.figure(figsize=(8, 5))
    for label, d, style in [
        ("MAPPO", baseline, "--"),
        ("MAPPO+Guide", guide, "-"),
    ]:
        curve = d.get("reward_curve", [])
        if not curve:
            continue
        steps = [2048 * (i + 1) for i in range(len(curve))]
        plt.plot(steps, curve, style, label=label)
    plt.xlabel("Frames")
    plt.ylabel("Reward")
    plt.title("Figure 1 — Baseline vs Guide")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "figure1_baseline_vs_guide.png", dpi=150)
    plt.close()


def plot_comparison_bar(summaries: list[dict], out_dir: Path):
    import matplotlib.pyplot as plt

    labels = [s.get("experiment_id", "?") for s in summaries]
    finals = [s.get("final_reward", 0) or 0 for s in summaries]
    if not labels:
        return
    plt.figure(figsize=(10, 5))
    plt.bar(labels, finals)
    plt.ylabel("Final Episode Reward")
    plt.title("Experiment Comparison")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_final_reward.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot figures from results/")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summaries = load_summaries(results_dir)
    if not summaries:
        print(f"No summaries found in {results_dir}")
        return

    plot_reward_curves(summaries, figures_dir)
    plot_multi_seed_curves(
        results_dir, "guide_gate3_seed*/summary.json", figures_dir,
        "Guide Gate3 Mean Std",
    )
    plot_baseline_vs_guide(results_dir, figures_dir)
    plot_comparison_bar(summaries, figures_dir)
    print(f"Figures saved to {figures_dir}")


if __name__ == "__main__":
    main()

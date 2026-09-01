"""Paper figures from 5-seed baseline16 results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 3407, 2026, 1234, 8888]
METHODS = ["mappo", "gat", "transformer", "dsgf"]
LABELS = {
    "mappo": "MAPPO",
    "gat": "GAT",
    "transformer": "Transformer",
    "dsgf": "DSGF",
}
COLORS = {
    "mappo": "#4C72B0",
    "gat": "#DD8452",
    "transformer": "#55A868",
    "dsgf": "#C44E52",
}
FPB = 2048


def load_curve(method: str, seed: int) -> list[float] | None:
    p = ROOT / "results" / "baseline16_seeds" / method / f"s{seed}" / "summary.json"
    if not p.exists() and seed == 42:
        p = ROOT / "results" / "baseline16" / method / "summary.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f).get("reward_curve", [])


def compute_auc(curve: list[float]) -> float:
    if not curve:
        return 0.0
    x = np.arange(1, len(curve) + 1) * FPB
    y = np.array(curve, dtype=np.float64)
    return float(np.trapezoid(y, x) / (x[-1] - x[0] + FPB))


def plot_learning_curves(out: Path):
    plt.figure(figsize=(9, 5.5))
    for method in METHODS:
        curves = []
        for seed in SEEDS:
            c = load_curve(method, seed)
            if c:
                curves.append(c)
        if not curves:
            continue
        min_len = min(len(c) for c in curves)
        arr = np.array([c[:min_len] for c in curves])
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        steps = [FPB * (i + 1) for i in range(min_len)]
        plt.plot(steps, mean, label=LABELS[method], color=COLORS[method], linewidth=2)
        plt.fill_between(steps, mean - std, mean + std, color=COLORS[method], alpha=0.15)

    plt.xlabel("Training Frames")
    plt.ylabel("Episode Reward (train)")
    plt.title("16-UAV Learning Curves (5 seeds, mean ± std)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "fig2_learning_curve_5seed.png", dpi=150)
    plt.close()


def plot_metric_bars(out: Path, metric: str, ylabel: str, fname: str, load_fn):
    means, stds, labels = [], [], []
    for method in METHODS:
        vals = []
        for seed in SEEDS:
            v = load_fn(method, seed)
            if v is not None:
                vals.append(v)
        if not vals:
            continue
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        labels.append(LABELS[method])

    if not labels:
        return
    x = np.arange(len(labels))
    plt.figure(figsize=(8, 5))
    plt.bar(x, means, yerr=stds, capsize=5, color=[COLORS[m] for m in METHODS if LABELS[m] in labels])
    plt.xticks(x, labels)
    plt.ylabel(ylabel)
    plt.title(f"16-UAV {ylabel} (5 seeds, mean ± std)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / fname, dpi=150)
    plt.close()


def load_success(method: str, seed: int) -> float | None:
    p = ROOT / "results" / "baseline16_seeds" / method / f"s{seed}" / "summary.json"
    if not p.exists() and seed == 42:
        p = ROOT / "results" / "baseline16" / method / "summary.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    return d.get("paper_metrics", d).get("success")


def main():
    out = ROOT / "paper" / "figures"
    out.mkdir(parents=True, exist_ok=True)

    plot_learning_curves(out)
    plot_metric_bars(out, "success", "Success Rate", "fig1_success_5seed.png", load_success)
    plot_metric_bars(out, "aulc", "AULC (normalized reward AUC)", "fig_auc_baseline16.png", lambda m, s: compute_auc(load_curve(m, s) or []))

    print(f"Figures saved to {out}")


if __name__ == "__main__":
    main()

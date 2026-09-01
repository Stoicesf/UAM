"""P1-1 — Four-panel ablation figure (Success / Reward / Collision / Comm).

Usage:
  python scripts/plot_ablation_figure.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "paper" / "ablation_manifest.json"
OUT_DIR = ROOT / "paper" / "figures"
TABLE_OUT = ROOT / "paper" / "tables" / "table2_ablation.csv"

COLORS = ["#8DA0CB", "#FC8D62", "#66C2A5", "#E78AC3"]
METRICS = [
    ("success", "Success Rate", "(a) Success", True),
    ("reward", "Eval Reward", "(b) Reward", False),
    ("collision", "Collision Rate", "(c) Collision", False),
    ("communication_cost", "Comm Cost (mean edges)", "(d) Communication", False),
]


def load_summary(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def mean_comm_cost(run_dir: Path) -> float:
    comm_path = run_dir / "communication.csv"
    if not comm_path.exists():
        return 0.0
    vals = []
    with open(comm_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vals.append(float(row["communication_cost"]))
    return float(np.mean(vals)) if vals else 0.0


def collect_rows(manifest: dict) -> list[dict]:
    rows = []
    for v in manifest["variants"]:
        summary_path = ROOT / v["summary"]
        s = load_summary(summary_path)
        pm = s.get("paper_metrics", s)
        run_dir = summary_path.parent
        rows.append({
            "id": v["id"],
            "label": v["label"],
            "success": pm.get("success", 0.0) * 100,
            "reward": pm.get("reward", 0.0),
            "collision": pm.get("collision", 0.0) * 100,
            "communication_cost": mean_comm_cost(run_dir),
            "path_length": pm.get("path_length", 0.0),
        })
    return rows


def export_table(rows: list[dict]):
    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = ["label", "success_pct", "reward", "collision_pct", "communication_cost", "path_length"]
    with open(TABLE_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({
                "label": r["label"],
                "success_pct": round(r["success"], 4),
                "reward": round(r["reward"], 4),
                "collision_pct": round(r["collision"], 4),
                "communication_cost": round(r["communication_cost"], 4),
                "path_length": round(r["path_length"], 4),
            })


def plot_figure(rows: list[dict], protocol: dict):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = [r["label"] for r in rows]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(
        f"Component Contribution (4 UAV, seed={protocol['seed']}, {protocol['frames']} frames)",
        fontsize=12,
        y=0.98,
    )

    for ax, (key, ylabel, title, pct) in zip(axes.flat, METRICS):
        vals = [r[key] for r in rows]
        bars = ax.bar(x, vals, color=COLORS, edgecolor="black", linewidth=0.6)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        if key == "success":
            ax.set_ylim(0, max(vals) * 1.25 + 0.5)
        for bar, val in zip(bars, vals):
            fmt = f"{val:.2f}%" if pct else f"{val:.2f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                fmt,
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = OUT_DIR / "fig4_ablation_components.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved {out_png}")


def main():
    with open(MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)

    rows = collect_rows(manifest)
    export_table(rows)
    plot_figure(rows, manifest["protocol"])

    print("\n=== Ablation (4 UAV, seed=42) ===")
    for r in rows:
        print(
            f"{r['label']:<14} success={r['success']:.2f}%  "
            f"reward={r['reward']:.2f}  collision={r['collision']:.2f}%"
        )
    print(f"\nTable -> {TABLE_OUT}")


if __name__ == "__main__":
    main()

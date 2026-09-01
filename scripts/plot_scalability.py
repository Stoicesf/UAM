"""Figure 1 — Scalability: Success Rate vs number of UAV."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    summary_path = ROOT / "results" / "scalability" / "summary.json"
    if not summary_path.exists():
        print("Run scripts/run_scalability.py first.")
        sys.exit(1)

    with open(summary_path, encoding="utf-8") as f:
        rows = json.load(f)

    rows = sorted(rows, key=lambda r: r["uav"])
    xs = [r["uav"] for r in rows]
    ys = [r["success"] * 100 for r in rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs, ys, "o-", color="#1f77b4", linewidth=2, markersize=8, label="DSGF v2")
    ax.set_xlabel("Number of UAVs")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Figure 1 — Scalability (DSGF v2, 102k, seed=42)")
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out = ROOT / "figures" / "figure1_scalability.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

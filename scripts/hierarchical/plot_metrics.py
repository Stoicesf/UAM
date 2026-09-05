#!/usr/bin/env python3
"""Plot hierarchical training metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="plots.png")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = ["coverage", "collision", "role_entropy", "light_scout"]
    titles = ["Coverage", "Collision Rate", "Role Entropy", "Light Scout Ratio"]
    for ax, m, t in zip(axes.flatten(), metrics, titles):
        ax.plot(data.get("step", list(range(len(data.get(m, []))))), data.get(m, []))
        ax.set_title(t)
        ax.set_xlabel("env step")
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()

"""Draw environment panel: agents + goals (+ optional obstacles)."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def render_env(
    ax,
    positions: np.ndarray,
    goals: np.ndarray | None,
    title: str = "",
    xlim=None,
    ylim=None,
    show_ids: bool = True,
):
    n = positions.shape[0]
    for i in range(n):
        c = COLORS[i % len(COLORS)]
        ax.scatter(positions[i, 0], positions[i, 1], s=100, color=c, zorder=3, edgecolors="k", linewidths=0.4)
        if show_ids:
            ax.text(positions[i, 0] + 0.05, positions[i, 1] + 0.05, f"U{i+1}", fontsize=8, color=c)
    if goals is not None:
        g = np.asarray(goals)
        if g.ndim == 1:
            ax.scatter(g[0], g[1], s=200, marker="*", color="#DAA520", zorder=4)
        else:
            for i in range(g.shape[0]):
                ax.scatter(g[i, 0], g[i, 1], s=160, marker="*", color="#DAA520", zorder=4)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    if title:
        ax.set_title(title, fontsize=10)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])

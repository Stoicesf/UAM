"""Draw communication graph panel from adjacency / gate matrix."""

from __future__ import annotations

import numpy as np

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def edge_list(adj: np.ndarray, thr: float = 0.05) -> list[tuple[int, int, float]]:
    n = adj.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            w = float(max(adj[i, j], adj[j, i]))
            if w > thr:
                edges.append((i, j, w))
    return edges


def active_edge_count(adj: np.ndarray, thr: float = 0.05) -> int:
    return len(edge_list(adj, thr))


def render_comm_graph(
    ax,
    positions: np.ndarray,
    adj: np.ndarray,
    title: str = "",
    thr: float = 0.05,
    xlim=None,
    ylim=None,
):
    n = positions.shape[0]
    edges = edge_list(adj, thr)
    for i, j, w in edges:
        ax.plot(
            [positions[i, 0], positions[j, 0]],
            [positions[i, 1], positions[j, 1]],
            color="#555555",
            alpha=min(0.9, 0.2 + 0.7 * w),
            linewidth=0.8 + 2.0 * min(w, 1.0),
            zorder=1,
        )
    for i in range(n):
        c = COLORS[i % len(COLORS)]
        ax.scatter(positions[i, 0], positions[i, 1], s=90, color=c, zorder=3, edgecolors="k", linewidths=0.3)
        ax.text(positions[i, 0] + 0.04, positions[i, 1] + 0.04, f"U{i+1}", fontsize=7, color=c)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    n_e = len(edges)
    ax.set_title(title or f"Comm graph  edges={n_e}", fontsize=10)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    return n_e

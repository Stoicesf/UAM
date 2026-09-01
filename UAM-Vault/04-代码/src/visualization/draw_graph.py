"""Draw agents, goals, and communication graph for paper demos."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from guidance.graph_builder import build_adjacency

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def adjacency_from_positions(
    positions: torch.Tensor | np.ndarray,
    comm_radius: float,
) -> np.ndarray:
    """Return (N, N) adjacency for a single env."""
    if isinstance(positions, np.ndarray):
        positions = torch.as_tensor(positions, dtype=torch.float32)
    if positions.dim() == 2:
        positions = positions.unsqueeze(0)
    adj = build_adjacency(positions, comm_radius)
    return adj[0].detach().cpu().numpy()


def draw_frame(
    positions: np.ndarray,
    goals: np.ndarray | None,
    adj: np.ndarray | None,
    out_path: Path | None = None,
    ax=None,
    title: str = "",
    show_ids: bool = True,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
):
    """Draw one 2D frame. positions: (N, 2), goals: (N, 2) or (2,)."""
    owned = ax is None
    if owned:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    n = positions.shape[0]
    if adj is not None:
        for i in range(n):
            for j in range(i + 1, n):
                w = float(adj[i, j])
                if w <= 0:
                    continue
                ax.plot(
                    [positions[i, 0], positions[j, 0]],
                    [positions[i, 1], positions[j, 1]],
                    color="gray",
                    alpha=min(0.85, 0.25 + 0.6 * w),
                    linewidth=0.8 + 1.5 * w,
                    zorder=1,
                )

    for i in range(n):
        c = COLORS[i % len(COLORS)]
        ax.scatter(positions[i, 0], positions[i, 1], s=90, color=c, zorder=3)
        if show_ids:
            ax.text(
                positions[i, 0] + 0.04,
                positions[i, 1] + 0.04,
                f"U{i+1}",
                fontsize=8,
                color=c,
            )

    if goals is not None:
        g = np.asarray(goals)
        if g.ndim == 1:
            ax.scatter(g[0], g[1], s=220, marker="*", color="gold", zorder=4, label="Goal")
        else:
            for i in range(g.shape[0]):
                ax.scatter(g[i, 0], g[i, 1], s=140, marker="*", color="gold", zorder=4)
            ax.scatter([], [], s=140, marker="*", color="gold", label="Goal")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    if title:
        ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)

    if owned:
        fig.tight_layout()
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=120)
            plt.close(fig)
        return fig, ax
    return fig, ax


def plot_trajectories(
    trajectories: list[np.ndarray],
    goals: np.ndarray | None,
    out_path: Path,
    title: str = "Trajectory",
):
    """trajectories: list of (T, 2) arrays."""
    fig, ax = plt.subplots(figsize=(7, 7))
    for i, traj in enumerate(trajectories):
        c = COLORS[i % len(COLORS)]
        ax.plot(traj[:, 0], traj[:, 1], "-", color=c, lw=1.6, label=f"UAV{i+1}")
        ax.scatter(traj[0, 0], traj[0, 1], color=c, s=50, marker="o", zorder=5)
        ax.scatter(traj[-1, 0], traj[-1, 1], color=c, s=50, marker="s", zorder=5)

    if goals is not None:
        g = np.asarray(goals)
        if g.ndim == 1:
            ax.scatter(g[0], g[1], s=220, marker="*", color="gold", label="Goal", zorder=6)
        else:
            ax.scatter(g[:, 0], g[:, 1], s=160, marker="*", color="gold", label="Goal", zorder=6)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

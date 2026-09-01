"""Bottom metric strips for communication-aware demo."""

from __future__ import annotations

import numpy as np


METHOD_COLORS = {
    "gat": "#DD8452",
    "dsgf": "#4C72B0",
    "ac_dsgf": "#C44E52",
}


def render_metrics(
    axes,
    traces: dict[str, dict],
    t: int,
    methods: list[str],
):
    """axes: [comm_cost, active_edges, success/goal_dist]."""
    ax_c, ax_e, ax_s = axes
    for ax in axes:
        ax.clear()

    for m in methods:
        tr = traces[m]
        times = np.asarray(tr["time"][: t + 1], dtype=float)
        color = METHOD_COLORS.get(m, "gray")
        label = {"gat": "GAT", "dsgf": "DSGF", "ac_dsgf": "AC-DSGF"}.get(m, m)
        ax_c.plot(times, tr["comm_cost"][: t + 1], color=color, lw=1.8, label=label)
        ax_e.plot(times, tr["active_edges"][: t + 1], color=color, lw=1.8, label=label)
        ax_s.plot(times, tr["success"][: t + 1], color=color, lw=1.8, label=label)

    ax_c.set_ylabel("Comm cost", fontsize=8)
    ax_c.set_title("Communication Cost", fontsize=9)
    ax_c.grid(True, alpha=0.3)
    ax_c.legend(loc="upper right", fontsize=7, ncol=3)

    ax_e.set_ylabel("Edges", fontsize=8)
    ax_e.set_title("Active Edges", fontsize=9)
    ax_e.grid(True, alpha=0.3)

    ax_s.set_ylabel("Success", fontsize=8)
    ax_s.set_xlabel("t", fontsize=8)
    ax_s.set_title("Success Progress", fontsize=9)
    ax_s.set_ylim(-0.05, 1.05)
    ax_s.grid(True, alpha=0.3)

    for ax in axes:
        ax.tick_params(labelsize=7)

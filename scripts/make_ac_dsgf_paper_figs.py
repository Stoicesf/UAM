"""Generate AC-DSGF framework + Pareto figures for the IEEE draft."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "ac_dsgf" / "figures"
PAPER = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
PAPER.mkdir(parents=True, exist_ok=True)


def box(ax, x, y, w, h, text, fc="#F7F7F7", ec="#333"):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=fc,
        edgecolor=ec,
        lw=1.5,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)


def framework():
    fig, ax = plt.subplots(figsize=(11.2, 4.2))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    ax.set_title("AC-DSGF Framework", fontsize=13, pad=8)

    box(ax, 0.3, 1.4, 1.6, 1.4, "Local obs\n$o_i$", fc="#E8F1FA")
    box(ax, 2.2, 1.4, 1.8, 1.4, "Radius mask $A$\n+ quality $q_{ij}$", fc="#E8F1FA")
    box(ax, 4.3, 1.4, 1.9, 1.4, "Gate $g_{ij}$\n$\\sigma(f_\\theta(\\cdot))$", fc="#FDEDEC")
    box(ax, 6.5, 1.4, 1.7, 1.4, "Budget\nTop-K $\\rho$", fc="#FCF3CF")
    box(ax, 8.5, 2.35, 2.3, 1.3, "Sparse DSGF\nAttn + GRU $\\rightarrow \\Phi$", fc="#E8F8F5")
    box(ax, 8.5, 0.55, 2.3, 1.3, "Residual policy\n$a=\\pi(o)+\\beta\\Delta(\\Phi)$", fc="#E8F8F5")

    for x0, x1 in [(1.9, 2.2), (4.0, 4.3), (6.2, 6.5), (8.2, 8.5)]:
        ax.annotate(
            "",
            xy=(x1, 2.1),
            xytext=(x0, 2.1),
            arrowprops=dict(arrowstyle="->", lw=1.4, color="#444"),
        )
    ax.annotate(
        "",
        xy=(9.65, 1.85),
        xytext=(9.65, 2.35),
        arrowprops=dict(arrowstyle="->", lw=1.4, color="#444"),
    )
    ax.text(
        5.25,
        0.45,
        r"Loss: $J = R - \lambda_c C_{comm}$,   $C_{comm}=\sum g_{ij}$",
        ha="center",
        fontsize=9,
        color="#922B21",
    )
    ax.text(
        5.25,
        3.75,
        "MAPPO / GAT  →  DSGF v2  →  AC-DSGF",
        ha="center",
        fontsize=9,
        style="italic",
    )
    fig.savefig(OUT / "framework.png", dpi=180, bbox_inches="tight")
    fig.savefig(PAPER / "fig_ac_dsgf_framework.png", dpi=180, bbox_inches="tight")
    try:
        fig.savefig(OUT / "framework.pdf", bbox_inches="tight")
    except Exception:
        pass
    plt.close(fig)
    print("Saved framework")


def pareto():
    methods = [
        ("MAPPO", 2.40, 0.0),
        ("GAT", 2.04, 38.80),
        ("Transformer", 0.70, 240.0),
        ("DSGF", 4.01, 39.27),
        ("AC-DSGF", 3.95, 0.43),
    ]
    colors = {
        "MAPPO": "#888888",
        "GAT": "#DD8452",
        "Transformer": "#8172B3",
        "DSGF": "#4C72B0",
        "AC-DSGF": "#C44E52",
    }
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    for name, s, c in methods:
        cx = c if c > 0 else 0.05
        ax.scatter(
            cx, s, s=120, color=colors[name], zorder=3, edgecolors="k", linewidths=0.5
        )
        ax.annotate(name, (cx, s), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Communication cost (log scale)")
    ax.set_ylabel("Success (%)")
    ax.set_title("Communication–Performance Pareto (Table I, N=16)")
    ax.grid(True, alpha=0.35, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "fig_comm_pareto.png", dpi=160, bbox_inches="tight")
    fig.savefig(PAPER / "fig_comm_pareto.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Saved pareto")


def mirror():
    for name in ("fig_comm_budget16.png", "fig_comm_trigger.png"):
        src = PAPER / name
        if src.exists():
            shutil.copy(src, OUT / name)
            print("Copied", name)


if __name__ == "__main__":
    framework()
    pareto()
    mirror()

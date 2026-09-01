"""IEEE Paper Hardening figures: framework / degradation / Pareto / scale."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "ac_dsgf" / "figures"
PAPER = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def box(ax, x, y, w, h, text, fc="#F7F7F7", ec="#333", fs=9):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=fc, edgecolor=ec, lw=1.6,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def fig_framework():
    fig, ax = plt.subplots(figsize=(7.2, 8.2))
    ax.set_xlim(0, 7.2)
    ax.set_ylim(0, 8.2)
    ax.axis("off")
    ax.set_title("AC-DSGF Framework (three layers)", fontsize=13, pad=10)

    box(ax, 1.8, 7.0, 3.6, 0.85, "Observation  $o_i$", fc="#E8F1FA", fs=11)
    box(ax, 1.1, 4.55, 5.0, 2.0, "", fc="#FDEDEC", fs=10)
    ax.text(3.6, 6.25, "Adaptive Communication Layer", ha="center", fontsize=11, fontweight="bold")
    box(ax, 1.4, 5.35, 2.0, 0.7, "Gate  $g_{ij}$", fc="#FADBD8", fs=10)
    box(ax, 3.8, 5.35, 2.0, 0.7, "Budget Top-$K$", fc="#FCF3CF", fs=10)
    ax.text(3.6, 4.8, r"$A^{AC}=A\odot E^{K}$   $\rightarrow$   $e_{ij}=A^{AC} q_{ij} g_{ij}$",
            ha="center", fontsize=9)
    box(ax, 1.5, 2.7, 4.2, 1.35, "Residual DSGF Policy\nSparse Attn + GRU  $\\rightarrow\\Phi$\n"
        r"$a=\pi(o)+\beta\Delta(\Phi)$", fc="#E8F8F5", fs=10)
    box(ax, 2.2, 1.2, 2.8, 0.85, "Action  $a_i$", fc="#E8F1FA", fs=11)

    for y0, y1 in [(7.0, 6.55), (4.55, 4.05), (2.7, 2.05)]:
        ax.annotate("", xy=(3.6, y1), xytext=(3.6, y0),
                    arrowprops=dict(arrowstyle="->", lw=1.8, color="#444"))

    ax.text(3.6, 0.45,
            r"Joint objective:  $\max\,\mathbb{E}[\sum r_t]-\lambda_c\,\mathbb{E}[\sum C_t]$",
            ha="center", fontsize=10, color="#922B21")
    fig.savefig(OUT / "framework.png", dpi=200, bbox_inches="tight")
    fig.savefig(PAPER / "fig_ac_dsgf_framework.png", dpi=200, bbox_inches="tight")
    try:
        fig.savefig(OUT / "framework.pdf", bbox_inches="tight")
    except Exception:
        pass
    plt.close(fig)
    print("framework")


def fig_degradation():
    """Honest motivation: residual ablation (documented Table II), not invented GAT numbers."""
    labels = ["Full DSGF\n(residual)", "w/o Residual\n(guidance domination)"]
    vals = [9.27, 0.22]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    bars = ax.bar(labels, vals, color=["#4C72B0", "#C44E52"], width=0.55, edgecolor="k", linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.25, f"{v:.2f}%", ha="center", fontsize=11)
    ax.set_ylabel("Success (%)")
    ax.set_title("Long-horizon degradation of guidance-dominated MARL\n"
                 "(4-UAV residual ablation; ~42× collapse)")
    ax.set_ylim(0, 11)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_motivation_degradation.png", dpi=180, bbox_inches="tight")
    fig.savefig(PAPER / "fig_motivation_degradation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("degradation")


def fig_pareto():
    methods = [
        ("MAPPO", 2.40, 0.05),
        ("GAT", 2.04, 38.80),
        ("Transformer", 0.70, 240.0),
        ("DSGF", 4.01, 39.27),
        ("AC-DSGF", 3.95, 0.43),
    ]
    colors = {
        "MAPPO": "#888888", "GAT": "#DD8452", "Transformer": "#8172B3",
        "DSGF": "#4C72B0", "AC-DSGF": "#C44E52",
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for name, s, c in methods:
        ax.scatter(c, s, s=140 if name == "AC-DSGF" else 110,
                   color=colors[name], zorder=3, edgecolors="k", linewidths=0.6)
        ax.annotate(name, (c, s), textcoords="offset points",
                    xytext=(7, 5), fontsize=10,
                    fontweight="bold" if name == "AC-DSGF" else "normal")
    ax.set_xscale("log")
    ax.set_xlabel("Communication cost (log)")
    ax.set_ylabel("Success (%)")
    ax.set_title("Main Result: Communication–Performance Pareto ($N$=16)")
    ax.grid(True, which="both", alpha=0.35)
    # highlight region
    ax.axvspan(0.1, 1.0, color="#C44E52", alpha=0.08)
    fig.tight_layout()
    fig.savefig(OUT / "fig_comm_pareto.png", dpi=180, bbox_inches="tight")
    fig.savefig(PAPER / "fig_comm_pareto.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("pareto")


def fig_comm_scale():
    """Schematic + empirical DSGF scalability Comm vs theoretical dense."""
    Ns = np.array([4, 8, 16, 32])
    # Empirical DSGF from scalability runs (where available)
    dsgf_c = {8: 9.64, 16: 39.69, 32: 22.95}
    # AC at 16 from Table I; schematic linear-ish budgeted curve
    ac16 = 0.43
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    # theoretical dense (relative scale)
    dense = (Ns * (Ns - 1)) * 0.15  # scaled for plot readability
    ax.plot(Ns, dense, "--", color="#8172B3", lw=2, label="Dense / Transformer (schematic $O(N^2)$)")
    ax.plot(Ns, Ns * (Ns - 1) * 0.05, ":", color="#DD8452", lw=2, label="GAT radius (schematic)")
    xs, ys = zip(*sorted(dsgf_c.items()))
    ax.plot(xs, ys, "-o", color="#4C72B0", lw=2, label="DSGF (empirical)")
    ax.scatter([16], [ac16], s=160, color="#C44E52", zorder=5, label="AC-DSGF @16 (Table I)")
    ax.annotate("AC-DSGF", (16, ac16), textcoords="offset points", xytext=(8, -12),
                color="#C44E52", fontsize=10, fontweight="bold")
    ax.set_xlabel("Number of UAVs")
    ax.set_ylabel("Communication cost")
    ax.set_title("Communication vs. Team Size")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(OUT / "fig_comm_scale.png", dpi=180, bbox_inches="tight")
    fig.savefig(PAPER / "fig_comm_scale.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("comm-scale")


if __name__ == "__main__":
    fig_framework()
    fig_degradation()
    fig_pareto()
    fig_comm_scale()

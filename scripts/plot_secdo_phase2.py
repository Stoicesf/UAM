#!/usr/bin/env python
"""Generate Phase 2 paper figures from phase2_summary.json (+ series if present)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "ac_dsgf_v2" / "experiments" / "phase2_results" / "figures"


def _load():
    p = ROOT / "paper" / "ac_dsgf_v2" / "experiments" / "phase2_results" / "phase2_summary.json"
    return json.loads(p.read_text(encoding="utf-8"))


def fig_architecture():
    """Lightweight architecture schematic (boxes + arrows)."""
    fig, ax = plt.subplots(figsize=(9.2, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    boxes = [
        (0.3, 1.2, 1.8, 1.0, r"$s_t$", "#e8f1f8"),
        (2.4, 1.2, 1.8, 1.0, r"$z_t=\phi(s_t)$" "\n" r"$h_t=\mathrm{GRU}$", "#dceee6"),
        (4.5, 1.2, 2.0, 1.0, r"$\hat c_{t+1}=\mathcal{F}_\phi(h_t)$" "\n" r"$\hat{\mathcal{B}}_{t+1}$", "#f7e8d4"),
        (6.9, 1.2, 2.6, 1.0, r"$y_t=x_t-\eta\nabla F$" "\n" r"$x_{t+1}=\Pi_{\hat B}(y_t)$", "#f0d6d6"),
    ]
    for x, y, w, h, txt, c in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=c, edgecolor="#333", lw=1.2))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=9)
    for x0, x1 in [(2.1, 2.4), (4.2, 4.5), (6.5, 6.9)]:
        ax.annotate("", xy=(x1, 1.7), xytext=(x0, 1.7), arrowprops=dict(arrowstyle="->", color="#333"))
    ax.text(5, 2.7, "SECDO Algorithm 1", ha="center", fontsize=12, fontweight="bold")
    ax.text(
        5,
        0.45,
        r"Theory: $\delta$-accurate $\hat B$ $\Rightarrow$ anticipatory advantage when $\delta<\rho$",
        ha="center",
        fontsize=9,
        color="#444",
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig1_architecture.png", dpi=160)
    plt.close(fig)


def fig_viol_vs_time(rows):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for method, color in [("secdo", "#c0392b"), ("reactive", "#2980b9"), ("oracle", "#27ae60")]:
        r = next((x for x in rows if x["regime"] == "fast" and x["method"] == method), None)
        if r is None or "series" not in r:
            continue
        v = np.array(r["series"]["viol"], dtype=float)
        ax.plot(v, label=method, color=color, lw=1.6)
    ax.set_xlabel("t")
    ax.set_ylabel("constraint violation")
    ax.set_title("E2 fast: violation vs time")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_violation_vs_time.png", dpi=160)
    plt.close(fig)


def fig_gap_vs_pred(rows):
    scatter = [
        (r["mean_eps_plus_delta"], r["mean_gap"], r["regime"])
        for r in rows
        if r["method"] == "secdo"
    ]
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    xs = [s[0] for s in scatter]
    ys = [s[1] for s in scatter]
    labs = [s[2] for s in scatter]
    ax.scatter(xs, ys, s=70, c="#c0392b", zorder=3)
    for x, y, lab in scatter:
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
    if len(xs) >= 2:
        coef = np.polyfit(xs, ys, 1)
        xx = np.linspace(min(xs), max(xs), 50)
        ax.plot(xx, np.polyval(coef, xx), "--", color="#666", label=r"trend $\propto \epsilon+\delta$")
        ax.legend(frameon=False)
    ax.set_xlabel(r"$\frac{1}{T}\sum(\epsilon_t+\delta_t)$")
    ax.set_ylabel(r"Gap$(T)$")
    ax.set_title("Theory prediction plot")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_gap_vs_prediction_error.png", dpi=160)
    plt.close(fig)


def fig_delta_vs_rho(rows):
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    r = next((x for x in rows if x["regime"] == "fast" and x["method"] == "secdo"), None)
    if r is not None and "series" in r:
        d = np.array(r["series"]["delta"], dtype=float)
        rho = np.array(r["series"]["rho"], dtype=float)
        ax.scatter(rho, d, s=18, alpha=0.65, c="#8e44ad", label="SECDO (fast)")
        lim = max(float(rho.max()), float(d.max()), 1e-6) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=1, label=r"$\delta=\rho$")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
    else:
        # fallback: summary means
        for row in rows:
            if row["method"] != "secdo":
                continue
            ax.scatter(row["mean_rho"], row["mean_delta"], s=80, label=row["regime"])
        lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, lim], [0, lim], "k--", lw=1, label=r"$\delta=\rho$")
    ax.set_xlabel(r"$\rho_t$ (drift)")
    ax.set_ylabel(r"$\delta_t$ (pred. error)")
    ax.set_title(r"Certificate region: below diagonal $\delta<\rho$")
    ax.legend(frameon=False, fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_delta_vs_rho.png", dpi=160)
    plt.close(fig)


def fig_ablation(rows):
    keys = [
        ("secdo", "SECDO"),
        ("reactive", "Abl-C\n" r"$\Pi_B$"),
        ("no_latent", "Abl-A\nno latent"),
        ("no_constraint_dyn", "Abl-B\nno $\mathcal{F}$"),
        ("oracle", "Oracle"),
    ]
    fast = {r["method"]: r for r in rows if r["regime"] == "fast"}
    labels, gaps, viols = [], [], []
    for k, lab in keys:
        if k not in fast:
            continue
        labels.append(lab)
        gaps.append(fast[k]["mean_gap"])
        viols.append(fast[k]["mean_viol"])
    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.bar(x - w / 2, gaps, w, label="Gap", color="#34495e")
    ax.bar(x + w / 2, viols, w, label="Violation", color="#e67e22")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("mean metric")
    ax.set_title("Theory-term ablations (E2 fast)")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_ablation.png", dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = _load()
    fig_architecture()
    fig_viol_vs_time(rows)
    fig_gap_vs_pred(rows)
    fig_delta_vs_rho(rows)
    fig_ablation(rows)
    print("Wrote figures to", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

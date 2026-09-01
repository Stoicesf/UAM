"""Paper figures from results.json (PDF + PNG)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_results(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fig_architecture(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.2, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    boxes = [
        (0.3, 1.2, 1.8, 1.0, r"$s_t$", "#e8f1f8"),
        (2.4, 1.2, 1.8, 1.0, r"$z_t,h_t$", "#dceee6"),
        (4.5, 1.2, 2.0, 1.0, r"$\hat c_{t+1}$" "\n" r"$\hat B$", "#f7e8d4"),
        (6.9, 1.2, 2.6, 1.0, r"$x_{t+1}=\Pi_{\hat B}(y_t)$", "#f0d6d6"),
    ]
    for x, y, w, h, txt, c in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=c, edgecolor="#333", lw=1.2))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=10)
    for x0, x1 in [(2.1, 2.4), (4.2, 4.5), (6.5, 6.9)]:
        ax.annotate("", xy=(x1, 1.7), xytext=(x0, 1.7), arrowprops=dict(arrowstyle="->", color="#333"))
    ax.text(5, 2.7, "SECDO Algorithm 1", ha="center", fontsize=12, fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig1_architecture.{ext}", dpi=160)
    plt.close(fig)
    return out_dir / "fig1_architecture.pdf"


def fig_violation(rows: list[dict], out_dir: Path, regime: str = "fast") -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    colors = {"secdo": "#c0392b", "reactive": "#2980b9", "oracle": "#27ae60", "dsgf": "#7f8c8d", "ac_dsgf": "#8e44ad"}
    plotted = False
    for r in rows:
        if r.get("regime") != regime or "series" not in r:
            continue
        ax.plot(r["series"]["viol"], label=r["method"], color=colors.get(r["method"], "#333"), lw=1.5)
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlabel("t")
    ax.set_ylabel("violation")
    ax.set_title(f"{regime}: violation vs time")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig2_violation.{ext}", dpi=160)
    plt.close(fig)
    return out_dir / "fig2_violation.pdf"


def fig_delta_rho(rows: list[dict], out_dir: Path, regime: str = "fast") -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    r = next((x for x in rows if x.get("regime") == regime and x.get("method") == "secdo" and "series" in x), None)
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    if r is None:
        for row in rows:
            if row.get("method") != "secdo":
                continue
            ax.scatter(row["rho"], row["delta"], s=80, label=row.get("regime", ""))
        lim = max(ax.get_xlim()[1], ax.get_ylim()[1], 1e-3)
        ax.plot([0, lim], [0, lim], "k--", lw=1)
    else:
        d = np.array(r["series"]["delta"])
        rho = np.array(r["series"]["rho"])
        ax.scatter(rho, d, s=16, alpha=0.65, c="#8e44ad")
        lim = max(float(rho.max()), float(d.max()), 1e-6) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=1, label=r"$\delta=\rho$")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
    ax.set_xlabel(r"$\chi_t$ / $\rho_t$")
    ax.set_ylabel(r"$\delta_t$")
    ax.set_title(r"Certificate: $\delta<\chi$ below diagonal")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.25)
    try:
        ax.set_aspect("equal", adjustable="datalim")
    except Exception:
        pass
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig3_delta_rho.{ext}", dpi=140)
    plt.close(fig)
    return out_dir / "fig3_delta_rho.pdf"


def fig_gap_error(rows: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    pts = [
        (r["mean_eps_plus_delta"], r["gap"], r.get("regime", ""))
        for r in rows
        if r.get("method") == "secdo" and "mean_eps_plus_delta" in r
    ]
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    if pts:
        xs, ys, labs = zip(*pts)
        ax.scatter(xs, ys, s=70, c="#c0392b")
        for x, y, lab in pts:
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
        if len(xs) >= 2:
            coef = np.polyfit(xs, ys, 1)
            xx = np.linspace(min(xs), max(xs), 50)
            ax.plot(xx, np.polyval(coef, xx), "--", color="#666")
    ax.set_xlabel(r"avg$(\epsilon+\delta)$")
    ax.set_ylabel(r"Gap$(T)$")
    ax.set_title("Theory prediction plot")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig4_gap_error.{ext}", dpi=160)
    plt.close(fig)
    return out_dir / "fig4_gap_error.pdf"


def fig_ablation(rows: list[dict], out_dir: Path, regime: str = "fast") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    order = ["secdo", "reactive", "oracle", "dsgf", "ac_dsgf"]
    fast = {r["method"]: r for r in rows if r.get("regime") == regime}
    labels, gaps, viols = [], [], []
    for k in order:
        if k not in fast:
            continue
        labels.append(k)
        gaps.append(fast[k]["gap"])
        viols.append(fast[k]["violation"])
    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.bar(x - w / 2, gaps, w, label="Gap", color="#34495e")
    ax.bar(x + w / 2, viols, w, label="Violation", color="#e67e22")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean metric")
    ax.set_title(f"Baselines ({regime})")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig5_ablation.{ext}", dpi=160)
    plt.close(fig)
    return out_dir / "fig5_ablation.pdf"


def render_all(results_path: str | Path, out_dir: str | Path) -> list[str]:
    rows = load_results(results_path)
    out = Path(out_dir)
    paths = [
        fig_architecture(out),
        fig_violation(rows, out),
        fig_delta_rho(rows, out),
        fig_gap_error(rows, out),
        fig_ablation(rows, out),
    ]
    return [str(p) for p in paths if p is not None]

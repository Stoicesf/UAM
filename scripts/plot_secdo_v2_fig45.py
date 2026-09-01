"""Paper Fig.4 (synthetic + theory bound) and Fig.5 (UAV comparison).

Outputs → results/secdo_v2/paper_figures/
         paper/ac_dsgf_v2/experiments/phase2_results/figures/
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "secdo_v2" / "paper_figures"
PAPER = ROOT / "paper" / "ac_dsgf_v2" / "experiments" / "phase2_results" / "figures"
UAV = ROOT / "results" / "secdo_v2" / "uav"

C_SECDO = "#c0392b"
C_BOUND = "#1f4e79"
C_METHODS = {
    "secdo": "#c0392b",
    "reactive": "#2980b9",
    "oracle": "#27ae60",
    "dsgf": "#7f8c8d",
    "ac_dsgf": "#8e44ad",
}


def _save(fig: plt.Figure, stem: str) -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext, dpi in (("pdf", None), ("png", 300)):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        shutil.copy2(p, PAPER / p.name)
        paths.append(p)
    plt.close(fig)
    return paths


def run_synthetic_T_sweep(horizons=(40, 80, 120, 160, 200), noise=0.05, seed=0) -> list[dict]:
    from secdo.experiments.synthetic_convex.env import SynthConfig, SyntheticConvexEnv
    from secdo.experiments.synthetic_convex.run import _run_method

    rows = []
    for T in horizons:
        cfg = SynthConfig(horizon=int(T), drift_amp=0.2, noise_sigma=noise, seed=seed)
        env = SyntheticConvexEnv(cfg)
        row = _run_method(env, "secdo", pred_noise=noise)
        row["T"] = float(T)
        row["noise"] = noise
        # theory proxy (Thm.2 shape): a√(T(1+P_T)) + b·T·(mean δ)  [ε≈0 in this synth]
        P = float(row["P_T"])
        d = float(row["delta"])
        row["proxy_sqrt"] = float(np.sqrt(T * (1.0 + P)))
        row["proxy_pred"] = float(T * d)
        rows.append(row)
        print(
            f"[synth T={T}] Reg={row['Reg_T']:.4f} P_T={P:.3f} "
            f"sqrt={row['proxy_sqrt']:.3f} delta={d:.4f}",
            flush=True,
        )
    return rows


def fit_upper_bound(rows: list[dict]) -> tuple[float, float]:
    """Minimal a,b≥0 s.t. Reg ≤ a·√(T(1+P))+b·Tδ for all points (LP-ish grid)."""
    R = np.array([r["Reg_T"] for r in rows])
    S = np.array([r["proxy_sqrt"] for r in rows])
    D = np.array([r["proxy_pred"] for r in rows])
    best = (1e9, 0.0, 0.0)
    for a in np.linspace(0.01, 2.0, 80):
        # need a*S + b*D >= R ⇒ b >= max_i (R_i - a*S_i)/D_i for D_i>0
        need = (R - a * S) / np.maximum(D, 1e-8)
        b = float(max(0.0, need.max()))
        score = float((a * S + b * D).sum())
        if score < best[0] and np.all(a * S + b * D + 1e-9 >= R):
            best = (score, a, b)
    if best[0] >= 1e9:
        # fallback: scale to cover max ratio
        a = float(np.max(R / np.maximum(S, 1e-8)) * 1.05)
        b = 0.0
        return a, b
    return best[1], best[2]


def fig4_synthetic(rows: list[dict]) -> list[Path]:
    a, b = fit_upper_bound(rows)
    T = np.array([r["T"] for r in rows])
    R = np.array([r["Reg_T"] for r in rows])
    S = np.array([r["proxy_sqrt"] for r in rows])
    D = np.array([r["proxy_pred"] for r in rows])
    bound = a * S + b * D

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.plot(T, R, "o-", color=C_SECDO, lw=2.0, ms=7, label=r"SECDO $\mathrm{Reg}_T$ (empirical)")
    ax.plot(
        T,
        bound,
        "s--",
        color=C_BOUND,
        lw=1.8,
        ms=6,
        label=rf"Thm.2 proxy $a\sqrt{{T(1+P_T)}}+b\sum\delta$ "
        + f"\n$(a={a:.3f},\\,b={b:.3f})$",
    )
    ax.set_xlabel(r"horizon $T$")
    ax.set_ylabel(r"cumulative dynamic regret $\mathrm{Reg}_T$")
    ax.set_title("Theory validation: regret scales with $\\sqrt{T(1+P_T)}$ (synthetic)")
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()

    meta = {
        "a": a,
        "b": b,
        "rows": [
            {
                "T": r["T"],
                "Reg_T": r["Reg_T"],
                "P_T": r["P_T"],
                "delta": r["delta"],
                "bound": float(bd),
            }
            for r, bd in zip(rows, bound)
        ],
        "note": "a,b fitted as minimal nonnegative upper envelope (not claimed universal constants).",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fig4_theory_bound_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return _save(fig, "fig4_synthetic_regret_theory")


def fig5_uav() -> list[Path]:
    methods = ["dsgf", "ac_dsgf", "reactive", "oracle", "secdo"]
    regimes = ["slow_drift", "fast_drift", "stress_test"]
    labels = ["Slow", "Fast", "Stress"]
    # load summaries
    data = {m: {"gap": [], "viol": []} for m in methods}
    for reg in regimes:
        rows = json.loads((UAV / reg / "summary.json").read_text(encoding="utf-8"))
        by_m = {r["method"]: r for r in rows}
        for m in methods:
            data[m]["gap"].append(float(by_m[m]["gap"]))
            data[m]["viol"].append(float(by_m[m]["violation"]))

    x = np.arange(len(labels))
    width = 0.15
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))

    for i, m in enumerate(methods):
        axes[0].bar(
            x + (i - 2) * width,
            data[m]["viol"],
            width,
            label=m,
            color=C_METHODS[m],
            edgecolor="#333",
            linewidth=0.4,
        )
        axes[1].bar(
            x + (i - 2) * width,
            data[m]["gap"],
            width,
            label=m,
            color=C_METHODS[m],
            edgecolor="#333",
            linewidth=0.4,
        )

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("mean violation")
    axes[0].set_title("Constraint violation")
    axes[0].set_yscale("log")
    axes[0].grid(True, axis="y", alpha=0.28, which="both")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("mean optimality gap")
    axes[1].set_title("Allocation gap (utility proxy)")
    axes[1].set_yscale("log")
    axes[1].grid(True, axis="y", alpha=0.28, which="both")

    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper center", ncol=5, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("UAV system evaluation (5 seeds; demonstrate effectiveness)", y=1.08, fontsize=11)
    fig.tight_layout()
    return _save(fig, "fig5_uav_comparison")


def fig_ablation_bars() -> list[Path]:
    """Optional appendix figure from ablation summary."""
    p = ROOT / "results" / "secdo_v2" / "ablation" / "summary.json"
    if not p.is_file():
        return []
    s = json.loads(p.read_text(encoding="utf-8"))
    uav = s["uav_fast_drift"]
    crash = s["crash_recovery"]
    order = ["full_secdo", "A1_reactive", "A2_no_anticipatory", "A3_fixed_alpha1"]
    labels = ["Full", "A1", "A2", "A3"]
    reg = [uav[k]["Reg_T_mean"] for k in order]
    rise = [crash[k]["cum_window_rise"] for k in order]

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    axes[0].bar(labels, reg, color=[C_SECDO, "#2980b9", "#2980b9", "#7f8c8d"])
    axes[0].set_ylabel(r"Fast $\mathrm{Reg}_T$")
    axes[0].set_title("Ablation: anticipation → lower regret")
    axes[0].grid(True, axis="y", alpha=0.28)

    axes[1].bar(labels, rise, color=[C_SECDO, "#2980b9", "#2980b9", "#7f8c8d"])
    axes[1].set_ylabel("Crash window cum-violation rise")
    axes[1].set_title(r"Ablation: adaptive $\alpha$ → Cor.4")
    axes[1].grid(True, axis="y", alpha=0.28)
    fig.tight_layout()
    return _save(fig, "figE_ablation")


def main() -> int:
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 10, "axes.labelsize": 10})
    print("Fig.4 synthetic T-sweep…", flush=True)
    rows = run_synthetic_T_sweep()
    (OUT).mkdir(parents=True, exist_ok=True)
    (OUT / "fig4_synth_rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    p4 = fig4_synthetic(rows)
    print("Fig.5 UAV…", flush=True)
    p5 = fig5_uav()
    print("Fig.E ablation…", flush=True)
    pe = fig_ablation_bars()
    print("Wrote", p4, p5, pe, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

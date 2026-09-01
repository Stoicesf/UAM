"""SECDO v2 paper evidence figures (Thm2 / Thm3 / Cor4) from landed results.

Outputs (PDF + PNG @ 300 dpi):
  results/secdo_v2/paper_figures/fig1_thm2_drift_regret.*
  results/secdo_v2/paper_figures/fig2_thm3_pi_alpha.*
  results/secdo_v2/paper_figures/fig3_cor4_graceful_degradation.*
  paper/ac_dsgf_v2/experiments/phase2_results/figures/ (copies)
  results/secdo_v2/EVIDENCE_CHAIN.json
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

# Venue-safe palette (no purple glow / cream serif kit)
C_REGRET = "#1f4e79"
C_CHI = "#c0392b"
C_ALPHA = "#1f77b4"
C_PI = "#c0392b"
C_SECDO = "#c0392b"
C_FIXED = "#7f8c8d"
C_SHADE = "#bdc3c7"


def _save(fig: plt.Figure, stem: str) -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext, dpi in (("pdf", None), ("png", 300)):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        paths.append(p)
        shutil.copy2(p, PAPER / p.name)
    plt.close(fig)
    return paths


def _secdo_rows(regime: str) -> list[dict]:
    rows = json.loads((UAV / regime / "results.json").read_text(encoding="utf-8"))
    return [r for r in rows if r["method"] == "secdo"]


def collect_uav_evidence() -> dict:
    out = {}
    for regime in ("slow_drift", "fast_drift", "stress_test"):
        rows = _secdo_rows(regime)
        reg_t = np.array([float(np.sum(r["series"]["gap"])) for r in rows])
        viol = np.array([float(r["violation"]) for r in rows])
        chi = np.array([float(r["v2_chi"]) for r in rows])
        pi = np.array([float(r["v2_PI"]) for r in rows])
        alpha = np.array([float(r["v2_mean_alpha"]) for r in rows])
        out[regime] = {
            "n_seeds": len(rows),
            "Reg_T_mean": float(reg_t.mean()),
            "Reg_T_std": float(reg_t.std()),
            "violation_mean": float(viol.mean()),
            "violation_std": float(viol.std()),
            "chi_mean": float(chi.mean()),
            "chi_std": float(chi.std()),
            "PI_mean": float(pi.mean()),
            "mean_alpha": float(alpha.mean()),
        }
    slow, fast = out["slow_drift"]["Reg_T_mean"], out["fast_drift"]["Reg_T_mean"]
    out["slow_to_fast"] = {
        "regret_ratio": float(fast / max(slow, 1e-12)),
        "regret_rel_increase": float((fast - slow) / max(slow, 1e-12)),
        "chi_ratio": float(out["fast_drift"]["chi_mean"] / max(out["slow_drift"]["chi_mean"], 1e-12)),
    }
    return out


def fig1_thm2(evidence: dict) -> list[Path]:
    regimes = ["slow_drift", "fast_drift", "stress_test"]
    labels = ["Slow", "Fast", "Stress"]
    reg = np.array([evidence[r]["Reg_T_mean"] for r in regimes])
    reg_err = np.array([evidence[r]["Reg_T_std"] for r in regimes])
    chi = np.array([evidence[r]["chi_mean"] for r in regimes])

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(labels))
    w = 0.55
    bars = ax.bar(
        x,
        reg,
        w,
        yerr=reg_err,
        color=C_REGRET,
        edgecolor="#0d2b45",
        linewidth=0.8,
        capsize=4,
        error_kw=dict(ecolor="#333", lw=1.0),
        label=r"Dynamic regret $\mathrm{Reg}_T=\sum_t g_t$",
        zorder=3,
    )
    ax.set_ylabel(r"Dynamic regret $\mathrm{Reg}_T$ (log)", color=C_REGRET)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="y", labelcolor=C_REGRET)
    ax.set_yscale("log")
    ax.set_ylim(max(reg.min() * 0.35, 1e-5), max(reg + reg_err) * 2.2)
    ax.grid(True, axis="y", alpha=0.28, zorder=0, which="both")

    ax2 = ax.twinx()
    ax2.plot(x, chi, "o-", color=C_CHI, lw=2.0, ms=7, label=r"Mean drift $\bar\chi$")
    ax2.set_ylabel(r"Mean environmental drift $\bar\chi$", color=C_CHI)
    ax2.tick_params(axis="y", labelcolor=C_CHI)
    ax2.set_ylim(0, max(chi) * 1.35)

    ratio = evidence["slow_to_fast"]["regret_ratio"]
    ax.annotate(
        "",
        xy=(1.0, reg[1]),
        xytext=(0.0, reg[0] + reg_err[0]),
        arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.4),
    )
    ax.text(
        0.48,
        reg[1] * 0.55,
        rf"$\times${ratio:.1f} regret" "\n" r"as $\chi$ increases",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#2c3e50",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#95a5a6", alpha=0.92),
    )

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False, fontsize=9)
    ax.set_title("Thm.2: dynamic regret scales with environmental drift")
    fig.tight_layout()
    return _save(fig, "fig1_thm2_drift_regret")


def _run_protocol_series(method: str = "secdo", T1: int = 40, T2: int = 55, T: int = 100, noise: float = 0.05):
    from secdo.experiments.synthetic_convex.env import SynthConfig, SyntheticConvexEnv
    from secdo.experiments.synthetic_convex.run import _peek_next_c
    from secdo.optimizer.projected_gradient import gradient_step
    from secdo.optimizer.reactive_projection import reactive_project
    from secdo.optimizer.secdo_optimizer import SECDOOptimizer
    from secdo.evaluation.violation import ViolationTracker

    cfg = SynthConfig(horizon=T, drift_amp=0.2, seed=2)
    env = SyntheticConvexEnv(cfg)
    obs = env.reset(batch=32)
    pref = obs["pref"]
    x = reactive_project(pref.clone(), obs["c_teacher"])
    opt = SECDOOptimizer(eta=0.2)
    vtr = ViolationTracker()
    alpha_s, pi_s, cum = [], [], []
    while not env.done:
        c_t = obs["c_teacher"]
        c_true_next = _peek_next_c(env)
        c_hat = c_true_next + noise * torch.randn_like(c_true_next)
        if T1 <= env.t < T2:
            c_hat = c_hat * 4.0
        chi = (c_true_next - c_t).abs().clamp(min=1e-2)
        delta_hat = (c_hat - c_true_next).abs()
        if method == "reactive":
            x = reactive_project(gradient_step(x, pref, 0.2), c_t)
            alpha = torch.zeros_like(c_t)
            pi = delta_hat / chi
        elif method == "fixed_alpha":
            x, alpha, pi = opt.step(
                x, None, c_hat, c_t, delta_hat=delta_hat, chi_hat=chi, pref=pref, fixed_alpha=1.0
            )
        else:
            x, alpha, pi = opt.step(
                x, None, c_hat, c_t, delta_hat=delta_hat, chi_hat=chi, pref=pref
            )
        obs = env.step()
        c_next = obs["c_teacher"]
        pref = obs["pref"]
        delta = float((c_hat - c_next).abs().mean())
        chi_v = float((c_next - c_t).abs().mean())
        vtr.update(x, c_next, delta=delta, chi=chi_v, PI=float(pi.mean()), alpha=float(alpha.mean()))
        alpha_s.append(float(alpha.mean()))
        pi_s.append(float(pi.mean()))
        cum.append(sum(vtr.viol))
    summary = vtr.summary()
    return {
        "method": method,
        **summary,
        "alpha": alpha_s,
        "PI": pi_s,  # series (must not be overwritten by summary scalar PI)
        "mean_PI": float(summary.get("PI", np.mean(pi_s))),
        "cum_violation": cum,
        "corrupt": (T1, T2),
    }


def fig2_thm3(series: dict) -> list[Path]:
    alpha = np.asarray(series["alpha"], dtype=float)
    pi = np.asarray(series["PI"], dtype=float)
    t = np.arange(len(alpha))
    T1, T2 = series["corrupt"]

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    # shade contiguous PI > 1 (conservative regime)
    above = np.asarray(pi > 1.0, dtype=bool)
    labeled_pi = False
    i = 0
    n = len(above)
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j < n and above[j]:
            j += 1
        ax.axvspan(
            i - 0.5,
            j - 0.5,
            color=C_SHADE,
            alpha=0.40,
            zorder=0,
            label=(r"$PI_t>1$ (conservative)" if not labeled_pi else None),
        )
        labeled_pi = True
        i = j

    ax.axvspan(T1, T2, color="#f5b7b1", alpha=0.28, zorder=0, label="predictor corrupt")
    ax.plot(t, alpha, color=C_ALPHA, lw=2.0, label=r"$\alpha_t=1/(1+PI_t^2)$")
    ax.set_ylabel(r"Anticipation weight $\alpha_t$", color=C_ALPHA)
    ax.set_ylim(-0.05, 1.08)
    ax.tick_params(axis="y", labelcolor=C_ALPHA)
    ax.set_xlabel("time step $t$")
    ax.grid(True, alpha=0.28)

    ax2 = ax.twinx()
    ax2.plot(t, np.clip(pi, 0, None), color=C_PI, lw=1.6, ls="--", label=r"$PI_t=\hat\delta_t/\hat\chi_t$")
    ax2.axhline(1.0, color="#7f8c8d", ls=":", lw=1.0)
    ax2.set_ylabel(r"Predictability index $PI_t$ (log)", color=C_PI)
    ax2.tick_params(axis="y", labelcolor=C_PI)
    ax2.set_yscale("log")
    ax2.set_ylim(max(1e-2, float(np.min(pi[pi > 0])) * 0.8), max(float(np.max(pi)) * 1.2, 2.0))

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", frameon=False, fontsize=8.5)
    ax.set_title("Thm.3 / PAP: continuous PI-conditioned interpolation (not hard switching)")
    fig.tight_layout()
    return _save(fig, "fig2_thm3_pi_alpha")


def fig3_cor4(crash: dict[str, dict]) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    sec = crash["secdo_series"]
    # fixed_alpha from instrumented run
    fixed = _run_protocol_series("fixed_alpha")
    t = np.arange(len(sec["cum_violation"]))
    T1, T2 = sec["corrupt"]

    ax.plot(t, sec["cum_violation"], color=C_SECDO, lw=2.2, label="SECDO (adaptive $\\alpha$)")
    ax.plot(
        t,
        fixed["cum_violation"],
        color=C_FIXED,
        lw=1.8,
        ls="--",
        label=r"Fixed predictor ($\alpha\equiv 1$)",
    )
    ax.axvspan(T1, T2, color="#f5b7b1", alpha=0.4, label="predictor corrupt")
    # annotate steepest rise of fixed vs flat secdo in window
    mid = (T1 + T2) // 2
    ax.annotate(
        "fixed predictor rises\n(no fallback)",
        xy=(mid, fixed["cum_violation"][mid]),
        xytext=(mid - 22, max(fixed["cum_violation"]) * 0.72),
        fontsize=8.5,
        color=C_FIXED,
        arrowprops=dict(arrowstyle="->", color=C_FIXED, lw=1.0),
    )
    ax.annotate(
        "SECDO flattens\n(Cor.4 rollback)",
        xy=(mid + 2, sec["cum_violation"][mid]),
        xytext=(mid + 12, max(sec["cum_violation"]) * 0.45),
        fontsize=8.5,
        color=C_SECDO,
        arrowprops=dict(arrowstyle="->", color=C_SECDO, lw=1.0),
    )
    ax.set_xlabel("time step $t$")
    ax.set_ylabel("cumulative constraint violation")
    ax.set_title("Cor.4: graceful degradation under prediction crash")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.28)
    fig.tight_layout()
    # persist series for reproducibility
    dump = {
        "secdo": {
            "cum_violation": sec["cum_violation"],
            "sum_violation": sec.get("sum_violation"),
            "mean_alpha": sec.get("mean_alpha"),
            "PI": sec.get("PI") if isinstance(sec.get("PI"), float) else float(np.mean(sec["PI"])),
        },
        "fixed_alpha": {
            "cum_violation": fixed["cum_violation"],
            "sum_violation": fixed.get("sum_violation"),
            "mean_alpha": fixed.get("mean_alpha"),
            "PI": float(np.mean(fixed["PI"])),
        },
    }
    (OUT / "fig3_crash_series.json").write_text(json.dumps(dump, indent=2), encoding="utf-8")
    cr_dir = ROOT / "results" / "secdo" / "crash_recovery"
    cr_dir.mkdir(parents=True, exist_ok=True)
    (cr_dir / "results_with_series.json").write_text(json.dumps(dump, indent=2), encoding="utf-8")
    return _save(fig, "fig3_cor4_graceful_degradation")


def main() -> int:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 120,
        }
    )
    print("Collecting UAV evidence…", flush=True)
    evidence = collect_uav_evidence()
    print(json.dumps({k: evidence[k] for k in evidence}, indent=2), flush=True)

    print("Figure 1…", flush=True)
    p1 = fig1_thm2(evidence)

    print("Crash protocol series for Fig 2/3…", flush=True)
    sec_series = _run_protocol_series("secdo")
    crash = {"secdo_series": sec_series}

    print("Figure 2…", flush=True)
    p2 = fig2_thm3(sec_series)

    print("Figure 3…", flush=True)
    p3 = fig3_cor4(crash)

    chain = {
        "note": (
            "Narrative throughput/regret scalars (e.g. 28.65 / 95.2 / +64%) are NOT in "
            "MASTER_SUMMARY; figures use landed UAV gap-series Reg_T and Cor.4 crash protocol."
        ),
        "uav_secdo": evidence,
        "suggested_results_sentence": (
            f"As predicted by Theorem 2, SECDO's cumulative dynamic regret rises from "
            f"{evidence['slow_drift']['Reg_T_mean']:.4f} (slow drift) to "
            f"{evidence['fast_drift']['Reg_T_mean']:.4f} (fast drift) — a "
            f"{evidence['slow_to_fast']['regret_ratio']:.1f}× increase as mean χ grows "
            f"{evidence['slow_to_fast']['chi_ratio']:.1f}× — while mean violation stays "
            f"below {100 * evidence['fast_drift']['violation_mean']:.2f}% . "
            f"Under stress, regret reaches {evidence['stress_test']['Reg_T_mean']:.4f} with "
            f"violation {100 * evidence['stress_test']['violation_mean']:.2f}% without collapse; "
            f"Cor.4 crash tests show PI-adaptive α rollback vs fixed α≡1."
        ),
        "figures": {
            "fig1": [str(p) for p in p1],
            "fig2": [str(p) for p in p2],
            "fig3": [str(p) for p in p3],
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "secdo_v2" / "EVIDENCE_CHAIN.json").write_text(
        json.dumps(chain, indent=2), encoding="utf-8"
    )
    print("Wrote", OUT, flush=True)
    print(chain["suggested_results_sentence"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

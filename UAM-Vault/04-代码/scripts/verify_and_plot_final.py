"""Verify landed SECDO v2 evidence chain and print LaTeX table rows.

Compatible with results/secdo_v2/EVIDENCE_CHAIN.json (key: uav_secdo).
Does NOT assume legacy keys final_train / tasks / regime_metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results" / "secdo_v2" / "EVIDENCE_CHAIN.json"
FIG_DIR = ROOT / "results" / "secdo_v2" / "paper_figures"


def main() -> int:
    data = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    uav = data["uav_secdo"]
    metrics = {
        "Slow": {
            "regret": uav["slow_drift"]["Reg_T_mean"],
            "violation": uav["slow_drift"]["violation_mean"],
            "drift": uav["slow_drift"]["chi_mean"],
        },
        "Fast": {
            "regret": uav["fast_drift"]["Reg_T_mean"],
            "violation": uav["fast_drift"]["violation_mean"],
            "drift": uav["fast_drift"]["chi_mean"],
        },
        "Stress": {
            "regret": uav["stress_test"]["Reg_T_mean"],
            "violation": uav["stress_test"]["violation_mean"],
            "drift": uav["stress_test"]["chi_mean"],
        },
    }
    ratio = uav["slow_to_fast"]

    print("=== Evidence lock ===")
    print(f"chi Slow→Fast: ×{ratio['chi_ratio']:.2f}")
    print(f"Reg_T Slow→Fast: ×{ratio['regret_ratio']:.2f}")
    for name, m in metrics.items():
        print(
            f"{name}: chi={m['drift']:.4f}  Reg_T={m['regret']:.6f}  "
            f"viol={100*m['violation']:.2f}%"
        )

    required = [
        "fig1_thm2_drift_regret.pdf",
        "fig2_thm3_pi_alpha.pdf",
        "fig3_cor4_graceful_degradation.pdf",
    ]
    missing = [f for f in required if not (FIG_DIR / f).is_file()]
    if missing:
        print("MISSING figures:", missing)
        print("Run: python scripts/plot_secdo_v2_evidence_figures.py")
        return 1
    print("Figures OK:", ", ".join(required))

    # Lightweight consistency redraw (does not replace venue figures)
    regimes = list(metrics.keys())
    regrets = [metrics[r]["regret"] for r in regimes]
    drifts = [metrics[r]["drift"] for r in regimes]
    fig, ax1 = plt.subplots(figsize=(6.5, 3.8))
    ax1.bar(regimes, regrets, color="#1f4e79", alpha=0.85, label="Dynamic Regret")
    ax1.set_yscale("log")
    ax1.set_ylabel(r"$\mathrm{Reg}_T$ (log)")
    ax2 = ax1.twinx()
    ax2.plot(regimes, drifts, "o--", color="#c0392b", label=r"$\bar\chi$")
    ax2.set_ylabel(r"$\bar\chi$")
    ax1.set_title("Verify: Regret vs Drift (from EVIDENCE_CHAIN)")
    fig.tight_layout()
    out = FIG_DIR / "fig1_verify_from_evidence.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out)

    print("\n=== Final Table Data for Paper (LaTeX) ===")
    for name in regimes:
        m = metrics[name]
        print(
            f"{name} & {m['drift']:.3f} & {m['regret']:.2e} & "
            f"{100*m['violation']:.2f}\\% \\\\"
        )
    print("\nDraft prose:", data.get("suggested_results_sentence", "")[:200], "...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

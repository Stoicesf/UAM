# -*- coding: utf-8 -*-
"""Finish formal §6.3 from saved AB CSVs (after GBK print crash); optional re-agg only."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.collect_tro_theory_evidence import (  # noqa: E402
    FIG,
    OUT,
    aggregate_AB,
    plot_fig1,
    plot_fig2,
)

RUN = ROOT / "runs" / "tro_evidence_6_3_formal_20260719_121020"


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ab = _read(RUN / "phaseAB_per_seed.csv")
    for r in ab:
        for k in (
            "K",
            "seed",
            "mean_D_G",
            "mean_epsilon_G",
            "mean_delta_a",
            "mean_Delta_A_gamma",
            "mean_closed_loop_reward",
        ):
            if k in r and r[k] != "":
                r[k] = float(r[k]) if k != "K" and k != "seed" else int(float(r[k]))

    points_raw = _read(RUN / "phaseA_points.csv")
    points = []
    for p in points_raw:
        points.append(
            {
                "seed": int(float(p["seed"])),
                "K": int(float(p["K"])),
                "episode": int(float(p["episode"])),
                "D_G": float(p["D_G"]),
                "epsilon_G": float(p["epsilon_G"]),
                "delta_a": float(p["delta_a"]) if p.get("delta_a") not in ("", None) else None,
                "t": int(float(p["t"])),
            }
        )

    agg, fits = aggregate_AB(ab)
    fit = fits["fit_on_K_means"]
    print(
        "Fig1 linear fit: alpha={:.4f} beta={:.4f} R^2={:.4f}".format(
            fit["alpha"], fit["beta"], fit["R2"]
        )
    )

    with (RUN / "phaseAB_agg.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        w.writeheader()
        w.writerows(agg)

    plot_fig1(points, agg, fit, FIG / "Fig1_topology_information.png")
    plot_fig1(points, agg, fit, RUN / "Fig1_topology_information.png")
    plot_fig1(points, agg, fit, OUT / "Fig1_topology_information.png")

    scatter = [p for p in points if p.get("delta_a") is not None]
    for path in (
        FIG / "Fig2_information_action.png",
        RUN / "Fig2_information_action.png",
        OUT / "Fig2_information_action.png",
        FIG / "Fig2_information_performance.png",
    ):
        plot_fig2(agg, scatter, path, 0.99)

    b_rows = [
        {
            "K": r["K"],
            "epsilon_G": r["mean_epsilon_G"],
            "delta_a": r["mean_delta_a"],
            "Delta_A_gamma": r["mean_Delta_A_gamma"],
        }
        for r in agg
    ]
    with (RUN / "phaseB_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(b_rows[0].keys()))
        w.writeheader()
        w.writerows(b_rows)

    report = {
        "protocol": "formal_6_3",
        "source_run": str(RUN),
        "G_star_definition": "full-support reference topology before budget projection",
        "note": "Fig2 = discounted action discrepancy Delta_A_gamma, not task return gap. "
        "Phase C pending re-run (--phase C only).",
        "phaseAB_agg": agg,
        "linear_fit_Lemma2": fits,
        "seeds": sorted({int(r["seed"]) for r in ab}),
        "K": sorted({int(r["K"]) for r in ab}),
        "episodes_per_seed": 32,
    }
    out = OUT / "evidence_6_3_formal_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (RUN / "evidence_6_3_formal_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Wrote", out)
    print("Fig1/Fig2 finalized from saved AB CSVs.")


if __name__ == "__main__":
    main()

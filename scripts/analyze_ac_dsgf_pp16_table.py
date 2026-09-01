"""Phase 6 Table — assemble DSGF / AC-DSGF / AC-DSGF++ @ 16UAV × 5 seeds.

Writes:
  paper/tables/table_pp16_main.csv
  paper/tables/table_pp16_paper.csv
  results/ac_dsgf_pp/uav16/table_pp16_detail.json

Also plots utility alignment curve if logs exist:
  paper/figures/fig_utility_alignment.png
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 3407, 2026, 1234, 8888]
EPS = 1e-6

AC_ROOT = ROOT / "results" / "ac_dsgf" / "uav16"
PP_ROOT = ROOT / "results" / "ac_dsgf_pp" / "uav16"
BASELINE = ROOT / "results" / "baseline16_seeds"
# Prefer frozen Table I if present
TABLE1 = ROOT / "paper" / "tables" / "table1_final.csv"

METHOD_ORDER = ["gat", "dsgf", "ac_dsgf", "ac_dsgf_pp"]


def _mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return m, 0.0
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(v)


def cei(success: float, comm: float) -> float:
    return success / (comm + EPS)


def load_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _late_log_mean(run: Path, name: str, key: str, n: int = 5) -> float | None:
    p = run / "logs" / name
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    vals = [float(r[key]) for r in rows[-n:] if key in r and r[key] != ""]
    return sum(vals) / len(vals) if vals else None


def metrics_from_run(method: str, seed: int) -> dict | None:
    if method == "ac_dsgf_pp":
        run = PP_ROOT / f"s{seed}"
        data = load_summary(run / "summary.json")
    elif method == "ac_dsgf":
        run = AC_ROOT / f"s{seed}"
        data = load_summary(run / "summary.json")
    else:
        run = BASELINE / method / f"s{seed}"
        data = load_summary(run / "summary.json")
    if data is None:
        print(f"[missing] {method} s{seed}")
        return None

    paper = data.get("paper_metrics", data)
    success = float(paper.get("success", data.get("success", 0.0)))
    # Prefer fraction; some summaries store already as fraction 0–1
    if success > 1.0:
        success = success / 100.0
    comm = data.get("communication_cost_mean")
    if comm is None:
        comm = paper.get("communication_cost", 0.0)
    comm = float(comm or 0.0)

    row = {
        "method": method,
        "seed": seed,
        "success": success,
        "comm": comm,
        "cei": cei(success, comm),
        "collision": float(paper.get("collision", data.get("collision", 0.0))),
        "reward": float(paper.get("reward", data.get("reward", 0.0))),
        "utility_corr": None,
        "comm_precision": None,
    }
    if method == "ac_dsgf_pp":
        row["utility_corr"] = _late_log_mean(run, "utility_corr.csv", "corr_u_ustar")
        row["comm_precision"] = _late_log_mean(run, "comm_precision.csv", "comm_precision")
        if row["comm_precision"] is None:
            row["comm_precision"] = _late_log_mean(run, "utility_corr.csv", "comm_precision")
    return row


def plot_utility_alignment():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = False
    for seed in SEEDS:
        p = PP_ROOT / f"s{seed}" / "logs" / "utility_corr.csv"
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        xs = [int(r["step"]) for r in rows]
        ys = [float(r["corr_u_ustar"]) for r in rows]
        ax.plot(xs, ys, alpha=0.7, label=f"s{seed}")
        plotted = True
    if not plotted:
        return
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="target 0.5")
    ax.set_xlabel("Training steps")
    ax.set_ylabel("corr(U, U*)")
    ax.set_title("Utility Alignment (AC-DSGF++)")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.2, 1.0)
    out = ROOT / "paper" / "figures" / "fig_utility_alignment.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_comm_precision():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = False
    for seed in SEEDS:
        p = PP_ROOT / f"s{seed}" / "logs" / "comm_precision.csv"
        if not p.exists():
            p = PP_ROOT / f"s{seed}" / "logs" / "utility_corr.csv"
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows or "comm_precision" not in rows[0]:
            continue
        xs = [int(r["step"]) for r in rows]
        ys = [float(r["comm_precision"]) for r in rows]
        ax.plot(xs, ys, alpha=0.7, label=f"s{seed}")
        plotted = True
    if not plotted:
        return
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Communication Precision")
    ax.set_title("Comm Precision: P(U*>thr | g>0.5)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    out = ROOT / "paper" / "figures" / "fig_comm_precision.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def classify_case(pp: dict) -> str:
    """Case A/B/C from 5-seed (or partial) PP summary vs frozen AC-DSGF."""
    s = pp.get("success_mean", float("nan"))
    c = pp.get("comm_mean", float("nan"))
    corr = pp.get("utility_corr_mean", float("nan"))
    if not (s == s and c == c):
        return "PENDING"
    over = c < 0.05 and s < 0.035
    corr_ok = corr == corr and corr >= 0.5
    if over or (corr == corr and corr < 0.4 and s < 0.035):
        return "C"
    if s >= 0.035 and corr_ok and c < 0.20:
        return "A"
    if s >= 0.032 and corr_ok and c <= 0.25:
        return "B"
    if s >= 0.035 and c > 0.25:
        return "B-weak"
    return "C"


def inject_frozen_baselines(all_rows: list[dict]) -> list[dict]:
    """If baseline seed dirs missing, inject Table I frozen means as single synthetic rows."""
    have = {r["method"] for r in all_rows}
    if TABLE1.exists():
        with open(TABLE1, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                m = row["method"]
                if m not in ("gat", "dsgf", "ac_dsgf"):
                    continue
                if m in have:
                    continue
                all_rows.append(
                    {
                        "method": m,
                        "seed": -1,
                        "success": float(row["success_mean"]),
                        "comm": float(row["comm_mean"]),
                        "cei": float(row["cei_mean"]) if row.get("cei_mean") else float("nan"),
                        "collision": float(row.get("collision_mean") or 0),
                        "reward": float(row.get("reward_mean") or 0),
                        "utility_corr": None,
                        "comm_precision": None,
                        "frozen_table1": True,
                    }
                )
                have.add(m)
    return all_rows


def main():
    all_rows = []
    for method in METHOD_ORDER:
        for seed in SEEDS:
            m = metrics_from_run(method, seed)
            if m:
                all_rows.append(m)
    all_rows = inject_frozen_baselines(all_rows)

    summary = []
    for method in METHOD_ORDER:
        xs = [r for r in all_rows if r["method"] == method]
        if not xs:
            continue
        sm, ss = _mean_std([r["success"] for r in xs])
        cm, cs = _mean_std([r["comm"] for r in xs])
        cei_m, cei_s = _mean_std([r["cei"] for r in xs])
        entry = {
            "method": method,
            "n_seeds": len(xs),
            "success_mean": sm,
            "success_std": ss,
            "comm_mean": cm,
            "comm_std": cs,
            "cei_mean": cei_m,
            "cei_std": cei_s,
        }
        if method == "ac_dsgf_pp":
            uc = [r["utility_corr"] for r in xs if r["utility_corr"] is not None]
            cp = [r["comm_precision"] for r in xs if r["comm_precision"] is not None]
            if uc:
                entry["utility_corr_mean"], entry["utility_corr_std"] = _mean_std(uc)
            if cp:
                entry["comm_precision_mean"], entry["comm_precision_std"] = _mean_std(cp)
        summary.append(entry)
        print(
            f"{method:12s}  S={sm*100:.2f}±{ss*100:.2f}%  "
            f"C={cm:.2f}±{cs:.2f}  CEI={cei_m:.4f}"
        )

    out_dir = ROOT / "paper" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "method",
        "n_seeds",
        "success_mean",
        "success_std",
        "comm_mean",
        "comm_std",
        "cei_mean",
        "cei_std",
        "utility_corr_mean",
        "utility_corr_std",
        "comm_precision_mean",
        "comm_precision_std",
    ]
    for name in ("table_pp16_main.csv", "table1_pp_main.csv"):
        path = out_dir / name
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for e in summary:
                w.writerow(e)
        print(f"Wrote {path}")

    for name in ("table_pp16_paper.csv", "table1_pp_paper.csv"):
        path = out_dir / name
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Method", "Success (%)", "Comm", "CEI", "corr(U,U*)", "Comm Prec."])
            for e in summary:
                s = f"{e['success_mean']*100:.2f}±{e['success_std']*100:.2f}"
                c = f"{e['comm_mean']:.2f}±{e['comm_std']:.2f}"
                cei_s = f"{e['cei_mean']:.4f}"
                uc = (
                    f"{e.get('utility_corr_mean', float('nan')):.3f}"
                    if "utility_corr_mean" in e
                    else "—"
                )
                cp = (
                    f"{e.get('comm_precision_mean', float('nan')):.3f}"
                    if "comm_precision_mean" in e
                    else "—"
                )
                w.writerow([e["method"], s, c, cei_s, uc, cp])
        print(f"Wrote {path}")

    pp = next((e for e in summary if e["method"] == "ac_dsgf_pp"), None)
    case = classify_case(pp) if pp else "PENDING"
    print(f"\n=== CASE VERDICT: {case} ===")
    print("Priority: Success hold > corr(U,U*) > Comm reduction")

    detail = PP_ROOT / "table_pp16_detail.json"
    detail.parent.mkdir(parents=True, exist_ok=True)
    with open(detail, "w", encoding="utf-8") as f:
        json.dump({"rows": all_rows, "summary": summary, "case": case}, f, indent=2)
    print(f"Wrote {detail}")

    # Append verdict to analysis_pp.md
    analysis = ROOT / "paper" / "docs" / "planning" / "analysis_pp.md"
    if analysis.exists() and pp:
        with open(analysis, "a", encoding="utf-8") as f:
            f.write(
                f"\n\n## Auto verdict ({__import__('datetime').datetime.now().date()})\n"
                f"- Case: **{case}**\n"
                f"- Success: {pp['success_mean']*100:.2f}%\n"
                f"- Comm: {pp['comm_mean']:.4f}\n"
                f"- corr: {pp.get('utility_corr_mean', float('nan'))}\n"
                f"- precision: {pp.get('comm_precision_mean', float('nan'))}\n"
            )

    plot_utility_alignment()
    plot_comm_precision()
    print("\nClaim check: Success ≈ AC-DSGF, Comm ↓, CEI ↑ (not Success leadership)")


if __name__ == "__main__":
    main()

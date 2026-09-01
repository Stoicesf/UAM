"""Experiment runner: regimes × methods × seeds → results.json + figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch

from experiments.secdo_uav.env_uav_bandwidth import UAVBandwidthConfig, UAVBandwidthEnv
from secdo.baselines import get_solver
from secdo.evaluation.plots import render_all
from secdo.evaluation.theory_metrics import results_payload
from secdo.utils.seed import set_seed


REGIME_PRESETS = {
    "slow": dict(regime="slow", vel_scale=0.4),
    "slow_drift": dict(regime="slow", vel_scale=0.4),
    "fast": dict(regime="fast", vel_scale=0.4),
    "fast_drift": dict(regime="fast", vel_scale=0.4),
    "stress": dict(regime="stress", vel_scale=0.5, stress_eps=0.35, stress_delta=0.25),
    "stress_test": dict(regime="stress", vel_scale=0.5, stress_eps=0.35, stress_delta=0.25),
}


def run_experiment(
    *,
    exp: str = "fast_drift",
    methods: Iterable[str] = ("secdo", "reactive", "oracle"),
    seeds: Iterable[int] = (0,),
    horizon: int = 48,
    batch: int = 8,
    eta: float = 0.25,
    ckpt: str = "checkpoints/secdo_platform/pretrain/best.pt",
    out_dir: str | Path = "results/secdo",
    device: str = "cuda:0",
    plot: bool = True,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preset = REGIME_PRESETS.get(exp, REGIME_PRESETS["fast"])
    rows = []

    for seed in seeds:
        set_seed(int(seed))
        for method in methods:
            env_cfg = UAVBandwidthConfig(
                horizon=horizon,
                seed=int(seed),
                **{k: v for k, v in preset.items()},
            )
            env = UAVBandwidthEnv(env_cfg, device=device if torch.cuda.is_available() else "cpu")
            kwargs = {"eta": eta, "device": device}
            if method == "secdo":
                kwargs["ckpt"] = ckpt
            solver = get_solver(method, **kwargs)
            result = solver.solve(env, batch=batch)
            row = results_payload(
                method=result.method,
                regime=exp,
                seed=int(seed),
                summary=result.metrics,
                series=result.series,
            )
            rows.append(row)
            print(
                f"[{exp}|seed={seed}|{method}] "
                f"gap={row['gap']:.5f} viol={row['violation']:.5f} "
                f"delta={row['delta']:.5f} rho={row['rho']:.5f} "
                f"frac(d<r)={row['frac_delta_lt_rho']:.2f}",
                flush=True,
            )

    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("Wrote", results_path, flush=True)

    # seed-averaged table
    summary = _aggregate(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if plot:
        fig_dir = out_dir / "figures"
        try:
            paths = render_all(results_path, fig_dir)
            paper_fig = Path("paper/ac_dsgf_v2/experiments/phase2_results/figures")
            paper_fig.mkdir(parents=True, exist_ok=True)
            import shutil

            for p in paths:
                src = Path(p)
                if src.suffix == ".pdf":
                    shutil.copy2(src, paper_fig / src.name)
                    png = src.with_suffix(".png")
                    if png.is_file():
                        shutil.copy2(png, paper_fig / png.name)
            print("Figures →", fig_dir, "and", paper_fig, flush=True)
        except Exception as e:
            print(f"WARN: plotting failed ({e}); results.json kept", flush=True)
    return results_path


def _aggregate(rows: list[dict]) -> list[dict]:
    from collections import defaultdict

    buckets: dict[tuple, list] = defaultdict(list)
    for r in rows:
        buckets[(r["regime"], r["method"])].append(r)
    out = []
    keys = ["gap", "violation", "delta", "rho", "epsilon", "frac_delta_lt_rho", "mean_eps_plus_delta"]
    for (regime, method), rs in sorted(buckets.items()):
        agg = {"regime": regime, "method": method, "n_seeds": len(rs)}
        for k in keys:
            if k in rs[0]:
                agg[k] = float(sum(x[k] for x in rs) / len(rs))
        out.append(agg)
    return out

# -*- coding: utf-8 -*-
"""§6.4 Scalability observation (NO training, NO Thm3 write-up).

Fixed degree budget K (default 4). N ∈ {16,32,64,128}.
Questions: C_N growth, J_N stability, eta_N = J_N/C_N vs refs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchrl.envs.utils import step_mdp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tro.collector.common import ROOT, get_adapter, load_method, resolve_device, spawn_kwargs as _spawn_kwargs

from algorithms.guided.mappo_guided import ACGuideAdapter, build_guided_mappo
from utils.comm_metrics import compute_sparse_communication_stats
from utils.experiment import load_experiment_config
from utils.seed import set_seed
from utils.tro_run import create_run_dir, write_config

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore

OUT = ROOT / "paper" / "ac_dsgf_tro" / "experiments" / "evidence_6_4_scalability"
FIG_OUT = OUT / "figures"
FIG_PAPER = ROOT / "paper" / "ac_dsgf_tro" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG_OUT.mkdir(parents=True, exist_ok=True)
FIG_PAPER.mkdir(parents=True, exist_ok=True)

DEFAULT_SEEDS = (1234, 2026, 3407, 42, 8888)
DEFAULT_SIZES = (16, 32, 64, 128)

METHODS = {
    "ac_dsgf": (
        "AC-DSGF",
        "configs/ac_dsgf/ac_dsgf_16uav.yaml",
        "results/ac_dsgf/uav16/s{seed}/checkpoints/final.pt",
    ),
    "dsgf": (
        "DSGF",
        "configs/baseline16/dsgf_v2.yaml",
        "results/baseline16_seeds/dsgf/s{seed}/checkpoints/final.pt",
    ),
    "full_attn": (
        "Full-Attention",
        "configs/baseline16/full_attention.yaml",
        "results/baseline16_seeds/transformer/s{seed}/checkpoints/final.pt",
    ),
}


def _as_float(x) -> float:
    if torch.is_tensor(x):
        return float(x.detach().float().reshape(-1)[0].item())
    return float(x)


@torch.no_grad()
def rollout(
    env,
    policy,
    *,
    method: str,
    n: int,
    k_fixed: int | None,
    episodes: int,
    max_steps: int,
    success_thr: float,
    comm_radius: float,
) -> dict:
    adapter = get_adapter(policy)
    if adapter is not None:
        adapter.set_ablation_mode("full")
        if method == "ac_dsgf" and k_fixed is not None:
            adapter.set_fixed_k(int(k_fixed))
            adapter.set_budget_ratio(None)
        else:
            adapter.set_fixed_k(None)
            adapter.set_budget_ratio(None)

    rewards, successes, costs, rhos = [], [], [], []
    t0 = time.perf_counter()
    dense = float(n * (n - 1))

    for _ in range(episodes):
        td = env.reset()
        ep_r, ep_s, ep_c, n_c = 0.0, 0.0, 0.0, 0
        for _ in range(max_steps):
            td = policy(td)
            act = td.get(("agents", "action"))
            if act is not None and torch.is_tensor(act) and bool(act.isnan().any()):
                td[("agents", "action")] = torch.nan_to_num(act, nan=0.0)

            if method == "full_attn":
                ep_c += dense
                n_c += 1
            elif adapter is not None and adapter.last_topology_diag is not None:
                diag = adapter.last_topology_diag
                if "C_t" in diag:
                    ep_c += _as_float(diag["C_t"])
                    n_c += 1
                elif adapter.last_soft_edges is not None:
                    ep_c += float(adapter.last_soft_edges)
                    n_c += 1
            else:
                obs = td.get(("agents", "observation"))
                if obs is not None:
                    obs_b = obs.unsqueeze(0) if obs.dim() == 2 else obs
                    stats = compute_sparse_communication_stats(obs_b, comm_radius)
                    ep_c += float(stats["communication_cost"])
                    n_c += 1

            td = env.step(td)
            obs_n = td.get(("next", "agents", "observation"))
            if obs_n is not None:
                goal = obs_n[0, :, 4:6] if obs_n.dim() == 3 else obs_n[:, 4:6]
                ep_s = float((goal.norm(dim=-1) < success_thr).float().mean().item())
            rew = td.get(("next", "agents", "reward"))
            if rew is not None:
                ep_r += float(rew.mean().item())
            done = td.get(("next", "done"))
            if done is not None and bool(done.any()):
                break
            td = step_mdp(td)

        mean_c = ep_c / max(n_c, 1)
        rewards.append(ep_r)
        successes.append(ep_s)
        costs.append(mean_c)
        rhos.append(mean_c / max(dense, 1.0))

    elapsed = time.perf_counter() - t0
    C = float(np.mean(costs))
    J = float(np.mean(rewards))
    return {
        "reward": J,
        "success": float(np.mean(successes)),
        "C": C,
        "rho": float(np.mean(rhos)),
        "C_over_N": C / max(n, 1),
        "eta": J / max(C, 1e-8),
        "runtime_s": elapsed,
        "n_episodes": episodes,
    }


FIELDS = [
    "method",
    "display",
    "seed",
    "N",
    "K",
    "reward",
    "success",
    "C",
    "rho",
    "C_over_N",
    "eta",
    "runtime_s",
    "n_episodes",
]


def _write_csv(rows: list[dict]) -> None:
    path = OUT / "scalability_statistics.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def aggregate(rows: list[dict]) -> list[dict]:
    from collections import defaultdict

    g: dict[tuple, list] = defaultdict(list)
    for r in rows:
        g[(r["method"], int(r["N"]), str(r.get("K", "")))].append(r)
    out = []
    for (method, N, K), rs in sorted(g.items(), key=lambda x: (x[0][0], x[0][1])):
        def mean(k):
            return float(np.mean([float(x[k]) for x in rs]))

        def std(k):
            return float(np.std([float(x[k]) for x in rs]))

        out.append(
            {
                "method": method,
                "display": rs[0]["display"],
                "N": N,
                "K": K,
                "n_seeds": len(rs),
                "reward_mean": mean("reward"),
                "reward_std": std("reward"),
                "success_mean": mean("success"),
                "success_std": std("success"),
                "C_mean": mean("C"),
                "C_std": std("C"),
                "rho_mean": mean("rho"),
                "C_over_N_mean": mean("C_over_N"),
                "eta_mean": mean("eta"),
                "runtime_s_mean": mean("runtime_s"),
            }
        )
    return out


def plot_figures(agg: list[dict]) -> tuple[Path | None, Path | None]:
    colors = {"ac_dsgf": "#C44E52", "dsgf": "#4C72B0", "full_attn": "#55A868"}
    labels = {"ac_dsgf": "AC-DSGF (K=4)", "dsgf": "DSGF", "full_attn": "Full-Attn"}

    # Fig.7 C vs N
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for m, c in colors.items():
        pts = sorted([r for r in agg if r["method"] == m], key=lambda x: int(x["N"]))
        if not pts:
            continue
        xs = np.array([int(p["N"]) for p in pts], dtype=float)
        ys = np.array([p["C_mean"] for p in pts], dtype=float)
        yerr = np.array([p["C_std"] for p in pts], dtype=float)
        ax.errorbar(xs, ys, yerr=yerr, marker="o", color=c, label=labels[m], capsize=3)
        if m == "ac_dsgf" and len(xs) >= 2:
            coef = np.polyfit(xs, ys, 1)
            xfit = np.linspace(xs.min(), xs.max(), 50)
            ax.plot(xfit, coef[0] * xfit + coef[1], "--", color=c, alpha=0.5, label=rf"AC fit $\propto N$")
    ax.set_xlabel(r"Swarm size $N$")
    ax.set_ylabel(r"Communication cost $C_N$")
    ax.set_title("Communication scaling")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    p7 = FIG_PAPER / "Fig7_communication_scaling.png"
    fig.savefig(p7, dpi=200)
    fig.savefig(FIG_OUT / "Fig7_communication_scaling.png", dpi=200)
    plt.close(fig)

    # Fig.8 eta = J/C vs N (communication efficiency; not a performance-preservation claim)
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for m, c in colors.items():
        pts = sorted([r for r in agg if r["method"] == m], key=lambda x: int(x["N"]))
        if not pts:
            continue
        xs = [int(p["N"]) for p in pts]
        ys = [p["eta_mean"] for p in pts]
        ax.plot(xs, ys, marker="o", color=c, label=labels[m])
    ax.set_xlabel(r"Swarm size $N$")
    ax.set_ylabel(r"Efficiency $\eta_N=J_N/C_N$")
    ax.set_title("Communication efficiency scaling")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    p8 = FIG_PAPER / "Fig8_efficiency_scaling.png"
    fig.savefig(p8, dpi=200)
    fig.savefig(FIG_OUT / "Fig8_efficiency_scaling.png", dpi=200)
    plt.close(fig)
    return p7, p8


def thm3_gate(agg: list[dict]) -> dict:
    """Post-hoc observation only — strong Thm3 cancelled; Prop. complexity preferred."""
    ac = sorted([r for r in agg if r["method"] == "ac_dsgf"], key=lambda x: int(x["N"]))
    if len(ac) < 2:
        return {"recommend_thm3": False, "reason": "insufficient AC-DSGF points"}
    c_over_n = [r["C_over_N_mean"] for r in ac]
    j = [r["reward_mean"] for r in ac]
    c_ratio = max(c_over_n) / max(min(c_over_n), 1e-8)
    j_drop = (j[0] - min(j)) / max(abs(j[0]), 1e-8)
    return {
        "recommend_thm3": False,
        "decision": "empirical_scalability_plus_complexity_proposition",
        "C_over_N": c_over_n,
        "J_N": j,
        "C_over_N_max_min_ratio": c_ratio,
        "J_relative_drop_from_N16": j_drop,
        "note": "Strong Thm3 cancelled (2026-07-24). Use proposition_complexity.md + §6.4 observations.",
    }


def _write_progress(
    *,
    done_n: int,
    total_n: int,
    current: str,
    last_result: str = "",
) -> None:
    """Always-visible progress file (works even when stdout is redirected)."""
    pct = 100.0 * done_n / max(total_n, 1)
    bar_w = 28
    filled = int(bar_w * done_n / max(total_n, 1))
    bar = "#" * filled + "-" * (bar_w - filled)
    text = (
        f"§6.4 scalability  [{bar}] {done_n}/{total_n} ({pct:5.1f}%)\n"
        f"current: {current}\n"
        f"last:    {last_result}\n"
    )
    (OUT / "progress.txt").write_text(text, encoding="utf-8")
    (OUT / "progress.json").write_text(
        json.dumps(
            {
                "done": done_n,
                "total": total_n,
                "pct": round(pct, 2),
                "current": current,
                "last": last_result,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main():
    ap = argparse.ArgumentParser(description="T-RO §6.4 scalability observation")
    ap.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--methods",
        nargs="+",
        default=list(METHODS.keys()),
        choices=list(METHODS.keys()),
    )
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    device = resolve_device(args.device)
    run_dir = create_run_dir(prefix="tro_evidence_6_4_scalability")
    write_config(
        run_dir,
        {
            "section": "6.4",
            "purpose": "scalability_observation",
            "sizes": args.sizes,
            "K": args.K,
            "seeds": args.seeds,
            "episodes": args.episodes,
            "methods": args.methods,
            "device": device,
            "training": False,
            "thm3": "deferred",
        },
    )
    print(f"device={device} | K={args.K} | sizes={args.sizes} | run={run_dir}", flush=True)

    rows: list[dict] = []
    done: set[tuple] = set()
    csv_path = OUT / "scalability_statistics.csv"
    if args.resume and csv_path.exists():
        with csv_path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(dict(r))
                done.add((r["method"], str(r["seed"]), str(r["N"]), str(r.get("K", ""))))
        print(f"resume: {len(rows)} rows", flush=True)

    jobs: list[tuple[str, int, int, str]] = []
    for method in args.methods:
        for seed in args.seeds:
            for N in args.sizes:
                K_tag = str(args.K) if method == "ac_dsgf" else ""
                key = (method, str(seed), str(N), K_tag)
                if key in done:
                    continue
                jobs.append((method, seed, N, K_tag))

    total = len(done) + len(jobs)
    print(f"progress: {len(done)} done / {total} total | pending={len(jobs)}", flush=True)
    _write_progress(
        done_n=len(done),
        total_n=total,
        current="(starting)",
        last_result=f"resume loaded {len(done)} cells",
    )

    pbar = None
    if tqdm is not None and jobs:
        pbar = tqdm(
            total=len(jobs),
            desc="§6.4 cells",
            unit="cell",
            dynamic_ncols=True,
            file=sys.stderr,
        )

    for method, seed, N, K_tag in jobs:
        set_seed(seed)
        label = f"{method} seed={seed} N={N}"
        _write_progress(done_n=len(done), total_n=total, current=label, last_result="")
        if pbar is not None:
            pbar.set_postfix_str(label, refresh=True)
        print(f"[{method}] seed={seed} N={N} …", flush=True)
        try:
            cfg, env, policy, display = load_method(METHODS, method, seed, device, n_agents=N)
        except Exception as e:
            msg = f"SKIP load: {e}"
            print(f"  {msg}", flush=True)
            _write_progress(done_n=len(done), total_n=total, current=label, last_result=msg)
            if pbar is not None:
                pbar.update(1)
            continue
        thr = float(cfg.get("env", {}).get("success_threshold", 0.3))
        max_steps = int(cfg.get("env", {}).get("max_steps", 128) or 128)
        radius = float(cfg.get("env", {}).get("comm_radius", 0.5))
        k_fixed = args.K if method == "ac_dsgf" else None
        try:
            r = rollout(
                env,
                policy,
                method=method,
                n=N,
                k_fixed=k_fixed,
                episodes=args.episodes,
                max_steps=max_steps,
                success_thr=thr,
                comm_radius=radius,
            )
        except Exception as e:
            msg = f"SKIP rollout: {e}"
            print(f"  {msg}", flush=True)
            _write_progress(done_n=len(done), total_n=total, current=label, last_result=msg)
            del env, policy
            if pbar is not None:
                pbar.update(1)
            continue
        row = {
            "method": method,
            "display": display if method != "ac_dsgf" else f"{display} (K={args.K})",
            "seed": seed,
            "N": N,
            "K": K_tag,
            **r,
        }
        rows.append(row)
        done.add((method, str(seed), str(N), K_tag))
        last = (
            f"J={r['reward']:.3f} S={r['success']:.3f} C={r['C']:.2f} "
            f"rho={r['rho']:.4f} C/N={r['C_over_N']:.3f}"
        )
        print(f"  {last}", flush=True)
        _write_csv(rows)
        _write_progress(
            done_n=len(done),
            total_n=total,
            current=label,
            last_result=last,
        )
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix_str(f"C={r['C']:.1f} J={r['reward']:.2f}", refresh=True)
        del env, policy
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    if pbar is not None:
        pbar.close()

    agg = aggregate(rows)
    fig7, fig8 = plot_figures(agg) if agg else (None, None)
    gate = thm3_gate(agg)
    report = {
        "n_rows": len(rows),
        "table_III": agg,
        "fig7": str(fig7) if fig7 else None,
        "fig8": str(fig8) if fig8 else None,
        "thm3_gate": gate,
        "claim_mode": "empirical_observation_only",
        "run_dir": str(run_dir),
    }
    (OUT / "evidence_6_4_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _write_progress(
        done_n=len(done),
        total_n=total,
        current="(finished)",
        last_result=f"wrote report; thm3_gate={gate.get('recommend_thm3')}",
    )
    print("Wrote", OUT / "scalability_statistics.csv", flush=True)
    print("Wrote", OUT / "evidence_6_4_report.json", flush=True)
    print("Progress file:", OUT / "progress.txt", flush=True)
    print("Thm3 gate (heuristic):", gate, flush=True)


if __name__ == "__main__":
    main()

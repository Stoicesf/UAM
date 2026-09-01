# -*- coding: utf-8 -*-
"""§6.6 Phase A — Communication MARL baseline comparison (NO training).

Methods (frozen 16-UAV checkpoints, same seeds):
  A   MAPPO
  B   GAT-MAPPO, DSGF
  G2  Full Attention (transformer)
  Ours AC-DSGF (+ optional hard-K slices)

Modes:
  summary — aggregate existing summary.json paper_metrics
  reeval  — unified closed-loop evaluation

Claim hygiene: joint (J, C); no Class C numbers until wrappers exist.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchrl.envs.utils import step_mdp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tro.collector.common import ROOT, get_adapter, load_method, resolve_device

from algorithms.baseline.mappo import build_mappo
from algorithms.guided.mappo_guided import ACGuideAdapter, build_guided_mappo
from utils.comm_metrics import compute_sparse_communication_stats
from utils.experiment import load_experiment_config
from utils.seed import set_seed
from utils.tro_run import create_run_dir, write_config

OUT = ROOT / "paper" / "ac_dsgf_tro" / "experiments" / "evidence_6_6_baselines"
FIG_OUT = OUT / "figures"
FIG_PAPER = ROOT / "paper" / "ac_dsgf_tro" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG_OUT.mkdir(parents=True, exist_ok=True)
FIG_PAPER.mkdir(parents=True, exist_ok=True)

DEFAULT_SEEDS = (1234, 2026, 3407, 42, 8888)

# method_key -> (class, display, config, ckpt_template)
METHODS = {
    "mappo": (
        "A",
        "MAPPO",
        "configs/baseline16/mappo.yaml",
        "results/baseline16_seeds/mappo/s{seed}/checkpoints/final.pt",
    ),
    "gat": (
        "B",
        "GAT-MAPPO",
        "configs/baseline16/gat_mappo.yaml",
        "results/baseline16_seeds/gat/s{seed}/checkpoints/final.pt",
    ),
    "dsgf": (
        "B",
        "DSGF",
        "configs/baseline16/dsgf_v2.yaml",
        "results/baseline16_seeds/dsgf/s{seed}/checkpoints/final.pt",
    ),
    "full_attn": (
        "G2",
        "Full-Attention",
        "configs/baseline16/full_attention.yaml",
        "results/baseline16_seeds/transformer/s{seed}/checkpoints/final.pt",
    ),
    "ac_dsgf": (
        "Ours",
        "AC-DSGF",
        "configs/ac_dsgf/ac_dsgf_16uav.yaml",
        "results/ac_dsgf/uav16/s{seed}/checkpoints/final.pt",
    ),
}


def _goal_success(obs: torch.Tensor, thr: float) -> float:
    if obs.dim() == 3:
        goal = obs[0, :, 4:6]
    else:
        goal = obs[:, 4:6]
    return float((goal.norm(dim=-1) < thr).float().mean().item())


@torch.no_grad()
def reeval_policy(
    env,
    policy,
    *,
    method: str,
    episodes: int,
    max_steps: int,
    success_thr: float,
    comm_radius: float,
    fixed_k: int | None = None,
    n_agents: int = 16,
) -> dict:
    adapter = get_adapter(policy)
    if adapter is not None:
        adapter.set_ablation_mode("full")
        if fixed_k is not None:
            adapter.set_fixed_k(int(fixed_k))
            adapter.set_budget_ratio(None)
        else:
            adapter.set_fixed_k(None)
            adapter.set_budget_ratio(getattr(adapter.encoder, "budget_ratio", None))

    rewards, successes, collisions, costs, rhos = [], [], [], [], []
    dense_c = float(n_agents * (n_agents - 1))
    for _ in range(episodes):
        td = env.reset()
        ep_r, ep_s, ep_c, ep_cost, n_c = 0.0, 0.0, 0.0, 0.0, 0
        for _ in range(max_steps):
            td = policy(td)
            # Method-aware communication cost (never silently use radius for MAPPO / full-attn)
            if method == "mappo":
                ep_cost += 0.0
                n_c += 1
            elif method == "full_attn":
                ep_cost += dense_c
                n_c += 1
            elif adapter is not None and adapter.last_topology_diag is not None and "C_t" in adapter.last_topology_diag:
                diag = adapter.last_topology_diag
                ep_cost += float(
                    diag["C_t"].item() if torch.is_tensor(diag["C_t"]) else diag["C_t"]
                )
                n_c += 1
            elif adapter is not None and adapter.last_soft_edges is not None:
                ep_cost += float(adapter.last_soft_edges)
                n_c += 1
            else:
                # GAT / DSGF: radius-neighborhood edge count as support cost
                obs = td.get(("agents", "observation"))
                if obs is not None:
                    obs_b = obs.unsqueeze(0) if obs.dim() == 2 else obs
                    stats = compute_sparse_communication_stats(obs_b, comm_radius)
                    ep_cost += float(stats["communication_cost"])
                    n_c += 1

            td = env.step(td)
            obs_n = td.get(("next", "agents", "observation"))
            if obs_n is not None:
                ep_s = _goal_success(obs_n, success_thr)
            info = td.get(("next", "agents", "info"), default=None)
            if info is not None and "agent_collisions" in info.keys():
                coll = info.get("agent_collisions")
                ep_c = max(ep_c, float((coll < 0).float().mean().item()))
            rew = td.get(("next", "agents", "reward"))
            if rew is not None:
                ep_r += float(rew.mean().item())
            done = td.get(("next", "done"))
            if done is not None and bool(done.any()):
                break
            td = step_mdp(td)

        rewards.append(ep_r)
        successes.append(ep_s)
        collisions.append(ep_c)
        mean_c = ep_cost / max(n_c, 1)
        costs.append(mean_c)
        rhos.append(mean_c / max(dense_c, 1.0))

    return {
        "reward": float(np.mean(rewards)),
        "success": float(np.mean(successes)),
        "collision": float(np.mean(collisions)),
        "C": float(np.mean(costs)),
        "rho": float(np.mean(rhos)),
        "n_episodes": episodes,
    }


def aggregate_summaries(seeds: list[int]) -> list[dict]:
    rows = []
    for method, (cls, name, _cfg, ckpt_tmpl) in METHODS.items():
        for seed in seeds:
            # summary lives next to checkpoints
            ckpt = ROOT / ckpt_tmpl.format(seed=seed)
            summary = ckpt.parents[1] / "summary.json"
            if not summary.exists():
                print(f"WARN missing summary {summary}", flush=True)
                continue
            d = json.loads(summary.read_text(encoding="utf-8"))
            pm = d.get("paper_metrics") or d
            rows.append(
                {
                    "mode": "summary",
                    "method": method,
                    "display": name,
                    "class": cls,
                    "seed": seed,
                    "budget": "train_default",
                    "K": "",
                    "reward": float(pm.get("reward", d.get("reward", float("nan")))),
                    "success": float(pm.get("success", d.get("success", float("nan")))),
                    "collision": float(pm.get("collision", d.get("collision", float("nan")))),
                    "C": float(pm.get("communication_cost", d.get("communication_cost", float("nan")))),
                    "rho": "",
                    "C_source": pm.get("communication_cost_source", "summary"),
                }
            )
    return rows


def run_reeval(
    seeds: list[int],
    episodes: int,
    device: str,
    ac_K: list[int],
    methods: list[str],
) -> list[dict]:
    rows = []
    for method in methods:
        for seed in seeds:
            set_seed(seed)
            print(f"[reeval] {method} seed={seed}", flush=True)
            cfg, env, policy, cls, name = load_method(METHODS, method, seed, device)
            thr = float(cfg.get("env", {}).get("success_threshold", 0.3))
            radius = float(cfg.get("env", {}).get("comm_radius", 0.5))
            max_steps = int(cfg.get("env", {}).get("max_steps", 128) or 128)

            budgets: list[tuple[str, int | None]] = [("train_default", None)]
            if method == "ac_dsgf":
                budgets += [(f"fixed_k_{K}", K) for K in ac_K]

            for budget_tag, K in budgets:
                r = reeval_policy(
                    env,
                    policy,
                    method=method,
                    episodes=episodes,
                    max_steps=max_steps,
                    success_thr=thr,
                    comm_radius=radius,
                    fixed_k=K,
                    n_agents=int(cfg["env"].get("num_agents", 16)),
                )
                c_source = {
                    "mappo": "zero_no_comm",
                    "full_attn": "dense_N_N-1",
                    "ac_dsgf": "topology_diag_C_t",
                }.get(method, "radius_support_edges")
                row = {
                    "mode": "reeval",
                    "method": method,
                    "display": name if K is None else f"{name} (K={K})",
                    "class": cls,
                    "seed": seed,
                    "budget": budget_tag,
                    "K": K if K is not None else "",
                    "reward": r["reward"],
                    "success": r["success"],
                    "collision": r["collision"],
                    "C": r["C"],
                    "rho": r["rho"],
                    "C_source": c_source,
                    "n_episodes": r["n_episodes"],
                }
                rows.append(row)
                print(
                    f"  {row['display']}: J={r['reward']:.3f} succ={r['success']:.4f} "
                    f"C={r['C']:.3f}",
                    flush=True,
                )
                _write_csv(rows)

            del env, policy
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    return rows


FIELDS = [
    "mode",
    "method",
    "display",
    "class",
    "seed",
    "budget",
    "K",
    "reward",
    "success",
    "collision",
    "C",
    "rho",
    "C_source",
    "n_episodes",
]


def _write_csv(rows: list[dict]) -> None:
    path = OUT / "baseline_comparison.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def aggregate(rows: list[dict]) -> list[dict]:
    from collections import defaultdict

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["method"], r["budget"], r["display"], r["class"])
        groups[key].append(r)
    out = []
    for (method, budget, display, cls), rs in sorted(groups.items()):
        def col(name: str) -> list[float]:
            return [float(x[name]) for x in rs if x.get(name) not in ("", None)]

        rew, suc, c = col("reward"), col("success"), col("C")
        out.append(
            {
                "method": method,
                "display": display,
                "class": cls,
                "budget": budget,
                "n_seeds": len(rs),
                "reward_mean": float(np.mean(rew)),
                "reward_std": float(np.std(rew)),
                "success_mean": float(np.mean(suc)),
                "success_std": float(np.std(suc)),
                "C_mean": float(np.mean(c)),
                "C_std": float(np.std(c)),
            }
        )
    return out


def plot_pareto(agg: list[dict]) -> Path | None:
    if not agg:
        return None
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    markers = {"A": "s", "B": "o", "G2": "^", "Ours": "D"}
    for r in agg:
        ax.errorbar(
            r["C_mean"],
            r["reward_mean"],
            xerr=r["C_std"],
            yerr=r["reward_std"],
            fmt=markers.get(r["class"], "o"),
            capsize=3,
            label=r["display"],
        )
    ax.set_xlabel(r"Communication cost $C$")
    ax.set_ylabel(r"Return $J$")
    ax.set_title("Baseline comparison on the $(C,J)$ plane")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, frameon=False, loc="best")
    fig.tight_layout()
    path = FIG_PAPER / "Fig5_baseline_pareto.png"
    fig.savefig(path, dpi=200)
    fig.savefig(FIG_OUT / "Fig5_baseline_pareto.png", dpi=200)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description="T-RO §6.6 Phase A baseline comparison")
    ap.add_argument("--mode", choices=("summary", "reeval", "both"), default="both")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--ac-K", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument(
        "--methods",
        nargs="+",
        default=list(METHODS.keys()),
        choices=list(METHODS.keys()),
    )
    args = ap.parse_args()

    device = resolve_device(args.device)
    run_dir = create_run_dir(prefix="tro_evidence_6_6_baselines")
    write_config(
        run_dir,
        {
            "section": "6.6",
            "phase": "A",
            "mode": args.mode,
            "seeds": args.seeds,
            "episodes": args.episodes,
            "ac_K": args.ac_K,
            "methods": args.methods,
            "device": device,
            "training": False,
            "class_C": "deferred",
        },
    )
    print(f"device={device} | run={run_dir}", flush=True)

    rows: list[dict] = []
    if args.mode in ("summary", "both"):
        print("=== summary aggregation ===", flush=True)
        rows.extend(aggregate_summaries(args.seeds))
        _write_csv(rows)

    if args.mode in ("reeval", "both"):
        print("=== unified reeval ===", flush=True)
        rows.extend(
            run_reeval(args.seeds, args.episodes, device, args.ac_K, args.methods)
        )
        _write_csv(rows)

    # Prefer reeval rows for Table II if present
    formal = [r for r in rows if r["mode"] == "reeval"] or rows
    agg = aggregate(formal)
    fig = plot_pareto(agg)

    report = {
        "phase": "A",
        "class_C": "deferred",
        "n_rows": len(rows),
        "table_II": agg,
        "fig5": str(fig) if fig else None,
        "run_dir": str(run_dir),
        "narrative": [
            "Class A: communication necessity vs MAPPO",
            "Class B / G2: dynamic or projected topology vs fixed / dense on (J,C)",
            "Class C: deferred until TarMAC/IC3Net/ATOC wrappers exist",
        ],
    }
    (OUT / "evidence_6_6_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("Wrote", OUT / "baseline_comparison.csv", flush=True)
    print("Wrote", OUT / "evidence_6_6_report.json", flush=True)
    if fig:
        print("Wrote", fig, flush=True)


if __name__ == "__main__":
    main()

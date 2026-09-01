# -*- coding: utf-8 -*-
"""§6.2 Formal budget feasibility statistics (NO training, NO tuning).

Theorem 1 implementation evidence:
    G_t = Π_{B_t}(S_t)  ⇒  C(G_t) ≤ B_t  (degree-wise: d_i ≤ k_i)

Metrics:
  Vmax = max_t V_B(t)
  VR   = (1/T) sum_t 1[V_B(t) > 0]
  E_d  = |mean_degree - K|   (fixed-K; also vs mean k_cap when support < K)
  rho  = |E| / [N(N-1)]

Claim hygiene: demonstrates practical feasibility — does not prove Theorem 1.
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

from tro.collector.common import (
    ROOT,
    as_float as _as_float,
    get_adapter,
    list_ac_dsgf_ckpts as list_ckpts,
    load_frozen_ac_dsgf as load_frozen,
    resolve_device,
    write_csv,
)

from algorithms.guided.mappo_guided import ACGuideAdapter, build_guided_mappo
from models.communication.budget_layer import (
    apply_topk_budget,
    apply_topk_fixed_k,
    degree_budget_violation,
    hard_adjacency,
)
from utils.experiment import load_experiment_config
from utils.seed import set_seed
from utils.tro_run import create_run_dir, write_config

OUT = ROOT / "paper" / "ac_dsgf_tro" / "experiments" / "evidence_6_2_budget"
FIG_OUT = OUT / "figures"
FIG_PAPER = ROOT / "paper" / "ac_dsgf_tro" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG_OUT.mkdir(parents=True, exist_ok=True)
FIG_PAPER.mkdir(parents=True, exist_ok=True)

DEFAULT_SEEDS = (1234, 2026, 3407, 42, 8888)
DEFAULT_SIZES = (8, 16, 32, 64)
DEFAULT_K = (1, 2, 4, 6, 8)
DEFAULT_RATIOS = (0.05, 0.1, 0.2, 0.4)


def get_encoder(policy):
    return get_adapter(policy).encoder


def _spawn_scale(n: int) -> float:
    if n >= 64:
        return 3.0
    if n >= 32:
        return 2.0
    if n >= 16:
        return 1.5
    return 1.0


@torch.no_grad()
def collect_encoder_geom(
    encoder,
    *,
    N: int,
    obs_dim: int,
    budget_type: str,
    K: int | None,
    ratio: float | None,
    n_steps: int,
    device: str,
    seed: int,
) -> dict:
    """Frozen scorer + hard Π on geometric samples (no VMAS). For N≥32 speed path.

    Still uses learned edge scores; only the state sampler changes.
    """
    from models.ac_dsgf import ACDSGF

    assert isinstance(encoder, ACDSGF)
    encoder.eval()
    encoder.ablation_mode = "full"
    if budget_type == "fixed_k":
        encoder.fixed_k = int(K)
        encoder.budget_ratio = None
    else:
        encoder.fixed_k = None
        encoder.budget_ratio = float(ratio)

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) * 1009 + N * 17 + (K or 0) * 3 + int((ratio or 0) * 1000))
    scale = _spawn_scale(N)
    vbs: list[float] = []
    degrees: list[float] = []
    k_caps: list[float] = []
    rhos: list[float] = []
    degree_vecs: list[np.ndarray] = []

    for t in range(n_steps):
        pos = (torch.rand(1, N, 2, generator=g) * 2.0 - 1.0) * scale
        vel = (torch.rand(1, N, 2, generator=g) * 2.0 - 1.0) * 0.05
        obs = torch.zeros(1, N, obs_dim)
        obs[..., :2] = pos
        obs[..., 2:4] = vel
        # remaining dims stay 0 (goals / extras unused for topology scores)
        pos_d = pos.to(device)
        obs_d = obs.to(device)
        _, _, diag = encoder(obs_d, positions=pos_d, apply_budget=True)
        vb = _as_float(diag["V_B"])
        vbs.append(vb)
        degrees.append(_as_float(diag["mean_degree"]))
        rhos.append(_as_float(diag["rho_t"]))
        kc = diag.get("k_caps")
        if kc is not None and torch.is_tensor(kc):
            k_caps.append(float(kc.float().mean().item()))
            d = diag.get("degrees")
            if d is not None and torch.is_tensor(d):
                degree_vecs.append(d.detach().float().cpu().numpy().reshape(-1))

    T = len(vbs)
    vmax = float(max(vbs)) if vbs else float("nan")
    vr = float(sum(1 for v in vbs if v > 0) / T) if T else float("nan")
    avg_d = float(np.mean(degrees)) if degrees else float("nan")
    avg_cap = float(np.mean(k_caps)) if k_caps else float("nan")
    avg_rho = float(np.mean(rhos)) if rhos else float("nan")
    out = {
        "avg_degree": avg_d,
        "avg_k_cap": avg_cap,
        "edge_density": avg_rho,
        "Vmax": vmax,
        "violation_rate": vr,
        "n_steps": T,
        "degree_std": float(np.std(degrees)) if degrees else float("nan"),
        "eval_mode": "geom",
    }
    if budget_type == "fixed_k" and K is not None:
        out["degree_error_vs_K"] = abs(avg_d - float(K))
        out["degree_error_vs_cap"] = abs(avg_d - avg_cap) if np.isfinite(avg_cap) else float("nan")
    else:
        out["degree_error_vs_K"] = ""
        out["degree_error_vs_cap"] = abs(avg_d - avg_cap) if np.isfinite(avg_cap) else float("nan")
    if degree_vecs:
        flat = np.concatenate(degree_vecs)
        out["degree_p50"] = float(np.median(flat))
        out["degree_p95"] = float(np.percentile(flat, 95))
    return out


# ---------------------------------------------------------------------------
# Synthetic projection (debug / Gate 1 style) — no env
# ---------------------------------------------------------------------------


def _random_mask(B: int, N: int, p: float, gen: torch.Generator) -> torch.Tensor:
    m = (torch.rand(B, N, N, generator=gen) < p).float()
    eye = torch.eye(N).unsqueeze(0)
    return m * (1.0 - eye)


def run_synthetic(
    sizes: list[int],
    Ks: list[int],
    ratios: list[float],
    trials: int = 64,
) -> list[dict]:
    rows = []
    for N in sizes:
        for K in Ks:
            gen = torch.Generator().manual_seed(1000 + N * 17 + K)
            vmax, viol, degs, rhos, caps = 0.0, 0, [], [], []
            for _ in range(trials):
                mask = _random_mask(2, N, p=0.5, gen=gen)
                scores = torch.rand(2, N, N, generator=gen)
                A = hard_adjacency(apply_topk_fixed_k(scores, mask, K))
                feas = degree_budget_violation(A, mask, k_fixed=K)
                vb = float(feas["V_B"])
                vmax = max(vmax, vb)
                if vb > 0:
                    viol += 1
                degs.append(float(feas["mean_degree"]))
                rhos.append(float(feas["rho"]))
                caps.append(float(feas["k_caps"].float().mean().item()))
            T = trials
            rows.append(
                {
                    "source": "synthetic",
                    "N": N,
                    "budget_type": "fixed_k",
                    "K": K,
                    "ratio": "",
                    "avg_degree": float(np.mean(degs)),
                    "avg_k_cap": float(np.mean(caps)),
                    "degree_error_vs_K": abs(float(np.mean(degs)) - K),
                    "degree_error_vs_cap": abs(float(np.mean(degs)) - float(np.mean(caps))),
                    "edge_density": float(np.mean(rhos)),
                    "Vmax": vmax,
                    "violation_rate": viol / T,
                    "n_steps": T,
                    "seed": -1,
                }
            )
        for ratio in ratios:
            gen = torch.Generator().manual_seed(2000 + N * 17 + int(ratio * 1000))
            vmax, viol, degs, rhos, caps = 0.0, 0, [], [], []
            for _ in range(trials):
                mask = _random_mask(2, N, p=0.5, gen=gen)
                scores = torch.rand(2, N, N, generator=gen)
                A = hard_adjacency(apply_topk_budget(scores, mask, ratio))
                feas = degree_budget_violation(A, mask, budget_ratio=ratio)
                vb = float(feas["V_B"])
                vmax = max(vmax, vb)
                if vb > 0:
                    viol += 1
                degs.append(float(feas["mean_degree"]))
                rhos.append(float(feas["rho"]))
                caps.append(float(feas["k_caps"].float().mean().item()))
            T = trials
            rows.append(
                {
                    "source": "synthetic",
                    "N": N,
                    "budget_type": "ratio",
                    "K": "",
                    "ratio": ratio,
                    "avg_degree": float(np.mean(degs)),
                    "avg_k_cap": float(np.mean(caps)),
                    "degree_error_vs_K": "",
                    "degree_error_vs_cap": abs(float(np.mean(degs)) - float(np.mean(caps))),
                    "edge_density": float(np.mean(rhos)),
                    "Vmax": vmax,
                    "violation_rate": viol / T,
                    "n_steps": T,
                    "seed": -1,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Encoder / env rollouts (formal evidence)
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_budget_rollout(
    env,
    policy,
    *,
    budget_type: str,
    K: int | None,
    ratio: float | None,
    episodes: int,
    max_steps: int,
) -> dict:
    adapter = get_adapter(policy)
    adapter.set_ablation_mode("full")
    if budget_type == "fixed_k":
        adapter.set_fixed_k(int(K))
        adapter.set_budget_ratio(None)
    else:
        adapter.set_fixed_k(None)
        adapter.set_budget_ratio(float(ratio))

    vbs: list[float] = []
    degrees: list[float] = []
    k_caps: list[float] = []
    rhos: list[float] = []
    degree_vecs: list[np.ndarray] = []

    for _ in range(episodes):
        td = env.reset()
        for _ in range(max_steps):
            td = policy(td)
            diag = adapter.last_topology_diag
            if diag is None:
                raise RuntimeError("last_topology_diag missing — budget wiring broken")
            vb = _as_float(diag["V_B"])
            vbs.append(vb)
            degrees.append(_as_float(diag["mean_degree"]))
            rhos.append(_as_float(diag["rho_t"]))
            kc = diag.get("k_caps")
            if kc is not None and torch.is_tensor(kc):
                k_caps.append(float(kc.float().mean().item()))
                d = diag.get("degrees")
                if d is not None and torch.is_tensor(d):
                    degree_vecs.append(d.detach().float().cpu().numpy().reshape(-1))
            td = env.step(td)
            done = td.get(("next", "done"))
            if done is not None and bool(done.any()):
                break
            td = step_mdp(td)

    T = len(vbs)
    vmax = float(max(vbs)) if vbs else float("nan")
    vr = float(sum(1 for v in vbs if v > 0) / T) if T else float("nan")
    avg_d = float(np.mean(degrees)) if degrees else float("nan")
    avg_cap = float(np.mean(k_caps)) if k_caps else float("nan")
    avg_rho = float(np.mean(rhos)) if rhos else float("nan")
    out = {
        "avg_degree": avg_d,
        "avg_k_cap": avg_cap,
        "edge_density": avg_rho,
        "Vmax": vmax,
        "violation_rate": vr,
        "n_steps": T,
        "degree_std": float(np.std(degrees)) if degrees else float("nan"),
        "eval_mode": "env",
    }
    if budget_type == "fixed_k" and K is not None:
        out["degree_error_vs_K"] = abs(avg_d - float(K))
        out["degree_error_vs_cap"] = abs(avg_d - avg_cap) if np.isfinite(avg_cap) else float("nan")
    else:
        out["degree_error_vs_K"] = ""
        out["degree_error_vs_cap"] = abs(avg_d - avg_cap) if np.isfinite(avg_cap) else float("nan")
    if degree_vecs:
        flat = np.concatenate(degree_vecs)
        out["degree_p50"] = float(np.median(flat))
        out["degree_p95"] = float(np.percentile(flat, 95))
    return out


def run_encoder_campaign(
    sizes: list[int],
    Ks: list[int],
    ratios: list[float],
    seeds: list[int],
    episodes: int,
    device: str,
    max_steps: int | None = None,
    env_max_n: int = 16,
    geom_steps: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Env rollouts for N≤env_max_n; geometric encoder samples for larger N.

    Theorem 1 only requires learned scores + hard Π; geom path avoids VMAS cost at N=32/64.
    """
    budget_rows: list[dict] = []
    degree_rows: list[dict] = []
    ckpts = list_ckpts(seeds)
    g_steps = geom_steps if geom_steps is not None else max(200, episodes * (max_steps or 50))

    for seed, ckpt in ckpts:
        set_seed(seed)
        # Always load N=16 once for encoder weights + obs_dim (geom path)
        print(f"[encoder] seed={seed} loading backbone N=16…", flush=True)
        cfg16, env16, policy16 = load_frozen(16, seed, ckpt, device=device)
        encoder = get_encoder(policy16)
        obs_dim = int(cfg16.get("_obs_dim") or 0)
        if obs_dim <= 0:
            # infer from env observation
            td0 = env16.reset()
            obs0 = td0.get(("agents", "observation"))
            obs_dim = int(obs0.shape[-1])
        ms_default = max_steps if max_steps is not None else int(cfg16.get("env", {}).get("max_steps", 100) or 100)

        for N in sizes:
            use_env = N <= env_max_n
            if use_env:
                if N == 16:
                    env, policy = env16, policy16
                else:
                    print(f"[encoder] seed={seed} N={N} env loading…", flush=True)
                    _, env, policy = load_frozen(N, seed, ckpt, device=device)
                ms = ms_default
            else:
                print(f"[encoder] seed={seed} N={N} geom steps={g_steps}…", flush=True)
                env, policy = None, None

            settings: list[tuple[str, int | None, float | None]] = [
                ("fixed_k", K, None) for K in Ks
            ] + [("ratio", None, float(r)) for r in ratios]

            for budget_type, K, ratio in settings:
                if use_env:
                    r = collect_budget_rollout(
                        env,
                        policy,
                        budget_type=budget_type,
                        K=K,
                        ratio=ratio,
                        episodes=episodes,
                        max_steps=ms,
                    )
                else:
                    r = collect_encoder_geom(
                        encoder,
                        N=N,
                        obs_dim=obs_dim,
                        budget_type=budget_type,
                        K=K,
                        ratio=ratio,
                        n_steps=g_steps,
                        device=device,
                        seed=seed,
                    )
                row = {
                    "source": "encoder",
                    "N": N,
                    "budget_type": budget_type,
                    "K": K if budget_type == "fixed_k" else "",
                    "ratio": ratio if budget_type == "ratio" else "",
                    "seed": seed,
                    **r,
                }
                budget_rows.append(row)
                degree_rows.append(
                    {
                        "source": "encoder",
                        "N": N,
                        "budget_type": budget_type,
                        "K": K if budget_type == "fixed_k" else "",
                        "ratio": ratio if budget_type == "ratio" else "",
                        "seed": seed,
                        "avg_degree": r["avg_degree"],
                        "avg_k_cap": r["avg_k_cap"],
                        "degree_std": r["degree_std"],
                        "degree_error_vs_K": r["degree_error_vs_K"],
                        "degree_error_vs_cap": r["degree_error_vs_cap"],
                        "degree_p50": r.get("degree_p50", ""),
                        "degree_p95": r.get("degree_p95", ""),
                        "eval_mode": r.get("eval_mode", ""),
                    }
                )
                status = "PASS" if r["Vmax"] == 0.0 and r["violation_rate"] == 0.0 else "FAIL"
                tag = f"K={K}" if budget_type == "fixed_k" else f"rho_B={ratio}"
                print(
                    f"  [{status}] {r.get('eval_mode')} N={N} {budget_type} {tag} seed={seed}  "
                    f"Vmax={r['Vmax']:.4g} VR={r['violation_rate']:.4g}  "
                    f"d={r['avg_degree']:.3f} rho={r['edge_density']:.4f}",
                    flush=True,
                )
                _flush_csv(budget_rows, degree_rows)

            if use_env and N != 16:
                del env, policy
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()

        del env16, policy16, encoder
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    return budget_rows, degree_rows


BUDGET_FIELDS = [
    "source",
    "N",
    "budget_type",
    "K",
    "ratio",
    "seed",
    "eval_mode",
    "avg_degree",
    "avg_k_cap",
    "degree_error_vs_K",
    "degree_error_vs_cap",
    "edge_density",
    "Vmax",
    "violation_rate",
    "n_steps",
]

DEGREE_FIELDS = [
    "source",
    "N",
    "budget_type",
    "K",
    "ratio",
    "seed",
    "avg_degree",
    "avg_k_cap",
    "degree_std",
    "degree_error_vs_K",
    "degree_error_vs_cap",
    "degree_p50",
    "degree_p95",
]


def _flush_csv(budget_rows: list[dict], degree_rows: list[dict]) -> None:
    _write_csv(OUT / "budget_statistics.csv", budget_rows, BUDGET_FIELDS)
    _write_csv(OUT / "degree_statistics.csv", degree_rows, DEGREE_FIELDS)


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def aggregate_encoder(budget_rows: list[dict]) -> list[dict]:
    """Mean over seeds for Table I."""
    from collections import defaultdict

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in budget_rows:
        if r.get("source") != "encoder":
            continue
        key = (int(r["N"]), r["budget_type"], str(r["K"]), str(r["ratio"]))
        groups[key].append(r)

    agg = []
    for (N, btype, K, ratio), rows in sorted(groups.items()):
        vmax = max(float(r["Vmax"]) for r in rows)
        vr = max(float(r["violation_rate"]) for r in rows)  # any seed violation fails
        agg.append(
            {
                "N": N,
                "budget_type": btype,
                "K": K if btype == "fixed_k" else "",
                "ratio": ratio if btype == "ratio" else "",
                "avg_degree": float(np.mean([float(r["avg_degree"]) for r in rows])),
                "edge_density": float(np.mean([float(r["edge_density"]) for r in rows])),
                "Vmax": vmax,
                "violation_rate": vr,
                "n_seeds": len(rows),
                "pass": vmax == 0.0 and vr == 0.0,
            }
        )
    return agg


def plot_fig4(agg_rows: list[dict]) -> Path | None:
    """rho vs N for fixed-K (no scalability claim)."""
    fixed = [r for r in agg_rows if r["budget_type"] == "fixed_k" and r.get("pass")]
    if not fixed:
        return None
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    Ks = sorted({int(float(r["K"])) for r in fixed if r["K"] != ""})
    for K in Ks:
        pts = sorted(
            [r for r in fixed if r["K"] != "" and int(float(r["K"])) == K],
            key=lambda x: int(x["N"]),
        )
        if not pts:
            continue
        xs = [int(r["N"]) for r in pts]
        ys = [float(r["edge_density"]) for r in pts]
        ax.plot(xs, ys, marker="o", label=f"K={K}")
    # O(1/N) guide for K=4 if present
    Ns = sorted({int(r["N"]) for r in fixed})
    if Ns:
        guide_N = np.array(Ns, dtype=float)
        # rho ≈ K / (N-1) for out-degree K
        ax.plot(guide_N, 4.0 / (guide_N - 1.0), "k--", alpha=0.4, label=r"$4/(N-1)$ guide")
    ax.set_xlabel(r"Swarm size $N$")
    ax.set_ylabel(r"Communication density $\rho$")
    ax.set_title("Communication density under increasing swarm size")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = FIG_PAPER / "Fig4_communication_scaling.png"
    fig.savefig(path, dpi=200)
    fig.savefig(FIG_OUT / "Fig4_communication_scaling.png", dpi=200)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description="T-RO §6.2 budget feasibility statistics")
    ap.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    ap.add_argument("--K", type=int, nargs="+", default=list(DEFAULT_K))
    ap.add_argument("--ratios", type=float, nargs="+", default=list(DEFAULT_RATIOS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--env-max-n",
        type=int,
        default=16,
        help="Use VMAS env rollouts for N<=this; geom encoder samples for larger N",
    )
    ap.add_argument(
        "--geom-steps",
        type=int,
        default=None,
        help="Geometric sample count for N>env-max-n (default: max(200, ep*max_steps))",
    )
    ap.add_argument(
        "--phase",
        choices=("synthetic", "encoder", "all"),
        default="all",
        help="synthetic=projection math only; encoder=frozen ckpt rollouts; all=both",
    )
    ap.add_argument("--synthetic-trials", type=int, default=64)
    args = ap.parse_args()

    device = resolve_device(args.device)
    run_dir = create_run_dir(prefix="tro_evidence_6_2_budget")
    write_config(
        run_dir,
        {
            "section": "6.2",
            "purpose": "budget_feasibility_table",
            "sizes": args.sizes,
            "K": args.K,
            "ratios": args.ratios,
            "seeds": args.seeds,
            "episodes": args.episodes,
            "env_max_n": args.env_max_n,
            "geom_steps": args.geom_steps,
            "device": device,
            "phase": args.phase,
            "training": False,
        },
    )
    print(f"device={device} | run={run_dir}", flush=True)
    print(
        f"sizes={args.sizes} K={args.K} ratios={args.ratios} ep={args.episodes} "
        f"env_max_n={args.env_max_n}",
        flush=True,
    )

    all_budget: list[dict] = []
    all_degree: list[dict] = []

    if args.phase in ("synthetic", "all"):
        print("=== Phase synthetic ===", flush=True)
        syn = run_synthetic(args.sizes, args.K, args.ratios, trials=args.synthetic_trials)
        for r in syn:
            r.setdefault("eval_mode", "synthetic")
        all_budget.extend(syn)
        for r in syn:
            status = "PASS" if r["Vmax"] == 0.0 and r["violation_rate"] == 0.0 else "FAIL"
            tag = f"K={r['K']}" if r["budget_type"] == "fixed_k" else f"rho={r['ratio']}"
            print(f"  [{status}] synthetic N={r['N']} {r['budget_type']} {tag} Vmax={r['Vmax']}", flush=True)
        _flush_csv(all_budget, all_degree)

    if args.phase in ("encoder", "all"):
        print("=== Phase encoder (frozen uav16) ===", flush=True)
        enc_b, enc_d = run_encoder_campaign(
            args.sizes,
            args.K,
            args.ratios,
            args.seeds,
            args.episodes,
            device,
            max_steps=args.max_steps,
            env_max_n=args.env_max_n,
            geom_steps=args.geom_steps,
        )
        all_budget.extend(enc_b)
        all_degree.extend(enc_d)
        _flush_csv(all_budget, all_degree)

    agg = aggregate_encoder(all_budget)
    all_pass = all(
        float(r["Vmax"]) == 0.0 and float(r["violation_rate"]) == 0.0
        for r in all_budget
        if r.get("source") == "encoder" or args.phase == "synthetic"
    )
    # stricter: every row
    all_pass = all(
        float(r["Vmax"]) == 0.0 and float(r["violation_rate"]) == 0.0 for r in all_budget
    )

    fig4 = plot_fig4(agg) if agg else None

    report = {
        "all_pass": all_pass,
        "n_budget_rows": len(all_budget),
        "table_I_aggregate": agg,
        "fig4": str(fig4) if fig4 else None,
        "run_dir": str(run_dir),
        "claim": (
            "The projection layer consistently satisfies the predefined communication "
            "budgets across different swarm sizes and constraints, demonstrating the "
            "practical feasibility of the proposed constrained topology decision mechanism."
        ),
    }
    (OUT / "evidence_6_2_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (run_dir / "metrics" / "evidence_6_2_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("=== §6.2 gate:", "PASS" if all_pass else "FAIL", "===")
    print("Wrote", OUT / "budget_statistics.csv")
    print("Wrote", OUT / "evidence_6_2_report.json")
    if fig4:
        print("Wrote", fig4)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()

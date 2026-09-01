# -*- coding: utf-8 -*-
"""§6.3 Formal theory evidence (frozen checkpoints, no tuning).

Protocol freeze:
  G*  = full-support reference topology BEFORE budget projection (pre-Π scores)
  Fig1: D_G → ε_G + linear fit (α, R²)     — Lemma 2 trend
  Fig2: ε_G → ΔA_γ                          — Lemma 1 / discounted action discrepancy
        ΔA_γ := Σ_t γ^t ||a*_t - a_t||      — NOT task return gap
  Fig3: B → (J, success, C)                 — Pareto (E0)
  K   ∈ {1,2,4,6} by default (avoid N=16 radius saturation at K≥8)
  Multi-seed over frozen uav16 checkpoints

Claim hygiene: consistent with theory — does not prove theorems.
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
    get_adapter,
    list_ac_dsgf_ckpts as list_ckpts,
    load_frozen_ac_dsgf as load_frozen,
    resolve_device,
)

from algorithms.guided.mappo_guided import ACGuideAdapter, build_guided_mappo
from models.ac_dsgf import ACDSGF
from models.residual_policy import ResidualGuidanceActor
from tro.twin.action_compare import action_mean_l2, actions_from_residual_actor
from tro.twin.message_compare import message_mean_l2
from utils.experiment import load_experiment_config
from utils.seed import set_seed
from utils.tro_run import create_run_dir, write_config

OUT = ROOT / "paper" / "ac_dsgf_tro" / "experiments" / "evidence_6_3_formal"
FIG = ROOT / "paper" / "ac_dsgf_tro" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

DEFAULT_SEEDS = (1234, 2026, 3407, 42, 8888)
DEFAULT_K = (1, 2, 4, 6)  # avoid K-saturation on N=16 radius neighborhoods


def get_encoder(policy) -> ACDSGF:
    return get_adapter(policy).encoder  # type: ignore[return-value]


def get_residual_actor(policy) -> ResidualGuidanceActor | None:
    for m in policy.modules():
        if isinstance(m, ResidualGuidanceActor):
            return m
    return None


def frobenius_adj(A_star: torch.Tensor, A_t: torch.Tensor) -> float:
    d = (A_star - A_t).reshape(A_star.shape[0], -1).norm(dim=-1)
    return float(d.mean().item())


def linear_fit(x: np.ndarray, y: np.ndarray) -> dict:
    """ε_G ≈ α D_G + β ; report slope, intercept, R²."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return {"alpha": float("nan"), "beta": float("nan"), "R2": float("nan"), "n": int(len(x))}
    A = np.vstack([x, np.ones_like(x)]).T
    alpha, beta = np.linalg.lstsq(A, y, rcond=None)[0]
    y_hat = alpha * x + beta
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return {"alpha": float(alpha), "beta": float(beta), "R2": float(r2), "n": int(len(x))}


@torch.no_grad()
def twin_metrics_on_obs(encoder, actor, obs, *, k_fixed: int, gamma: float = 0.99) -> dict:
    twin = encoder.twin_forward(obs, sparse_mode="fixed_k", k_fixed=k_fixed)
    eps = message_mean_l2(twin["full"]["message"], twin["sparse"]["message"])
    d_g = frobenius_adj(twin["full"]["A_t"], twin["sparse"]["A_t"])
    delta_a = None
    if actor is not None:
        a_f, a_s = actions_from_residual_actor(
            actor, twin["ctx"]["obs"], twin["full"]["phi"], twin["sparse"]["phi"]
        )
        delta_a = action_mean_l2(a_f, a_s)
    return {
        "epsilon_G": eps,
        "D_G": d_g,
        "delta_a": delta_a,
        "mean_degree": float((twin["sparse"]["A_t"] > 0).float().sum(dim=-1).mean()),
    }


@torch.no_grad()
def collect_twin_trajectory(env, policy, encoder, actor, *, k_fixed, max_steps, gamma):
    adapter = get_adapter(policy)
    adapter.set_ablation_mode("full")
    adapter.set_fixed_k(k_fixed)
    adapter.set_budget_ratio(None)

    td = env.reset()
    rows = []
    delta_A_gamma = 0.0
    ep_reward = 0.0
    for t in range(max_steps):
        obs = td.get(("agents", "observation"))
        obs_b = obs.unsqueeze(0) if obs.dim() == 2 else obs
        m = twin_metrics_on_obs(encoder, actor, obs_b, k_fixed=k_fixed, gamma=gamma)
        rows.append({"t": t, **m})
        if m["delta_a"] is not None:
            delta_A_gamma += (gamma**t) * m["delta_a"]

        td = policy(td)
        td = env.step(td)
        rew = td.get(("next", "agents", "reward"))
        if rew is not None:
            ep_reward += float(rew.mean().item())
        done = td.get(("next", "done"))
        if done is not None and bool(done.any()):
            break
        td = step_mdp(td)

    eps = np.array([r["epsilon_G"] for r in rows], dtype=np.float64)
    dgs = np.array([r["D_G"] for r in rows], dtype=np.float64)
    das = np.array([r["delta_a"] for r in rows if r["delta_a"] is not None], dtype=np.float64)
    return {
        "K": k_fixed,
        "mean_epsilon_G": float(eps.mean()) if len(eps) else float("nan"),
        "mean_D_G": float(dgs.mean()) if len(dgs) else float("nan"),
        "mean_delta_a": float(das.mean()) if len(das) else float("nan"),
        "Delta_A_gamma": float(delta_A_gamma),
        "closed_loop_reward": ep_reward,
        "steps": rows,
    }


@torch.no_grad()
def closed_loop_eval(env, policy, *, k_fixed, budget_ratio, episodes, max_steps, success_thr):
    adapter = get_adapter(policy)
    adapter.set_ablation_mode("full")
    if k_fixed is not None:
        adapter.set_fixed_k(k_fixed)
        adapter.set_budget_ratio(None)
    elif budget_ratio is not None and budget_ratio < 1.0:
        adapter.set_fixed_k(None)
        adapter.set_budget_ratio(budget_ratio)
    else:
        adapter.set_fixed_k(None)
        adapter.set_budget_ratio(None)
        # pre-projection full-support reference (all radius edges), not "infinite comm"
        adapter.set_ablation_mode("no_budget")

    rewards, successes, costs = [], [], []
    for _ in range(episodes):
        td = env.reset()
        ep_r, ep_s, ep_c, n_c = 0.0, 0.0, 0.0, 0
        for _ in range(max_steps):
            td = policy(td)
            diag = get_adapter(policy).last_topology_diag
            if diag is not None and "C_t" in diag:
                ep_c += float(diag["C_t"])
                n_c += 1
            td = env.step(td)
            obs = td.get(("next", "agents", "observation"))
            if obs is not None:
                goal = obs[0, :, 4:6] if obs.dim() == 3 else obs[:, 4:6]
                ep_s = float((goal.norm(dim=-1) < success_thr).float().mean())
            rew = td.get(("next", "agents", "reward"))
            if rew is not None:
                ep_r += float(rew.mean().item())
            done = td.get(("next", "done"))
            if done is not None and bool(done.any()):
                break
            td = step_mdp(td)
        rewards.append(ep_r)
        successes.append(ep_s)
        costs.append(ep_c / max(n_c, 1))
    return {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "success_mean": float(np.mean(successes)),
        "success_std": float(np.std(successes)),
        "comm_cost_mean": float(np.mean(costs)),
        "comm_cost_std": float(np.std(costs)),
    }


def run_seed_AB(env, policy, encoder, actor, Ks, episodes, max_steps, gamma, seed):
    points, summary = [], []
    for K in Ks:
        ep_stats = []
        for ep in range(episodes):
            traj = collect_twin_trajectory(
                env, policy, encoder, actor, k_fixed=K, max_steps=max_steps, gamma=gamma
            )
            ep_stats.append(traj)
            for step in traj["steps"]:
                points.append(
                    {
                        "seed": seed,
                        "K": K,
                        "episode": ep,
                        "D_G": step["D_G"],
                        "epsilon_G": step["epsilon_G"],
                        "delta_a": step["delta_a"],
                        "t": step["t"],
                    }
                )
        row = {
            "seed": seed,
            "K": K,
            "mean_D_G": float(np.mean([e["mean_D_G"] for e in ep_stats])),
            "mean_epsilon_G": float(np.mean([e["mean_epsilon_G"] for e in ep_stats])),
            "mean_delta_a": float(np.mean([e["mean_delta_a"] for e in ep_stats])),
            "mean_Delta_A_gamma": float(np.mean([e["Delta_A_gamma"] for e in ep_stats])),
            "mean_closed_loop_reward": float(np.mean([e["closed_loop_reward"] for e in ep_stats])),
        }
        summary.append(row)
        print(
            f"  [A] seed={seed} K={K}: D_G={row['mean_D_G']:.4f} "
            f"ε_G={row['mean_epsilon_G']:.4f} ΔAγ={row['mean_Delta_A_gamma']:.4f}"
        )
    return summary, points


def run_seed_C(env, policy, Ks, fractions, episodes, max_steps, thr, seed):
    rows = []
    full = closed_loop_eval(
        env, policy, k_fixed=None, budget_ratio=1.0, episodes=episodes, max_steps=max_steps, success_thr=thr
    )
    rows.append({"seed": seed, "mode": "full_support_ref", "K": "", "fraction": 1.0, **full})
    print(f"  [C] seed={seed} full-support ref: R={full['reward_mean']:.3f} S={full['success_mean']:.3f}")

    for K in Ks:
        r = closed_loop_eval(
            env, policy, k_fixed=K, budget_ratio=None, episodes=episodes, max_steps=max_steps, success_thr=thr
        )
        frac = r["comm_cost_mean"] / max(full["comm_cost_mean"], 1e-8)
        rows.append({"seed": seed, "mode": "fixed_k", "K": K, "fraction": frac, **r})
        print(f"  [C] seed={seed} K={K}: R={r['reward_mean']:.3f} S={r['success_mean']:.3f} C={r['comm_cost_mean']:.2f}")

    for frac in fractions:
        if frac >= 1.0:
            continue
        r = closed_loop_eval(
            env, policy, k_fixed=None, budget_ratio=frac, episodes=episodes, max_steps=max_steps, success_thr=thr
        )
        rows.append({"seed": seed, "mode": "ratio", "K": "", "fraction": frac, **r})
    return rows


def aggregate_AB(per_seed_summary: list[dict]) -> tuple[list[dict], dict]:
    """Mean±std across seeds for each K; linear fit on seed-mean points."""
    by_k: dict[int, list] = {}
    for r in per_seed_summary:
        by_k.setdefault(int(r["K"]), []).append(r)
    agg = []
    for K in sorted(by_k):
        rows = by_k[K]
        def mstd(key):
            vals = np.array([x[key] for x in rows], dtype=np.float64)
            return float(vals.mean()), float(vals.std())

        dg_m, dg_s = mstd("mean_D_G")
        eg_m, eg_s = mstd("mean_epsilon_G")
        da_m, da_s = mstd("mean_delta_a")
        ag_m, ag_s = mstd("mean_Delta_A_gamma")
        agg.append(
            {
                "K": K,
                "mean_D_G": dg_m,
                "std_D_G": dg_s,
                "mean_epsilon_G": eg_m,
                "std_epsilon_G": eg_s,
                "mean_delta_a": da_m,
                "std_delta_a": da_s,
                "mean_Delta_A_gamma": ag_m,
                "std_Delta_A_gamma": ag_s,
                "n_seeds": len(rows),
            }
        )
    fit = linear_fit(
        np.array([r["mean_D_G"] for r in agg]),
        np.array([r["mean_epsilon_G"] for r in agg]),
    )
    # also fit on all seed-level K means
    fit_all = linear_fit(
        np.array([r["mean_D_G"] for r in per_seed_summary]),
        np.array([r["mean_epsilon_G"] for r in per_seed_summary]),
    )
    return agg, {"fit_on_K_means": fit, "fit_on_seed_K": fit_all}


def aggregate_C(per_seed_C: list[dict]) -> list[dict]:
    """Group by mode+K (fixed_k) or mode+rounded ratio; fraction varies slightly per seed."""
    groups: dict[tuple, list] = {}
    for r in per_seed_C:
        mode = r["mode"]
        if mode == "fixed_k":
            key = ("fixed_k", str(int(float(r["K"]))), None)
        elif mode == "ratio":
            key = ("ratio", "", round(float(r["fraction"]), 1))
        else:
            key = ("full_support_ref", "", 1.0)
        groups.setdefault(key, []).append(r)
    out = []
    for (mode, K, frac), rows in sorted(
        groups.items(),
        key=lambda x: (
            0 if x[0][0] == "full_support_ref" else 1 if x[0][0] == "fixed_k" else 2,
            x[0][1] or "0",
            x[0][2] if x[0][2] is not None else 0,
        ),
    ):
        def mstd(key):
            vals = np.array([float(x[key]) for x in rows], dtype=np.float64)
            return float(vals.mean()), float(vals.std(ddof=0))

        rm, rs = mstd("reward_mean")
        sm, ss = mstd("success_mean")
        cm, cs = mstd("comm_cost_mean")
        fracs = [float(x["fraction"]) for x in rows]
        out.append(
            {
                "mode": mode,
                "K": K,
                "fraction": float(np.mean(fracs)) if frac is None else float(frac),
                "reward_mean": rm,
                "reward_std": rs,
                "success_mean": sm,
                "success_std": ss,
                "comm_cost_mean": cm,
                "comm_cost_std": cs,
                "n_seeds": len(rows),
            }
        )
    return out


def plot_fig1(points, agg, fit, out_path: Path):
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    if points:
        ax.scatter(
            [p["D_G"] for p in points],
            [p["epsilon_G"] for p in points],
            s=6,
            alpha=0.15,
            c="#4C78A8",
            label="steps",
        )
    xs = np.array([r["mean_D_G"] for r in agg])
    ys = np.array([r["mean_epsilon_G"] for r in agg])
    xerr = [r["std_D_G"] for r in agg]
    yerr = [r["std_epsilon_G"] for r in agg]
    ax.errorbar(xs, ys, xerr=xerr, yerr=yerr, fmt="o", color="#E45756", capsize=3, label="mean±std by K")
    for r in agg:
        ax.annotate(f"K={r['K']}", (r["mean_D_G"], r["mean_epsilon_G"]), fontsize=8)
    if np.isfinite(fit.get("alpha", np.nan)):
        x_line = np.linspace(0, max(xs.max(), 1e-6), 50)
        y_line = fit["alpha"] * x_line + fit["beta"]
        ax.plot(
            x_line,
            y_line,
            "--",
            color="#B279A2",
            label=rf"fit: $\varepsilon_G={fit['alpha']:.3f}D_G{fit['beta']:+.3f}$ ($R^2={fit['R2']:.3f}$)",
        )
    ax.set_xlabel(r"$D_G=\|A_t-A_t^\star\|_F$")
    ax.set_ylabel(r"$\varepsilon_G=\|M(G^\star)-M(G_t)\|$")
    ax.set_title("Fig.1 Topology–information (Lemma 2 trend)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig2(agg, scatter, out_path: Path, gamma: float):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1))
    ax = axes[0]
    if scatter:
        ax.scatter(
            [p["epsilon_G"] for p in scatter],
            [p["delta_a"] for p in scatter],
            s=6,
            alpha=0.15,
            c="#72B7B2",
        )
    ax.errorbar(
        [r["mean_epsilon_G"] for r in agg],
        [r["mean_delta_a"] for r in agg],
        xerr=[r["std_epsilon_G"] for r in agg],
        yerr=[r["std_delta_a"] for r in agg],
        fmt="o-",
        color="#E45756",
        capsize=3,
    )
    for r in agg:
        ax.annotate(f"K={r['K']}", (r["mean_epsilon_G"], r["mean_delta_a"]), fontsize=8)
    ax.set_xlabel(r"$\varepsilon_G$")
    ax.set_ylabel(r"$\Delta a=\|a^\star-a\|$")
    ax.set_title("Lemma 1: information → action")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    xs = np.array([r["mean_epsilon_G"] for r in agg])
    ys = np.array([r["mean_Delta_A_gamma"] for r in agg])
    ax.errorbar(
        xs,
        ys,
        xerr=[r["std_epsilon_G"] for r in agg],
        yerr=[r["std_Delta_A_gamma"] for r in agg],
        fmt="o-",
        color="#4C78A8",
        capsize=3,
        label=r"$\Delta A_\gamma=\sum_t\gamma^t\|a_t^\star-a_t\|$",
    )
    ax.set_xlabel(r"$\varepsilon_G$")
    ax.set_ylabel(r"$\Delta A_\gamma$ (shared-state)")
    ax.set_title(r"Discounted action discrepancy (not task $\Delta J$)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.suptitle("Fig.2 Information → action discrepancy (E1 twin)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig3(rows, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1))
    ax = axes[0]
    ax.errorbar(
        [r["fraction"] for r in rows],
        [r["reward_mean"] for r in rows],
        yerr=[r["reward_std"] for r in rows],
        fmt="o-",
        color="#4C78A8",
        capsize=3,
        label="reward",
    )
    for r in rows:
        lab = "ref" if r["mode"] == "full_support_ref" else (f"K={r['K']}" if r["K"] not in ("", None) else f"ρ={r['fraction']}")
        ax.annotate(lab, (r["fraction"], r["reward_mean"]), fontsize=7)
    ax.set_xlabel(r"Comm. fraction $\approx C/C_{\mathrm{ref}}$")
    ax.set_ylabel("Episode reward (mean±std over seeds)")
    ax.set_title("Budget → reward")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.errorbar(
        [r["fraction"] for r in rows],
        [r["success_mean"] for r in rows],
        yerr=[r["success_std"] for r in rows],
        fmt="s-",
        color="#E45756",
        capsize=3,
        label="success",
    )
    ax.set_xlabel(r"Comm. fraction $\approx C/C_{\mathrm{ref}}$")
    ax.set_ylabel("Success rate")
    ax.set_title("Budget → success")
    ax.grid(True, alpha=0.3)
    fig.suptitle("Fig.3 Budget–performance Pareto (E0)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="ABC")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=64)
    ap.add_argument("--K", type=int, nargs="+", default=list(DEFAULT_K))
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda | cuda:0 | cpu | auto (default: cuda if available)",
    )
    args = ap.parse_args()

    device = resolve_device(args.device)
    ckpts = list_ckpts(args.seeds)
    run_dir = create_run_dir(prefix="tro_evidence_6_3_formal")
    write_config(
        run_dir,
        {
            "protocol": "formal_6_3",
            "G_star": "full-support reference topology before budget projection",
            "Delta_A_gamma": "sum_t gamma^t ||a*_t - a_t||  (NOT task return gap)",
            "K": args.K,
            "seeds": [s for s, _ in ckpts],
            "episodes": args.episodes,
            "N": args.N,
            "device": device,
            "tuning": False,
            "claim": "consistent_with_theory_not_proof",
        },
    )
    print(
        f"Formal 6.3 | device={device} | seeds={[s for s,_ in ckpts]} | "
        f"K={args.K} | ep={args.episodes}"
    )
    if device.startswith("cuda"):
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Run dir: {run_dir}")

    phase = args.phase.upper()
    all_AB, all_points, all_C = [], [], []

    for seed, ckpt in ckpts:
        set_seed(seed)
        cfg, env, policy = load_frozen(args.N, seed, ckpt, device=device)
        encoder = get_encoder(policy)
        actor = get_residual_actor(policy)
        thr = float(cfg["env"].get("success_threshold", 0.3))
        max_steps = min(args.max_steps, int(cfg["env"].get("max_steps", 128)))
        print(f"=== seed {seed} | {ckpt.name} ===")

        if "A" in phase or "B" in phase:
            summary, points = run_seed_AB(
                env, policy, encoder, actor, args.K, args.episodes, max_steps, args.gamma, seed
            )
            all_AB.extend(summary)
            all_points.extend(points)
            _write_csv(run_dir / "phaseAB_per_seed.csv", all_AB)
            _write_csv(run_dir / "phaseA_points.csv", all_points)
        if "C" in phase:
            all_C.extend(
                run_seed_C(env, policy, args.K, args.fractions, args.episodes, max_steps, thr, seed)
            )
            _write_csv(run_dir / "phaseC_per_seed.csv", all_C)

    report = {
        "protocol": "formal_6_3",
        "G_star_definition": "full-support reference topology before budget projection",
        "note": "Fig2 reports discounted action discrepancy ΔA_γ, not task return gap. "
        "Consistent with Lemma 1 / Lemma 2 trends; does not prove theorems.",
        "seeds": [s for s, _ in ckpts],
        "K": args.K,
        "episodes_per_seed": args.episodes,
    }

    if all_AB:
        _write_csv(run_dir / "phaseAB_per_seed.csv", all_AB)
        _write_csv(run_dir / "phaseA_points.csv", all_points)
        agg, fits = aggregate_AB(all_AB)
        _write_csv(run_dir / "phaseAB_agg.csv", agg)
        report["phaseAB_agg"] = agg
        report["linear_fit_Lemma2"] = fits
        fit = fits["fit_on_K_means"]
        plot_fig1(all_points, agg, fit, FIG / "Fig1_topology_information.png")
        plot_fig1(all_points, agg, fit, run_dir / "Fig1_topology_information.png")
        plot_fig1(all_points, agg, fit, OUT / "Fig1_topology_information.png")
        print(
            "  Fig1 linear fit: alpha={:.4f} beta={:.4f} R^2={:.4f}".format(
                fit["alpha"], fit["beta"], fit["R2"]
            )
        )
        if "B" in phase:
            scatter = [p for p in all_points if p.get("delta_a") is not None]
            _write_csv(
                run_dir / "phaseB_summary.csv",
                [
                    {
                        "K": r["K"],
                        "epsilon_G": r["mean_epsilon_G"],
                        "delta_a": r["mean_delta_a"],
                        "Delta_A_gamma": r["mean_Delta_A_gamma"],
                    }
                    for r in agg
                ],
            )
            for path in (
                FIG / "Fig2_information_action.png",
                run_dir / "Fig2_information_action.png",
                OUT / "Fig2_information_action.png",
                FIG / "Fig2_information_performance.png",
            ):
                plot_fig2(agg, scatter, path, args.gamma)
            print("  saved Fig2 (eps_G -> delta_a / Delta_A_gamma)")

    if all_C:
        _write_csv(run_dir / "phaseC_per_seed.csv", all_C)
        aggC = aggregate_C(all_C)
        _write_csv(run_dir / "phaseC_agg.csv", aggC)
        report["phaseC_agg"] = aggC
        for path in (
            FIG / "Fig3_budget_performance.png",
            run_dir / "Fig3_budget_performance.png",
            OUT / "Fig3_budget_performance.png",
        ):
            plot_fig3(aggC, path)
        print("  saved Fig3")

    out_json = OUT / "evidence_6_3_formal_report.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (run_dir / "evidence_6_3_formal_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("Wrote", out_json)


if __name__ == "__main__":
    main()

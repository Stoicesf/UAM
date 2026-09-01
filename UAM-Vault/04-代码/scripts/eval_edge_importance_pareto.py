# -*- coding: utf-8 -*-
"""Eval-only: Edge Importance (Fig.12) + Communication-Performance Pareto (Fig.13).

No training / no algorithm change. Uses frozen AC-DSGF + DSGF checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchrl.envs.utils import step_mdp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithms.guided.mappo_guided import ACGuideAdapter, build_guided_mappo
from utils.communication_analysis import radius_mask_from_positions, soft_comm_mass
from utils.experiment import load_experiment_config
from utils.seed import set_seed

FIG = ROOT / "paper" / "ac_dsgf_cn" / "figures"
FIG_EN = ROOT / "paper" / "ac_dsgf" / "figures"
TAB = ROOT / "paper" / "tables"
OUT = ROOT / "results" / "ac_dsgf" / "importance_pareto"
for d in (FIG, FIG_EN, TAB, OUT):
    d.mkdir(parents=True, exist_ok=True)


def load_policy(kind: str, n: int, seed: int):
    if kind == "ac":
        cfg_path = ROOT / "configs/ac_dsgf/ac_dsgf_16uav.yaml"
        ckpt_path = ROOT / f"results/ac_dsgf/uav16/s{seed}/checkpoints/final.pt"
    else:
        cfg_path = ROOT / "configs/dsgf/dsgf_v2_frozen.yaml"
        # fallback common locations
        candidates = [
            ROOT / f"results/baseline16_seeds/dsgf/s{seed}/checkpoints/final.pt",
            ROOT / f"results/dsgf/uav16/s{seed}/checkpoints/final.pt",
        ]
        ckpt_path = next((p for p in candidates if p.exists()), candidates[0])
    cfg = load_experiment_config(str(cfg_path))
    cfg["env"]["num_agents"] = n
    cfg["env"]["num_envs"] = 1
    cfg["env"]["device"] = "cpu"
    cfg["env"].setdefault("max_steps", 128)
    # DSGF configs may not have guidance block used by AC builder — AC path preferred
    if kind == "ac":
        components = build_guided_mappo(cfg["train"], cfg["env"], cfg["guidance"])
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        components.policy.load_state_dict(ckpt["policy"], strict=False)
        components.policy.eval()
        return cfg, components.env, components.policy
    # For Pareto DSGF point: try guided builder with dsgf guidance if present
    try:
        components = build_guided_mappo(cfg["train"], cfg["env"], cfg.get("guidance", {}))
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        components.policy.load_state_dict(ckpt["policy"], strict=False)
        components.policy.eval()
        return cfg, components.env, components.policy
    except Exception as e:
        print("DSGF load skipped:", e)
        return cfg, None, None


def rollout_metrics(policy, env, episodes, max_steps, thr, collect_gates=False):
    suc, masses, gates = [], [], []
    with torch.no_grad():
        for _ in range(episodes):
            td = env.reset()
            ep_s, ep_c, n_c = 0.0, 0.0, 0
            for _ in range(max_steps):
                td = policy(td)
                td = env.step(td)
                obs = td.get(("next", "agents", "observation"))
                goal = obs[0, :, 4:6] if obs.dim() == 3 else obs[:, 4:6]
                ep_s = float((goal.norm(dim=-1) < thr).float().mean())
                for m in policy.modules():
                    if isinstance(m, ACGuideAdapter) and m.last_gate_matrix is not None:
                        g = m.last_gate_matrix
                        ep_c += soft_comm_mass(g)
                        n_c += 1
                        if collect_gates:
                            gates.append(g.detach().cpu().numpy().copy())
                        break
                done = td.get(("next", "done"))
                if done is not None and bool(done.any()):
                    break
                td = step_mdp(td)
            suc.append(ep_s)
            masses.append(ep_c / max(n_c, 1))
    g_mean = np.mean(np.stack(gates, axis=0), axis=0) if gates else None
    return float(np.mean(suc)), float(np.mean(masses)), g_mean


def set_multiplier(policy, mul: torch.Tensor | None):
    for m in policy.modules():
        if isinstance(m, ACGuideAdapter):
            m.encoder.eval_gate_multiplier = mul


def edge_importance(policy, env, cfg, episodes_base: int, episodes_ablate: int, n_sample: int, seed: int):
    thr = float(cfg["env"].get("success_threshold", 0.3))
    max_steps = int(cfg["env"].get("max_steps", 128))
    n = int(cfg["env"]["num_agents"])
    set_multiplier(policy, None)
    for m in policy.modules():
        if isinstance(m, ACGuideAdapter):
            m.set_ablation_mode("full")
            m.set_budget_ratio(None)

    s0, c0, g_mean = rollout_metrics(
        policy, env, episodes_base, max_steps, thr, collect_gates=True
    )
    print(f"baseline Success={s0:.4f} SoftMass={c0:.6f}")
    assert g_mean is not None
    # sample edges on support of mean gate
    flat = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            flat.append((float(g_mean[i, j]), i, j))
    flat.sort(reverse=True, key=lambda x: x[0])
    # stratified: top / mid / bottom among positive-ish
    pos = [e for e in flat if e[0] > 0] or flat
    idxs = np.linspace(0, len(pos) - 1, num=min(n_sample, len(pos)), dtype=int)
    chosen = [pos[k] for k in idxs]

    rows = []
    rng = np.random.default_rng(seed)
    for g_val, i, j in chosen:
        mul = torch.ones(n, n)
        mul[i, j] = 0.0
        set_multiplier(policy, mul)
        s, c, _ = rollout_metrics(policy, env, episodes_ablate, max_steps, thr, False)
        dR = s0 - s  # drop in Success when removing edge
        rows.append(
            {
                "i": i,
                "j": j,
                "g": round(g_val, 8),
                "success_ablate": round(s, 4),
                "delta_success": round(dR, 4),
                "soft_mass_ablate": round(c, 6),
            }
        )
        print(f"  zero ({i},{j}) g={g_val:.2e} ΔS={dR:.4f}")

    set_multiplier(policy, None)
    g_arr = np.array([r["g"] for r in rows], dtype=np.float64)
    d_arr = np.array([r["delta_success"] for r in rows], dtype=np.float64)
    if len(g_arr) >= 3 and np.std(g_arr) > 0 and np.std(d_arr) > 0:
        corr = float(np.corrcoef(g_arr, d_arr)[0, 1])
    else:
        corr = float("nan")
    # also Spearman
    try:
        from scipy.stats import spearmanr

        spear = float(spearmanr(g_arr, d_arr).correlation)
    except Exception:
        rg = g_arr.argsort().argsort().astype(np.float64)
        rd = d_arr.argsort().argsort().astype(np.float64)
        spear = float(np.corrcoef(rg, rd)[0, 1]) if np.std(rg) > 0 else float("nan")

    with open(TAB / "table_edge_importance.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    meta = {
        "baseline_success": s0,
        "baseline_soft_mass": c0,
        "pearson_g_vs_deltaS": corr,
        "spearman_g_vs_deltaS": spear,
        "n_edges": len(rows),
    }
    (OUT / "edge_importance.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("correlation pearson", corr, "spearman", spear)

    # Fig.12
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.scatter(g_arr, d_arr, c="#0b6e4f", s=36, alpha=0.85, edgecolors="#063d2c")
    if np.isfinite(corr) and np.std(g_arr) > 0:
        coef = np.polyfit(g_arr, d_arr, 1)
        xs = np.linspace(g_arr.min(), g_arr.max(), 50)
        ax.plot(xs, np.polyval(coef, xs), "--", color="#b35c00", lw=1.5, label="trend")
    ax.set_xlabel(r"Mean gate score $g_{ij}$")
    ax.set_ylabel(r"Task contribution $\Delta S = S - S^{(-ij)}$")
    ax.set_title("Fig.12  Edge Importance Consistency", loc="left", fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.text(
        0.98,
        0.05,
        f"Pearson $\\rho$={corr:.2f}\nSpearman={spear:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#0b6e4f"),
    )
    ax.legend(fontsize=8)
    for dest in (FIG, FIG_EN):
        fig.savefig(dest / "Fig12_edge_importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved Fig12")
    return meta, rows


def pareto_curve(policy, env, cfg, episodes: int):
    """Frozen-policy operating points (budget / Top-K / open) — no λ retrain."""
    thr = float(cfg["env"].get("success_threshold", 0.3))
    max_steps = int(cfg["env"].get("max_steps", 128))
    rows = []

    def _run(name, setup_fn):
        setup_fn()
        s, c, _ = rollout_metrics(policy, env, episodes, max_steps, thr, False)
        rows.append({"point": name, "success": round(s, 4), "soft_mass": round(c, 6)})
        print(name, s, c)

    def soft():
        set_multiplier(policy, None)
        for m in policy.modules():
            if isinstance(m, ACGuideAdapter):
                m.set_ablation_mode("full")
                m.set_budget_ratio(None)

    def open_all():
        set_multiplier(policy, None)
        for m in policy.modules():
            if isinstance(m, ACGuideAdapter):
                m.set_ablation_mode("no_budget")
                m.set_budget_ratio(None)

    def random_drop():
        set_multiplier(policy, None)
        for m in policy.modules():
            if isinstance(m, ACGuideAdapter):
                m.set_ablation_mode("random")
                m.set_budget_ratio(None)

    def budget(rho):
        set_multiplier(policy, None)
        for m in policy.modules():
            if isinstance(m, ACGuideAdapter):
                m.set_ablation_mode("full")
                m.set_budget_ratio(rho)

    _run("AC-soft", soft)
    _run("AC-budget50%", lambda: budget(0.5))
    _run("AC-budget10%", lambda: budget(0.1))
    _run("AC-open(g=A)", open_all)
    _run("AC-random", random_drop)

    # reuse Top-K via budget_layer like theory script
    import models.communication.budget_layer as budget_layer
    from scripts.eval_theory_binding_figures import _apply_topk_fixed_k

    orig = budget_layer.apply_topk_budget
    for K in (1, 2, 3):
        def _fn(scores, adj, ratio, k_fixed=K):
            return _apply_topk_fixed_k(scores, adj, ratio, k_fixed)

        budget_layer.apply_topk_budget = _fn
        def _setup(k=K):
            set_multiplier(policy, None)
            for m in policy.modules():
                if isinstance(m, ACGuideAdapter):
                    m.set_ablation_mode("full")
                    m.set_budget_ratio(0.5)

        _run(f"AC-Top{K}", _setup)
    budget_layer.apply_topk_budget = orig
    soft()

    # Published DSGF / GAT anchors from paper tables (frozen numbers)
    rows.append({"point": "DSGF(ref)", "success": 0.217, "soft_mass": 40.29})
    rows.append({"point": "GAT(ref)", "success": 0.213, "soft_mass": 44.14})

    with open(TAB / "table_comm_pareto_points.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["point", "success", "soft_mass"])
        w.writeheader()
        w.writerows(rows)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for r in rows:
        x, y = r["soft_mass"], 100.0 * r["success"]
        if r["point"].startswith("AC"):
            ax.scatter(x, y, c="#0b6e4f", s=70, zorder=3)
            ax.annotate(r["point"].replace("AC-", ""), (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
        else:
            ax.scatter(x, y, c="#b35c00", s=80, marker="s", zorder=3)
            ax.annotate(r["point"], (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Soft Comm. Mass $C$ (log)")
    ax.set_ylabel("Success (%)")
    ax.set_title(
        "Fig.13  Communication–Performance Pareto (frozen operating points)",
        loc="left",
        fontweight="bold",
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.text(
        0.02,
        0.98,
        "Goal: same Success, AC farther left\n(not Success maximization)",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#0b6e4f"),
    )
    for dest in (FIG, FIG_EN):
        fig.savefig(dest / "Fig13_comm_pareto.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved Fig13")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes", type=int, default=24)
    p.add_argument("--ablate-episodes", type=int, default=12)
    p.add_argument("--n-sample", type=int, default=36)
    p.add_argument("--skip-importance", action="store_true")
    p.add_argument("--skip-pareto", action="store_true")
    args = p.parse_args()
    set_seed(args.seed)

    cfg, env, policy = load_policy("ac", 16, args.seed)
    if not args.skip_importance:
        print("=== Edge importance ===")
        edge_importance(
            policy,
            env,
            cfg,
            episodes_base=args.episodes,
            episodes_ablate=args.ablate_episodes,
            n_sample=args.n_sample,
            seed=args.seed,
        )
    if not args.skip_pareto:
        print("=== Pareto operating points ===")
        pareto_curve(policy, env, cfg, episodes=args.episodes)
    print("Done →", FIG)


if __name__ == "__main__":
    main()

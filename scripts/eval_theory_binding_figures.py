# -*- coding: utf-8 -*-
"""Theory-binding figures + Top-K deploy eval (no algo change / no training).

Produces:
  Fig9  communication density scaling (C vs N, η_N)  [theory]
  Fig10 gate distribution histogram
  Fig11 Top-K deployment Success
  paper/tables/table_comm_density_eta.csv
  paper/tables/table_topk_deploy.csv
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
import models.communication.budget_layer as budget_layer
from utils.communication_analysis import radius_mask_from_positions
from utils.experiment import load_experiment_config
from utils.seed import set_seed

FIG = ROOT / "paper" / "ac_dsgf_cn" / "figures"
TAB = ROOT / "paper" / "tables"
OUT = ROOT / "results" / "ac_dsgf" / "theory_binding"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

# From table_scalability_interpretability.csv (frozen eval)
SCALE = {
    "dsgf": {8: 11.0091, 16: 22.6198, 32: 56.0599},
    "ac_dsgf": {8: 0.0020, 16: 0.0059, 32: 0.0188},
}
SCALE_S = {
    "dsgf": {8: 0.3333, 16: 0.1979, 32: 0.1120},
    "ac_dsgf": {8: 0.4062, 16: 0.2109, 32: 0.1029},
}


def eta(c: float, n: int) -> float:
    return c / (n * (n - 1))


def fig10_scaling_law():
    ns = [8, 16, 32]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))

    # log-log C vs N
    ax = axes[0]
    cd = [SCALE["dsgf"][n] for n in ns]
    ca = [SCALE["ac_dsgf"][n] for n in ns]
    ax.loglog(ns, cd, "s-", color="#b35c00", lw=2.2, markersize=8, label="DSGF")
    ax.loglog(ns, ca, "D-", color="#0b6e4f", lw=2.2, markersize=8, label="AC-DSGF")
    # reference slopes
    nref = np.array([8.0, 32.0])
    ax.loglog(nref, 0.15 * nref**2, "--", color="#888", lw=1, label=r"$\propto N^{2}$")
    ax.loglog(nref, 0.0004 * nref, ":", color="#888", lw=1.2, label=r"$\propto N$")
    ax.set_xlabel(r"Swarm size $N$")
    ax.set_ylabel(r"Soft Comm. Mass $C$")
    ax.set_title("Fig.9a  Communication scaling law", loc="left", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)

    # η_N
    ax = axes[1]
    ed = [eta(SCALE["dsgf"][n], n) for n in ns]
    ea = [eta(SCALE["ac_dsgf"][n], n) for n in ns]
    ax.semilogy(ns, ed, "s-", color="#b35c00", lw=2.2, markersize=8, label="DSGF")
    ax.semilogy(ns, ea, "D-", color="#0b6e4f", lw=2.2, markersize=8, label="AC-DSGF")
    ax.set_xlabel(r"Swarm size $N$")
    ax.set_ylabel(r"Normalized density $\eta_N=C/(N(N-1))$")
    ax.set_title("Fig.9b  Communication density $\\eta_N$", loc="left", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "Fig9_comm_density_scaling.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for n in ns:
        rows.append(
            {
                "N": n,
                "dsgf_C": SCALE["dsgf"][n],
                "dsgf_eta": eta(SCALE["dsgf"][n], n),
                "ac_C": SCALE["ac_dsgf"][n],
                "ac_eta": eta(SCALE["ac_dsgf"][n], n),
                "dsgf_success": SCALE_S["dsgf"][n],
                "ac_success": SCALE_S["ac_dsgf"][n],
            }
        )
    with open(TAB / "table_comm_density_eta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Saved Fig9 (density scaling) + table_comm_density_eta.csv")
    for r in rows:
        print(
            f"  N={r['N']}: η_DSGF={r['dsgf_eta']:.4f}  η_AC={r['ac_eta']:.2e}"
        )


def load_ac(n: int = 16, seed: int = 42):
    cfg = load_experiment_config(str(ROOT / "configs/ac_dsgf/ac_dsgf_16uav.yaml"))
    cfg["env"]["num_agents"] = n
    cfg["env"]["num_envs"] = 1
    cfg["env"]["device"] = "cpu"
    cfg["env"].setdefault("max_steps", 128)
    components = build_guided_mappo(cfg["train"], cfg["env"], cfg["guidance"])
    ckpt = torch.load(
        str(ROOT / f"results/ac_dsgf/uav16/s{seed}/checkpoints/final.pt"),
        map_location="cpu",
        weights_only=False,
    )
    components.policy.load_state_dict(ckpt["policy"], strict=False)
    components.policy.eval()
    return cfg, components.env, components.policy


def collect_gates(policy, env, episodes: int, max_steps: int, thr: float):
    vals = []
    with torch.no_grad():
        for _ in range(episodes):
            td = env.reset()
            for _ in range(max_steps):
                td = policy(td)
                td = env.step(td)
                g = None
                for m in policy.modules():
                    if isinstance(m, ACGuideAdapter):
                        g = m.last_gate_matrix
                        break
                if g is not None:
                    obs = td.get(("next", "agents", "observation"))
                    pos = obs[0, :, :2] if obs.dim() == 3 else obs[:, :2]
                    mask = radius_mask_from_positions(pos, 0.5).to(g.device)
                    g2 = g if g.dim() == 2 else g[0]
                    m2 = mask[0] if mask.dim() == 3 else mask
                    # only radius-support edges (include g≈0); DSGF counterpart = 1 on same set
                    on = m2.bool()
                    if on.any():
                        vals.append(g2[on].detach().cpu().numpy())
                done = td.get(("next", "done"))
                if done is not None and bool(done.any()):
                    break
                td = step_mdp(td)
    return np.concatenate(vals) if vals else np.array([])


def fig9_gate_hist(gate_vals: np.ndarray):
    """Compare DSGF (g≡1 on radius support) vs AC continuous sparse gates."""
    support = np.asarray(gate_vals, dtype=np.float64).ravel()
    n_sup = int(len(support))  # all radius-support edges (incl. g≈0)
    # DSGF: all radius edges fully open
    dsgf = np.ones(max(n_sup, 1), dtype=np.float64)
    ac = support  # keep zeros so edge counts match DSGF panel

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=False)

    ax = axes[0]
    ax.hist(dsgf, bins=[0.95, 1.05], color="#b35c00", alpha=0.9, density=False, label="DSGF $g=1$")
    ax.set_xlim(0.0, 1.15)
    ax.set_xlabel(r"Gate value $g_{ij}$")
    ax.set_ylabel("Edge count")
    ax.set_title("Fig.10a  DSGF (dense on radius)", loc="left", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.text(
        0.05,
        0.95,
        f"mass at $g=1$: {n_sup} edges\n(full open on support)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#b35c00"),
    )

    ax = axes[1]
    # dB-ish log of count: use hist without density for edge count narrative
    ax.hist(ac, bins=50, color="#0b6e4f", alpha=0.85, density=False, label="AC-DSGF $g_{ij}|A_{ij}=1$")
    ax.axvline(0.5, color="#b35c00", ls="--", lw=1.5, label=r"hard $\tau=0.5$")
    ax.axvline(0.01, color="#3d5a80", ls=":", lw=1.5, label=r"soft $\tau=0.01$")
    ax.set_xlabel(r"Gate value $g_{ij}$")
    ax.set_ylabel("Edge count")
    ax.set_title("Fig.10b  AC (selective sparse)", loc="left", fontweight="bold")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    p_hi = float((ac > 0.5).mean()) if len(ac) else 0.0
    p_mid = float(((ac > 0.01) & (ac <= 0.5)).mean()) if len(ac) else 0.0
    # relative ranking among tiny gates still carries structure → Fig.11
    ax.text(
        0.98,
        0.95,
        f"share $g>0.5$: {p_hi:.2%}\n"
        f"share $0.01<g\\leq0.5$: {p_mid:.2%}\n"
        r"(structure via Top-$K$, Fig.11)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#0b6e4f"),
    )

    fig.suptitle(
        "Gate distribution: dense open vs selective sparse (not silence collapse)",
        fontsize=11,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIG / "Fig10_gate_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved Fig10", "n_support", n_sup, "ac_samples", len(ac), "p(g>0.5)", p_hi)


def _apply_topk_fixed_k(scores, adj_mask, budget_ratio, k_fixed: int):
    masked = scores * adj_mask.float()
    neg_inf = torch.finfo(masked.dtype).min
    masked = masked.masked_fill(adj_mask <= 0, neg_inf)
    deg = adj_mask.float().sum(dim=-1).clamp(min=1.0)
    k = torch.minimum(
        torch.full_like(deg, float(k_fixed)), deg
    ).long().clamp(min=1)
    k_max = int(k.max().item())
    k_max = max(1, min(k_max, masked.shape[-1]))
    top_vals, top_idx = torch.topk(masked, k=k_max, dim=-1)
    selected = torch.zeros_like(scores)
    arange_k = torch.arange(k_max, device=scores.device).view(1, 1, k_max)
    valid = arange_k < k.unsqueeze(-1)
    selected.scatter_(-1, top_idx, top_vals.masked_fill(~valid, 0.0))
    selected = selected * adj_mask.float()
    n = scores.shape[-1]
    eye = torch.eye(n, device=scores.device, dtype=scores.dtype).unsqueeze(0)
    return selected * (1.0 - eye)


def eval_success(policy, env, episodes, max_steps, thr):
    suc = []
    with torch.no_grad():
        for _ in range(episodes):
            td = env.reset()
            ep_s = 0.0
            for _ in range(max_steps):
                td = policy(td)
                td = env.step(td)
                obs = td.get(("next", "agents", "observation"))
                goal = obs[0, :, 4:6] if obs.dim() == 3 else obs[:, 4:6]
                ep_s = float((goal.norm(dim=-1) < thr).float().mean())
                done = td.get(("next", "done"))
                if done is not None and bool(done.any()):
                    break
                td = step_mdp(td)
            suc.append(ep_s)
    return float(np.mean(suc))


def fig11_topk(policy, env, cfg, episodes: int):
    thr = float(cfg["env"].get("success_threshold", 0.3))
    max_steps = int(cfg["env"].get("max_steps", 128))
    rows = []

    # Soft baseline (no hard Top-K)
    for m in policy.modules():
        if isinstance(m, ACGuideAdapter):
            m.set_budget_ratio(None)
            m.set_ablation_mode("full")
    s0 = eval_success(policy, env, episodes, max_steps, thr)
    rows.append({"mode": "soft_gate", "K": "soft", "success": round(s0, 4)})
    print("soft", s0)

    orig = budget_layer.apply_topk_budget
    for K in (1, 2, 3):
        def _fn(scores, adj, ratio, k_fixed=K):
            return _apply_topk_fixed_k(scores, adj, ratio, k_fixed)

        budget_layer.apply_topk_budget = _fn
        for m in policy.modules():
            if isinstance(m, ACGuideAdapter):
                # ratio<1 triggers Top-K path in encoder
                m.set_budget_ratio(0.5)
        s = eval_success(policy, env, episodes, max_steps, thr)
        rows.append({"mode": f"top{K}", "K": K, "success": round(s, 4)})
        print(f"top-{K}", s)

    budget_layer.apply_topk_budget = orig
    for m in policy.modules():
        if isinstance(m, ACGuideAdapter):
            m.set_budget_ratio(None)

    with open(TAB / "table_topk_deploy.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # bar figure
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    labels = [str(r["K"]) for r in rows]
    ys = [100 * r["success"] for r in rows]
    colors = ["#0b6e4f", "#3d5a80", "#5c6b73", "#b35c00"][: len(rows)]
    ax.bar(labels, ys, color=colors, edgecolor="#1a1a1a")
    ax.set_ylabel("Success (%)")
    ax.set_xlabel("Deployment rule (per-agent Top-$K$ on learned $g$)")
    ax.set_title("Fig.11  Top-$K$ deployment from soft gates", loc="left", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(FIG / "Fig11_topk_deploy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved Fig11 + table_topk_deploy.csv")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--skip-topk", action="store_true")
    parser.add_argument("--skip-hist", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)

    fig10_scaling_law()

    cfg, env, policy = load_ac(16, args.seed)
    if not args.skip_hist:
        print("Collecting gates for Fig10...")
        gates = collect_gates(
            policy,
            env,
            episodes=min(12, args.episodes),
            max_steps=int(cfg["env"].get("max_steps", 128)),
            thr=float(cfg["env"].get("success_threshold", 0.3)),
        )
        np.save(OUT / "gate_samples_n16.npy", gates)
        fig9_gate_hist(gates)  # writes Fig10_gate_distribution.png

    if not args.skip_topk:
        print("Top-K deployment eval...")
        rows = fig11_topk(policy, env, cfg, episodes=args.episodes)
        (OUT / "topk_deploy.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("Done →", FIG)


if __name__ == "__main__":
    main()

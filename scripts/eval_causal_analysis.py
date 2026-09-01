"""Causal communication validation — Phase 7 killer experiment.

Collects (U_pred, Δa_true) pairs per active edge and compares:
  full (AC++) | random | distance | wo_utility | random_utility | distance_utility

Outputs:
  paper/tables/table_causal.csv
  paper/figures/fig_causal_validity.png

Usage:
  python scripts/eval_causal_analysis.py \\
    --checkpoint results/ac_dsgf_pp/ac_dsgf_pp_smoke/checkpoints/final.pt \\
    --exp configs/ac_dsgf_pp/ac_dsgf_pp_smoke_v0.yaml \\
    --episodes 64 --plot
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchrl.envs.utils import step_mdp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithms.guided.mappo_guided import ACGuideAdapterPP
from utils.counterfactual import pearson_correlation
from utils.experiment import load_experiment_config
from utils.policy_loader import load_policy_for_eval, resolve_checkpoint
from utils.seed import set_seed

VARIANTS = (
    "full",
    "random",
    "distance",
    "wo_utility",
    "random_utility",
    "distance_utility",
)


def _find_pp_adapter(policy) -> ACGuideAdapterPP:
    for m in policy.modules():
        if isinstance(m, ACGuideAdapterPP):
            return m
    raise RuntimeError("ACGuideAdapterPP not found in policy")


def set_variant(adapter: ACGuideAdapterPP, mode: str):
    adapter.set_ablation_mode(None if mode == "full" else mode)
    adapter.set_budget_ratio(None)
    adapter.encoder.ablation_mode = None if mode == "full" else mode


def collect_pairs(
    env,
    policy,
    adapter: ACGuideAdapterPP,
    mode: str,
    num_episodes: int,
    max_steps: int,
) -> tuple[list[float], list[float], float]:
    set_variant(adapter, mode)
    policy.eval()
    u_preds: list[float] = []
    deltas: list[float] = []
    edges: list[float] = []

    with torch.no_grad():
        for _ in range(num_episodes):
            td = env.reset()
            for _ in range(max_steps):
                td = policy(td)
                obs = td.get(("agents", "observation"))
                if obs is None:
                    break
                if obs.dim() == 2:
                    obs_b = obs.unsqueeze(0)
                else:
                    obs_b = obs

                u_pred, u_star, mask = adapter.encoder.compute_utility_targets(
                    obs_b,
                    obs_b[..., :2],
                    residual_delta_fn=getattr(adapter._residual_actor, "delta_net", None),
                    beta=float(adapter._residual_actor.beta.item())
                    if adapter._residual_actor is not None
                    else 1.0,
                )
                m = mask > 0
                g = adapter.last_gate_matrix
                if g is not None:
                    if g.dim() == 2:
                        g = g.unsqueeze(0)
                    active = m & (g > 0.05)
                    edges.append(float(g.sum(dim=(-2, -1)).mean()))
                else:
                    active = m

                if active.any():
                    u_preds.extend(u_pred[active].cpu().tolist())
                    deltas.extend(u_star[active].cpu().tolist())

                td = env.step(td)
                done = td.get(("next", "done"))
                if done is not None and bool(done.any()):
                    break
                td = step_mdp(td)

    mean_edges = sum(edges) / len(edges) if edges else 0.0
    return u_preds, deltas, mean_edges


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--exp", default="configs/ac_dsgf_pp/ac_dsgf_pp_smoke_v0.yaml")
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--out-table", default="paper/tables/table_causal.csv")
    parser.add_argument("--out-fig", default="paper/figures/fig_causal_validity.png")
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = load_experiment_config(args.exp)
    env_cfg = cfg["env"]
    max_steps = env_cfg.get("max_steps", 128)

    ckpt = resolve_checkpoint(args.checkpoint)
    env, policy = load_policy_for_eval(cfg, ckpt)
    adapter = _find_pp_adapter(policy)

    rows = []
    scatter = {}
    for var in VARIANTS:
        u_list, d_list, mean_g = collect_pairs(
            env, policy, adapter, var, args.episodes, max_steps
        )
        if len(u_list) < 2:
            corr = 0.0
        else:
            corr = pearson_correlation(
                torch.tensor(u_list), torch.tensor(d_list)
            )
        mean_delta = sum(d_list) / len(d_list) if d_list else 0.0
        mean_u = sum(u_list) / len(u_list) if u_list else 0.0
        rows.append(
            {
                "variant": var,
                "pearson_u_delta": round(corr, 4),
                "mean_utility": round(mean_u, 4),
                "mean_delta_action": round(mean_delta, 4),
                "n_pairs": len(u_list),
                "mean_gate_mass": round(mean_g, 4),
            }
        )
        scatter[var] = (u_list, d_list)
        print(
            f"{var:18s}  corr={corr:.3f}  Δa={mean_delta:.4f}  "
            f"U={mean_u:.4f}  pairs={len(u_list)}  G={mean_g:.3f}"
        )

    out_table = ROOT / args.out_table
    out_table.parent.mkdir(parents=True, exist_ok=True)
    with open(out_table, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_table}")

    if args.plot:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        plot_vars = ("full", "random", "distance")
        for ax, var in zip(axes, plot_vars):
            u, d = scatter.get(var, ([], []))
            if u:
                ax.scatter(u, d, s=8, alpha=0.35)
            ax.set_title(var)
            ax.set_xlabel("U_pred")
            ax.set_ylabel("Δa (U*)")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        fig.suptitle("Causal validity: utility vs action influence")
        fig.tight_layout()
        out_fig = ROOT / args.out_fig
        out_fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_fig, dpi=150)
        plt.close(fig)
        print(f"Wrote {out_fig}")

    summary = ROOT / "results/ac_dsgf_pp/causal_analysis_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    with open(summary, "w", encoding="utf-8") as f:
        json.dump({"variants": rows}, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

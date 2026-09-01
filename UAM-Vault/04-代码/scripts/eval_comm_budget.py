"""Exp-2 — Communication Budget Sweep (AC-DSGF killer experiment).

Evaluate frozen checkpoints under hard edge budgets:
  budget_ratio ∈ {1.0, 0.75, 0.5, 0.25, 0.1}

Methods:
  - GAT-A (distance top-k under budget)
  - DSGF v2 (quality top-k under budget)
  - AC-DSGF (learned g_ij top-k under budget)

CE = Success / CommunicationCost

Usage:
  python scripts/eval_comm_budget.py --episodes 64 --plot
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

from algorithms.guided.mappo_guided import ACGuideAdapter, GraphGuideAdapter
from utils.comm_metrics import compute_sparse_communication_stats
from utils.experiment import load_experiment_config
from utils.policy_loader import load_policy_for_eval, resolve_checkpoint
from utils.seed import set_seed

BUDGETS = [1.0, 0.75, 0.5, 0.25, 0.1]

METHODS = {
    "gat": {
        "exp": "configs/demo/gat_4uav.yaml",
        "ckpt": "results/graph/gat_a_lambda003/checkpoints/checkpoint_20k.pt",
    },
    "dsgf": {
        "exp": "configs/demo/dsgf_4uav.yaml",
        "ckpt": "results/dsgf/dsfg_v2_102k/checkpoints/checkpoint_20k.pt",
    },
    "ac_dsgf": {
        "exp": "configs/ac_dsgf/ac_dsgf_smoke_v0.yaml",
        "ckpt": "results/ac_dsgf/ac_dsgf_smoke_v0/checkpoints/final.pt",
    },
}


def _unwrap_core(guide_module):
    """TensorDictModule → adapter → core encoder."""
    # policy.module is TensorDictSequential; first module is guide TensorDictModule
    return guide_module


def set_budget_on_policy(policy, mode: str, budget_ratio: float | None):
    """Attach budget_ratio to the underlying graph encoder."""
    # ProbabilisticActor → module (TensorDictSequential)
    seq = policy.module
    guide_td = seq.module[0] if hasattr(seq, "module") else seq[0]
    adapter = guide_td.module  # GraphGuideAdapter or ACGuideAdapter

    if isinstance(adapter, ACGuideAdapter):
        adapter.set_budget_ratio(budget_ratio)
        return adapter

    if isinstance(adapter, GraphGuideAdapter):
        core = adapter.encoder
        core.budget_ratio = budget_ratio
        return adapter

    # Fallback: walk
    for m in policy.modules():
        if isinstance(m, ACGuideAdapter):
            m.set_budget_ratio(budget_ratio)
            return m
        if hasattr(m, "budget_ratio"):
            m.budget_ratio = budget_ratio
            return m
    raise RuntimeError(f"Cannot set budget for mode={mode}")


def eval_with_budget(
    env,
    policy,
    mode: str,
    budget_ratio: float,
    num_episodes: int,
    max_steps: int,
    success_threshold: float,
    comm_radius: float,
) -> dict:
    set_budget_on_policy(policy, mode, budget_ratio)
    policy.eval()

    successes, collisions, rewards, edge_sums = [], [], [], []

    with torch.no_grad():
        for _ in range(num_episodes):
            td = env.reset()
            ep_reward = 0.0
            ep_collision = 0.0
            ep_success = 0.0
            ep_edges = []

            for _ in range(max_steps):
                td = policy(td)
                td = env.step(td)

                obs = td.get(("next", "agents", "observation"))
                if obs.dim() == 3:
                    goal_rel = obs[0, :, 4:6]
                else:
                    goal_rel = obs[:, 4:6]
                ep_success = float((goal_rel.norm(dim=-1) < success_threshold).float().mean())

                rew = td.get(("next", "agents", "reward"))
                ep_reward += float(rew.mean())

                info = td.get(("next", "agents", "info"), default=None)
                if info is not None and "agent_collisions" in info.keys():
                    coll = info.get("agent_collisions")
                    ep_collision = max(ep_collision, float((coll < 0).float().mean()))

                # Communication accounting
                if mode == "ac_dsgf":
                    for m in policy.modules():
                        if isinstance(m, ACGuideAdapter):
                            ep_edges.append(m.last_soft_edges)
                            break
                else:
                    stats = compute_sparse_communication_stats(obs, comm_radius)
                    # Approximate under budget: scale by ratio (upper bound); better hard count via mask
                    # Recompute top-k geometric edges for DSGF/GAT budget cost
                    if obs.dim() == 3:
                        pos = obs[..., :2]
                    else:
                        pos = obs[:, :2].unsqueeze(0)
                    from guidance.graph_builder import build_adjacency
                    from models.communication.budget_layer import apply_topk_budget, budget_edge_count

                    adj = build_adjacency(pos, comm_radius)
                    dist = (pos.unsqueeze(2) - pos.unsqueeze(1)).norm(dim=-1)
                    scores = adj * (1.0 / (dist + 1e-6))
                    gated = apply_topk_budget(scores, adj, budget_ratio)
                    ep_edges.append(budget_edge_count(gated))

                done = td.get(("next", "done"))
                if done is not None and bool(done.any()):
                    break
                td = step_mdp(td)

            successes.append(ep_success)
            collisions.append(ep_collision)
            rewards.append(ep_reward)
            edge_sums.append(sum(ep_edges) / max(len(ep_edges), 1))

    success = sum(successes) / len(successes)
    comm = sum(edge_sums) / len(edge_sums)
    ce = success / comm if comm > 1e-8 else success
    return {
        "success": round(success, 4),
        "collision": round(sum(collisions) / len(collisions), 4),
        "reward": round(sum(rewards) / len(rewards), 4),
        "comm_cost": round(comm, 4),
        "ce": round(ce, 6),
        "budget_ratio": budget_ratio,
    }


def plot_budget(rows: list[dict], out: Path):
    colors = {"gat": "#DD8452", "dsgf": "#4C72B0", "ac_dsgf": "#C44E52"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for method in sorted({r["method"] for r in rows}):
        pts = sorted([r for r in rows if r["method"] == method], key=lambda x: -x["budget_ratio"])
        xs = [100 * p["budget_ratio"] for p in pts]
        axes[0].plot(
            xs, [100 * p["success"] for p in pts],
            marker="o", lw=2, label=method.upper(), color=colors.get(method),
        )
        axes[1].plot(
            xs, [p["ce"] for p in pts],
            marker="s", lw=2, label=method.upper(), color=colors.get(method),
        )

    axes[0].set_xlabel("Communication Budget (%)")
    axes[0].set_ylabel("Success Rate (%)")
    axes[0].set_title("Success vs Communication Budget")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Communication Budget (%)")
    axes[1].set_ylabel("CE = Success / Comm Cost")
    axes[1].set_title("Communication Efficiency")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle("Exp-2: Adaptive Communication under Budget Constraints", fontsize=12)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", nargs="+", default=["gat", "dsgf", "ac_dsgf"])
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--out-dir", default="results/ac_dsgf/budget_sweep")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for method in args.methods:
        spec = METHODS[method]
        exp_cfg = load_experiment_config(str(ROOT / spec["exp"]))
        exp_cfg["env"]["num_envs"] = 1
        exp_cfg["env"]["device"] = "cpu"
        ckpt = resolve_checkpoint(str(ROOT / spec["ckpt"]))
        print(f"\n=== {method} ckpt={ckpt}")
        env, policy = load_policy_for_eval(exp_cfg, ckpt)
        mode = exp_cfg["guidance"]["mode"]
        if mode in ("ac-dsgf",):
            mode = "ac_dsgf"
        if method == "ac_dsgf":
            mode = "ac_dsgf"
        elif method == "dsgf":
            mode = "dsfg"
        elif method == "gat":
            mode = "gat"

        comm_radius = float(
            exp_cfg.get("guidance", {}).get("comm_radius", exp_cfg["env"].get("comm_radius", 0.5))
        )
        max_steps = int(exp_cfg["env"].get("max_steps", 128))
        thr = float(exp_cfg["env"].get("success_threshold", 0.3))

        for b in BUDGETS:
            print(f"  budget={b:.0%} ...", end=" ", flush=True)
            stats = eval_with_budget(
                env, policy, mode, b, args.episodes, max_steps, thr, comm_radius
            )
            row = {"method": method, **stats}
            rows.append(row)
            print(
                f"S={stats['success']:.2%} C={stats['comm_cost']:.2f} CE={stats['ce']:.4f}"
            )

    csv_path = out_dir / "budget_sweep.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {csv_path}")

    json_path = out_dir / "budget_sweep.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    paper_csv = ROOT / "paper" / "tables" / "table_budget_sweep.csv"
    paper_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(paper_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["method", "budget_ratio", "success", "comm_cost", "ce", "collision"],
        )
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})
    print(f"Saved {paper_csv}")

    if args.plot and rows:
        plot_budget(rows, ROOT / "paper" / "figures" / "fig_comm_budget.png")
        plot_budget(rows, out_dir / "fig_comm_budget.png")


if __name__ == "__main__":
    main()

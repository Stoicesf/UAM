"""Communication Trigger Analysis — why AC-DSGF communicates when it does.

Answers: sparsity is task-related, not random silence.

Aggregates multi-episode rollouts of AC-DSGF (4UAV smoke ckpt for clarity)
and correlates Comm / active edges with:
  - mean goal distance
  - agent spatial dispersion
  - proximity risk (inverse nearest-neighbor distance)

Writes:
  paper/figures/fig_comm_trigger.png
  paper/tables/table_comm_trigger.csv
  results/ac_dsgf/comm_trigger/summary.json

Usage:
  python scripts/analyze_comm_trigger.py --episodes 16 --plot
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

from algorithms.guided.mappo_guided import ACGuideAdapter
from demo.render_comm_graph import active_edge_count
from utils.experiment import load_experiment_config
from utils.policy_loader import load_policy_for_eval, resolve_checkpoint
from utils.seed import set_seed
from visualization.draw_graph import adjacency_from_positions

EPS = 1e-6


def _pos(obs: torch.Tensor) -> np.ndarray:
    if obs.dim() == 3:
        obs = obs[0]
    return obs[:, :2].detach().cpu().numpy()


def _goal_dist(obs: torch.Tensor) -> float:
    if obs.dim() == 3:
        obs = obs[0]
    return float(obs[:, 4:6].norm(dim=-1).mean())


def _dispersion(pos: np.ndarray) -> float:
    return float(np.linalg.norm(pos.std(axis=0)))


def _nn_risk(pos: np.ndarray) -> float:
    """Higher when agents are closer (1 / mean nearest-neighbor dist)."""
    n = pos.shape[0]
    if n < 2:
        return 0.0
    dists = []
    for i in range(n):
        d = np.linalg.norm(pos - pos[i], axis=1)
        d[i] = np.inf
        dists.append(d.min())
    return float(1.0 / (np.mean(dists) + 1e-3))


def _get_gate(policy) -> np.ndarray | None:
    for m in policy.modules():
        if isinstance(m, ACGuideAdapter) and m.last_gate_matrix is not None:
            g = m.last_gate_matrix.detach().cpu().numpy()
            return np.maximum(g, g.T)
    return None


def _comm(policy) -> float:
    for m in policy.modules():
        if isinstance(m, ACGuideAdapter):
            return float(m.last_comm_cost)
    return 0.0


def rollout_series(env, policy, max_steps: int, radius: float) -> list[dict]:
    rows = []
    policy.eval()
    with torch.no_grad():
        td = env.reset()
        obs = td.get(("agents", "observation"))
        td = policy(td)
        for t in range(max_steps + 1):
            if t > 0:
                td = env.step(td)
                obs = td.get(("next", "agents", "observation"))
                done = td.get(("next", "done"))
                td = step_mdp(td)
                td = policy(td)
                if done is not None and bool(done.any()):
                    break
            pos = _pos(obs)
            g = _get_gate(policy)
            if g is None:
                g = adjacency_from_positions(pos, radius)
            rows.append({
                "t": t,
                "comm": _comm(policy),
                "edges": active_edge_count(g, thr=0.02),
                "goal_dist": _goal_dist(obs),
                "dispersion": _dispersion(pos),
                "nn_risk": _nn_risk(pos),
            })
    return rows


def phase_bins(rows: list[dict], n_bins: int = 3) -> list[dict]:
    """Early / mid / late by time thirds."""
    if not rows:
        return []
    T = rows[-1]["t"] + 1
    cuts = [0, T // 3, 2 * T // 3, T]
    labels = ["early (disperse)", "mid (approach)", "late (finish)"]
    out = []
    for i, lab in enumerate(labels):
        chunk = [r for r in rows if cuts[i] <= r["t"] < cuts[i + 1]]
        if not chunk:
            continue
        out.append({
            "phase": lab,
            "mean_comm": float(np.mean([r["comm"] for r in chunk])),
            "mean_edges": float(np.mean([r["edges"] for r in chunk])),
            "mean_goal_dist": float(np.mean([r["goal_dist"] for r in chunk])),
            "mean_dispersion": float(np.mean([r["dispersion"] for r in chunk])),
            "mean_nn_risk": float(np.mean([r["nn_risk"] for r in chunk])),
            "n_steps": len(chunk),
        })
    return out


def corr(xs: np.ndarray, ys: np.ndarray) -> float:
    if xs.std() < 1e-8 or ys.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(xs, ys)[0, 1])


def plot_trigger(agg_ts: dict, phase_rows: list[dict], corrs: dict, out: Path):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))

    t = np.asarray(agg_ts["t"])
    axes[0, 0].plot(t, agg_ts["comm"], color="#C44E52", lw=2, label="Comm cost")
    axes[0, 0].set_title("Communication over time (mean± episode)")
    axes[0, 0].set_xlabel("t")
    axes[0, 0].set_ylabel("Comm")
    axes[0, 0].grid(True, alpha=0.3)
    # shade phases
    T = t.max() + 1 if len(t) else 1
    for i, (a, b, c) in enumerate([
        (0, T / 3, "#eeeeee"),
        (T / 3, 2 * T / 3, "#ffe6e6"),
        (2 * T / 3, T, "#e6f0ff"),
    ]):
        axes[0, 0].axvspan(a, b, color=c, alpha=0.45, zorder=0)

    axes[0, 1].plot(t, agg_ts["edges"], color="#4C72B0", lw=2)
    axes[0, 1].set_title("Active edges over time")
    axes[0, 1].set_xlabel("t")
    axes[0, 1].set_ylabel("#edges")
    axes[0, 1].grid(True, alpha=0.3)

    # Phase bars
    labs = [p["phase"] for p in phase_rows]
    axes[1, 0].bar(labs, [p["mean_comm"] for p in phase_rows], color="#C44E52", alpha=0.85)
    axes[1, 0].set_title("Mean Comm by episode phase")
    axes[1, 0].tick_params(axis="x", labelsize=8)
    axes[1, 0].grid(True, axis="y", alpha=0.3)

    # Correlation text panel / scatter goal_dist vs comm
    axes[1, 1].scatter(agg_ts["goal_dist"], agg_ts["comm"], s=12, alpha=0.5, c="#C44E52")
    axes[1, 1].set_xlabel("Mean goal distance")
    axes[1, 1].set_ylabel("Comm")
    axes[1, 1].set_title(
        f"Triggers  corr(goal,comm)={corrs['goal_dist']:.2f}\n"
        f"corr(disp,comm)={corrs['dispersion']:.2f}  "
        f"corr(risk,comm)={corrs['nn_risk']:.2f}"
    )
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(
        "Communication Trigger Analysis — AC-DSGF\n"
        "(task-related sparsity, not random silence)",
        fontsize=11,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", action="store_true", default=True)
    parser.add_argument(
        "--ckpt",
        default="results/ac_dsgf/ac_dsgf_smoke_v0/checkpoints/final.pt",
    )
    parser.add_argument("--exp", default="configs/ac_dsgf/ac_dsgf_smoke_v0.yaml")
    args = parser.parse_args()

    set_seed(args.seed)
    ckpt = resolve_checkpoint(str(ROOT / args.ckpt))
    exp_cfg = load_experiment_config(str(ROOT / args.exp))
    exp_cfg["env"]["num_envs"] = 1
    exp_cfg["env"]["device"] = "cpu"
    env, policy = load_policy_for_eval(exp_cfg, ckpt)
    for m in policy.modules():
        if isinstance(m, ACGuideAdapter):
            m.set_ablation_mode("full")
            m.set_budget_ratio(None)
    radius = float(
        exp_cfg.get("guidance", {}).get("comm_radius", exp_cfg["env"].get("comm_radius", 0.5))
    )

    all_rows: list[dict] = []
    phase_acc: list[dict] = []
    for ep in range(args.episodes):
        set_seed(args.seed + ep * 13)
        rows = rollout_series(env, policy, args.max_steps, radius)
        all_rows.extend(rows)
        phase_acc.extend(phase_bins(rows))
        print(f"ep={ep} steps={len(rows)} mean_comm={np.mean([r['comm'] for r in rows]):.4f}")

    # Align by t (average across episodes)
    by_t: dict[int, list[dict]] = {}
    for r in all_rows:
        by_t.setdefault(r["t"], []).append(r)
    ts = sorted(by_t)
    agg = {
        "t": ts,
        "comm": [float(np.mean([x["comm"] for x in by_t[t]])) for t in ts],
        "edges": [float(np.mean([x["edges"] for x in by_t[t]])) for t in ts],
        "goal_dist": [float(np.mean([x["goal_dist"] for x in by_t[t]])) for t in ts],
        "dispersion": [float(np.mean([x["dispersion"] for x in by_t[t]])) for t in ts],
        "nn_risk": [float(np.mean([x["nn_risk"] for x in by_t[t]])) for t in ts],
    }

    # Aggregate phases across episodes
    phase_names = ["early (disperse)", "mid (approach)", "late (finish)"]
    phase_rows = []
    for name in phase_names:
        chunk = [p for p in phase_acc if p["phase"] == name]
        if not chunk:
            continue
        phase_rows.append({
            "phase": name,
            "mean_comm": float(np.mean([p["mean_comm"] for p in chunk])),
            "mean_edges": float(np.mean([p["mean_edges"] for p in chunk])),
            "mean_goal_dist": float(np.mean([p["mean_goal_dist"] for p in chunk])),
            "mean_dispersion": float(np.mean([p["mean_dispersion"] for p in chunk])),
            "mean_nn_risk": float(np.mean([p["mean_nn_risk"] for p in chunk])),
            "n_episodes": len(chunk),
        })

    xs_c = np.asarray([r["comm"] for r in all_rows], dtype=float)
    corrs = {
        "goal_dist": corr(np.asarray([r["goal_dist"] for r in all_rows]), xs_c),
        "dispersion": corr(np.asarray([r["dispersion"] for r in all_rows]), xs_c),
        "nn_risk": corr(np.asarray([r["nn_risk"] for r in all_rows]), xs_c),
    }
    print("Correlations with Comm:", corrs)
    for p in phase_rows:
        print(
            f"  {p['phase']:20} Comm={p['mean_comm']:.4f} "
            f"edges={p['mean_edges']:.2f} goal_d={p['mean_goal_dist']:.3f}"
        )

    out_dir = ROOT / "results" / "ac_dsgf" / "comm_trigger"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "claim": "AC communication varies with task phase — not random silence",
        "correlations": corrs,
        "phases": phase_rows,
        "n_episodes": args.episodes,
        "ckpt": str(ckpt),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    paper_csv = ROOT / "paper" / "tables" / "table_comm_trigger.csv"
    paper_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(paper_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(phase_rows[0].keys()))
        w.writeheader()
        w.writerows(phase_rows)
        # also dump correlations as extra rows via second file
    corr_csv = ROOT / "paper" / "tables" / "table_comm_trigger_corr.csv"
    with open(corr_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["factor", "corr_with_comm"])
        w.writeheader()
        for k, v in corrs.items():
            w.writerow({"factor": k, "corr_with_comm": round(v, 4)})
    print(f"Saved {paper_csv}")
    print(f"Saved {corr_csv}")

    if args.plot:
        plot_trigger(
            agg, phase_rows, corrs,
            ROOT / "paper" / "figures" / "fig_comm_trigger.png",
        )
        plot_trigger(agg, phase_rows, corrs, out_dir / "fig_comm_trigger.png")


if __name__ == "__main__":
    main()

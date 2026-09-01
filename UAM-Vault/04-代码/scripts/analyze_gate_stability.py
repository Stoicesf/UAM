"""Gate stability from frozen AC-DSGF checkpoint (eval only, no training).

Reports Var(C_t) and edge-change rate ΔE_t = |E_t - E_{t-1}|.
Outputs:
  paper/figures/fig_gate_stability.png
  paper/tables/table_gate_stability.csv
"""

from __future__ import annotations

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


def main():
    set_seed(42)
    ckpt = resolve_checkpoint("results/ac_dsgf/ac_dsgf_smoke_v0/checkpoints/final.pt")
    exp_cfg = load_experiment_config(str(ROOT / "configs/ac_dsgf/ac_dsgf_smoke_v0.yaml"))
    exp_cfg["env"]["num_envs"] = 1
    exp_cfg["env"]["device"] = "cpu"
    env, policy = load_policy_for_eval(exp_cfg, ckpt)
    for m in policy.modules():
        if isinstance(m, ACGuideAdapter):
            m.set_ablation_mode("full")
            m.set_budget_ratio(None)

    n_ep, max_steps = 12, 100
    all_c, all_e, all_de = [], [], []
    series_c, series_e = [], []

    policy.eval()
    with torch.no_grad():
        for ep in range(n_ep):
            set_seed(42 + ep * 11)
            td = env.reset()
            td = policy(td)
            cs, es = [], []
            for t in range(max_steps + 1):
                if t > 0:
                    td = env.step(td)
                    done = td.get(("next", "done"))
                    td = step_mdp(td)
                    td = policy(td)
                    if done is not None and bool(done.any()):
                        break
                for m in policy.modules():
                    if isinstance(m, ACGuideAdapter):
                        c = float(m.last_comm_cost)
                        g = m.last_gate_matrix
                        if g is not None:
                            gnp = np.maximum(g.detach().cpu().numpy(), g.detach().cpu().numpy().T)
                            e = active_edge_count(gnp, thr=0.05)
                        else:
                            e = 0
                        cs.append(c)
                        es.append(e)
                        break
            de = np.abs(np.diff(es)).tolist() if len(es) > 1 else [0.0]
            all_c.extend(cs)
            all_e.extend(es)
            all_de.extend(de)
            series_c.append(cs)
            series_e.append(es)

    # Align by t
    T = min(len(s) for s in series_c)
    mean_c = np.mean([s[:T] for s in series_c], axis=0)
    mean_e = np.mean([s[:T] for s in series_e], axis=0)
    std_c = np.std([s[:T] for s in series_c], axis=0)

    stats = {
        "n_episodes": n_ep,
        "mean_C": float(np.mean(all_c)),
        "std_C": float(np.std(all_c)),
        "var_C": float(np.var(all_c)),
        "mean_abs_dE": float(np.mean(all_de)),
        "frac_dE_zero": float(np.mean(np.asarray(all_de) == 0)),
        "mean_edges": float(np.mean(all_e)),
    }
    print(json.dumps(stats, indent=2))

    out_dir = ROOT / "results" / "ac_dsgf" / "gate_stability"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    csv_path = ROOT / "paper" / "tables" / "table_gate_stability.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(stats.keys()))
        w.writeheader()
        w.writerow(stats)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    t = np.arange(T)
    axes[0].plot(t, mean_c, color="#C44E52", lw=2, label=r"mean $C_t$")
    axes[0].fill_between(t, mean_c - std_c, mean_c + std_c, color="#C44E52", alpha=0.2)
    axes[0].set_title(rf"Communication mass  $\mathrm{{Var}}(C)={stats['var_C']:.4f}$")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel(r"$C_t$")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    dE_mean = np.mean(
        [np.abs(np.diff(s[:T])) for s in series_e if len(s) >= T], axis=0
    )
    axes[1].plot(t[1:], dE_mean, color="#4C72B0", lw=2)
    axes[1].set_title(
        rf"Edge change  mean $|\Delta E_t|={stats['mean_abs_dE']:.3f}$"
        f"\n({100*stats['frac_dE_zero']:.0f}% steps with $\\Delta E=0$)"
    )
    axes[1].set_xlabel("t")
    axes[1].set_ylabel(r"$|\Delta E_t|$")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Gate Stability — task-driven, not random switching", fontsize=11)
    fig.tight_layout()
    for path in (
        ROOT / "paper" / "figures" / "fig_gate_stability.png",
        ROOT / "paper" / "ac_dsgf" / "figures" / "fig_gate_stability.png",
        out_dir / "fig_gate_stability.png",
    ):
        fig.savefig(path, dpi=160, bbox_inches="tight")
        print("Saved", path)
    plt.close(fig)


if __name__ == "__main__":
    main()

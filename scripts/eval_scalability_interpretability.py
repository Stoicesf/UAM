"""Eval-only scalability + communication interpretability (no training / no algo change).

Loads frozen 16-UAV *policy* weights into envs with N ∈ {8,16,32}.
Centralized critic weights depend on N and are skipped (eval does not need critic).

Usage:
  python scripts/eval_scalability_interpretability.py --seed 42 --episodes 32 --ns 8 16 32
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from torchrl.envs.utils import step_mdp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithms.guided.mappo_guided import ACGuideAdapter, build_guided_mappo
from utils.communication_analysis import (
    active_edge_ratio,
    expected_packet_survival,
    mass_ratio_on_support,
    packet_survival_on_active,
    radius_mask_from_positions,
    soft_comm_mass,
)
from utils.comm_metrics import compute_sparse_communication_stats
from utils.experiment import load_experiment_config
from utils.seed import set_seed

SPECS = {
    "dsgf": {
        "exp": "configs/baseline16/dsgf_v2.yaml",
        "ckpt": "results/baseline16_seeds/dsgf/s{seed}/checkpoints/final.pt",
    },
    "ac_dsgf": {
        "exp": "configs/ac_dsgf/ac_dsgf_16uav.yaml",
        "ckpt": "results/ac_dsgf/uav16/s{seed}/checkpoints/final.pt",
    },
}


def load_policy_eval_any_n(method: str, seed: int, n_agents: int):
    """Build env at target N; load policy weights with strict=False."""
    spec = SPECS[method]
    ckpt_path = ROOT / spec["ckpt"].format(seed=seed)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    cfg = load_experiment_config(str(ROOT / spec["exp"]))
    cfg["env"]["num_agents"] = int(n_agents)
    cfg["env"]["num_envs"] = 1
    cfg["env"]["device"] = "cpu"
    cfg["env"].setdefault("max_steps", 128)
    # Larger spawn box for N≥32 to avoid VMAS spawn hang
    if n_agents >= 32:
        cfg["env"]["world_spawning_x"] = 2.0
        cfg["env"]["world_spawning_y"] = 2.0
    elif n_agents >= 16:
        cfg["env"].setdefault("world_spawning_x", 1.5)
        cfg["env"].setdefault("world_spawning_y", 1.5)

    components = build_guided_mappo(cfg["train"], cfg["env"], cfg.get("guidance", {}))
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    incompat = components.policy.load_state_dict(ckpt["policy"], strict=False)
    print(
        f"  policy load strict=False: missing={len(incompat.missing_keys)} "
        f"unexpected={len(incompat.unexpected_keys)}",
        flush=True,
    )
    components.policy.eval()
    return cfg, components.env, components.policy


def evaluate(
    method: str,
    env,
    policy,
    episodes: int,
    max_steps: int,
    thr: float,
    comm_radius: float,
    gate_thr: float,
    n_agents: int,
) -> dict:
    policy.eval()
    successes, soft_masses, edge_ratios, mass_ratios, runtimes = [], [], [], [], []
    ratios_lo, ratios_mid = [], []  # τ=0.01 and 0.1
    recv_at_loss = {0.0: [], 0.3: [], 0.7: []}

    with torch.no_grad():
        for _ in range(episodes):
            td = env.reset()
            ep_s = 0.0
            ep_soft, ep_ratio, ep_mass_r = [], [], []
            ep_r01, ep_r10 = [], []
            ep_recv = {k: [] for k in recv_at_loss}
            for _ in range(max_steps):
                t0 = time.perf_counter()
                td = policy(td)
                t1 = time.perf_counter()
                runtimes.append(1000.0 * (t1 - t0))

                td = env.step(td)
                obs = td.get(("next", "agents", "observation"))
                if obs.dim() == 3:
                    pos = obs[0, :, :2]
                    goal = obs[0, :, 4:6]
                else:
                    pos = obs[:, :2]
                    goal = obs[:, 4:6]
                ep_s = float((goal.norm(dim=-1) < thr).float().mean())

                if method == "ac_dsgf":
                    g = None
                    for m in policy.modules():
                        if isinstance(m, ACGuideAdapter):
                            g = m.last_gate_matrix
                            break
                    if g is not None:
                        mask = radius_mask_from_positions(pos, comm_radius).to(g.device)
                        g2 = g if g.dim() == 2 else g[0]
                        m2 = mask if mask.dim() == 2 else mask[0]
                        ep_soft.append(soft_comm_mass(g2))
                        ep_ratio.append(active_edge_ratio(g2, m2, threshold=gate_thr))
                        ep_r01.append(active_edge_ratio(g2, m2, threshold=0.01))
                        ep_r10.append(active_edge_ratio(g2, m2, threshold=0.1))
                        ep_mass_r.append(mass_ratio_on_support(g2, m2))
                        # survival uses soft-positive edges (τ=0.01), not 0.5
                        for p in recv_at_loss:
                            ep_recv[p].append(
                                packet_survival_on_active(
                                    g2, m2, loss_rate=p, threshold=0.01
                                )
                            )
                else:
                    stats = compute_sparse_communication_stats(obs, comm_radius)
                    ep_soft.append(float(stats["sparse_edges"]))
                    ep_ratio.append(1.0)
                    ep_r01.append(1.0)
                    ep_r10.append(1.0)
                    ep_mass_r.append(1.0)
                    for p in recv_at_loss:
                        ep_recv[p].append(expected_packet_survival(p))

                done = td.get(("next", "done"))
                if done is not None and bool(done.any()):
                    break
                td = step_mdp(td)

            successes.append(ep_s)
            soft_masses.append(sum(ep_soft) / max(len(ep_soft), 1))
            edge_ratios.append(sum(ep_ratio) / max(len(ep_ratio), 1))
            ratios_lo.append(sum(ep_r01) / max(len(ep_r01), 1))
            ratios_mid.append(sum(ep_r10) / max(len(ep_r10), 1))
            mass_ratios.append(sum(ep_mass_r) / max(len(ep_mass_r), 1))
            for p in recv_at_loss:
                recv_at_loss[p].append(sum(ep_recv[p]) / max(len(ep_recv[p]), 1))

    return {
        "method": method,
        "num_agents": n_agents,
        "success": round(sum(successes) / len(successes), 4),
        "soft_comm_mass": round(sum(soft_masses) / len(soft_masses), 4),
        "active_edge_ratio_t0.5": round(sum(edge_ratios) / len(edge_ratios), 4),
        "active_edge_ratio_t0.1": round(sum(ratios_mid) / len(ratios_mid), 4),
        "active_edge_ratio_t0.01": round(sum(ratios_lo) / len(ratios_lo), 4),
        "mean_gate_on_support": round(sum(mass_ratios) / len(mass_ratios), 6),
        "runtime_ms": round(sum(runtimes) / max(len(runtimes), 1), 3),
        "gate_threshold": gate_thr,
        "recv_frac_loss0": round(sum(recv_at_loss[0.0]) / len(recv_at_loss[0.0]), 4),
        "recv_frac_loss30": round(sum(recv_at_loss[0.3]) / len(recv_at_loss[0.3]), 4),
        "recv_frac_loss70": round(sum(recv_at_loss[0.7]) / len(recv_at_loss[0.7]), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--ns", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--gate-thr", type=float, default=0.5)
    parser.add_argument("--methods", nargs="+", default=["dsgf", "ac_dsgf"])
    args = parser.parse_args()
    set_seed(args.seed)

    rows = []
    for n in args.ns:
        for method in args.methods:
            print(f"\n=== N={n}  {method}", flush=True)
            try:
                cfg, env, policy = load_policy_eval_any_n(method, args.seed, n)
            except Exception as e:
                print(f"[skip] {type(e).__name__}: {e}")
                continue
            thr = float(cfg["env"].get("success_threshold", 0.3))
            rc = float(
                cfg["env"].get(
                    "comm_radius", cfg.get("guidance", {}).get("comm_radius", 0.5)
                )
            )
            max_steps = int(cfg["env"].get("max_steps", 128))
            row = evaluate(
                method, env, policy, args.episodes, max_steps, thr, rc, args.gate_thr, n
            )
            row["seed"] = args.seed
            row["episodes"] = args.episodes
            print(row, flush=True)
            rows.append(row)

    out_dir = ROOT / "results" / "ac_dsgf" / "scalability_interpretability"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scalability_interpretability.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    paper = ROOT / "paper" / "tables" / "table_scalability_interpretability.csv"
    if rows:
        with open(paper, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("Saved", paper)
    print("Saved", out_dir / "scalability_interpretability.json")


if __name__ == "__main__":
    main()

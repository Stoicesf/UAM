# -*- coding: utf-8 -*-
"""Hard Top-K fixed-budget comparison (eval-only, no training).

Answers: under identical |E_i|≤K, does learned scoring beat random / distance?

Methods (same frozen AC-DSGF checkpoint + residual actor):
  - Random Top-K : random scores on radius support, keep Top-K per agent
  - Distance Top-K: score = -distance, keep Top-K per agent
  - AC Top-K     : learned g_ij scores, keep Top-K per agent

Usage:
  conda run -n dpg_hrl python scripts/eval_hard_topk_fixed.py --seed 42 --episodes 64 --K 2
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torchrl.envs.utils import step_mdp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithms.guided.mappo_guided import ACGuideAdapter, build_guided_mappo
import models.communication.budget_layer as budget_layer
from scripts.eval_theory_binding_figures import _apply_topk_fixed_k
from utils.experiment import load_experiment_config
from utils.seed import set_seed

TAB = ROOT / "paper" / "tables"
OUT = ROOT / "results" / "ac_dsgf" / "hard_topk"
FIG = ROOT / "paper" / "ac_dsgf" / "figures"
FIG_CN = ROOT / "paper" / "ac_dsgf_cn" / "figures"
for d in (TAB, OUT, FIG, FIG_CN):
    d.mkdir(parents=True, exist_ok=True)


def load_ac(n: int, seed: int):
    cfg_path = ROOT / "configs/ac_dsgf/ac_dsgf_16uav.yaml"
    ckpt_path = ROOT / f"results/ac_dsgf/uav16/s{seed}/checkpoints/final.pt"
    if not ckpt_path.exists():
        # fallback common seeds
        for s in (42, 1234, 2026, 3407, 8888):
            p = ROOT / f"results/ac_dsgf/uav16/s{s}/checkpoints/final.pt"
            if p.exists():
                ckpt_path = p
                seed = s
                break
    cfg = load_experiment_config(str(cfg_path))
    cfg["env"]["num_agents"] = n
    cfg["env"]["num_envs"] = 1
    cfg["env"]["device"] = "cpu"
    components = build_guided_mappo(cfg["train"], cfg["env"], cfg["guidance"])
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    components.policy.load_state_dict(ckpt["policy"], strict=False)
    components.policy.eval()
    return cfg, components.env, components.policy, seed, ckpt_path


def _set_mode(policy, ablation: str):
    for m in policy.modules():
        if isinstance(m, ACGuideAdapter):
            m.set_ablation_mode(ablation)
            m.set_budget_ratio(0.5)  # triggers hard Top-K path
            return


def _patch_topk(K: int, score_mode: str):
    """score_mode: ac | random | distance"""
    orig = budget_layer.apply_topk_budget

    def _fn(scores, adj, ratio, k_fixed=K, mode=score_mode):
        if mode == "random":
            ranking = torch.rand_like(scores) * adj.float()
        elif mode == "distance":
            # scores unused; invert by using -|log| placeholder — caller passes dist via scores
            ranking = scores
        else:
            ranking = scores
        return _apply_topk_fixed_k(ranking, adj, ratio, k_fixed)

    budget_layer.apply_topk_budget = _fn
    return orig


def _inject_distance_scores(policy):
    """Monkey-patch ACDSGF forward path: use -dist as ranking before Top-K.

    We override ablation to full, then replace g with -dist * mask after controller
    by setting a hook on apply_topk via wrapping scores in budget call.

    Simpler approach: set ablation random/full and pass -dist through
    eval_gate by wrapping budget_layer to ignore scores and rebuild from positions
    stored on adapter — not available.

    Instead: patch CommunicationController temporarily is heavy.
    Use ablation_mode full + replace apply_topk to recompute -dist from adj only
    is insufficient without positions.

    Practical: for distance, set scores = -ones * idx order is wrong.
    Patch ACDSGF.forward locally via module attribute.
    """
    from models.ac_dsgf import ACDSGF

    for m in policy.modules():
        if isinstance(m, ACDSGF):
            m._hard_topk_score_mode = "distance"
            return
    raise RuntimeError("ACDSGF not found")


def _clear_score_mode(policy):
    from models.ac_dsgf import ACDSGF

    for m in policy.modules():
        if isinstance(m, ACDSGF) and hasattr(m, "_hard_topk_score_mode"):
            delattr(m, "_hard_topk_score_mode")


def _install_forward_hook():
    """Patch ACDSGF.forward to swap ranking scores before budget when requested."""
    from models.ac_dsgf import ACDSGF

    if getattr(ACDSGF, "_hard_topk_patched", False):
        return
    _orig = ACDSGF.forward

    def _forward(self, obs, positions=None, z_prev=None, apply_budget=False):
        mode = getattr(self, "_hard_topk_score_mode", None)
        if mode is None:
            return _orig(self, obs, positions=positions, z_prev=z_prev, apply_budget=apply_budget)

        # Run original but intercept: temporarily set ablation and post-process g
        # Cleaner: call original pieces — duplicate minimal path
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)
        if positions is None:
            positions = obs[..., :2]
        elif positions.dim() == 2:
            positions = positions.unsqueeze(0)

        velocities = obs[..., 2:4]
        weighted_adj, q_ij = self.dynamic_graph(positions, velocities)
        dist = (positions.unsqueeze(2) - positions.unsqueeze(1)).norm(dim=-1)
        radius_mask = (dist < self.comm_radius).float()
        eye = torch.eye(radius_mask.shape[-1], device=obs.device).unsqueeze(0)
        radius_mask = radius_mask * (1.0 - eye)

        h = self.node_embed(obs)
        edge_state = self._edge_state(positions, q_ij)

        if mode == "random":
            g = radius_mask * torch.rand_like(radius_mask)
        elif mode == "distance":
            # closer = higher score
            g = (-dist) * radius_mask
            # shift to positive for topk stability
            g = g - g.min() + 1e-6
            g = g * radius_mask
        else:  # ac
            g = self.controller(h, edge_state, adj_mask=radius_mask)

        ratio = self.budget_ratio
        if ratio is not None and ratio < 1.0:
            g = budget_layer.apply_topk_budget(g, radius_mask, float(ratio))

        A_tilde = radius_mask * q_ij * (g > 0).float()  # hard links for message pass
        # keep soft magnitude on selected edges for AC; binary for random/distance
        if mode == "ac":
            A_tilde = radius_mask * q_ij * g
        else:
            A_tilde = radius_mask * q_ij * (g > 0).float()

        x = h
        for layer in self.spatial_layers:
            x = torch.tanh(layer(x, A_tilde) + x)
        if self.temporal is not None:
            z = self.temporal(x, z_prev)
        else:
            z = x
        phi = self.phi_head(z)
        direction = phi[..., :2]
        phi = phi.clone()
        phi[..., :2] = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

        from models.communication.cost import communication_cost

        diagnostics = {
            "g": g,
            "A_tilde": A_tilde,
            "comm_cost": communication_cost((g > 0).float() if mode != "ac" else g),
            "hard_edges": torch.tensor(
                budget_layer.budget_edge_count(g), device=g.device
            ),
        }
        return phi, g, diagnostics

    ACDSGF.forward = _forward
    ACDSGF._hard_topk_patched = True
    ACDSGF._hard_topk_orig_forward = _orig


def eval_method(policy, env, episodes, max_steps, thr, K, method: str):
    from models.ac_dsgf import ACDSGF

    _install_forward_hook()
    for m in policy.modules():
        if isinstance(m, ACDSGF):
            m._hard_topk_score_mode = method
        if isinstance(m, ACGuideAdapter):
            m.set_ablation_mode("full")
            m.set_budget_ratio(0.5)

    orig = _patch_topk(K, "passthrough")
    # passthrough: scores already prepared in patched forward
    budget_layer.apply_topk_budget = lambda scores, adj, ratio, k_fixed=K: _apply_topk_fixed_k(
        scores, adj, ratio, k_fixed
    )

    suc, hard_edges = [], []
    with torch.no_grad():
        for _ in range(episodes):
            td = env.reset()
            ep_s, ep_h, n = 0.0, 0.0, 0
            for _ in range(max_steps):
                td = policy(td)
                td = env.step(td)
                obs = td.get(("next", "agents", "observation"))
                goal = obs[0, :, 4:6] if obs.dim() == 3 else obs[:, 4:6]
                ep_s = float((goal.norm(dim=-1) < thr).float().mean())
                for m in policy.modules():
                    if isinstance(m, ACGuideAdapter):
                        g = m.last_gate_matrix
                        if g is not None:
                            g2 = g if g.dim() == 2 else g[0]
                            ep_h += float((g2 > 0).float().sum().item())
                            n += 1
                        break
                done = td.get(("next", "done"))
                if done is not None and bool(done.any()):
                    break
                td = step_mdp(td)
            suc.append(ep_s)
            hard_edges.append(ep_h / max(n, 1))

    budget_layer.apply_topk_budget = orig
    _clear_score_mode(policy)

    s_arr = np.asarray(suc, dtype=np.float64)
    e_arr = np.asarray(hard_edges, dtype=np.float64)
    return {
        "method": method,
        "K": K,
        "success_mean": float(s_arr.mean()),
        "success_std": float(s_arr.std(ddof=0)),
        "hard_edges_mean": float(e_arr.mean()),
        "hard_edges_std": float(e_arr.std(ddof=0)),
        "n_episodes": episodes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--n", type=int, default=16)
    args = parser.parse_args()
    set_seed(args.seed)

    cfg, env, policy, used_seed, ckpt = load_ac(args.n, args.seed)
    thr = float(cfg["env"].get("success_threshold", 0.3))
    max_steps = int(cfg["env"].get("max_steps", 128))
    print("ckpt", ckpt, "seed", used_seed, "K", args.K, "episodes", args.episodes)

    rows = []
    for method in ("random", "distance", "ac"):
        print("eval", method, "...")
        row = eval_method(policy, env, args.episodes, max_steps, thr, args.K, method)
        print(row)
        rows.append(row)

    out_csv = TAB / "table_hard_topk_k2.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (OUT / "hard_topk.json").write_text(
        json.dumps({"seed": used_seed, "ckpt": str(ckpt), "rows": rows}, indent=2),
        encoding="utf-8",
    )

    # simple bar figure
    try:
        import matplotlib.pyplot as plt

        labels = {"random": "Random Top-K", "distance": "Distance Top-K", "ac": "AC Top-K"}
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        xs = [labels[r["method"]] for r in rows]
        ys = [100 * r["success_mean"] for r in rows]
        yerr = [100 * r["success_std"] for r in rows]
        colors = ["#8d99ae", "#e09f3e", "#0b6e4f"]
        ax.bar(xs, ys, yerr=yerr, color=colors, edgecolor="#1a1a1a", capsize=4)
        ax.set_ylabel("Success (%)")
        ax.set_title(f"Fixed hard budget: per-agent Top-$K$={args.K}", loc="left", fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        for dest in (FIG / "Fig14_hard_topk.png", FIG_CN / "Fig14_hard_topk.png"):
            fig.savefig(dest, dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print("plot skipped:", e)

    print("Wrote", out_csv)


if __name__ == "__main__":
    main()

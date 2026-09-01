"""Packet-loss robustness (eval only). Week-2 recommended for RA-L.

Drops random edges with probability p on frozen 16-UAV checkpoints.
  p ∈ {0, 0.1, 0.3, 0.5, 0.7}

Usage:
  python scripts/eval_packet_loss.py --episodes 64 --plot
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
from models.ac_dsgf import ACDSGF
from models.dsgf import DSGF
from models.guide_gnn import GATGuideEncoder
from utils.experiment import load_experiment_config
from utils.policy_loader import load_policy_for_eval, resolve_checkpoint
from utils.seed import set_seed

LOSS_RATES = [0.0, 0.1, 0.3, 0.5, 0.7]

METHODS = {
    "mappo": {
        "exp": "configs/baseline16/mappo.yaml",
        "ckpt": "results/baseline16_seeds/mappo/s42/checkpoints/final.pt",
    },
    "gat": {
        "exp": "configs/baseline16/gat_mappo.yaml",
        "ckpt": "results/baseline16_seeds/gat/s42/checkpoints/final.pt",
    },
    "dsgf": {
        "exp": "configs/baseline16/dsgf_v2.yaml",
        "ckpt": "results/baseline16_seeds/dsgf/s42/checkpoints/final.pt",
    },
    "ac_dsgf": {
        "exp": "configs/ac_dsgf/ac_dsgf_16uav.yaml",
        "ckpt": "results/ac_dsgf/uav16/s42/checkpoints/final.pt",
    },
}


def install_hooks():
    if getattr(install_hooks, "_done", False):
        return

    _ac = ACDSGF.forward

    def ac_fwd(self, obs, positions=None, z_prev=None, apply_budget=False):
        p = float(getattr(self, "packet_loss_p", 0.0) or 0.0)
        if p <= 0.0 or self.training:
            return _ac(self, obs, positions, z_prev, apply_budget)
        ctrl = self.controller
        _c = ctrl.forward

        def noisy(*a, **k):
            g = _c(*a, **k)
            return g * (torch.rand_like(g) > p).float()

        ctrl.forward = noisy  # type: ignore
        try:
            return _ac(self, obs, positions, z_prev, apply_budget)
        finally:
            ctrl.forward = _c  # type: ignore

    ACDSGF.forward = ac_fwd

    _dsgf = DSGF.forward

    def dsgf_fwd(self, obs, positions=None, comm_mask=None, z_prev=None):
        p = float(getattr(self, "packet_loss_p", 0.0) or 0.0)
        if p <= 0.0 or self.training:
            return _dsgf(self, obs, positions, comm_mask, z_prev)
        dg = self.dynamic_graph
        _dg = dg.forward

        def noisy(*a, **k):
            wadj, q = _dg(*a, **k)
            keep = (torch.rand_like(wadj) > p).float()
            return wadj * keep, q * keep

        dg.forward = noisy  # type: ignore
        try:
            return _dsgf(self, obs, positions, comm_mask, z_prev)
        finally:
            dg.forward = _dg  # type: ignore

    DSGF.forward = dsgf_fwd

    _gat = GATGuideEncoder.forward

    def gat_fwd(self, obs, positions=None):
        p = float(getattr(self, "packet_loss_p", 0.0) or 0.0)
        if p <= 0.0 or self.training:
            return _gat(self, obs, positions)
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)
        if positions is None:
            positions = obs[..., :2]
        elif positions.dim() == 2:
            positions = positions.unsqueeze(0)
        from guidance.graph_builder import build_adjacency
        from models.communication.budget_layer import apply_topk_budget

        adj = build_adjacency(positions, self.comm_radius)
        if self.budget_ratio is not None and self.budget_ratio < 1.0:
            dist = (positions.unsqueeze(2) - positions.unsqueeze(1)).norm(dim=-1)
            scores = adj * (1.0 / (dist + 1e-6))
            adj = (apply_topk_budget(scores, adj, float(self.budget_ratio)) > 0).float()
        keep = (torch.rand_like(adj) > p).float()
        adj = adj * keep
        h = torch.tanh(self.input_proj(obs))
        for layer in self.gat_layers:
            h = torch.tanh(layer(h, adj) + h)
        return self.guidance_head(h)

    GATGuideEncoder.forward = gat_fwd
    install_hooks._done = True


def set_packet_loss(policy, p: float):
    for m in policy.modules():
        if isinstance(m, ACGuideAdapter):
            m.encoder.packet_loss_p = float(p)
        elif isinstance(m, GraphGuideAdapter):
            m.encoder.packet_loss_p = float(p)
        elif isinstance(m, (ACDSGF, DSGF, GATGuideEncoder)):
            m.packet_loss_p = float(p)


def eval_success(env, policy, episodes: int, max_steps: int, thr: float) -> float:
    policy.eval()
    successes = []
    with torch.no_grad():
        for _ in range(episodes):
            td = env.reset()
            ep_s = 0.0
            for _ in range(max_steps):
                td = policy(td)
                td = env.step(td)
                obs = td.get(("next", "agents", "observation"))
                if obs.dim() == 3:
                    goal = obs[0, :, 4:6]
                else:
                    goal = obs[:, 4:6]
                ep_s = float((goal.norm(dim=-1) < thr).float().mean())
                done = td.get(("next", "done"))
                if done is not None and bool(done.any()):
                    break
                td = step_mdp(td)
            successes.append(ep_s)
    return sum(successes) / max(len(successes), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--methods", nargs="+", default=["mappo", "gat", "dsgf", "ac_dsgf"])
    args = parser.parse_args()
    set_seed(args.seed)
    install_hooks()

    rows = []
    for name in args.methods:
        spec = METHODS[name]
        ckpt_path = ROOT / spec["ckpt"]
        if not ckpt_path.exists():
            print(f"[skip] missing {ckpt_path}")
            continue
        exp_cfg = load_experiment_config(str(ROOT / spec["exp"]))
        exp_cfg["env"]["num_envs"] = 1
        exp_cfg["env"]["device"] = "cpu"
        print(f"\n=== {name}")
        env, policy = load_policy_for_eval(exp_cfg, resolve_checkpoint(str(ckpt_path)))
        max_steps = int(exp_cfg["env"].get("max_steps", 128))
        thr = float(exp_cfg["env"].get("success_threshold", 0.3))
        for p in LOSS_RATES:
            set_packet_loss(policy, p)
            s = eval_success(env, policy, args.episodes, max_steps, thr)
            rows.append({"method": name, "packet_loss": p, "success": round(s, 4)})
            print(f"  p={p:.0%}  S={s:.2%}")

    out_dir = ROOT / "results" / "ac_dsgf" / "packet_loss"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "packet_loss.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out_dir / "packet_loss.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    paper = ROOT / "paper" / "tables" / "table_packet_loss.csv"
    with open(paper, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Saved", paper)

    if args.plot and rows:
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        colors = {"mappo": "#888888", "gat": "#DD8452", "dsgf": "#4C72B0", "ac_dsgf": "#C44E52"}
        labels = {"mappo": "MAPPO", "gat": "GAT", "dsgf": "DSGF", "ac_dsgf": "AC-DSGF"}
        for m in ["mappo", "gat", "dsgf", "ac_dsgf"]:
            pts = sorted([r for r in rows if r["method"] == m], key=lambda x: x["packet_loss"])
            if not pts:
                continue
            ax.plot(
                [100 * p["packet_loss"] for p in pts],
                [100 * p["success"] for p in pts],
                marker="o",
                lw=2,
                color=colors[m],
                label=labels[m],
            )
        ax.set_xlabel("Packet loss (%)")
        ax.set_ylabel("Success (%)")
        ax.set_title("Task performance under unreliable communication")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        for path in (
            ROOT / "paper" / "figures" / "fig_packet_loss.png",
            ROOT / "paper" / "ac_dsgf" / "figures" / "fig_packet_loss.png",
            out_dir / "fig_packet_loss.png",
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print("Saved", path)
        plt.close(fig)


if __name__ == "__main__":
    main()

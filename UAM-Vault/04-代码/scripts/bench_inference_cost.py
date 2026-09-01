"""Measure per-step inference time / soft edges (no training). Table V computational cost."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.experiment import load_experiment_config
from utils.policy_loader import load_policy_for_eval, resolve_checkpoint
from utils.seed import set_seed

SPECS = {
    "gat": {
        "exp": "configs/baseline16/gat_mappo.yaml",
        "ckpt": "results/baseline16_seeds/gat/s42/checkpoints/final.pt",
    },
    "transformer": {
        "exp": "configs/baseline16/full_attention.yaml",
        "ckpt": "results/baseline16_seeds/transformer/s42/checkpoints/final.pt",
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


def bench(method: str, warmup: int = 5, iters: int = 50) -> dict:
    spec = SPECS[method]
    ckpt = ROOT / spec["ckpt"]
    if not ckpt.exists():
        return {"method": method, "error": f"missing {ckpt}"}
    exp_cfg = load_experiment_config(str(ROOT / spec["exp"]))
    n = int(exp_cfg["env"]["num_agents"])
    exp_cfg["env"]["num_envs"] = 1
    exp_cfg["env"]["device"] = "cpu"
    env, policy = load_policy_for_eval(exp_cfg, resolve_checkpoint(str(ckpt)))
    policy.eval()
    td = env.reset()
    with torch.no_grad():
        for _ in range(warmup):
            td = policy(td)
        # timed
        t0 = time.perf_counter()
        for _ in range(iters):
            td = policy(td)
        t1 = time.perf_counter()
    ms = 1000.0 * (t1 - t0) / iters
    # try last soft edges for AC
    edges = None
    from algorithms.guided.mappo_guided import ACGuideAdapter

    for m in policy.modules():
        if isinstance(m, ACGuideAdapter):
            edges = float(m.last_soft_edges)
            break
    return {
        "method": method,
        "num_agents": n,
        "inference_ms_per_step": round(ms, 3),
        "soft_edges_last": edges,
        "device": "cpu",
        "iters": iters,
    }


def main():
    set_seed(42)
    rows = []
    for m in SPECS:
        print(f"bench {m} ...", flush=True)
        rows.append(bench(m))
        print(rows[-1])
    out = ROOT / "results" / "ac_dsgf" / "compute_cost"
    out.mkdir(parents=True, exist_ok=True)
    (out / "compute_cost.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    csv_path = ROOT / "paper" / "tables" / "table5_compute_cost.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        keys = sorted({k for r in rows for k in r.keys()})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("Saved", csv_path)


if __name__ == "__main__":
    main()

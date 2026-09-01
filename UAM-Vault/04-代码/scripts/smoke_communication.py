"""P1-3 Day-1 smoke — verify comm edge counts at R=1/2/full without full training.

Usage:
  python scripts/smoke_communication.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.comm_metrics import compute_sparse_communication_stats
from utils.communication_config import build_comm_experiment_config, list_runs


def smoke_comm_metrics():
    """Synthetic 16-agent positions in unit square."""
    torch.manual_seed(42)
    positions = torch.rand(4, 16, 2) * 2.0 - 1.0
    obs = torch.zeros(4, 16, 18)
    obs[..., :2] = positions

    print("=== Synthetic comm edge counts (16 agents, batch=4) ===")
    for key in ("0", "1", "2", "full"):
        from utils.communication_config import load_radius_spec
        spec = load_radius_spec(key)
        r = spec["comm_radius"]
        stats = compute_sparse_communication_stats(obs, r)
        print(
            f"  R={spec['label']:>4} (r={r:>6.1f}): "
            f"edges={stats['communication_cost']:.2f}  "
            f"ratio={stats['communication_ratio']:.3f}"
        )


def smoke_configs():
    print("\n=== Generated experiment configs ===")
    for spec in list_runs("smoke"):
        cfg = build_comm_experiment_config(spec["method"], spec["radius"])
        r = cfg["comm_sweep"]["comm_radius"]
        mode = cfg["guidance"]["mode"]
        print(f"  {spec['run_name']}: method={spec['method']} mode={mode} R={r}")


def main():
    smoke_comm_metrics()
    smoke_configs()
    print("\n[OK] Smoke checks passed. Run training:")
    print("  python scripts/run_communication_train.py --profile smoke --gate 2")


if __name__ == "__main__":
    main()

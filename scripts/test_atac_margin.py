#!/usr/bin/env python3
"""Phase-3 ATAC: capacity margin fixed radius vs optimized radius."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from models.transport.atac.atac_optimizer import ATACOptimizer


def main() -> int:
    opt = ATACOptimizer(n_agents=4, t_max=20.0, L0=3.0, radius_bounds=(1.5, 4.5))
    # demanding force along +x
    f_des = torch.tensor([25.0, 5.0])
    r_fixed = 3.0
    eta_fixed = float(opt.compute_capacity_margin(f_des, opt._ring_dirs(r_fixed)))
    out = opt.optimize(f_des, radius_init=r_fixed, n_grid=15)
    eta_opt = float(out["margin"])
    print(
        f"atac compare: fixed_r={r_fixed:.2f} eta={eta_fixed:.3f} | "
        f"opt_r={out['radius']:.2f} eta={eta_opt:.3f}"
    )
    assert eta_opt >= eta_fixed - 1e-6, (eta_fixed, eta_opt)
    gain = (eta_opt - eta_fixed) / max(abs(eta_fixed), 1e-3)
    # when fixed is poor, optimizer should improve by >=10% relative or absolute
    assert eta_opt - eta_fixed >= 0.1 * max(abs(eta_fixed), 1.0) or gain >= 0.1, (
        eta_fixed,
        eta_opt,
        gain,
    )

    out_dir = ROOT / "experiment_results" / "atac"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(["fixed r", "ATAC"], [eta_fixed, eta_opt], color=["#888", "#6a2"])
    ax.set_ylabel("capacity margin η")
    ax.set_title("Wrench capacity margin vs ring radius")
    fig.tight_layout()
    png = out_dir / "vmas_compare.png"
    fig.savefig(png, dpi=120)
    plt.close(fig)
    print(f"wrote {png}")
    print("test_atac_margin: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

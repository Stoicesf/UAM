#!/usr/bin/env python3
"""Phase-3 SC-RISE vs plain RISE under wind: tracking + CBF residual."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from environments.scenarios.cooperative_transport import CooperativeTransportEnv
from models.transport.control.hierarchical import TransportHierarchicalController


def _run(use_scrise: bool, wind: float = 0.5, steps: int = 150, seed: int = 1) -> dict:
    env = CooperativeTransportEnv(
        n_agents=16,
        seed=seed,
        max_steps=steps,
        use_hybrid=True,
        wind_force=wind,
        target_pos=(5.0, 5.0),
    )
    obs, _ = env.reset(seed=seed)
    ctl = TransportHierarchicalController(
        env,
        use_rise=True,
        use_scrise=use_scrise,
        use_shield=False,
    )
    min_resid = 1e9
    for _ in range(steps):
        obs, _, done, _, info = env.step(ctl.act(obs))
        min_resid = min(min_resid, float(getattr(ctl, "_last_cbf_residual", 0.0)))
        if done:
            break
    return {
        "payload_d": float(info["payload_distance"]),
        "min_cbf_residual": min_resid,
        "steps": int(info["step"]),
    }


def main() -> int:
    rise = _run(use_scrise=False)
    scrise = _run(use_scrise=True)
    print(
        f"scrise compare: RISE d={rise['payload_d']:.3f} resid={rise['min_cbf_residual']:.3f} | "
        f"SC-RISE d={scrise['payload_d']:.3f} resid={scrise['min_cbf_residual']:.3f}"
    )
    assert scrise["payload_d"] < 0.5, scrise
    # SC-RISE should keep relaxed CBF residual non-negative (allow tiny float noise)
    assert scrise["min_cbf_residual"] >= -0.05, scrise

    out_dir = ROOT / "experiment_results" / "scrise"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(7, 3))
    ax[0].bar(["RISE", "SC-RISE"], [rise["payload_d"], scrise["payload_d"]], color=["#888", "#a62"])
    ax[0].set_title("payload distance")
    ax[1].bar(
        ["RISE", "SC-RISE"],
        [rise["min_cbf_residual"], scrise["min_cbf_residual"]],
        color=["#888", "#a62"],
    )
    ax[1].axhline(0.0, color="r", ls="--", lw=1)
    ax[1].set_title("min CBF residual")
    fig.tight_layout()
    png = out_dir / "vmas_compare.png"
    fig.savefig(png, dpi=120)
    plt.close(fig)
    print(f"wrote {png}")
    print("test_scrise_wind: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

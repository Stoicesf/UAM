#!/usr/bin/env python3
"""Phase-3 M-UBF formation/collision compare vs baseline hierarchical."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from environments.scenarios.cooperative_transport import CooperativeTransportEnv
from models.transport.control.hierarchical import TransportHierarchicalController


def _min_sep(pos: torch.Tensor) -> float:
    n = pos.shape[0]
    dmat = torch.cdist(pos, pos)
    eye = torch.eye(n)
    return float((dmat + eye * 1e6).min())


def _run(use_mubf: bool, steps: int = 120, seed: int = 0) -> dict:
    env = CooperativeTransportEnv(
        n_agents=16, seed=seed, max_steps=steps, use_hybrid=True, target_pos=(5.0, 5.0)
    )
    # pack agents closer to stress collisions
    obs, info = env.reset(seed=seed)
    env.pos = env.pos * 0.55 + env.payload_pos.unsqueeze(0)
    ctl = TransportHierarchicalController(
        env, use_rise=True, use_mubf=use_mubf, use_shield=False
    )
    form_errs = []
    min_seps = []
    for _ in range(steps):
        act = ctl.act(obs)
        obs, _, done, _, info = env.step(act)
        form_errs.append(float(info["formation_error"]))
        min_seps.append(_min_sep(env.pos))
        if done:
            break
    return {
        "form_err": form_errs[-1],
        "form_mean": sum(form_errs) / len(form_errs),
        "min_sep": min(min_seps),
        "payload_d": float(info["payload_distance"]),
    }


def main() -> int:
    base = _run(use_mubf=False)
    mubf = _run(use_mubf=True)
    print(
        f"mubf compare: base form={base['form_err']:.3f} sep={base['min_sep']:.3f} | "
        f"mubf form={mubf['form_err']:.3f} sep={mubf['min_sep']:.3f}"
    )
    assert mubf["min_sep"] >= 0.5 - 1e-3, mubf
    # formation should not be worse than ~baseline; prefer improvement
    assert mubf["form_err"] <= max(base["form_err"] * 1.05, 0.35), (base, mubf)

    out_dir = ROOT / "experiment_results" / "mubf"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(7, 3))
    ax[0].bar(["base", "M-UBF"], [base["form_err"], mubf["form_err"]], color=["#888", "#26a"])
    ax[0].set_title("formation error")
    ax[1].bar(["base", "M-UBF"], [base["min_sep"], mubf["min_sep"]], color=["#888", "#26a"])
    ax[1].axhline(0.5, color="r", ls="--", lw=1)
    ax[1].set_title("min separation")
    fig.tight_layout()
    png = out_dir / "vmas_compare.png"
    fig.savefig(png, dpi=120)
    plt.close(fig)
    print(f"wrote {png}")
    print("test_mubf_formation: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

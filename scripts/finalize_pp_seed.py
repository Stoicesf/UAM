"""Finalize one AC-DSGF++ seed when train finished but eval hung.

Writes summary.json + runs Case A/B/C health check.

Usage:
  python scripts/finalize_pp_seed.py --run results/ac_dsgf_pp/uav16/s42 --episodes 64
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.eval_rollout import evaluate_policy
from utils.experiment import load_experiment_config
from utils.policy_loader import load_policy_for_eval, resolve_checkpoint
from algorithms.guided.mappo_guided import ACGuideAdapterPP
import torch
from torchrl.envs.utils import step_mdp


def _late_mean(path: Path, key: str, n: int = 5) -> float | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    vals = [float(r[key]) for r in rows[-n:] if r.get(key) not in (None, "")]
    return sum(vals) / len(vals) if vals else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--exp", default="configs/ac_dsgf_pp/ac_dsgf_pp_16uav.yaml")
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--ckpt", default=None)
    args = parser.parse_args()

    run = Path(args.run)
    ckpt = Path(args.ckpt) if args.ckpt else run / "checkpoints" / "final.pt"
    if not ckpt.exists():
        print(f"Missing checkpoint: {ckpt}")
        return 1

    cfg = load_experiment_config(str(ROOT / args.exp) if not Path(args.exp).is_absolute() else args.exp)
    # Faster eval: single env
    cfg["env"] = {**cfg["env"], "num_envs": 1}
    cfg["train"] = {**cfg["train"], "eval_episodes": args.episodes}

    print(f"Eval {ckpt} episodes={args.episodes} num_envs=1 ...")
    env, policy = load_policy_for_eval(cfg, resolve_checkpoint(str(ckpt)))
    stats = evaluate_policy(
        env,
        policy,
        num_episodes=args.episodes,
        success_threshold=float(cfg["env"].get("success_threshold", 0.3)),
        max_steps=int(cfg["env"].get("max_steps", 128)),
    )

    adapter = None
    for m in policy.modules():
        if isinstance(m, ACGuideAdapterPP):
            adapter = m
            break
    gates = []
    with torch.no_grad():
        td = env.reset()
        for _ in range(64):
            td = policy(td)
            if adapter is not None and adapter.last_soft_edges is not None:
                gates.append(float(adapter.last_soft_edges))
            td = env.step(td)
            done = td.get(("next", "done"))
            if done is not None and bool(done.any()):
                td = env.reset()
            else:
                td = step_mdp(td)
    comm = sum(gates) / len(gates) if gates else _late_mean(run / "logs" / "gate_mass.csv", "gate_mass") or 0.0

    success = float(stats.get("success_rate", stats.get("success", 0.0)))
    if success > 1.0:
        success = success / 100.0

    corr = _late_mean(run / "logs" / "utility_corr.csv", "corr_u_ustar")
    prec = _late_mean(run / "logs" / "comm_precision.csv", "comm_precision")
    if prec is None:
        prec = _late_mean(run / "logs" / "utility_corr.csv", "comm_precision")

    paper = {
        "success": success,
        "collision": float(stats.get("collision_rate", 0.0)),
        "reward": float(stats.get("episode_reward_mean", stats.get("mean_reward", 0.0))),
        "path_length": float(stats.get("path_length", 0.0)),
        "alignment": float(stats.get("action_alignment", 0.0)),
        "communication_cost": comm,
        "eval_episodes": args.episodes,
    }
    summary = {
        "run": str(run),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "finalize_note": "post-hoc eval (train completed; original 200-ep eval interrupted)",
        "paper_metrics": paper,
        "success": success,
        "communication_cost_mean": comm,
        "eval_stats": stats,
        "utility_corr_late": corr,
        "comm_precision_late": prec,
        "tag": "AC_DSGF_PP_16UAV",
    }
    out = run / "summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out}")
    print(f"Success={success*100:.2f}%  Comm={comm:.4f}  corr={corr}  prec={prec}")

    rc = subprocess.call(
        [
            sys.executable,
            str(ROOT / "scripts" / "check_pp_seed_health.py"),
            "--run",
            str(run),
        ],
        cwd=str(ROOT),
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

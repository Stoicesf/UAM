"""Unified CLI for smoke / run / eval scripts.

Usage:
  python scripts/uam.py smoke secdo
  python scripts/uam.py run seeds --exp configs/experiments/exp0_baseline.yaml
  python scripts/uam.py eval generalization --dry-run
  python scripts/uam.py tro budget --seeds 42 1234
"""

from __future__ import annotations

import argparse
import importlib
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SMOKE = {
    "secdo": "scripts.smoke_secdo",
    "secdo_training": "scripts.smoke_secdo_training",
    "secdo_uav_teacher": "scripts.smoke_secdo_uav_teacher",
    "ac_dsgf": "scripts.smoke_ac_dsgf",
    "ac_dsgf_pp": "scripts.smoke_ac_dsgf_pp",
    "communication": "scripts.smoke_communication",
}

RUN = {
    "seeds": "scripts.run_seeds",
    "baseline16": "scripts.run_baseline16",
    "scalability": "scripts.run_scalability",
    "secdo_ablation": "scripts.run_secdo_ablation",
    "secdo_v2_full": "scripts.run_secdo_v2_full",
    "communication_train": "scripts.run_communication_train",
    "generalization_train": "scripts.run_generalization_train",
}

EVAL = {
    "communication": "scripts.eval_communication",
    "comm_budget": "scripts.eval_comm_budget",
    "comm_ablation": "scripts.eval_comm_ablation",
    "causal": "scripts.eval_causal_analysis",
    "generalization": "scripts.eval_generalization",
    "packet_loss": "scripts.eval_packet_loss",
    "scalability": "scripts.eval_scalability_interpretability",
}

TRO = {
    "budget": "scripts.collect_tro_budget_statistics",
    "theory": "scripts.collect_tro_theory_evidence",
    "channel": "scripts.collect_tro_channel_robustness",
    "scalability": "scripts.collect_tro_scalability",
    "baselines": "scripts.collect_tro_baseline_comparison",
}


def _run_module(module: str, argv: list[str]) -> int:
    sys.argv = [module.split(".")[-1], *argv]
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    except SystemExit as e:
        code = e.code
        return int(code) if isinstance(code, int) else (0 if code is None else 1)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="UAM unified CLI")
    sub = parser.add_subparsers(dest="group", required=True)

    for name, table in (
        ("smoke", SMOKE),
        ("run", RUN),
        ("eval", EVAL),
        ("tro", TRO),
    ):
        p = sub.add_parser(name, help=f"{name} scripts")
        p.add_argument("task", choices=sorted(table))
        p.add_argument("args", nargs=argparse.REMAINDER, help="passed to underlying script")

    args = parser.parse_args(argv)
    table = {"smoke": SMOKE, "run": RUN, "eval": EVAL, "tro": TRO}[args.group]
    module = table[args.task]
    extra = [a for a in args.args if a != "--"]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return _run_module(module, extra)


if __name__ == "__main__":
    raise SystemExit(main())

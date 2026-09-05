#!/usr/bin/env python3
"""Hierarchical RL training entry (high role selector + conditional low policy).

  python scripts/hierarchical/train.py --n 16 --rounds 1 --high_steps 5000 --low_steps 5000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from environments.dice_vmas_env import DICEVMASEnv
from models.hierarchical.high_level import HighLevelRoleSelector
from models.hierarchical.low_level import LowLevelConditionalPolicy
from models.hierarchical.trainer import HierarchicalTrainer
from models.hierarchical.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--high_steps", type=int, default=5000)
    parser.add_argument("--low_steps", type=int, default=5000)
    parser.add_argument("--decision_interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scout_weight", type=float, default=0.5)
    parser.add_argument("--max_steps", type=int, default=None, help="env episode length")
    parser.add_argument("--save_dir", type=str, default="experiment_results/hierarchical/")
    parser.add_argument("--config", type=str, default="configs/hierarchical/default.yaml")
    args = parser.parse_args()

    config: dict = {}
    cfg_path = Path(args.config)
    if cfg_path.exists():
        config = load_config(cfg_path)
    config.update({k: v for k, v in vars(args).items() if v is not None})
    ep_len = int(args.max_steps if args.max_steps is not None else config.get("max_steps", 200))
    config["max_steps"] = ep_len
    config["save_dir"] = args.save_dir

    env = DICEVMASEnv(
        n_agents=args.n,
        seed=args.seed,
        heterogeneous=True,
        hetero_role_bias=False,
        n_roles=3,
        scout_weight=args.scout_weight,
        max_steps=ep_len,
    )
    env.reset(seed=args.seed)
    global_obs_dim = env.get_global_obs().numel()

    high_policy = HighLevelRoleSelector(
        global_obs_dim, n_roles=3, decision_interval=args.decision_interval
    )
    low_policy = LowLevelConditionalPolicy(env.obs_dim, n_roles=3)

    trainer = HierarchicalTrainer(env, high_policy, low_policy, config)
    trainer.alternate_train(args.rounds, args.high_steps, args.low_steps)


if __name__ == "__main__":
    main()

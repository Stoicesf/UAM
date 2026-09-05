#!/usr/bin/env python3
"""Hierarchical PPO training entry (GPU-capable, reward_formula for Pareto tuning).

  python scripts/hierarchical/train_ppo.py --n 16 --device cuda \\
      --reward_formula "1.2*coverage - 0.5*collision"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from environments.dice_vmas_env import DICEVMASEnv
from models.hierarchical.ac_network import LowLevelActorCritic, RoleActorCritic
from models.hierarchical.trainer_ppo import HierarchicalPPOTrainer
from models.hierarchical.utils import eval_reward_formula, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--high_steps", type=int, default=30000)
    parser.add_argument("--low_steps", type=int, default=30000)
    parser.add_argument("--decision_interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scout_weight", type=float, default=0.5)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--gae_lambda", type=float, default=None)
    parser.add_argument("--clip_epsilon", type=float, default=None)
    parser.add_argument(
        "--reward_formula",
        type=str,
        default="coverage - 0.5*collision",
        help='e.g. "1.2*coverage - 0.5*collision" (only coverage/collision)',
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--hidden", type=int, default=128, help="high AC shared width")
    parser.add_argument("--save_dir", type=str, default="experiment_results/hierarchical_ppo/")
    parser.add_argument("--config", type=str, default="configs/hierarchical/ppo.yaml")
    args = parser.parse_args()

    # validate formula early
    _ = eval_reward_formula(args.reward_formula, 0.5, 0.1)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to cpu", flush=True)
        device = "cpu"

    config: dict = {}
    cfg_path = Path(args.config)
    if cfg_path.exists():
        config = load_config(cfg_path)
    config.update({k: v for k, v in vars(args).items() if v is not None})
    ep_len = int(args.max_steps if args.max_steps is not None else config.get("max_steps", 200))
    config["max_steps"] = ep_len
    config["save_dir"] = args.save_dir
    config["epochs"] = args.epochs
    config["device"] = device
    config["reward_formula"] = args.reward_formula
    # Pareto focus: formula replaces lsr/match shaping
    config["match_weight"] = 0.0
    config["lsr_coef"] = 0.0

    env = DICEVMASEnv(
        n_agents=args.n,
        seed=args.seed,
        heterogeneous=True,
        hetero_role_bias=False,
        n_roles=3,
        scout_weight=args.scout_weight,
        max_steps=ep_len,
        device="cpu",
    )
    env.reward_formula = args.reward_formula
    env.reset(seed=args.seed)
    global_obs_dim = env.get_global_obs().numel()

    high_ac = RoleActorCritic(
        global_obs_dim,
        n_roles=3,
        decision_interval=args.decision_interval,
        hidden=args.hidden,
    )
    low_ac = LowLevelActorCritic(env.obs_dim, n_roles=3)

    print(f"train device={device} formula={args.reward_formula!r} hidden={args.hidden}", flush=True)
    trainer = HierarchicalPPOTrainer(env, high_ac, low_ac, config)
    trainer.alternate_train(args.rounds, args.high_steps, args.low_steps)


if __name__ == "__main__":
    main()

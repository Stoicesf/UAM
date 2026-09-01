"""Quick check that 32-agent VMAS navigation spawns without hanging."""
from __future__ import annotations

import time

from env.vmas_env import make_torchrl_env
from utils.experiment import load_experiment_config


def main() -> None:
    cfg = load_experiment_config("configs/scalability/uav32.yaml")
    env_cfg = cfg["env"]
    print("scenario_kwargs:", env_cfg.get("scenario_kwargs"))
    print("num_agents:", env_cfg["num_agents"], "num_envs:", env_cfg["num_envs"])

    t0 = time.time()
    env = make_torchrl_env(env_cfg)
    env.reset()
    dt = time.time() - t0
    print(f"OK: reset in {dt:.1f}s, n_agents={env.n_agents}")


if __name__ == "__main__":
    main()

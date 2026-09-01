"""MAPPO — 基于 TorchRL 官方 multiagent PPO 教程。

参考: https://docs.pytorch.org/rl/stable/tutorials/multiagent_ppo.html
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from tensordict.nn import TensorDictModule, set_composite_lp_aggregate
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.collectors import Collector
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from torchrl.objectives.multiagent import MAPPOLoss

from algorithms.baseline.advantage import setup_gae
from algorithms.baseline.buffer import make_replay_buffer
from env.vmas_env import make_torchrl_env


@dataclass
class MAPPOComponents:
    env: Any
    policy: ProbabilisticActor
    critic: TensorDictModule
    loss_module: MAPPOLoss
    collector: Collector
    replay_buffer: Any
    optimizer: torch.optim.Optimizer
    gae: Any
    device: torch.device


def build_mappo(train_cfg: dict, env_cfg: dict) -> MAPPOComponents:
    """构建完整 MAPPO 训练组件 (DSGF 关闭)。"""
    set_composite_lp_aggregate(False).set()

    device = torch.device(
        env_cfg.get("device", "cpu")
        if torch.cuda.is_available() or env_cfg.get("device") == "cpu"
        else "cpu"
    )
    env = make_torchrl_env(env_cfg)
    vmas_device = device

    hidden = train_cfg.get("hidden_dim", 128)
    obs_dim = env.observation_spec["agents", "observation"].shape[-1]
    action_dim = env.full_action_spec[env.action_key].shape[-1]

    policy_net = torch.nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=obs_dim,
            n_agent_outputs=2 * action_dim,
            n_agents=env.n_agents,
            centralised=False,
            share_params=train_cfg.get("share_params_policy", True),
            device=device,
            depth=2,
            num_cells=hidden,
            activation_class=torch.nn.Tanh,
        ),
        NormalParamExtractor(),
    )
    policy_module = TensorDictModule(
        policy_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )
    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec_unbatched,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[env.action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.full_action_spec_unbatched[env.action_key].space.low,
            "high": env.full_action_spec_unbatched[env.action_key].space.high,
        },
        return_log_prob=True,
    )

    critic_net = MultiAgentMLP(
        n_agent_inputs=obs_dim,
        n_agent_outputs=1,
        n_agents=env.n_agents,
        centralised=True,
        share_params=train_cfg.get("share_params_critic", True),
        device=device,
        depth=2,
        num_cells=hidden,
        activation_class=torch.nn.Tanh,
    )
    critic = TensorDictModule(
        module=critic_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "state_value")],
    )

    loss_module = MAPPOLoss(
        actor_network=policy,
        critic_network=critic,
        clip_epsilon=train_cfg.get("clip_epsilon", 0.2),
        entropy_coeff=train_cfg.get("entropy_coef", 0.01),
        critic_coeff=train_cfg.get("critic_coef", 1.0),
    )
    loss_module.set_keys(
        reward=env.reward_key,
        action=env.action_key,
        value=("agents", "state_value"),
        done=("agents", "done"),
        terminated=("agents", "terminated"),
    )
    gae = setup_gae(
        loss_module,
        gamma=train_cfg.get("gamma", 0.99),
        gae_lambda=train_cfg.get("gae_lambda", 0.95),
    )

    frames_per_batch = train_cfg.get("frames_per_batch", 2048)
    total_frames = train_cfg.get("total_frames", 100_000)

    collector = Collector(
        env,
        policy,
        device=vmas_device,
        storing_device=device,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        auto_register_policy_transforms=True,
    )
    replay_buffer = make_replay_buffer(
        frames_per_batch,
        train_cfg.get("minibatch_size", 512),
        device,
    )
    optimizer = torch.optim.Adam(
        loss_module.parameters(),
        lr=train_cfg.get("learning_rate", 3e-4),
    )

    return MAPPOComponents(
        env=env,
        policy=policy,
        critic=critic,
        loss_module=loss_module,
        collector=collector,
        replay_buffer=replay_buffer,
        optimizer=optimizer,
        gae=gae,
        device=device,
    )

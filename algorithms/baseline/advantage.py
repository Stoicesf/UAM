"""GAE 优势估计 — 复用 TorchRL MAPPOLoss 内置 GAE。"""

from __future__ import annotations

from torchrl.objectives import ValueEstimators


def setup_gae(loss_module, gamma: float, gae_lambda: float):
    """为 MAPPO loss 挂载 GAE value estimator。"""
    loss_module.make_value_estimator(
        ValueEstimators.GAE,
        gamma=gamma,
        lmbda=gae_lambda,
    )
    return loss_module.value_estimator


def expand_agent_done_flags(tensordict_data, env):
    """将 team-level done 扩展到 per-agent shape (TorchRL MAPPO 要求)。"""
    reward_key = env.reward_key
    next_td = tensordict_data.get("next")

    done = next_td.get("done").unsqueeze(-1).expand(
        tensordict_data.get_item_shape(("next", reward_key))
    )
    terminated = next_td.get("terminated").unsqueeze(-1).expand(
        tensordict_data.get_item_shape(("next", reward_key))
    )
    tensordict_data.set(("next", "agents", "done"), done)
    tensordict_data.set(("next", "agents", "terminated"), terminated)
    return tensordict_data


def compute_gae(
    gae_module,
    tensordict_data,
    critic_params,
    target_critic_params,
):
    """计算 advantage 与 value_target，写入 tensordict。"""
    with __import__("torch").no_grad():
        gae_module(
            tensordict_data,
            params=critic_params,
            target_params=target_critic_params,
        )
    return tensordict_data

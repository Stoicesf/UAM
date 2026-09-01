"""MAPPO 训练循环 — 采样 / GAE / PPO Update / 日志 / Checkpoint。"""

from __future__ import annotations

import torch
from pathlib import Path
from typing import Any
from tqdm import tqdm

from algorithms.baseline.advantage import compute_gae, expand_agent_done_flags
from algorithms.baseline.mappo import MAPPOComponents, build_mappo
from utils.logger import Logger


def save_checkpoint(components: MAPPOComponents, path: str, step: int, reward: float):
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "reward": reward,
        "actor": components.policy.state_dict(),
        "critic": components.critic.state_dict(),
        "policy": components.policy.state_dict(),
        "loss_module": components.loss_module.state_dict(),
        "optimizer": components.optimizer.state_dict(),
    }
    if hasattr(components, "guide_encoder"):
        payload["guide_encoder"] = components.guide_encoder.state_dict()
    torch.save(payload, str(ckpt_path))
    if not ckpt_path.exists():
        raise RuntimeError(f"Checkpoint save failed: {ckpt_path}")


def load_checkpoint(components: MAPPOComponents, path: str) -> dict:
    ckpt = torch.load(path, map_location=components.device, weights_only=False)
    components.policy.load_state_dict(ckpt["policy"])
    components.critic.load_state_dict(ckpt["critic"])
    components.loss_module.load_state_dict(ckpt["loss_module"])
    components.optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


def train_mappo(
    train_cfg: dict[str, Any],
    env_cfg: dict[str, Any],
    run_name: str | None = None,
) -> list[float]:
    """运行 MAPPO baseline 训练，返回 episode reward 曲线。"""
    if train_cfg.get("use_guidance"):
        raise RuntimeError("DSGF must stay OFF until baseline is validated. Set use_guidance: false")

    components = build_mappo(train_cfg, env_cfg)
    env = components.env
    loss_module = components.loss_module
    replay_buffer = components.replay_buffer
    optimizer = components.optimizer

    frames_per_batch = train_cfg.get("frames_per_batch", 2048)
    num_epochs = train_cfg.get("num_epochs", 5)
    minibatch_size = train_cfg.get("minibatch_size", 512)
    max_grad_norm = train_cfg.get("max_grad_norm", 0.5)
    save_interval = train_cfg.get("save_interval", 10_000)

    log_dir = Path(train_cfg.get("log_dir", "logs"))
    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
    if run_name:
        log_dir = log_dir / run_name
        ckpt_dir = ckpt_dir / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(str(log_dir))
    reward_curve: list[float] = []
    global_step = 0
    n_minibatches = max(1, frames_per_batch // minibatch_size)

    pbar = tqdm(components.collector, desc="episode_reward=0.0")
    for i, tensordict_data in enumerate(pbar):
        expand_agent_done_flags(tensordict_data, env)
        compute_gae(
            components.gae,
            tensordict_data,
            loss_module.critic_network_params,
            loss_module.target_critic_network_params,
        )

        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view)

        for _ in range(num_epochs):
            for _ in range(n_minibatches):
                subdata = replay_buffer.sample()
                loss_vals = loss_module(subdata)
                loss = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

        components.collector.update_policy_weights_()

        done = tensordict_data.get(("next", "agents", "done"))
        ep_reward = tensordict_data.get(("next", "agents", "episode_reward"))
        if done.any():
            mean_reward = ep_reward[done].mean().item()
        else:
            mean_reward = ep_reward.mean().item()

        reward_curve.append(mean_reward)
        global_step += frames_per_batch

        logger.log_scalar("train/episode_reward_mean", mean_reward, global_step)
        logger.log_scalar("train/loss_objective", loss_vals["loss_objective"].item(), global_step)
        logger.log_scalar("train/loss_critic", loss_vals["loss_critic"].item(), global_step)
        logger.log_scalar("train/loss_entropy", loss_vals["loss_entropy"].item(), global_step)

        pbar.set_description(f"episode_reward={mean_reward:.3f}", refresh=False)

        if global_step % save_interval < frames_per_batch:
            save_checkpoint(
                components,
                str(ckpt_dir / "latest.pt"),
                global_step,
                mean_reward,
            )

    save_checkpoint(components, str(ckpt_dir / "final.pt"), global_step, reward_curve[-1] if reward_curve else 0.0)
    logger.close()
    return reward_curve


def train_mappo_with_experiment(
    train_cfg: dict[str, Any],
    env_cfg: dict[str, Any],
    exp_run: "ExperimentRun",
    resume: str | None = None,
) -> list[float]:
    """Baseline MAPPO with experiment artifact management."""
    components = build_mappo(train_cfg, env_cfg)
    if resume:
        load_checkpoint(components, resume)

    env = components.env
    loss_module = components.loss_module
    replay_buffer = components.replay_buffer
    optimizer = components.optimizer

    frames_per_batch = train_cfg.get("frames_per_batch", 2048)
    num_epochs = train_cfg.get("num_epochs", 5)
    minibatch_size = train_cfg.get("minibatch_size", 512)
    max_grad_norm = train_cfg.get("max_grad_norm", 0.5)
    save_interval = train_cfg.get("save_interval", 10_000)

    logger = Logger(str(exp_run.log_dir))
    reward_curve: list[float] = []
    metrics_history: list[dict] = []
    saved_milestones: set[int] = set()
    global_step = 0
    n_minibatches = max(1, frames_per_batch // minibatch_size)

    pbar = tqdm(components.collector, desc="episode_reward=0.0")
    for tensordict_data in pbar:
        expand_agent_done_flags(tensordict_data, env)
        compute_gae(
            components.gae,
            tensordict_data,
            loss_module.critic_network_params,
            loss_module.target_critic_network_params,
        )

        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view)

        for _ in range(num_epochs):
            for _ in range(n_minibatches):
                subdata = replay_buffer.sample()
                loss_vals = loss_module(subdata)
                loss = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

        components.collector.update_policy_weights_()

        done = tensordict_data.get(("next", "agents", "done"))
        ep_reward = tensordict_data.get(("next", "agents", "episode_reward"))
        mean_reward = ep_reward[done].mean().item() if done.any() else ep_reward.mean().item()

        reward_curve.append(mean_reward)
        global_step += frames_per_batch

        metrics = {
            "episode_reward_mean": mean_reward,
            "loss_objective": loss_vals["loss_objective"].item(),
            "loss_critic": loss_vals["loss_critic"].item(),
            "loss_entropy": loss_vals["loss_entropy"].item(),
        }
        metrics_history.append(metrics)
        for k, v in metrics.items():
            logger.log_scalar(f"train/{k}", v, global_step)
        exp_run.log_metrics(global_step, metrics)

        pbar.set_description(f"episode_reward={mean_reward:.3f}", refresh=False)

        if global_step % save_interval < frames_per_batch:
            save_checkpoint(
                components,
                str(exp_run.checkpoint_latest),
                global_step,
                mean_reward,
            )

        from utils.training_helpers import maybe_save_milestones
        saved_milestones = maybe_save_milestones(
            global_step, saved_milestones, components, exp_run.ckpt_dir, mean_reward, save_checkpoint
        )

    final_r = reward_curve[-1] if reward_curve else 0.0
    save_checkpoint(components, str(exp_run.checkpoint_final), global_step, final_r)

    from utils.eval_rollout import evaluate_policy

    eval_stats = evaluate_policy(
        env,
        components.policy,
        num_episodes=train_cfg.get("eval_episodes", 200),
        success_threshold=env_cfg.get("success_threshold", 0.3),
        max_steps=env_cfg.get("max_steps"),
    )

    logger.close()
    exp_run.finalize(
        reward_curve,
        metrics_history=metrics_history,
        eval_stats=eval_stats,
        extra={"guidance_mode": "none"},
    )
    return reward_curve

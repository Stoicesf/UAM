"""Guided MAPPO training loop — Stage 1~4."""

from __future__ import annotations

from typing import Any

import torch
from tqdm import tqdm

from algorithms.baseline.advantage import compute_gae, expand_agent_done_flags
from algorithms.baseline.runner import save_checkpoint
from algorithms.guided.mappo_guided import build_guided_mappo
from utils.eval_rollout import evaluate_policy
from utils.experiment import ExperimentRun
from utils.guidance_schedule import compute_guidance_coef, compute_residual_beta
from utils.logger import Logger
from reward.guidance_reward import guidance_alignment_reward
from utils.comm_metrics import compute_sparse_communication_stats
from utils.reward_decompose import (
    compute_guide_reward_metrics,
    extract_vmas_reward_parts,
)
from utils.training_helpers import maybe_save_milestones


def _apply_guidance_reward(tensordict_data, env, coef: float):
    if coef <= 0:
        return
    actions = tensordict_data.get(("agents", "action"))
    phi = tensordict_data.get(("agents", "phi"))
    if phi is None:
        return
    from reward.guidance_reward import guidance_alignment_reward

    guide = guidance_alignment_reward(actions, phi)
    reward_key = ("next", env.reward_key)
    reward = tensordict_data.get(reward_key)
    tensordict_data.set(reward_key, reward + coef * guide.unsqueeze(-1))


def train_guided_mappo(
    train_cfg: dict[str, Any],
    env_cfg: dict[str, Any],
    guidance_cfg: dict[str, Any],
    exp_run: ExperimentRun,
) -> list[float]:
    components = build_guided_mappo(train_cfg, env_cfg, guidance_cfg)
    env = components.env
    loss_module = components.loss_module
    replay_buffer = components.replay_buffer
    optimizer = components.optimizer

    frames_per_batch = train_cfg.get("frames_per_batch", 2048)
    total_frames = train_cfg.get("total_frames", 102_400)
    num_epochs = train_cfg.get("num_epochs", 5)
    minibatch_size = train_cfg.get("minibatch_size", 512)
    max_grad_norm = train_cfg.get("max_grad_norm", 0.5)
    save_interval = train_cfg.get("save_interval", 10_240)
    use_guide_reward = guidance_cfg.get("use_guidance_reward", True)
    max_guide_coef = guidance_cfg.get("guidance_reward_coef", 0.1)
    use_warmup = guidance_cfg.get("guidance_warmup", False)
    warmup_fraction = guidance_cfg.get("guidance_warmup_fraction", 0.2)
    schedule = guidance_cfg.get("guidance_schedule")
    if schedule is None:
        schedule = "warmup_ramp" if use_warmup else "fixed"
    min_coef = guidance_cfg.get("guidance_coef_min", 0.02)
    decay_k = guidance_cfg.get("guidance_decay_k", 3.0)
    use_residual = guidance_cfg.get("residual_policy", False)
    beta_max = guidance_cfg.get("beta_max", 1.0)
    beta_min = guidance_cfg.get("beta_min", 0.0)
    beta_warmup_fraction = guidance_cfg.get("beta_warmup_fraction", 0.2)
    beta_decay_k = guidance_cfg.get("beta_decay_k", 3.0)
    n_agents = env_cfg.get("num_agents", env.n_agents)
    success_threshold = env_cfg.get("success_threshold", 0.3)
    eval_episodes = train_cfg.get("eval_episodes", 32)
    guidance_mode = guidance_cfg.get("mode", "none")
    use_ac_comm = guidance_mode in (
        "ac_dsgf",
        "ac-dsgf",
        "ac_dsgf_pp",
        "ac-dsgf-pp",
        "ac_dsgf++",
    )
    use_ac_pp = guidance_mode in ("ac_dsgf_pp", "ac-dsgf-pp", "ac_dsgf++")
    lambda_comm_final = float(
        guidance_cfg.get("lambda_comm", guidance_cfg.get("lambda_c", 1e-3))
    )
    lambda_comm = lambda_comm_final
    utility_warmup_steps = int(guidance_cfg.get("utility_warmup_steps", 0)) if use_ac_pp else 0
    lambda_u = float(guidance_cfg.get("lambda_u", 0.1)) if use_ac_pp else 0.0
    lambda_rank = float(guidance_cfg.get("lambda_rank", 0.0)) if use_ac_pp else 0.0
    ranking_mode = str(guidance_cfg.get("ranking_mode", "hinge")) if use_ac_pp else "hinge"
    # Soft budget (v2d) + optional curriculum anneal (v2e)
    lambda_budget_final = float(guidance_cfg.get("lambda_budget", 0.0)) if use_ac_pp else 0.0
    lambda_budget = lambda_budget_final
    target_comm = float(guidance_cfg.get("target_comm", 0.5)) if use_ac_pp else 0.5
    from models.communication.budget_constraint import scheduled_budget

    ac_adapter = getattr(components, "guide_encoder", None) if use_ac_comm else None
    from algorithms.guided.mappo_guided import ACGuideAdapter, ACGuideAdapterPP

    if use_ac_comm and not isinstance(ac_adapter, ACGuideAdapter):
        ac_adapter = None
    ac_pp_adapter = ac_adapter if isinstance(ac_adapter, ACGuideAdapterPP) else None

    def _scheduled_lambda_comm(step: int) -> float:
        """Causal utility warm-up: λ_comm=0 until utility sees action differences."""
        if not use_ac_pp or utility_warmup_steps <= 0:
            return lambda_comm_final
        if step < utility_warmup_steps:
            return 0.0
        return lambda_comm_final

    def _scheduled_budget_pair(step: int) -> tuple[float, float]:
        """(λ_b, target_comm) — fixed or curriculum anneal."""
        if not use_ac_pp:
            return 0.0, target_comm
        return scheduled_budget(
            step,
            guidance_cfg,
            fallback_lambda=lambda_budget_final,
            fallback_target=target_comm,
            utility_warmup_steps=utility_warmup_steps,
        )

    def _scheduled_gate_alpha(step: int) -> float:
        """During warm-up keep more base gate; after, trust utility gate more."""
        if not use_ac_pp:
            return 0.5
        a0 = float(guidance_cfg.get("gate_residual_alpha", 0.5))
        a1 = float(guidance_cfg.get("gate_residual_alpha_after", 0.8))
        if utility_warmup_steps <= 0 or step < utility_warmup_steps:
            return a0
        return a1

    def _apply_gate_alpha(step: int) -> None:
        if ac_pp_adapter is None:
            return
        alpha = _scheduled_gate_alpha(step)
        ctrl = getattr(ac_pp_adapter.encoder, "controller", None)
        if ctrl is not None and hasattr(ctrl, "set_residual_alpha"):
            ctrl.set_residual_alpha(alpha)

    logger = Logger(str(exp_run.log_dir))
    pp_logger = None
    if use_ac_pp:
        from utils.ac_pp_logging import ACPlusPlusRunLogger

        pp_logger = ACPlusPlusRunLogger(exp_run.root)
    reward_curve: list[float] = []
    metrics_history: list[dict] = []
    saved_milestones: set[int] = set()
    global_step = 0
    n_minibatches = max(1, frames_per_batch // minibatch_size)

    pbar = tqdm(components.collector, desc="episode_reward=0.0")
    for tensordict_data in pbar:
        expand_agent_done_flags(tensordict_data, env)

        actions_for_align = tensordict_data.get(("agents", "action"))
        phi_for_align = tensordict_data.get(("agents", "phi"))
        current_alignment = 0.0
        if actions_for_align is not None and phi_for_align is not None:
            current_alignment = guidance_alignment_reward(
                actions_for_align, phi_for_align
            ).mean().item()

        effective_coef = compute_guidance_coef(
            global_step + frames_per_batch,
            total_frames,
            max_guide_coef,
            warmup_fraction,
            use_warmup,
            schedule=schedule,
            min_coef=min_coef,
            decay_k=decay_k,
            alignment=current_alignment,
        )

        residual_beta = 0.0
        if use_residual and hasattr(components, "residual_actor"):
            residual_beta = compute_residual_beta(
                global_step + frames_per_batch,
                total_frames,
                beta_max=beta_max,
                beta_min=beta_min,
                warmup_fraction=beta_warmup_fraction,
                use_warmup=True,
                decay_k=beta_decay_k,
            )
            components.residual_actor.set_beta(residual_beta)

        vmas_parts = extract_vmas_reward_parts(tensordict_data, env)
        guide_parts = compute_guide_reward_metrics(tensordict_data, effective_coef)

        actions = tensordict_data.get(("agents", "action"))
        path_length = actions[..., :2].norm(dim=-1).mean().item() if actions is not None else 0.0

        if use_guide_reward:
            _apply_guidance_reward(tensordict_data, env, effective_coef)

        compute_gae(
            components.gae,
            tensordict_data,
            loss_module.critic_network_params,
            loss_module.target_critic_network_params,
        )

        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view)

        # Schedule λ_comm / budget anneal + gate residual after utility warm-up
        next_step = global_step + frames_per_batch
        lambda_comm = _scheduled_lambda_comm(next_step)
        lambda_budget, target_comm = _scheduled_budget_pair(next_step)
        _apply_gate_alpha(next_step)

        for _ in range(num_epochs):
            for _ in range(n_minibatches):
                subdata = replay_buffer.sample()
                loss_vals = loss_module(subdata)
                loss = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                if use_ac_comm and ac_adapter is not None:
                    obs_mb = subdata.get(("agents", "observation"))
                    if obs_mb is not None:
                        if obs_mb.dim() == 2:
                            pass
                        # v2c: L = L_ppo + λ_c C + λ_u (MSE + λ_r Rank)
                        # v2d: L = L_ppo + λ_u L_u + λ_b L_b  (λ_c usually 0)
                        if lambda_comm > 0:
                            loss = loss + ac_adapter.communication_loss(
                                obs_mb, lambda_comm, n_agents=n_agents
                            )
                        if lambda_budget > 0:
                            loss = loss + ac_adapter.budget_loss(
                                obs_mb,
                                lambda_budget,
                                target_comm=target_comm,
                                n_agents=n_agents,
                            )
                        if ac_pp_adapter is not None and lambda_u > 0:
                            loss = loss + ac_pp_adapter.utility_loss(
                                obs_mb,
                                lambda_u,
                                n_agents=n_agents,
                                lambda_rank=lambda_rank,
                                ranking_mode=ranking_mode,
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

        comm_radius = guidance_cfg.get("comm_radius", env_cfg.get("comm_radius", 0.5))
        obs_batch = tensordict_data.get(("agents", "observation"))
        if use_ac_comm and ac_adapter is not None:
            soft_edges = ac_adapter.last_soft_edges
            full_edges = float(n_agents * (n_agents - 1))
            comm_stats = {
                "sparse_edges": soft_edges,
                "full_graph_edges": full_edges,
                "communication_ratio": soft_edges / max(full_edges, 1.0),
                "communication_cost": soft_edges,
            }
        elif obs_batch is not None and guidance_mode in ("dsfg", "gat", "graph", "full"):
            comm_stats = compute_sparse_communication_stats(obs_batch, comm_radius)
        else:
            n = n_agents
            full_edges = float(n * (n - 1))
            comm_stats = {
                "sparse_edges": full_edges,
                "full_graph_edges": full_edges,
                "communication_ratio": 1.0,
                "communication_cost": full_edges,
            }

        pp_stats: dict[str, float] = {}
        if ac_pp_adapter is not None and obs_batch is not None:
            with torch.no_grad():
                pp_stats = ac_pp_adapter.utility_diagnostics(
                    obs_batch, lambda_u=lambda_u, n_agents=n_agents
                )

        metrics = {
            "episode_reward_mean": mean_reward,
            "reward_goal": vmas_parts["reward_goal"],
            "reward_collision": vmas_parts["reward_collision"],
            "reward_guide": guide_parts["reward_guide"],
            "guidance_lambda": effective_coef,
            "residual_beta": residual_beta,
            "success_rate": vmas_parts["success_rate"],
            "collision_rate": vmas_parts["collision_rate"],
            "action_alignment": guide_parts["action_alignment"],
            "path_length": path_length,
            "sparse_edges": comm_stats["sparse_edges"],
            "full_graph_edges": comm_stats["full_graph_edges"],
            "communication_ratio": comm_stats["communication_ratio"],
            "loss_objective": loss_vals["loss_objective"].item(),
            "loss_critic": loss_vals["loss_critic"].item(),
            "loss_entropy": loss_vals["loss_entropy"].item(),
            "lambda_comm": lambda_comm if use_ac_comm else 0.0,
            "lambda_u": lambda_u if use_ac_pp else 0.0,
            "lambda_rank": lambda_rank if use_ac_pp else 0.0,
            "lambda_budget": lambda_budget if use_ac_pp else 0.0,
            "target_comm": target_comm if use_ac_pp else 0.0,
            "utility_warmup_steps": float(utility_warmup_steps) if use_ac_pp else 0.0,
        }
        if pp_stats:
            metrics["utility_mean"] = pp_stats.get("utility_mean", 0.0)
            metrics["utility_std"] = pp_stats.get("utility_std", 0.0)
            metrics["gate_mass"] = pp_stats.get("gate_mass", comm_stats["sparse_edges"])
            metrics["utility_corr"] = pp_stats.get("utility_corr", 0.0)
            metrics["comm_precision"] = pp_stats.get("comm_precision", 0.0)
            metrics["cud"] = pp_stats.get("cud", 0.0)
            if "utility_loss" in pp_stats:
                metrics["utility_loss"] = pp_stats["utility_loss"]
        metrics_history.append(metrics)

        for k, v in metrics.items():
            logger.log_scalar(f"train/{k}", v, global_step)
            if k.startswith("reward_"):
                logger.log_scalar(f"reward/{k.replace('reward_', '')}", v, global_step)
        logger.log_scalar("train/guidance_lambda", effective_coef, global_step)
        if use_residual:
            logger.log_scalar("train/residual_beta", residual_beta, global_step)
        if use_ac_comm:
            logger.log_scalar("train/lambda_comm", lambda_comm, global_step)
            logger.log_scalar("train/soft_edges", comm_stats["sparse_edges"], global_step)
        if use_ac_pp:
            logger.log_scalar("train/lambda_u", lambda_u, global_step)
            logger.log_scalar("train/lambda_budget", lambda_budget, global_step)
            logger.log_scalar("train/target_comm", target_comm, global_step)
            if pp_stats:
                logger.log_scalar("train/utility_mean", metrics["utility_mean"], global_step)
                logger.log_scalar("train/utility_corr", metrics["utility_corr"], global_step)
                logger.log_scalar("train/gate_mass", metrics["gate_mass"], global_step)
                logger.log_scalar("train/comm_precision", metrics["comm_precision"], global_step)
                logger.log_scalar("train/cud", metrics["cud"], global_step)
                if pp_logger is not None:
                    pp_logger.log_step(
                        global_step,
                        utility_mean=metrics["utility_mean"],
                        utility_std=metrics["utility_std"],
                        gate_mass=metrics["gate_mass"],
                        utility_corr=metrics["utility_corr"],
                        utility_loss=metrics.get("utility_loss"),
                        comm_precision=metrics.get("comm_precision"),
                        cud=metrics.get("cud"),
                    )

        exp_run.log_metrics(global_step, metrics)
        exp_run.log_action_alignment(
            global_step, guide_parts["action_alignment"], guide_parts["reward_guide"]
        )
        exp_run.log_communication(
            global_step,
            comm_stats["communication_cost"],
            comm_stats["communication_ratio"],
            sparse_edges=comm_stats["sparse_edges"],
            full_graph_edges=comm_stats["full_graph_edges"],
        )

        pbar.set_description(
            f"R={mean_reward:.2f} "
            f"{'beta' if use_residual else 'lam'}="
            f"{residual_beta if use_residual else effective_coef:.3f} "
            f"align={guide_parts['action_alignment']:.3f}"
            + (f" edges={comm_stats['sparse_edges']:.1f}" if use_ac_comm else "")
            + (
                f" U={pp_stats.get('utility_mean', 0):.2f}"
                f" r={pp_stats.get('utility_corr', 0):.2f}"
                if pp_stats
                else ""
            )
            + f" step={global_step}",
            refresh=False,
        )

        if global_step % save_interval < frames_per_batch:
            save_checkpoint(
                components, str(exp_run.checkpoint_latest), global_step, mean_reward
            )

        saved_milestones = maybe_save_milestones(
            global_step, saved_milestones, components, exp_run.ckpt_dir, mean_reward, save_checkpoint
        )

    final_r = reward_curve[-1] if reward_curve else 0.0
    save_checkpoint(components, str(exp_run.checkpoint_final), global_step, final_r)

    eval_stats = evaluate_policy(
        env, components.policy,
        num_episodes=eval_episodes,
        success_threshold=success_threshold,
        max_steps=env_cfg.get("max_steps"),
    )

    logger.close()
    if pp_logger is not None:
        pp_logger.close()
    exp_run.finalize(
        reward_curve,
        metrics_history=metrics_history,
        eval_stats=eval_stats,
        extra={
            "guidance_mode": guidance_cfg.get("mode"),
            "residual_policy": use_residual,
            "max_guide_coef": max_guide_coef,
            "guidance_warmup": use_warmup,
            "guidance_schedule": schedule,
            "guidance_coef_min": min_coef,
            "guidance_decay_k": decay_k,
            "beta_max": beta_max if use_residual else None,
            "beta_min": beta_min if use_residual else None,
            "lambda_comm": lambda_comm if use_ac_comm else None,
            "lambda_comm_final": lambda_comm_final if use_ac_comm else None,
            "lambda_u": lambda_u if use_ac_pp else None,
            "lambda_rank": lambda_rank if use_ac_pp else None,
            "ranking_mode": ranking_mode if use_ac_pp else None,
            "lambda_budget": lambda_budget_final if use_ac_pp else None,
            "target_comm": target_comm if use_ac_pp else None,
            "utility_warmup_steps": utility_warmup_steps if use_ac_pp else None,
            "use_utility_in_gate": (
                bool(guidance_cfg.get("use_utility_in_gate", True)) if use_ac_pp else None
            ),
            "use_utility_in_policy": (
                bool(guidance_cfg.get("use_utility_in_policy", True)) if use_ac_pp else None
            ),
            "utility_target": (
                str(guidance_cfg.get("utility_target", "action")) if use_ac_pp else None
            ),
            "tag": exp_run.tag,
        },
    )
    return reward_curve

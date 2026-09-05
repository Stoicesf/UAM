"""Hierarchical PPO trainer: type-conditioned high AC + low AC with GAE."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from dice.role_reward import RoleReward

from .ppo_buffer import RolloutBuffer
from .utils import (
    eval_reward_formula,
    freeze,
    light_scout_ratio,
    pack_actions,
    save_checkpoint,
    unfreeze,
)


class HierarchicalPPOTrainer:
    def __init__(self, env: Any, high_ac: Any, low_ac: Any, config: dict):
        self.env = env
        self.device = torch.device(config.get("device", "cpu"))
        self.high_ac = high_ac.to(self.device)
        self.low_ac = low_ac.to(self.device)
        self.config = config
        self.high_optim = torch.optim.Adam(self.high_ac.parameters(), lr=config.get("high_lr", 3e-4))
        self.low_optim = torch.optim.Adam(self.low_ac.parameters(), lr=config.get("low_lr", 3e-4))
        self.gamma = float(config.get("gamma", 0.99))
        self.gae_lambda = float(config.get("gae_lambda", 0.95))
        self.clip_eps = float(config.get("clip_epsilon", 0.2))
        self.ppo_epochs = int(config.get("epochs", 10))
        self.ent_coef = float(config.get("ent_coef", 0.01))
        self.vf_coef = float(config.get("vf_coef", 0.5))
        self.max_grad = float(config.get("max_grad_norm", 0.5))
        self.ep_max = int(config.get("max_steps", 200))
        self.match_weight = float(config.get("match_weight", 0.0))
        self.coll_coef = float(config.get("coll_coef", 0.5))
        self.lsr_coef = float(config.get("lsr_coef", 0.0))
        self.reward_formula = config.get("reward_formula") or getattr(env, "reward_formula", None)
        self._role_reward = RoleReward(int(config.get("n_roles", 3)))
        self.metrics: dict[str, list] = {
            "step": [],
            "coverage": [],
            "collision": [],
            "role_entropy": [],
            "light_scout": [],
        }
        self.save_dir = Path(config.get("save_dir", "experiment_results/hierarchical_ppo/"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._global_step = 0
        self._fixed_roles: torch.Tensor | None = None

    def _global_reward(self, info: dict, roles: torch.Tensor) -> float:
        cov = float(info.get("coverage", 0.0))
        coll = float(info.get("collision_rate", 0.0))
        if self.reward_formula:
            return eval_reward_formula(str(self.reward_formula), cov, coll)
        lsr = light_scout_ratio(roles, self.env.uav_type_list)
        match = float(
            self._role_reward.capability_match_bonus(roles, self.env.uav_type_list).mean()
        )
        return cov - self.coll_coef * coll + self.lsr_coef * lsr + self.match_weight * match

    def _to_dev(self, t: torch.Tensor) -> torch.Tensor:
        return t.to(self.device)

    def _log_metrics(self, info: dict, roles: torch.Tensor) -> None:
        self.metrics["step"].append(self._global_step)
        self.metrics["coverage"].append(float(info.get("coverage", 0.0)))
        self.metrics["collision"].append(float(info.get("collision_rate", 0.0)))
        with torch.no_grad():
            gobs = self._to_dev(self.env.get_global_obs())
            logits, _ = self.high_ac(gobs, self.env.uav_type_list)
            probs = torch.softmax(logits.mean(dim=0), dim=-1)
            ent = float(-(probs * torch.log(probs + 1e-8)).sum())
        self.metrics["role_entropy"].append(ent)
        self.metrics["light_scout"].append(light_scout_ratio(roles.cpu(), self.env.uav_type_list))

    def _normalize_adv(self, advantages: torch.Tensor) -> torch.Tensor:
        if advantages.numel() > 1:
            std = advantages.std()
            if torch.isfinite(std) and float(std) > 1e-8:
                return (advantages - advantages.mean()) / (std + 1e-8)
            return advantages - advantages.mean()
        return advantages * 0.0

    def _ppo_update_high(self, buf: RolloutBuffer) -> None:
        if not buf.states:
            return
        advantages, returns = buf.compute_gae(self.gamma, self.gae_lambda)
        advantages = self._normalize_adv(advantages).to(self.device)
        returns = returns.to(self.device)
        old_logps = torch.stack(buf.log_probs).to(self.device)
        for _ in range(self.ppo_epochs):
            new_logps = []
            new_ents = []
            new_vals = []
            for i, state in enumerate(buf.states):
                lp, ent, val = self.high_ac.evaluate(
                    self._to_dev(state), buf.types[i], self._to_dev(buf.actions[i])
                )
                new_logps.append(lp)
                new_ents.append(ent)
                new_vals.append(val)
            new_logps_t = torch.stack(new_logps)
            new_ents_t = torch.stack(new_ents)
            new_vals_t = torch.stack(new_vals)
            ratio = torch.exp(new_logps_t - old_logps)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.functional.mse_loss(new_vals_t, returns)
            loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * new_ents_t.mean()
            self.high_optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.high_ac.parameters(), self.max_grad)
            self.high_optim.step()

    def _ppo_update_low(self, buf: RolloutBuffer) -> None:
        if not buf.obs:
            return
        advantages, returns = buf.compute_gae(self.gamma, self.gae_lambda)
        advantages = self._normalize_adv(advantages).to(self.device)
        returns = returns.to(self.device)
        old_logps = torch.stack([lp.mean() for lp in buf.log_probs_vec]).to(self.device)
        for _ in range(self.ppo_epochs):
            new_logps = []
            new_ents = []
            new_vals = []
            for i in range(len(buf.obs)):
                lp, ent, val = self.low_ac.evaluate(
                    self._to_dev(buf.obs[i]),
                    self._to_dev(buf.roles[i]),
                    self._to_dev(buf.actions[i]),
                )
                new_logps.append(lp.mean())
                new_ents.append(ent.mean())
                new_vals.append(val.mean())
            new_logps_t = torch.stack(new_logps)
            new_ents_t = torch.stack(new_ents)
            new_vals_t = torch.stack(new_vals)
            ratio = torch.exp(new_logps_t - old_logps.detach())
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.functional.mse_loss(new_vals_t, returns)
            loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * new_ents_t.mean()
            self.low_optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.low_ac.parameters(), self.max_grad)
            self.low_optim.step()

    def train_high(self, env_steps: int) -> None:
        freeze(self.low_ac)
        unfreeze(self.high_ac)
        start = self._global_step
        ep = 0
        interval = int(self.high_ac.decision_interval)
        while self._global_step - start < env_steps:
            buf = RolloutBuffer()
            obs, _ = self.env.reset(seed=self.config.get("seed", 0) + ep)
            roles: torch.Tensor | None = None
            pending: dict[str, Any] | None = None
            info: dict = {}
            done = False
            while not done and self._global_step - start < env_steps:
                need = roles is None or (self.env.step_count % interval == 0)
                if need:
                    if pending is not None and roles is not None:
                        R = self._global_reward(info, roles)
                        buf.add_high(
                            pending["state"],
                            pending["types"],
                            pending["action"],
                            pending["log_prob"],
                            R,
                            False,
                            pending["value"],
                        )
                    gobs = self._to_dev(self.env.get_global_obs())
                    types = list(self.env.uav_type_list)
                    roles, lp, _ent, value = self.high_ac.get_action(gobs, types)
                    pending = {
                        "state": gobs.detach().cpu(),
                        "types": types,
                        "action": roles.detach().cpu(),
                        "log_prob": lp.detach().cpu(),
                        "value": value.detach().cpu(),
                    }
                assert roles is not None
                with torch.no_grad():
                    dxdy = self.low_ac(self._to_dev(obs), roles)
                actions = pack_actions(dxdy.cpu(), roles.cpu(), self.env.n_roles)
                obs, _r, done, trunc, info = self.env.step(actions)
                self._global_step += 1
                if trunc or done:
                    done = True

            if pending is not None and roles is not None:
                R = self._global_reward(info, roles)
                buf.add_high(
                    pending["state"],
                    pending["types"],
                    pending["action"],
                    pending["log_prob"],
                    R,
                    True,
                    pending["value"],
                )
            self._ppo_update_high(buf)
            if roles is not None:
                self._log_metrics(info, roles)
            if ep % 25 == 0 or self._global_step - start >= env_steps:
                print(
                    f"[high-ppo] ep={ep} step={self._global_step} "
                    f"cov={info.get('coverage', 0):.2f} "
                    f"coll={info.get('collision_rate', 0):.3f} "
                    f"ent={self.metrics['role_entropy'][-1]:.3f} "
                    f"light_scout={self.metrics['light_scout'][-1]:.2f}",
                    flush=True,
                )
            ep += 1

    def train_low(self, env_steps: int) -> None:
        freeze(self.high_ac)
        unfreeze(self.low_ac)
        start = self._global_step
        ep = 0
        while self._global_step - start < env_steps:
            buf = RolloutBuffer()
            obs, _ = self.env.reset(seed=self.config.get("seed", 0) + 50_000 + ep)
            with torch.no_grad():
                fixed_roles, _, _, _ = self.high_ac.get_action(
                    self._to_dev(self.env.get_global_obs()),
                    self.env.uav_type_list,
                    deterministic=True,
                )
            info: dict = {}
            done = False
            while not done and self._global_step - start < env_steps:
                obs_d = self._to_dev(obs)
                dxdy, lp, val = self.low_ac.act(obs_d, fixed_roles)
                actions = pack_actions(dxdy.cpu(), fixed_roles.cpu(), self.env.n_roles)
                next_obs, r, done, trunc, info = self.env.step(actions.detach())
                # low reward: env mean + optional global formula shaping
                r_cpu = r.detach().cpu()
                if self.reward_formula:
                    shape = eval_reward_formula(
                        str(self.reward_formula),
                        float(info.get("coverage", 0.0)),
                        float(info.get("collision_rate", 0.0)),
                    )
                    r_cpu = r_cpu + 0.1 * shape
                buf.add_low(
                    obs, fixed_roles.cpu(), dxdy.detach().cpu(), lp.cpu(), r_cpu, done or trunc, val.cpu()
                )
                obs = next_obs
                self._global_step += 1
                if trunc or done:
                    done = True
            self._ppo_update_low(buf)
            self._log_metrics(info, fixed_roles)
            if ep % 25 == 0 or self._global_step - start >= env_steps:
                print(
                    f"[low-ppo] ep={ep} step={self._global_step} "
                    f"cov={info.get('coverage', 0):.2f} "
                    f"coll={info.get('collision_rate', 0):.3f} "
                    f"ent={self.metrics['role_entropy'][-1]:.3f} "
                    f"light_scout={self.metrics['light_scout'][-1]:.2f}",
                    flush=True,
                )
            ep += 1

    def alternate_train(self, rounds: int, high_steps: int, low_steps: int) -> None:
        print(
            f"device={self.device} reward_formula={self.reward_formula!r} "
            f"gamma={self.gamma} gae_lambda={self.gae_lambda}",
            flush=True,
        )
        for r in range(rounds):
            print(f"=== round {r}: high-ppo ({high_steps} env steps) ===", flush=True)
            self.train_high(high_steps)
            save_checkpoint(self.high_ac, self.save_dir / f"round_{r}_high_ppo.pt")
            print(f"=== round {r}: low-ppo ({low_steps} env steps) ===", flush=True)
            self.train_low(low_steps)
            save_checkpoint(self.low_ac, self.save_dir / f"round_{r}_low_ppo.pt")
        self.save_metrics()

    def save_metrics(self) -> None:
        path = self.save_dir / "metrics.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2)
        print(f"metrics → {path}", flush=True)

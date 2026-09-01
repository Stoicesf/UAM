"""Guided MAPPO — MAPPO + optional Guide Field (Stage 1~4).

Baseline MAPPO logic unchanged; only Actor input and optional reward shaping differ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from tensordict.nn import TensorDictModule, TensorDictSequential, set_composite_lp_aggregate
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.collectors import Collector
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from torchrl.objectives.multiagent import MAPPOLoss

from algorithms.baseline.advantage import setup_gae
from algorithms.baseline.buffer import make_replay_buffer
from algorithms.baseline.mappo import MAPPOComponents
from env.vmas_env import make_torchrl_env
from guidance.factory import build_guidance_encoder
from models.ac_dsgf import ACDSGF
from models.ac_dsgf_pp import ACDSGFpp
from models.communication.budget_constraint import soft_budget_loss
from models.communication.cost import communication_cost
from models.residual_policy import ResidualGuidanceActor
from utils.multiagent_obs import reshape_multiagent_obs


def _cat_obs_phi(obs: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return torch.cat([obs, phi], dim=-1)


class GraphGuideAdapter(torch.nn.Module):
    """Adapter for DSGFEncoder: extract positions from obs (VMAS nav: pos at [:2])."""

    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        positions = obs[..., :2]
        return self.encoder(obs, positions)


class ACGuideAdapter(torch.nn.Module):
    """Adapter for ACDSGF: returns Φ only; exposes communication loss for training."""

    def __init__(self, encoder: ACDSGF):
        super().__init__()
        self.encoder = encoder
        self._last_cost = 0.0
        self._last_edges = 0.0
        self._last_g: torch.Tensor | None = None
        self._last_diag: dict | None = None

    def set_budget_ratio(self, ratio: float | None):
        self.encoder.budget_ratio = ratio

    def set_fixed_k(self, k: int | None):
        """T-RO hard degree budget |E_i|≤K (Theorem 1 specialization)."""
        self.encoder.fixed_k = k

    def set_ablation_mode(self, mode: str | None):
        """Eval modes: None/'full' | 'no_budget' | 'random' (Silence Collapse)."""
        self.encoder.ablation_mode = mode

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        positions = obs[..., :2]
        use_budget = (
            self.encoder.fixed_k is not None
            or (
                self.encoder.budget_ratio is not None
                and self.encoder.budget_ratio < 1.0
            )
        )
        phi, g, diag = self.encoder(obs, positions, apply_budget=use_budget)
        self._last_cost = float(diag["comm_cost"].detach())
        self._last_edges = float(g.detach().sum(dim=(-2, -1)).mean())
        # Single-env (N,N) for demo traces
        g_np = g.detach()
        self._last_g = g_np[0] if g_np.dim() == 3 else g_np
        self._last_diag = {
            k: (v.detach() if torch.is_tensor(v) else v)
            for k, v in diag.items()
        }
        return phi

    @property
    def last_gate_matrix(self) -> torch.Tensor | None:
        return self._last_g

    @property
    def last_topology_diag(self) -> dict | None:
        """T-RO logging: A_t, V_B, C_t, B_t, rho_t, degrees (Gate 1)."""
        return getattr(self, "_last_diag", None)

    def communication_loss(self, obs: torch.Tensor, lambda_c: float, n_agents: int = 0) -> torch.Tensor:
        obs = reshape_multiagent_obs(obs, n_agents) if n_agents else obs
        positions = obs[..., :2]
        _, g, _ = self.encoder(obs, positions)
        return lambda_c * communication_cost(g)

    def budget_loss(
        self,
        obs: torch.Tensor,
        lambda_b: float,
        target_comm: float,
        n_agents: int = 0,
    ) -> torch.Tensor:
        """Soft budget: λ_b · (C / C_target − 1)². Prefer over raising λ_comm."""
        if lambda_b <= 0:
            return obs.new_zeros(())
        obs = reshape_multiagent_obs(obs, n_agents) if n_agents else obs
        positions = obs[..., :2]
        _, g, _ = self.encoder(obs, positions)
        return soft_budget_loss(g, target_comm=target_comm, lambda_b=lambda_b)

    @property
    def last_comm_cost(self) -> float:
        return self._last_cost

    @property
    def last_soft_edges(self) -> float:
        return self._last_edges


class ACGuideAdapterPP(ACGuideAdapter):
    """Adapter for ACDSGF++: adds utility loss + last U matrix for demo."""

    def __init__(self, encoder: ACDSGFpp):
        super().__init__(encoder)  # type: ignore[arg-type]
        self._last_u: torch.Tensor | None = None
        self._residual_actor: ResidualGuidanceActor | None = None
        self._critic: torch.nn.Module | None = None

    def attach_residual_actor(self, actor: ResidualGuidanceActor | None):
        self._residual_actor = actor

    def attach_critic(self, critic: torch.nn.Module | None):
        """MAPPO critic (TensorDictModule or raw MLP) for CAU ΔV."""
        self._critic = critic

    def _value_fn(self, obs: torch.Tensor) -> torch.Tensor:
        """obs (B,N,D) → V (B,N)."""
        if self._critic is None:
            raise RuntimeError("critic not attached")
        mod = self._critic
        # TensorDictModule wraps MultiAgentMLP in .module
        mlp = getattr(mod, "module", mod)
        v = mlp(obs)
        if v.dim() == obs.dim():
            v = v.squeeze(-1)
        return v

    def _action_fn(self, obs: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Counterfactual action under given Φ (residual policy if available)."""
        if self._residual_actor is not None:
            loc, _ = self._residual_actor(obs, phi)
            return loc
        return phi[..., :2] if phi.shape[-1] >= 2 else phi

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        positions = obs[..., :2]
        use_budget = self.encoder.budget_ratio is not None and self.encoder.budget_ratio < 1.0
        phi, g, diag = self.encoder(obs, positions, apply_budget=use_budget)
        self._last_cost = float(diag["comm_cost"].detach())
        self._last_edges = float(g.detach().sum(dim=(-2, -1)).mean())
        g_np = g.detach()
        self._last_g = g_np[0] if g_np.dim() == 3 else g_np
        u = diag.get("U")
        if u is not None:
            u_np = u.detach()
            self._last_u = u_np[0] if u_np.dim() == 3 else u_np
        return phi

    @property
    def last_utility_matrix(self) -> torch.Tensor | None:
        return self._last_u

    def utility_loss(
        self,
        obs: torch.Tensor,
        lambda_u: float,
        n_agents: int = 0,
        lambda_rank: float = 0.0,
        ranking_mode: str = "hinge",
    ) -> torch.Tensor:
        if lambda_u <= 0:
            return obs.new_zeros(())
        obs = reshape_multiagent_obs(obs, n_agents) if n_agents else obs
        positions = obs[..., :2]
        residual_fn = None
        beta = 1.0
        if self._residual_actor is not None:
            residual_fn = self._residual_actor.delta_net
            beta = float(self._residual_actor.beta.item())
        value_fn = self._value_fn if self._critic is not None else None
        action_fn = self._action_fn if self._residual_actor is not None else None
        return lambda_u * self.encoder.utility_loss(
            obs,
            positions,
            residual_delta_fn=residual_fn,
            beta=beta,
            lambda_rank=lambda_rank,
            ranking_mode=ranking_mode,
            value_fn=value_fn,
            action_fn=action_fn,
        )

    def utility_diagnostics(
        self, obs: torch.Tensor, lambda_u: float = 1.0, n_agents: int = 0
    ) -> dict[str, float]:
        obs = reshape_multiagent_obs(obs, n_agents) if n_agents else obs
        positions = obs[..., :2]
        residual_fn = None
        beta = 1.0
        if self._residual_actor is not None:
            residual_fn = self._residual_actor.delta_net
            beta = float(self._residual_actor.beta.item())
        value_fn = self._value_fn if self._critic is not None else None
        action_fn = self._action_fn if self._residual_actor is not None else None
        stats = self.encoder.utility_diagnostics(
            obs,
            positions,
            residual_delta_fn=residual_fn,
            beta=beta,
            value_fn=value_fn,
            action_fn=action_fn,
        )
        if lambda_u > 0:
            stats["utility_loss"] = float(
                self.encoder.utility_loss(
                    obs,
                    positions,
                    residual_delta_fn=residual_fn,
                    beta=beta,
                    value_fn=value_fn,
                    action_fn=action_fn,
                ).detach()
            )
        return stats


def build_guided_mappo(train_cfg: dict, env_cfg: dict, guidance_cfg: dict) -> MAPPOComponents:
    """Build MAPPO with guidance encoder fused into Actor input."""
    set_composite_lp_aggregate(False).set()

    device = torch.device(
        env_cfg.get("device", "cpu")
        if torch.cuda.is_available() or env_cfg.get("device") == "cpu"
        else "cpu"
    )
    env = make_torchrl_env(env_cfg)

    hidden = train_cfg.get("hidden_dim", 128)
    obs_dim = env.observation_spec["agents", "observation"].shape[-1]
    action_dim = env.full_action_spec[env.action_key].shape[-1]
    guidance_dim = guidance_cfg.get("guidance_dim", 4)
    mode = guidance_cfg.get("mode", "mlp")

    guide_core = build_guidance_encoder(guidance_cfg, obs_dim)
    if guide_core is None:
        raise ValueError("guidance.mode must not be 'none' for guided MAPPO")

    guide_core = guide_core.to(device)

    if mode in ("ac_dsgf_pp", "ac-dsgf-pp", "ac_dsgf++"):
        pp_ablation = guidance_cfg.get("pp_ablation")
        if pp_ablation and hasattr(guide_core, "ablation_mode"):
            guide_core.ablation_mode = str(pp_ablation).lower()
        guide_core = ACGuideAdapterPP(guide_core)  # type: ignore[arg-type]
    elif mode in ("ac_dsgf", "ac-dsgf"):
        guide_core = ACGuideAdapter(guide_core)  # type: ignore[arg-type]
    elif mode in ("graph", "full", "gat", "dsfg"):
        guide_core = GraphGuideAdapter(guide_core)

    guide_module = TensorDictModule(
        guide_core,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "phi")],
    )

    use_residual = guidance_cfg.get("residual_policy", False)

    if use_residual:
        residual_actor = ResidualGuidanceActor(
            obs_dim=obs_dim,
            phi_dim=guidance_dim,
            action_dim=action_dim,
            n_agents=env.n_agents,
            hidden_dim=hidden,
            share_params=train_cfg.get("share_params_policy", True),
            device=device,
            beta_init=guidance_cfg.get("beta_max", 1.0),
        ).to(device)
        actor_body = TensorDictModule(
            residual_actor,
            in_keys=[("agents", "observation"), ("agents", "phi")],
            out_keys=[("agents", "loc"), ("agents", "scale")],
        )
        policy_module = TensorDictSequential(guide_module, actor_body)
    else:
        cat_module = TensorDictModule(
            _cat_obs_phi,
            in_keys=[("agents", "observation"), ("agents", "phi")],
            out_keys=[("agents", "observation_aug")],
        )
        policy_mlp = MultiAgentMLP(
            n_agent_inputs=obs_dim + guidance_dim,
            n_agent_outputs=2 * action_dim,
            n_agents=env.n_agents,
            centralised=False,
            share_params=train_cfg.get("share_params_policy", True),
            device=device,
            depth=2,
            num_cells=hidden,
            activation_class=torch.nn.Tanh,
        )
        policy_net = torch.nn.Sequential(policy_mlp, NormalParamExtractor())
        policy_body = TensorDictModule(
            policy_net,
            in_keys=[("agents", "observation_aug")],
            out_keys=[("agents", "loc"), ("agents", "scale")],
        )
        policy_module = TensorDictSequential(guide_module, cat_module, policy_body)
        residual_actor = None

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

    # Critic: raw obs only (MAPPO standard, unchanged)
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
    total_frames = train_cfg.get("total_frames", 102_400)

    collector = Collector(
        env,
        policy,
        device=device,
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

    components = MAPPOComponents(
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
    components.guide_encoder = guide_core  # type: ignore[attr-defined]
    if isinstance(guide_core, ACGuideAdapterPP):
        guide_core.attach_critic(critic)
    if residual_actor is not None:
        components.residual_actor = residual_actor  # type: ignore[attr-defined]
        if isinstance(guide_core, ACGuideAdapterPP):
            guide_core.attach_residual_actor(residual_actor)
    return components

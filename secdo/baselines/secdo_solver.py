"""SECDO anticipatory solver — Algorithm 1 v2 (PI / α / c_mix)."""

from __future__ import annotations

from pathlib import Path

import torch

from experiments.secdo_uav.env_uav_bandwidth import UAVBandwidthEnv
from secdo.baselines.base import SolveResult, Solver
from secdo.evaluation.theory_metrics import EpisodeMetrics
from secdo.evaluation.violation import ViolationTracker
from secdo.models import SECDO
from secdo.training.losses import eps_l2, violation


class SECDOSolver(Solver):
    name = "secdo"

    def __init__(
        self,
        eta: float = 0.25,
        device: str | torch.device = "cuda:0",
        ckpt: str = "checkpoints/secdo_platform/pretrain/best.pt",
        latent_dim: int = 32,
        fixed_alpha: float | None = None,
        corrupt_prediction: bool = False,
        corrupt_scale: float = 3.0,
    ):
        super().__init__(eta=eta, device=device)
        self.ckpt = ckpt
        self.latent_dim = latent_dim
        self.fixed_alpha = fixed_alpha
        self.corrupt_prediction = corrupt_prediction
        self.corrupt_scale = corrupt_scale
        self._model: SECDO | None = None

    def _get_model(self, feat_dim: int, n_agents: int) -> SECDO:
        if self._model is None:
            m = SECDO(
                feat_dim=feat_dim,
                n_agents=n_agents,
                latent_dim=self.latent_dim,
                eta=self.eta,
            ).to(self.device)
            if self.ckpt and Path(self.ckpt).is_file():
                m.load_predictor_checkpoint(self.ckpt, map_location=self.device)
            self._model = m
            self._model.eval()
        return self._model

    @torch.no_grad()
    def solve(self, env: UAVBandwidthEnv, *, batch: int = 8) -> SolveResult:
        obs, pref, x = self._init_alloc(env, batch)
        feat_dim = int(obs["feat"].shape[-1])
        model = self._get_model(feat_dim, env.cfg.n_agents)
        met = EpisodeMetrics()
        vtr = ViolationTracker()
        h = None
        c_prev = None
        while not env.done:
            feat = obs["feat"].to(self.device)
            c_t = obs["c_teacher"].to(self.device)
            out = model.forward_step(
                x,
                pref,
                feat,
                c_t,
                h,
                mode="secdo",
                c_prev=c_prev,
                fixed_alpha=self.fixed_alpha,
            )
            if self.corrupt_prediction:
                # Exp.5 crash: inflate ĉ then re-project mix
                from secdo.optimizer.secdo_optimizer import SECDOOptimizer

                bad_c = out["c_hat"] * self.corrupt_scale
                x, alpha, pi = SECDOOptimizer(self.eta).step(
                    x,
                    None,
                    bad_c,
                    c_t,
                    chi_hat=(c_t - c_prev).abs() if c_prev is not None else None,
                    pref=pref,
                )
                out["x"], out["alpha"], out["PI"] = x, alpha, pi
            else:
                x = out["x"]
            h = out["h"]
            c_prev = c_t
            obs = env.step()
            c_next = obs["c_teacher"].to(self.device)
            delta = float((out["c_hat"] - c_next).abs().mean())
            chi = float((c_next - c_t).abs().mean())
            met.update(
                eps=eps_l2(out["s_hat"], obs["feat"].to(self.device)),
                delta=delta,
                rho=chi,
                gap=self._gap(x, pref, c_next),
                viol=float(violation(x, c_next)),
            )
            vtr.update(
                x,
                c_next,
                delta=delta,
                chi=chi,
                PI=float(out["PI"].mean()) if torch.is_tensor(out["PI"]) else chi,
                alpha=float(out["alpha"].mean()) if torch.is_tensor(out["alpha"]) else 0.0,
            )
        extras = vtr.summary()
        return SolveResult(
            method=self.name,
            metrics={**met.summary(), **{f"v2_{k}": v for k, v in extras.items()}},
            series={**met.series(), "PI": vtr.PI, "alpha": vtr.alpha, "chi": vtr.chi},
            extras=extras,
        )

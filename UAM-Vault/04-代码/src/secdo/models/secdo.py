"""SECDO v2 — Algorithm 1 with PI-conditioned mixed projection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from secdo.models.dynamics.predictor import Predictor
from secdo.optimizer.projected_gradient import gradient_step
from secdo.optimizer.reactive_projection import reactive_project
from secdo.optimizer.secdo_optimizer import SECDOOptimizer


class SECDO(nn.Module):
    """
    Online inference (Algorithm 1 v2):

        predict ĉ → PI/α → c_mix → Π_{B(c_mix)}(y)
    """

    def __init__(
        self,
        feat_dim: int,
        n_agents: int = 8,
        latent_dim: int = 32,
        hidden: int = 64,
        eta: float = 0.25,
        residual: bool = True,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.n_agents = n_agents
        self.eta = eta
        self.predictor = Predictor(
            feat_dim=feat_dim, latent_dim=latent_dim, hidden=hidden, residual=residual
        )
        self.encoder = self.predictor.encoder
        self.gru = self.predictor.gru
        self.constraint_head = self.predictor.constraint_head
        self.state_decoder = self.predictor.state_decoder
        self.opt = SECDOOptimizer(eta=eta)

    def predict_constraint(
        self,
        feat: torch.Tensor,
        c_curr: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.predictor.forward_step(feat, c_curr, h)

    def update(
        self,
        x: torch.Tensor,
        pref: torch.Tensor,
        c_hat: torch.Tensor,
        c_now: torch.Tensor,
        *,
        chi_hat: torch.Tensor | None = None,
        delta_hat: torch.Tensor | None = None,
        mode: str = "secdo",
        fixed_alpha: float | None = None,
        detach_budget: bool = True,
    ) -> dict[str, torch.Tensor]:
        if mode == "reactive":
            x_next = self.opt.reactive_step(x, pref, c_now)
            z = torch.zeros_like(c_now)
            return {"x": x_next, "alpha": z, "PI": z, "c_mix": c_now}
        if mode == "oracle":
            # c_hat interpreted as true c_{t+1}
            x_next = self.opt.oracle_step(x, pref, c_hat)
            one = torch.ones_like(c_now)
            return {"x": x_next, "alpha": one, "PI": torch.zeros_like(c_now), "c_mix": c_hat}
        x_next, alpha, pi = self.opt.step(
            x,
            None,
            c_hat,
            c_now,
            delta_hat=delta_hat,
            chi_hat=chi_hat,
            pref=pref,
            detach_budget=detach_budget,
            fixed_alpha=fixed_alpha,
        )
        from secdo.optimizer.secdo_optimizer import mix_budget

        c_mix = mix_budget(c_hat.detach() if detach_budget else c_hat, c_now, alpha)
        return {"x": x_next, "alpha": alpha, "PI": pi, "c_mix": c_mix}

    def forward_step(
        self,
        x: torch.Tensor,
        pref: torch.Tensor,
        feat: torch.Tensor,
        c_curr: torch.Tensor,
        h: torch.Tensor | None = None,
        *,
        mode: str = "secdo",
        c_prev: torch.Tensor | None = None,
        fixed_alpha: float | None = None,
    ) -> dict[str, torch.Tensor]:
        pred = self.predict_constraint(feat, c_curr, h)
        chi = (c_curr - c_prev).abs() if c_prev is not None else None
        # online δ proxy before observing c_{t+1}: |ĉ − c_t|
        delta_proxy = (pred["c_hat"] - c_curr).abs()
        upd = self.update(
            x,
            pref,
            pred["c_hat"],
            c_curr,
            chi_hat=chi,
            delta_hat=delta_proxy,
            mode=mode,
            fixed_alpha=fixed_alpha,
            detach_budget=True,
        )
        return {
            **pred,
            **upd,
            "y": gradient_step(x, pref, self.eta),
        }

    def project(self, y, c_hat, *, mode="anticipatory", detach_budget=True):
        # back-compat shim
        c = c_hat.detach() if detach_budget else c_hat
        from secdo.optimizer.anticipatory_projection import project_sum_budget

        if mode in ("anticipatory", "oracle", "secdo"):
            return project_sum_budget(y, c)
        return reactive_project(y, c)

    def load_predictor_checkpoint(
        self, path: str | Path, map_location: str | torch.device = "cpu"
    ) -> dict[str, Any]:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        missing, unexpected = self.predictor.load_state_dict(state, strict=False)
        return {"missing": list(missing), "unexpected": list(unexpected), "path": str(path)}

    def as_modules_dict(self) -> dict:
        return {
            "enc": self.encoder,
            "rec": self.gru,
            "head": self.constraint_head,
            "dec": self.state_decoder,
            "feat_dim": self.feat_dim,
            "predictor": self.predictor,
            "secdo": self,
        }

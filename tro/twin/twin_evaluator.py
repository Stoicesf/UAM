"""Shared-state twin evaluator for T-RO Gate 2 / Theorem 2 chain.

            same s_t
               |
        ---------------
        |             |
     G_full        G_sparse
        |             |
     M(G*)          M(G)
        |             |
      a*              a

Never steps two environments.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.ac_dsgf import ACDSGF
from tro.twin.action_compare import action_mean_l2, actions_from_residual_actor
from tro.twin.message_compare import message_mean_l2
from utils.tro_run import create_run_dir, write_config


@dataclass
class TwinStepResult:
    t: int
    state_id: str
    epsilon_G: float
    delta_a: float | None
    D_G: float
    sparse_mode: str
    k_fixed: int | None = None
    mean_degree_sparse: float | None = None


class TwinEvaluator:
    """Eval-only twin on ACDSGF (+ optional ResidualGuidanceActor)."""

    def __init__(
        self,
        encoder: ACDSGF,
        actor: torch.nn.Module | None = None,
        run_dir: Path | None = None,
        run_prefix: str = "tro_gate2",
    ):
        self.encoder = encoder
        self.actor = actor
        self.run_dir = run_dir or create_run_dir(prefix=run_prefix)
        for sub in ("twin", "twin/message_full", "twin/message_sparse", "actions"):
            (self.run_dir / sub).mkdir(parents=True, exist_ok=True)
        self._csv = self.run_dir / "twin" / "epsilon_delta.csv"
        if not self._csv.exists():
            self._csv.write_text(
                "t,state_id,epsilon_G,delta_a,D_G,sparse_mode,k_fixed,mean_degree_sparse\n",
                encoding="utf-8",
            )

    def state_id(self, obs: torch.Tensor) -> str:
        raw = obs.detach().cpu().numpy().tobytes()
        return hashlib.sha1(raw).hexdigest()[:16]

    @torch.no_grad()
    def step(
        self,
        obs: torch.Tensor,
        *,
        t: int = 0,
        sparse_mode: str = "fixed_k",
        k_fixed: int | None = None,
        budget_ratio: float | None = None,
        positions: torch.Tensor | None = None,
        log: bool = True,
    ) -> TwinStepResult:
        twin = self.encoder.twin_forward(
            obs,
            positions,
            sparse_mode=sparse_mode,
            k_fixed=k_fixed,
            budget_ratio=budget_ratio,
        )
        m_full = twin["full"]["message"]
        m_sparse = twin["sparse"]["message"]
        eps = message_mean_l2(m_full, m_sparse)
        A_star = twin["full"]["A_t"]
        A_t = twin["sparse"]["A_t"]
        d_g = float((A_star - A_t).reshape(A_star.shape[0], -1).norm(dim=-1).mean().item())
        mean_deg = float((A_t > 0).float().sum(dim=-1).mean().item())

        delta_a = None
        a_full = a_sparse = None
        if self.actor is not None:
            obs_b = twin["ctx"]["obs"]
            a_full, a_sparse = actions_from_residual_actor(
                self.actor, obs_b, twin["full"]["phi"], twin["sparse"]["phi"]
            )
            delta_a = action_mean_l2(a_full, a_sparse)

        sid = self.state_id(twin["ctx"]["obs"])
        result = TwinStepResult(
            t=t,
            state_id=sid,
            epsilon_G=eps,
            delta_a=delta_a,
            D_G=d_g,
            sparse_mode=sparse_mode,
            k_fixed=k_fixed,
            mean_degree_sparse=mean_deg,
        )
        if log:
            self._log_step(result, m_full, m_sparse, a_full, a_sparse)
        return result

    def _log_step(
        self,
        result: TwinStepResult,
        m_full: torch.Tensor,
        m_sparse: torch.Tensor,
        a_full: torch.Tensor | None,
        a_sparse: torch.Tensor | None,
    ) -> None:
        t = result.t
        np.save(
            self.run_dir / "twin" / "message_full" / f"t{t:06d}.npy",
            m_full.detach().cpu().numpy(),
        )
        np.save(
            self.run_dir / "twin" / "message_sparse" / f"t{t:06d}.npy",
            m_sparse.detach().cpu().numpy(),
        )
        meta = {"state_id": result.state_id, "t": t}
        (self.run_dir / "twin" / f"state_{t:06d}.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )
        if a_full is not None and a_sparse is not None:
            np.save(
                self.run_dir / "actions" / f"action_full_t{t:06d}.npy",
                a_full.detach().cpu().numpy(),
            )
            np.save(
                self.run_dir / "actions" / f"action_sparse_t{t:06d}.npy",
                a_sparse.detach().cpu().numpy(),
            )
        with self._csv.open("a", encoding="utf-8") as f:
            f.write(
                f"{result.t},{result.state_id},{result.epsilon_G},"
                f"{'' if result.delta_a is None else result.delta_a},"
                f"{result.D_G},{result.sparse_mode},"
                f"{'' if result.k_fixed is None else result.k_fixed},"
                f"{'' if result.mean_degree_sparse is None else result.mean_degree_sparse}\n"
            )

    def write_summary(self, summary: dict[str, Any]) -> Path:
        write_config(self.run_dir, {"gate": 2, **summary})
        path = self.run_dir / "twin" / "gate2_summary.json"
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return path

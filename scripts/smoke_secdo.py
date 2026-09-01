#!/usr/bin/env python
"""SECDO Phase-1b smoke: constraint prediction + anticipatory vs reactive.

Does NOT chase task reward. Checks S1–S3 from phase1a_formal_spec.md:
  S1: delta_t (c prediction error) decreases
  S2: sum violation lower for anticipatory Π vs reactive Π
  S3: eps_t, delta_t logged

Usage:
  E:\\ANACONDA\\envs\\dpg_hrl\\python.exe -u scripts/smoke_secdo.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from secdo.api import (
    BudgetDynamics,
    ErrorTracker,
    FeasibleHead,
    LatentEncoder,
    LatentTransition,
    anticipatory_project,
    reactive_project,
)


def synthetic_batch(B: int, feat_dim: int, dim_x: int, drift: float, device):
    """Drifting bandwidth that often *shrinks* (anticipatory Π has a clear job)."""
    c = torch.rand(B, 1, device=device) * 2.0 + 1.5  # ~[1.5, 3.5]
    s = torch.randn(B, feat_dim, device=device)
    # Channel worsening proxy in s[:,0] → shrink factor in (0.55, 0.95)
    shrink = 0.55 + 0.40 * torch.sigmoid(-s[:, :1] * drift * 3.0)
    c_next = (c * shrink + 0.02 * torch.randn_like(c)).clamp(min=0.3)
    # Proposals that tend to saturate current budget (reactive then overshoots c_next)
    y = torch.rand(B, dim_x, device=device)
    y = y / y.sum(dim=-1, keepdim=True).clamp(min=1e-8) * (c * 1.05)
    a = torch.randn(B, 2, device=device)
    s_next = s + 0.1 * torch.randn_like(s) + 0.05 * a[:, :1]
    return s, a, s_next, c, c_next, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)

    feat_dim, dim_x, latent_dim = 6, 8, 16
    enc = LatentEncoder(feat_dim, latent_dim).to(device)
    trans = LatentTransition(latent_dim, action_dim=2).to(device)
    fphi = BudgetDynamics(feat_dim=latent_dim, c_dim=1).to(device)
    fhead = FeasibleHead("sum_budget")

    opt = torch.optim.Adam(
        list(enc.parameters()) + list(trans.parameters()) + list(fphi.parameters()),
        lr=1e-3,
    )

    tracker_anti = ErrorTracker()
    tracker_re = ErrorTracker()
    delta_hist: list[float] = []

    for step in range(args.steps):
        s, a, s_next, c, c_next, y = synthetic_batch(
            args.batch, feat_dim, dim_x, drift=0.15, device=device
        )
        z = enc(s)
        s_hat = trans.predict_next_feat(z, a)
        # align dims: decode to feat_dim
        if s_hat.shape[-1] != feat_dim:
            # project latent decode to feat via linear pad/crop for smoke
            if s_hat.shape[-1] > feat_dim:
                s_hat = s_hat[..., :feat_dim]
            else:
                pad = feat_dim - s_hat.shape[-1]
                s_hat = F.pad(s_hat, (0, pad))

        c_hat = fphi(c, z.detach())  # F_φ(c_t, z_t)
        loss = F.mse_loss(c_hat, c_next) + 0.1 * F.mse_loss(s_hat, s_next)

        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            # Anticipatory: project with predicted next budget
            x_anti = anticipatory_project(y, fhead(c_hat))
            # Reactive: project with current budget
            x_re = reactive_project(y, c)

            m_anti = tracker_anti.update(
                s_next=s_next, s_hat=s_hat, c_true=c_next, c_hat=c_hat, x=x_anti
            )
            tracker_re.update(
                s_next=s_next, s_hat=s_hat, c_true=c_next, c_hat=c.expand_as(c_hat), x=x_re
            )
            # For reactive violation vs true next budget c_next:
            viol_re = (x_re.sum(-1) - c_next.view(-1)).clamp(min=0).mean().item()
            viol_anti = (x_anti.sum(-1) - c_next.view(-1)).clamp(min=0).mean().item()
            delta_hist.append(m_anti["delta_t"])

            if step % 40 == 0 or step == args.steps - 1:
                print(
                    f"step={step:04d} loss={loss.item():.4f} "
                    f"delta={m_anti['delta_t']:.4f} eps={m_anti['eps_t']:.4f} "
                    f"viol_anti={viol_anti:.4f} viol_re={viol_re:.4f}",
                    flush=True,
                )

    # S1: last quarter mean delta < first quarter
    q = max(len(delta_hist) // 4, 1)
    d0 = sum(delta_hist[:q]) / q
    d1 = sum(delta_hist[-q:]) / q
    s1 = d1 < d0

    # S2a (must): oracle anticipatory Π_{B(c_{t+1})} vs reactive Π_{B(c_t)}
    # S2b (info): learned ĉ anticipatory vs reactive
    sum_v_oracle = sum_v_learned = sum_v_re = 0.0
    with torch.no_grad():
        for _ in range(20):
            s, a, s_next, c, c_next, y = synthetic_batch(
                args.batch, feat_dim, dim_x, 0.15, device
            )
            z = enc(s)
            c_hat = fphi(c, z)
            x_oracle = anticipatory_project(y, fhead(c_next))
            x_learned = anticipatory_project(y, fhead(c_hat))
            x_re = reactive_project(y, c)
            sum_v_oracle += (x_oracle.sum(-1) - c_next.view(-1)).clamp(min=0).mean().item()
            sum_v_learned += (x_learned.sum(-1) - c_next.view(-1)).clamp(min=0).mean().item()
            sum_v_re += (x_re.sum(-1) - c_next.view(-1)).clamp(min=0).mean().item()
    s2 = sum_v_oracle < sum_v_re  # operator benefit under shrinking budgets
    s2b = sum_v_learned < sum_v_re

    s3 = len(tracker_anti.eps) == args.steps

    print("--- smoke gates ---", flush=True)
    print(f"S1 constraint pred (delta↓): {s1}  ({d0:.4f} → {d1:.4f})", flush=True)
    print(
        f"S2 anticipatory (oracle) viol↓: {s2}  "
        f"(oracle={sum_v_oracle:.4f} re={sum_v_re:.4f})",
        flush=True,
    )
    print(
        f"S2b learned anticipatory (info): {s2b}  "
        f"(learned={sum_v_learned:.4f} re={sum_v_re:.4f})",
        flush=True,
    )
    print(f"S3 log eps/delta: {s3}  sums={tracker_anti.sums()}", flush=True)
    ok = s1 and s2 and s3
    print("PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""SECDO Phase 1b-1 smoke: UAV capacity teacher → F_φ → Π_{B(ĉ)}.

Data:  (s_t, c_t) from SINR/Shannon teacher (not learned)
Model: feat → ĉ_{t+1} via ConstraintHead (softplus)
Opt:   anticipatory Π vs reactive Π under shrinking channel capacity
Theory logs: ε (feat pred optional), δ=|c-ĉ|, d_H bound

Usage:
  E:\\ANACONDA\\envs\\dpg_hrl\\python.exe -u scripts/smoke_secdo_uav_teacher.py
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
    ConstraintHead,
    ErrorTracker,
    FeasibleHead,
    UAVChannelConfig,
    anticipatory_project,
    build_state_features,
    delta_capacity,
    hausdorff_bound_sum_budget,
    reactive_project,
    step_positions,
    system_capacity_teacher,
)


def rollout_batch(
    B: int,
    N: int,
    device: torch.device,
    cfg: UAVChannelConfig,
    T_inner: int = 2,
):
    """Simulate short UAV motion; return (feat_t, c_t, feat_{t+1}, c_{t+1}, y)."""
    pos = (torch.rand(B, N, 2, device=device) - 0.5) * 4.0
    vel = (torch.rand(B, N, 2, device=device) - 0.5) * 0.8
    pack0 = build_state_features(pos, vel, cfg=cfg)
    c0 = pack0["c_teacher"]
    feat0 = pack0["feat"]

    # Move — capacity typically changes with geometry
    for _ in range(T_inner):
        pos, vel = step_positions(pos, vel, dt=0.15, box=3.0)
    pack1 = build_state_features(pos, vel, cfg=cfg)
    c1 = pack1["c_teacher"]
    feat1 = pack1["feat"]

    # Allocation proposal that saturates current capacity (reactive overshoots if C shrinks)
    dim_x = N
    y = torch.rand(B, dim_x, device=device)
    y = y / y.sum(dim=-1, keepdim=True).clamp(min=1e-8) * (c0 * 1.05)
    return feat0, c0, feat1, c1, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--agents", type=int, default=8)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)
    cfg = UAVChannelConfig(bandwidth_hz=1.0, comm_radius=2.5)

    # feat dim from one probe
    f0, _, _, _, _ = rollout_batch(2, args.agents, device, cfg)
    feat_dim = f0.shape[-1]

    head = ConstraintHead(feat_dim=feat_dim, c_dim=1, hidden=64, horizon_k=1, residual=False).to(device)
    fhead = FeasibleHead("sum_budget")
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)

    tracker = ErrorTracker()
    delta_hist: list[float] = []
    dh_hist: list[float] = []

    for step in range(args.steps):
        feat0, c0, feat1, c1, y = rollout_batch(args.batch, args.agents, device, cfg)
        c_hat = head(feat0)  # F_φ(s_t) → ĉ_{t+1}
        loss = F.mse_loss(c_hat, c1)

        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            dlt = delta_capacity(c1, c_hat).mean()
            dh = hausdorff_bound_sum_budget(dlt)
            delta_hist.append(float(dlt))
            dh_hist.append(float(dh))

            x_anti = anticipatory_project(y, fhead(c_hat))
            # log theory terms (eps: feat change proxy)
            tracker.update(
                s_next=feat1,
                s_hat=feat0,  # myopic baseline for eps proxy in this smoke
                c_true=c1,
                c_hat=c_hat,
                x=x_anti,
            )

            if step % 50 == 0 or step == args.steps - 1:
                viol_anti = (x_anti.sum(-1) - c1.view(-1)).clamp(min=0).mean().item()
                x_re = reactive_project(y, c0)
                viol_re = (x_re.sum(-1) - c1.view(-1)).clamp(min=0).mean().item()
                print(
                    f"step={step:04d} loss={loss.item():.4f} "
                    f"delta={float(dlt):.4f} dH≤{float(dh):.4f} "
                    f"C_mean={c1.mean().item():.3f} "
                    f"viol_anti={viol_anti:.4f} viol_re={viol_re:.4f}",
                    flush=True,
                )

    q = max(len(delta_hist) // 4, 1)
    d0, d1 = sum(delta_hist[:q]) / q, sum(delta_hist[-q:]) / q
    s1 = d1 < d0

    # S2 oracle anticipatory on teacher c1
    sum_o = sum_l = sum_r = 0.0
    with torch.no_grad():
        for _ in range(25):
            feat0, c0, feat1, c1, y = rollout_batch(args.batch, args.agents, device, cfg)
            c_hat = head(feat0)
            x_o = anticipatory_project(y, fhead(c1))
            x_l = anticipatory_project(y, fhead(c_hat))
            x_r = reactive_project(y, c0)
            sum_o += (x_o.sum(-1) - c1.view(-1)).clamp(min=0).mean().item()
            sum_l += (x_l.sum(-1) - c1.view(-1)).clamp(min=0).mean().item()
            sum_r += (x_r.sum(-1) - c1.view(-1)).clamp(min=0).mean().item()
    s2 = sum_o < sum_r
    s2b = sum_l < sum_r
    s3 = len(tracker.delta) == args.steps
    # Closure: δ logged and d_H bound monotone with δ
    s4 = all(abs(dh_hist[i] - delta_hist[i]) < 1e-5 for i in range(0, len(delta_hist), 10))

    print("--- Phase 1b-1 gates ---", flush=True)
    print(f"D1 teacher→model (delta↓): {s1}  ({d0:.4f} → {d1:.4f})", flush=True)
    print(f"D2 oracle anti viol↓: {s2}  (o={sum_o:.4f} r={sum_r:.4f})", flush=True)
    print(f"D2b learned anti (info): {s2b}  (l={sum_l:.4f} r={sum_r:.4f})", flush=True)
    print(f"D3 log δ/ε: {s3}  {tracker.sums()}", flush=True)
    print(f"D4 δ→d_H bound (K=1): {s4}", flush=True)
    ok = s1 and s2 and s3 and s4
    print("PASS" if ok else "FAIL", flush=True)

    # Sanity: compact swarm (stronger links) vs spread swarm
    with torch.no_grad():
        ang = torch.linspace(0, 2 * 3.1415, args.agents, device=device)
        pos_near = torch.stack([0.3 * torch.cos(ang), 0.3 * torch.sin(ang)], dim=-1).unsqueeze(0)
        pos_far = torch.stack([2.5 * torch.cos(ang), 2.5 * torch.sin(ang)], dim=-1).unsqueeze(0)
        c_near = system_capacity_teacher(pos_near, cfg).item()
        c_far = system_capacity_teacher(pos_far, cfg).item()
        print(f"teacher sanity C(compact)={c_near:.3f} C(spread)={c_far:.3f}", flush=True)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

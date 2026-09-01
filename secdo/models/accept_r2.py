"""R2 acceptance: SECDO Alg.1 + load legacy checkpoints.

Usage:
  python -m secdo.models.accept_r2
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from secdo.models import SECDO
from secdo.optimizer import anticipatory_project, reactive_project


def main() -> int:
    torch.manual_seed(0)
    feat_dim = 7
    n = 8
    B = 4
    model = SECDO(feat_dim=feat_dim, n_agents=n, latent_dim=32, eta=0.25)

    ckpt = ROOT / "checkpoints" / "secdo_predictor" / "best.pt"
    if not ckpt.is_file():
        ckpt = ROOT / "checkpoints" / "secdo_joint" / "best.pt"
    loaded = False
    if ckpt.is_file():
        info = model.load_predictor_checkpoint(ckpt)
        loaded = True
        print(f"Loaded {ckpt}", flush=True)
        if info["missing"] or info["unexpected"]:
            print("FAIL: state_dict key mismatch", info, flush=True)
            return 1
        print("  state_dict: exact match", flush=True)
    else:
        print("WARN: no checkpoint — random init", flush=True)

    feat = torch.randn(B, feat_dim)
    c = torch.ones(B, 1) * 2.0
    pref = torch.ones(B, n) / n
    x = reactive_project(pref.clone(), c)

    out = model.forward_step(x, pref, feat, c, h=None, mode="anticipatory")

    # Predictor loss path MUST get grads
    model.zero_grad(set_to_none=True)
    out["c_hat"].sum().backward()
    pred_grad = any(
        p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.constraint_head.parameters()
    )

    # Projection path MUST NOT get grads into head (ĉ.detach())
    model.zero_grad(set_to_none=True)
    pred = model.predict_constraint(feat, c)
    y = x - 0.25 * (x - pref)
    x_proj = anticipatory_project(y, pred["c_hat"], detach_budget=True)
    no_grad_through_pi = not x_proj.requires_grad
    if x_proj.requires_grad:
        x_proj.sum().backward()
        leaked = any(
            p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.constraint_head.parameters()
        )
    else:
        leaked = False

    # Reactive vs anticipatory callable
    _ = reactive_project(y, c)
    _ = model.project(y, out["c_hat"].detach(), mode="oracle")

    print("R2 acceptance", flush=True)
    print(f"  checkpoint_loaded:     {loaded}", flush=True)
    print(f"  forward x:             {tuple(out['x'].shape)}", flush=True)
    print(f"  c_hat mean:            {float(out['c_hat'].mean()):.6f}", flush=True)
    print(f"  predictor grads:       {pred_grad} (want True)", flush=True)
    print(f"  x_proj requires_grad:  {x_proj.requires_grad} (want False)", flush=True)
    print(f"  grad leak through Π:   {leaked} (want False)", flush=True)

    if not pred_grad or leaked or x_proj.requires_grad:
        print("FAIL", flush=True)
        return 1
    print("PASS accept_r2", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

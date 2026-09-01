"""Forward smoke for AC-DSGF++ (no full train).

Checks:
  - ACDSGFpp builds
  - CausalUtility + causal gate forward
  - utility_loss finite
  - v1 ACDSGF still importable / unchanged API
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from models.ac_dsgf import ACDSGF
    from models.ac_dsgf_pp import ACDSGFpp

    b, n, obs_dim = 2, 4, 18
    obs = torch.randn(b, n, obs_dim)
    pos = obs[..., :2]

    v1 = ACDSGF(obs_dim=obs_dim, hidden_dim=64, guidance_dim=6, num_heads=2)
    phi1, g1, d1 = v1(obs, pos)
    assert phi1.shape == (b, n, 6)
    assert g1.shape == (b, n, n)
    assert "U" not in d1

    pp = ACDSGFpp(obs_dim=obs_dim, hidden_dim=64, guidance_dim=6, num_heads=2)
    phi, g, diag = pp(obs, pos)
    assert phi.shape == (b, n, 6)
    assert g.shape == (b, n, n)
    assert diag["U"].shape == (b, n, n)
    assert torch.isfinite(diag["U"]).all()

    phi0, g0, _ = pp(obs, pos, force_zero_comm=True)
    assert g0.abs().sum() == 0
    assert torch.isfinite(phi0).all()

    uloss = pp.utility_loss(obs, pos)
    assert torch.isfinite(uloss)
    print(
        f"[OK] AC-DSGF++ smoke forward | "
        f"phi={tuple(phi.shape)} g_mean={g.mean().item():.4f} "
        f"U_mean={diag['U'].mean().item():.4f} util_loss={uloss.item():.6f}"
    )
    print("[OK] AC-DSGF v1 API unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Synthetic perception quality for VMAS (no YOLO/LiDAR hard dependency)."""

from __future__ import annotations

import torch


def perception_quality(
    positions: torch.Tensor,
    *,
    obstacles: torch.Tensor | None = None,
    detection_conf: torch.Tensor | None = None,
    los_blocked: torch.Tensor | None = None,
    neighbor_radius: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Three-factor quality in [0,1]; higher is better.

    - detection_conf: stub confidence (default 0.9)
    - point_density: proxy from neighbor count within radius
    - occlusion: 1 - fraction of blocked LOS to neighbors
    q = mean of three factors
    """
    if positions.dim() == 2:
        positions = positions.unsqueeze(0)
    b, n, _ = positions.shape
    device = positions.device

    if detection_conf is None:
        det = torch.full((b, n), 0.9, device=device)
    else:
        det = detection_conf
        if det.dim() == 1:
            det = det.unsqueeze(0)

    dist = (positions.unsqueeze(2) - positions.unsqueeze(1)).norm(dim=-1)
    eye = torch.eye(n, device=device).unsqueeze(0)
    near = ((dist < neighbor_radius) & (eye < 0.5)).float()
    density = near.sum(dim=-1) / max(n - 1, 1)  # more neighbors → denser "cloud"
    # normalize density to prefer moderate crowding
    dens_q = (1.0 - (density - 0.3).abs()).clamp(0, 1)

    if los_blocked is None:
        # if obstacles given: block if midpoint near obstacle
        if obstacles is None or obstacles.numel() == 0:
            occ_q = torch.ones(b, n, device=device)
        else:
            # obstacles: (M, 2)
            mid = 0.5 * (positions.unsqueeze(2) + positions.unsqueeze(1))  # B,N,N,2
            obs = obstacles.to(device)
            d_obs = (mid.unsqueeze(-2) - obs.view(1, 1, 1, -1, 2)).norm(dim=-1)
            blocked = (d_obs.min(dim=-1).values < 0.3).float() * (1.0 - eye)
            frac = blocked.sum(dim=-1) / max(n - 1, 1)
            occ_q = (1.0 - frac).clamp(0, 1)
    else:
        occ_q = (1.0 - los_blocked).clamp(0, 1)
        if occ_q.dim() == 1:
            occ_q = occ_q.unsqueeze(0)

    q = (det + dens_q + occ_q) / 3.0
    return {
        "q": q,
        "detection_conf": det,
        "point_density": dens_q,
        "occlusion": occ_q,
    }


def self_check() -> None:
    pos = torch.rand(1, 5, 2)
    out = perception_quality(pos, obstacles=torch.tensor([[0.5, 0.5]]))
    assert out["q"].shape == (1, 5)
    assert float(out["q"].mean()) > 0
    print("quality_estimator: OK")


if __name__ == "__main__":
    self_check()

"""T-RO run layout + topology logging helpers (Gate 0 / Gate 1).

Does not train. Writes theory-recoverable artifacts under runs/<run_id>/.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def new_run_id(prefix: str = "exp") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"


def create_run_dir(run_id: str | None = None, prefix: str = "tro_gate1") -> Path:
    """
    runs/<run_id>/
      config.json
      topology/
      messages/
      actions/
      rewards/
      metrics/
    """
    rid = run_id or new_run_id(prefix)
    root = RUNS / rid
    for sub in ("topology", "messages", "actions", "rewards", "metrics"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def write_config(run_dir: Path, config: dict[str, Any]) -> None:
    path = run_dir / "config.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def append_topology_step(
    run_dir: Path,
    *,
    t: int,
    A_t: torch.Tensor | np.ndarray,
    S_t: torch.Tensor | np.ndarray | None = None,
    B_t: float,
    C_t: float,
    V_B: float,
    degrees: torch.Tensor | np.ndarray | None = None,
    rho_t: float | None = None,
    mean_degree: float | None = None,
    meta: dict[str, Any] | None = None,
) -> None:
    """One timestep topology record (Theorem 1)."""
    A = _to_numpy(A_t)
    if A.ndim == 3:
        A = A[0]
    rec = {
        "t": int(t),
        "node_num": int(A.shape[0]),
        "edge_count": int((A > 0).sum()),
        "B_t": float(B_t),
        "C_t": float(C_t),
        "V_B": float(V_B),
        "rho_t": float(rho_t) if rho_t is not None else float((A > 0).sum() / max(A.shape[0] * (A.shape[0] - 1), 1)),
        "mean_degree": float(mean_degree) if mean_degree is not None else float((A > 0).sum(axis=-1).mean()),
        "degrees": _to_list(degrees) if degrees is not None else (A > 0).sum(axis=-1).tolist(),
    }
    if meta:
        rec.update(meta)
    np.savez_compressed(
        run_dir / "topology" / f"t{t:06d}.npz",
        A_t=A,
        S_t=_to_numpy(S_t) if S_t is not None else np.array([]),
        **{k: np.asarray(v) for k, v in rec.items() if k not in ("degrees",)},
        degrees=np.asarray(rec["degrees"], dtype=np.float64),
    )
    # also append JSONL for easy metrics aggregation
    line = {k: v for k, v in rec.items()}
    with (run_dir / "metrics" / "topology_steps.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")


def summarize_v_b(run_dir: Path) -> dict[str, float]:
    path = run_dir / "metrics" / "topology_steps.jsonl"
    if not path.exists():
        return {"max_V_B": float("nan"), "n_steps": 0}
    vals = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            vals.append(float(json.loads(line)["V_B"]))
    return {
        "max_V_B": float(max(vals)) if vals else float("nan"),
        "mean_V_B": float(sum(vals) / len(vals)) if vals else float("nan"),
        "n_steps": len(vals),
        "gate1_pass": bool(vals) and max(vals) == 0.0,
    }


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_list(x: torch.Tensor | np.ndarray | list) -> list:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x).reshape(-1).tolist()

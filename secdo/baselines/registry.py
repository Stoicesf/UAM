"""Solver registry — experiments call get_solver(name), never if-chains."""

from __future__ import annotations

from typing import Callable

from secdo.baselines.ac_dsgf import ACDSGFSolver
from secdo.baselines.base import Solver
from secdo.baselines.dsgf import DSGFSolver
from secdo.baselines.oracle import OracleSolver
from secdo.baselines.reactive import ReactiveSolver
from secdo.baselines.secdo_solver import SECDOSolver

_REGISTRY: dict[str, Callable[..., Solver]] = {
    "secdo": SECDOSolver,
    "reactive": ReactiveSolver,
    "oracle": OracleSolver,
    "dsgf": DSGFSolver,
    "ac_dsgf": ACDSGFSolver,
}


def get_solver(name: str, **kwargs) -> Solver:
    key = name.lower().strip()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown solver {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[key](**kwargs)


def list_solvers() -> list[str]:
    return sorted(_REGISTRY)

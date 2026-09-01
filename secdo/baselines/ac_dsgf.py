"""AC-DSGF baseline adapter for SECDO UAV capacity loop.

Full AC-DSGF (`models/ac_dsgf.py`) Top-K topology path is not coupled here yet.
Until coupling, AC-DSGF runs as reactive projection (same certificate as Π_B),
tagged separately so experiment tables stay stable when Top-K is wired.
"""

from __future__ import annotations

from secdo.baselines.reactive import ReactiveSolver


class ACDSGFSolver(ReactiveSolver):
    name = "ac_dsgf"

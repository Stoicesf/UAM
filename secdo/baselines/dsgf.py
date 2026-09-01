"""DSGF baseline adapter for SECDO UAV capacity loop.

Full topology-learning DSGF (`models/dsgf.py`) is orthogonal to anticipatory
projection. Here DSGF = myopic reactive allocation under the same F / B(c),
used as the 'original method' control in the capacity-drift paper track.
"""

from __future__ import annotations

from secdo.baselines.reactive import ReactiveSolver


class DSGFSolver(ReactiveSolver):
    name = "dsgf"

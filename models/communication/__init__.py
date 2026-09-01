"""Adaptive communication modules for AC-DSGF / AC-DSGF++.

v1 (frozen): CommunicationController
++ (new): CausalUtility, CausalCommunicationController

DSGF v2 (`mode=dsfg`) and AC-DSGF v1 gate must stay untouched.
"""

from models.communication.budget_constraint import scheduled_budget, soft_budget_loss
from models.communication.budget_layer import (
    apply_topk_budget,
    apply_topk_fixed_k,
    budget_edge_count,
    degree_budget_violation,
)
from models.communication.causal_utility import CausalUtility
from models.communication.controller import CommunicationController
from models.communication.controller_causal import CausalCommunicationController
from models.communication.cost import communication_cost, edge_count
from models.communication.scheduler import CommunicationScheduler

__all__ = [
    "CommunicationController",
    "CausalCommunicationController",
    "CausalUtility",
    "communication_cost",
    "edge_count",
    "CommunicationScheduler",
    "apply_topk_budget",
    "apply_topk_fixed_k",
    "budget_edge_count",
    "degree_budget_violation",
    "soft_budget_loss",
    "scheduled_budget",
]

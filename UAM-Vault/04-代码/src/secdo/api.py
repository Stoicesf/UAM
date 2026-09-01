"""SECDO public API — replaces models.secdo."""

from secdo.datasets.uav.mobility import step_positions
from secdo.datasets.uav.teacher import (
    UAVChannelConfig,
    build_state_features,
    capacity_teacher,
    system_capacity_teacher,
)
from secdo.evaluation.error_tracker import ErrorTracker
from secdo.models.constraint.budget_dynamics import BudgetDynamics
from secdo.models.constraint.capacity_head import CapacityHead, ConstraintHead
from secdo.models.constraint.feasible_region import FeasibleHead
from secdo.models.dynamics.encoder import LatentEncoder
from secdo.models.dynamics.predictor import SecdoPredictor
from secdo.models.dynamics.transition import LatentTransition
from secdo.optimizer.anticipatory_projection import anticipatory_project, project_budget
from secdo.optimizer.reactive_projection import reactive_project
from secdo.optimizer.secdo_optimizer import SECDOOptimizer as SecdoOptimizer
from secdo.theory.feasible_set_geometry import delta_capacity, hausdorff_bound_sum_budget

__all__ = [
    "BudgetDynamics",
    "ConstraintHead",
    "CapacityHead",
    "FeasibleHead",
    "UAVChannelConfig",
    "build_state_features",
    "system_capacity_teacher",
    "capacity_teacher",
    "step_positions",
    "LatentEncoder",
    "LatentTransition",
    "SecdoPredictor",
    "anticipatory_project",
    "reactive_project",
    "project_budget",
    "SecdoOptimizer",
    "ErrorTracker",
    "delta_capacity",
    "hausdorff_bound_sum_budget",
]

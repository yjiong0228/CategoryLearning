"""Read-only and predictive diagnostics for completed searches."""

from .search import (
    evaluate_hyper_cd_convergence,
    evaluate_multiobjective_selection,
    evaluate_near_optimal_plateau,
)
from .predictive import (
    diagnose_hyper_accuracy_sampling,
    evaluate_volatility_calibration,
)

__all__ = [
    "diagnose_hyper_accuracy_sampling",
    "evaluate_hyper_cd_convergence",
    "evaluate_multiobjective_selection",
    "evaluate_near_optimal_plateau",
    "evaluate_volatility_calibration",
]

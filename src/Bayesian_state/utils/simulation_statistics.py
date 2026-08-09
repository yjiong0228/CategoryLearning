"""Compatibility exports for repeated-simulation statistics.

New code should import schema orchestration from
:mod:`src.Bayesian_state.simulation.repeated_simulation` and numerical definitions from
:mod:`src.Bayesian_state.metrics`.
"""

from ..metrics.behavior_metrics import (
    accuracy_curve_metrics,
    accuracy_scalar_metrics,
    history_kernel_metrics,
    switch_behavior_metrics,
)
from ..metrics._numeric import (
    finite_array,
    minimize_rank01,
    nanmean_or_nan,
    nanmedian_or_nan,
    nanquantile_or_nan,
    safe_float,
)
from ..metrics.trajectory_statistics import marginal_prediction_metrics_from_runs
from ..simulation.repeated_simulation import (
    MULTIOBJECTIVE_WEIGHT_DEFAULTS,
    SELECTION_METRIC_ALIASES,
    SIMULATION_STAT_DEFAULTS,
    compute_simulation_statistics,
    get_stat_value,
    resolve_selection_metric_path,
    resolve_simulation_stat_config,
)

__all__ = [
    "MULTIOBJECTIVE_WEIGHT_DEFAULTS",
    "SELECTION_METRIC_ALIASES",
    "SIMULATION_STAT_DEFAULTS",
    "accuracy_curve_metrics",
    "accuracy_scalar_metrics",
    "compute_simulation_statistics",
    "finite_array",
    "get_stat_value",
    "history_kernel_metrics",
    "marginal_prediction_metrics_from_runs",
    "minimize_rank01",
    "nanmean_or_nan",
    "nanmedian_or_nan",
    "nanquantile_or_nan",
    "resolve_selection_metric_path",
    "resolve_simulation_stat_config",
    "safe_float",
    "switch_behavior_metrics",
]

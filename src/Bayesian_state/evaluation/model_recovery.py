"""Compatibility exports for recovery; implementations live in their owning layers.

New callers should import simulation.recovery, optimization.recovery,
evaluation.recovery or workflows.recovery.design explicitly.
"""
from importlib import import_module

_EXPORTS = {'FEATURE_COLUMNS': 'src.Bayesian_state.simulation.recovery',
 'ORDER_COLUMNS': 'src.Bayesian_state.simulation.recovery',
 'SCHEDULE_COLUMNS': 'src.Bayesian_state.simulation.recovery',
 'MODEL_PARAMETER_NAMES': 'src.Bayesian_state.simulation.recovery',
 'CELL_FREE_PARAMETERS': 'src.Bayesian_state.simulation.recovery',
 'RecoveryDatasetSpec': 'src.Bayesian_state.simulation.recovery',
 'RecoveryDesign': 'src.Bayesian_state.simulation.recovery',
 'schedule_fingerprint': 'src.Bayesian_state.simulation.recovery',
 'synthetic_dataset_frame': 'src.Bayesian_state.simulation.recovery',
 '_truth_hyperparams': 'src.Bayesian_state.workflows.recovery.generation',
 'generate_synthetic_dataset': 'src.Bayesian_state.workflows.recovery.generation',
 '_resolve_relative': 'src.Bayesian_state.workflows.recovery.design',
 '_generation_seed': 'src.Bayesian_state.workflows.recovery.design',
 '_module_specs': 'src.Bayesian_state.workflows.recovery.design',
 '_parameter_specs': 'src.Bayesian_state.workflows.recovery.design',
 '_validate_parameter_truth_support': 'src.Bayesian_state.workflows.recovery.design',
 'load_recovery_design': 'src.Bayesian_state.workflows.recovery.design',
 '_declared_support': 'src.Bayesian_state.optimization.recovery_parameters',
 'model_0826_truth_hyperparams': 'src.Bayesian_state.optimization.recovery_parameters',
 '_load_yaml': 'src.Bayesian_state.utils.recovery_artifacts',
 '_canonical_fingerprint': 'src.Bayesian_state.utils.recovery_artifacts',
 '_atomic_json': 'src.Bayesian_state.utils.recovery_artifacts',
 '_atomic_csv': 'src.Bayesian_state.utils.recovery_artifacts',
 '_atomic_npz': 'src.Bayesian_state.utils.recovery_artifacts',
 '_atomic_yaml': 'src.Bayesian_state.utils.recovery_artifacts',
 '_write_immutable_yaml': 'src.Bayesian_state.utils.recovery_artifacts',
 'resolve_recovery_stage_budgets': 'src.Bayesian_state.optimization.recovery',
 'fit_recovery_dataset': 'src.Bayesian_state.optimization.recovery',
 'build_calibration_bank': 'src.Bayesian_state.evaluation.recovery',
 'resolve_calibration_filter_seeds': 'src.Bayesian_state.evaluation.recovery',
 '_frozen_readout_args': 'src.Bayesian_state.evaluation.recovery',
 'score_pf_bank': 'src.Bayesian_state.evaluation.recovery',
 '_score_pf_candidate_seed': 'src.Bayesian_state.evaluation.recovery',
 'score_pf_bank_parallel': 'src.Bayesian_state.evaluation.recovery',
 '_setting_rows': 'src.Bayesian_state.evaluation.recovery',
 '_compare_pf_settings': 'src.Bayesian_state.evaluation.recovery',
 'summarize_search_budget_retention': 'src.Bayesian_state.evaluation.recovery',
 '_budget_mcse_q95': 'src.Bayesian_state.evaluation.recovery',
 'summarize_pf_calibration': 'src.Bayesian_state.evaluation.recovery',
 'freeze_smallest_passing_budget': 'src.Bayesian_state.evaluation.recovery',
 'mean_probability_nll': 'src.Bayesian_state.evaluation.recovery',
 'score_frozen_candidate': 'src.Bayesian_state.evaluation.recovery',
 '_wilson_interval': 'src.Bayesian_state.evaluation.recovery',
 'summarize_module_recovery': 'src.Bayesian_state.evaluation.recovery',
 '_parameter_support_values': 'src.Bayesian_state.evaluation.recovery',
 '_workspace_support_values': 'src.Bayesian_state.evaluation.recovery',
 '_safe_spearman': 'src.Bayesian_state.evaluation.recovery',
 '_balanced_accuracy_binary': 'src.Bayesian_state.evaluation.recovery',
 'summarize_parameter_recovery': 'src.Bayesian_state.evaluation.recovery',
 '_save_png_atomic': 'src.Bayesian_state.evaluation.recovery',
 '_configure_recovery_figure_style': 'src.Bayesian_state.evaluation.recovery',
 'plot_module_recovery': 'src.Bayesian_state.evaluation.recovery',
 'plot_parameter_recovery': 'src.Bayesian_state.evaluation.recovery'}

__all__ = [
    "CELL_FREE_PARAMETERS",
    "FEATURE_COLUMNS",
    "MODEL_PARAMETER_NAMES",
    "ORDER_COLUMNS",
    "RecoveryDatasetSpec",
    "RecoveryDesign",
    "build_calibration_bank",
    "freeze_smallest_passing_budget",
    "fit_recovery_dataset",
    "generate_synthetic_dataset",
    "load_recovery_design",
    "mean_probability_nll",
    "model_0826_truth_hyperparams",
    "plot_module_recovery",
    "plot_parameter_recovery",
    "resolve_calibration_filter_seeds",
    "resolve_recovery_stage_budgets",
    "schedule_fingerprint",
    "score_frozen_candidate",
    "score_pf_bank",
    "score_pf_bank_parallel",
    "summarize_module_recovery",
    "summarize_parameter_recovery",
    "summarize_pf_calibration",
    "summarize_search_budget_retention",
    "synthetic_dataset_frame",
]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))

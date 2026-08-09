"""Compatibility exports for the pre-refactor optimization runtime path.

New code should import run results from
:mod:`src.Bayesian_state.simulation.state_model_execution`,
losses from :mod:`src.Bayesian_state.metrics`, and execution helpers from
:mod:`src.Bayesian_state.simulation.state_model_execution`.
"""

from ..simulation.state_model_execution import (
    SimulationResult,
    SingleRunResult,
    TrialArrays,
)
from ..metrics.behavior_metrics import exponential_smooth_curve
from ..metrics.losses import (
    ACCURACY_LOSS_METRIC_CHOICES,
    CHOICE_LOSS_METRIC_CHOICES,
    LOSS_METRIC_ACCURACY_BERHU,
    LOSS_METRIC_ACCURACY_BRIER,
    LOSS_METRIC_ACCURACY_CURVE_BERHU,
    LOSS_METRIC_ACCURACY_CURVE_FAMILY_MSE,
    LOSS_METRIC_ACCURACY_CURVE_MAE,
    LOSS_METRIC_ACCURACY_CURVE_MSE,
    LOSS_METRIC_ACCURACY_FAMILY_BRIER,
    LOSS_METRIC_ACCURACY_MAE,
    LOSS_METRIC_ACCURACY_MSE,
    LOSS_METRIC_ACCURACY_NLL,
    LOSS_METRIC_BERHU,
    LOSS_METRIC_CHOICE_BRIER,
    LOSS_METRIC_CHOICE_NLL,
    LOSS_METRIC_CHOICES,
    LOSS_METRIC_CONDITIONAL_WRONG_CHOICE_NLL,
    LOSS_METRIC_MAE,
    LOSS_METRIC_MSE,
    LOSS_METRIC_TARGET_PROB_BRIER,
    LOSS_METRIC_WRONG_CHOICE_NLL,
    PROBABILISTIC_LOSS_METRIC_CHOICES,
    AccuracyBrierLoss,
    AccuracyCurveBerHuLoss,
    AccuracyCurveFamilyMSELoss,
    AccuracyCurveMAELoss,
    AccuracyCurveMSELoss,
    AccuracyFamilyBrierLoss,
    AccuracyNLLLoss,
    ChoiceBrierLoss,
    ChoiceNLLLoss,
    ConditionalWrongChoiceNLLLoss,
    LossStrategy,
    TargetProbBrierLoss,
    WrongChoiceNLLLoss,
    attach_loss_metrics,
    build_loss_strategy,
    compute_loss_values,
)
from ..simulation.state_model_execution import (
    PREDICTION_MODE_BOTH,
    PREDICTION_MODE_CHOICES,
    PREDICTION_MODE_POSTERIOR_T_MINUS_1,
    PREDICTION_MODE_PRIOR_T,
    BaseStateOptimizer,
    compute_metrics_from_category_probabilities,
    compute_prediction_metrics,
    derive_run_seed,
    evaluate_state_model_run,
    get_hypothesis_transition_seed,
    inject_params,
    prepare_trial_sequence,
    sequential_importance_marginal,
    set_hypothesis_transition_seed,
)
from ..utils.seeding import (
    derive_hyper_candidate_seed,
    derive_module_seed,
    derive_simulation_point_seed,
    derive_trajectory_seed,
    inject_module_seed_from_trajectory,
    stable_seed,
)

__all__ = [name for name in globals() if not name.startswith("_")]

"""共享数值指标的轻量公共入口。

具体子模块只在首次访问对应名称时导入，避免普通 package import 提前加载
SciPy 等较重依赖；现有包级名称导入接口保持不变。
"""

from importlib import import_module


_EXPORT_MODULES = (
    "behavior",
    "group",
    "losses",
    "prediction",
    "residuals",
    "selection",
    "trajectory",
    "trial",
)


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    for module_name in _EXPORT_MODULES:
        module = import_module(f"{__name__}.{module_name}")
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"public metric {name!r} is not defined by a metric submodule")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

__all__ = [
    "MetricResult",
    "RunPrediction",
    "TrialPrediction",
    "ACCURACY_LOSS_METRIC_CHOICES",
    "CHOICE_LOSS_METRIC_CHOICES",
    "PROBABILISTIC_LOSS_METRIC_CHOICES",
    "LOSS_METRIC_CHOICES",
    "LOSS_METRIC_ACCURACY_CURVE_MAE",
    "LOSS_METRIC_ACCURACY_CURVE_MSE",
    "LOSS_METRIC_ACCURACY_CURVE_FAMILY_MSE",
    "LOSS_METRIC_ACCURACY_CURVE_BERHU",
    "LOSS_METRIC_ACCURACY_BRIER",
    "LOSS_METRIC_ACCURACY_FAMILY_BRIER",
    "LOSS_METRIC_ACCURACY_NLL",
    "LOSS_METRIC_CHOICE_BRIER",
    "LOSS_METRIC_CHOICE_NLL",
    "LOSS_METRIC_WRONG_CHOICE_NLL",
    "LOSS_METRIC_CONDITIONAL_WRONG_CHOICE_NLL",
    "LOSS_METRIC_TARGET_PROB_BRIER",
    "LOSS_METRIC_ACCURACY_MAE",
    "LOSS_METRIC_ACCURACY_MSE",
    "LOSS_METRIC_ACCURACY_BERHU",
    "LOSS_METRIC_MAE",
    "LOSS_METRIC_MSE",
    "LOSS_METRIC_BERHU",
    "LossStrategy",
    "AccuracyCurveMAELoss",
    "AccuracyCurveMSELoss",
    "AccuracyCurveFamilyMSELoss",
    "AccuracyCurveBerHuLoss",
    "AccuracyBerHuLoss",
    "AccuracyBrierLoss",
    "AccuracyFamilyBrierLoss",
    "AccuracyNLLLoss",
    "ChoiceBrierLoss",
    "ChoiceNLLLoss",
    "WrongChoiceNLLLoss",
    "ConditionalWrongChoiceNLLLoss",
    "TargetProbBrierLoss",
    "accuracy_curve_metrics",
    "accuracy_shape_metrics_from_runs",
    "accuracy_shape_score",
    "accuracy_curve_mae",
    "accuracy_curve_mse",
    "accuracy_curve_family_mse",
    "accuracy_curve_berhu",
    "accuracy_berhu",
    "accuracy_brier",
    "accuracy_family_brier",
    "accuracy_nll",
    "accuracy_metrics_from_info",
    "accuracy_scalar_metrics",
    "behavior_ppc_group_metrics",
    "benjamini_hochberg",
    "bernoulli_calibration_test",
    "causal_residual_state_feature",
    "centered_curve_metrics",
    "choice_brier",
    "choice_brier_loss",
    "choice_brier_curve_metrics_from_info",
    "conditional_behavioral_accuracy_band_metrics",
    "choice_nll",
    "choice_nll_loss",
    "choice_probability_metrics",
    "curve_discrepancy_metrics",
    "empirical_crps",
    "expected_calibration_error",
    "exponential_smooth_curve",
    "exponential_accuracy_metrics_from_info",
    "family_correct",
    "family_indices",
    "forward_residual_state_probe",
    "history_kernel_metrics",
    "history_kernel_metrics_from_runs",
    "history_kernel_score",
    "loss_metric_summary_from_runs",
    "logit_intercept_recalibration",
    "martingale_lag_tests",
    "marginal_prediction_metrics_from_runs",
    "paired_metric_summary",
    "predictive_accuracy_band_metrics",
    "predictive_interval_metrics",
    "representative_accuracy_shape_score",
    "representative_behavior_score",
    "representative_switch_score",
    "safe_pearson",
    "rolling_martingale_z",
    "simulation_error_summary",
    "sliding_binary_metrics",
    "switch_behavior_metrics",
    "switch_residual_test",
    "switch_behavior_metrics_from_runs",
    "switch_behavior_score",
    "target_majority_accuracy_metrics_from_info",
    "target_majority_indices",
    "validate_exp_accuracy_alpha",
    "target_prob_brier",
    "wrong_choice_nll",
    "conditional_wrong_choice_nll",
    "build_loss_strategy",
    "compute_loss_values",
    "attach_loss_metrics",
    "build_prediction_metric_bundle",
    "distribution_behavior_metrics_from_runs",
]

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from src.Bayesian_state.metrics import (
    RunPrediction,
    TrialPrediction,
    AccuracyCurveBerHuLoss as SharedAccuracyCurveBerHuLoss,
    LOSS_METRIC_CHOICES,
    accuracy_berhu,
    accuracy_curve_berhu,
    accuracy_curve_metrics,
    accuracy_metrics_from_info,
    attach_loss_metrics,
    benjamini_hochberg,
    centered_curve_metrics,
    choice_brier,
    choice_nll,
    empirical_crps,
    expected_calibration_error,
    exponential_smooth_curve,
    marginal_prediction_metrics_from_runs,
    paired_metric_summary,
    predictive_interval_metrics,
    build_prediction_metric_bundle,
    choice_brier_curve_metrics_from_info,
    compute_loss_values,
    representative_accuracy_shape_score,
    representative_behavior_score,
    representative_switch_score,
    simulation_error_summary,
    switch_behavior_metrics,
)
from src.Bayesian_state.simulation.state_model_execution import SingleRunResult
from src.Bayesian_state.model_evaluation.model_evaluation import ModelEval
from src.Bayesian_state.optimization.optimizer_common import (
    AccuracyCurveBerHuLoss as OptimizerAccuracyCurveBerHuLoss,
    exponential_smooth_curve as optimizer_exponential_smooth_curve,
)
from src.Bayesian_state.simulation.repeated_simulation import (
    compute_simulation_statistics,
)
from src.Bayesian_state.utils.simulation_statistics import (
    accuracy_curve_metrics as legacy_accuracy_curve_metrics,
    compute_simulation_statistics as legacy_compute_simulation_statistics,
    marginal_prediction_metrics_from_runs as legacy_marginal_prediction_metrics_from_runs,
)


def test_trial_prediction_validates_only_scored_probability_rows():
    prediction = TrialPrediction(
        category_probabilities=np.asarray(
            [
                [np.nan, np.nan],
                [0.8, 0.2],
                [0.25, 0.75],
            ]
        ),
        observed_choice_index=np.asarray([-1, 0, 1]),
        valid_trial_mask=np.asarray([False, True, True]),
    )

    assert prediction.n_trials == 3
    assert prediction.n_categories == 2
    assert prediction.n_valid == 2
    assert np.allclose(prediction.selected_probabilities(), [0.8, 0.75])
    assert not prediction.category_probabilities.flags.writeable

    with pytest.raises(ValueError, match="not normalized"):
        TrialPrediction(
            category_probabilities=np.asarray([[0.7, 0.2]]),
            observed_choice_index=np.asarray([0]),
        )


def test_choice_scores_and_ece_have_known_values():
    prediction = TrialPrediction(
        category_probabilities=np.asarray([[0.8, 0.2], [0.25, 0.75]]),
        observed_choice_index=np.asarray([0, 1]),
    )

    brier = choice_brier(prediction)
    nll = choice_nll(prediction)
    calibration = expected_calibration_error(prediction, n_bins=2)

    assert brier.n_observations == 2
    assert np.isclose(brier.value, (0.08 + 0.125) / 2.0)
    assert np.isclose(nll.value, -(math.log(0.8) + math.log(0.75)) / 2.0)
    assert np.isclose(calibration.value, 1.0 - np.mean([0.8, 0.75]))
    assert calibration.details["bins"][1]["count"] == 2


def test_learning_curve_helpers_are_shared_without_definition_drift():
    metrics = {
        "sliding_true_acc": np.asarray([0.0, 0.5, 1.0]),
        "sliding_pred_acc": np.asarray([0.1, 0.4, 0.8]),
    }
    shared = accuracy_curve_metrics(metrics)
    legacy = legacy_accuracy_curve_metrics(metrics)

    assert shared == legacy
    assert np.isclose(shared["acc_mae"], (0.1 + 0.1 + 0.2) / 3.0)
    centered = centered_curve_metrics(
        metrics["sliding_true_acc"],
        metrics["sliding_pred_acc"],
    )
    assert np.isclose(centered["level_bias"], -0.06666666666666665)
    assert centered["centered_mae"] < shared["acc_mae"]

    values = np.asarray([1.0, np.nan, 0.0])
    expected = np.asarray([0.75, 0.75, 0.375])
    assert np.allclose(
        exponential_smooth_curve(values, alpha=0.5, init_value=0.5),
        expected,
    )
    assert np.allclose(
        optimizer_exponential_smooth_curve(values, alpha=0.5, init_value=0.5),
        expected,
    )


def test_repeated_run_marginal_scores_use_mean_probabilities_before_scoring():
    choices = np.asarray([0, 1])
    valid = np.asarray([True, True])
    true_curve = np.asarray([0.0, 1.0])
    runs = [
        RunPrediction(
            trial=TrialPrediction(np.asarray([[0.8, 0.2], [0.2, 0.8]]), choices, valid),
            prediction_mode="prior_t",
            sliding_true_accuracy=true_curve,
            sliding_pred_accuracy=np.asarray([0.2, 0.8]),
        ),
        RunPrediction(
            trial=TrialPrediction(np.asarray([[0.6, 0.4], [0.4, 0.6]]), choices, valid),
            prediction_mode="prior_t",
            sliding_true_accuracy=true_curve,
            sliding_pred_accuracy=np.asarray([0.4, 0.6]),
        ),
    ]

    summary = marginal_prediction_metrics_from_runs(
        runs,
        selection_prediction_mode="prior_t",
    )
    legacy_summary = legacy_marginal_prediction_metrics_from_runs(
        runs,
        selection_prediction_mode="prior_t",
    )

    assert summary == legacy_summary
    assert summary["run_count"] == 2
    assert np.isclose(summary["choice_brier"], 0.18)
    assert np.isclose(summary["choice_nll"], -math.log(0.7))
    assert np.isclose(empirical_crps([0.0, 1.0], 0.5), 0.25)

    legacy_runs = [
        SingleRunResult(
            params={},
            mean_error=0.0,
            metrics_by_mode={"prior_t": run.to_metrics_mapping()},
            selection_prediction_mode="prior_t",
            loss_metric="choice_brier",
            loss_delta=None,
        )
        for run in runs
    ]
    object_summary = marginal_prediction_metrics_from_runs(
        legacy_runs,
        selection_prediction_mode="prior_t",
    )
    assert object_summary == summary


def test_predictive_intervals_report_coverage_width_and_crps_together():
    summary = predictive_interval_metrics(
        np.asarray([[0.0, 0.2], [0.5, 0.4], [1.0, 0.6]]),
        np.asarray([0.5, 0.9]),
        alpha=0.20,
    )

    assert summary["n_observations"] == 2
    assert np.isclose(summary["coverage"], 0.5)
    assert summary["mean_width"] > 0.0
    assert summary["mean_crps"] > 0.0


def test_switch_metrics_preserve_trial_order_and_previous_outcome_alignment():
    metrics = {
        "pred_category_probs": np.asarray(
            [[0.9, 0.1], [0.7, 0.3], [0.2, 0.8]]
        ),
        "observed_choice_index": np.asarray([0, 0, 1]),
        "true_acc": np.asarray([1.0, 1.0, 0.0]),
        "valid_trial_mask": np.asarray([True, True, True]),
    }
    summary = switch_behavior_metrics(metrics, min_trials=2)

    assert summary["n_pairs"] == 2
    assert np.isclose(summary["switch_human"], 0.5)
    assert np.isclose(summary["switch_model"], 0.55)
    assert np.isclose(summary["win_stay_human"], 0.5)
    assert np.isnan(summary["lose_shift_human"])


def test_representative_run_scores_are_canonical_metrics():
    metrics = {
        "sliding_true_acc": np.asarray([0.0, 0.5, 1.0]),
        "sliding_pred_acc": np.asarray([0.1, 0.4, 0.8]),
        "pred_category_probs": np.asarray(
            [[0.9, 0.1], [0.7, 0.3], [0.2, 0.8]]
        ),
        "observed_choice_index": np.asarray([0, 0, 1]),
        "valid_trial_mask": np.asarray([True, True, True]),
    }

    shape = representative_accuracy_shape_score(metrics)
    switch = representative_switch_score(metrics)
    expected_shape = np.mean([0.1, 0.1, 0.2]) + 0.06 * abs(np.log(0.7))

    assert np.isclose(shape, expected_shape)
    assert np.isclose(switch, abs(np.mean([0.3, 0.8]) - 0.5))
    assert np.isclose(representative_behavior_score(metrics), np.mean([shape, switch]))

    errors = simulation_error_summary([0.4, 0.2, 0.3])
    assert errors["best_index"] == 1
    assert np.isclose(errors["mean_error"], 0.3)
    assert np.isclose(errors["std_error"], np.std([0.4, 0.2, 0.3]))


def test_simulation_statistics_schema_delegates_to_metrics():
    metrics = {
        "sliding_true_acc": np.asarray([0.0, 0.5, 1.0]),
        "sliding_pred_acc": np.asarray([0.1, 0.4, 0.8]),
        "true_acc": np.asarray([0.0, 1.0, 1.0]),
        "pred_acc": np.asarray([0.1, 0.7, 0.8]),
        "pred_category_probs": np.asarray(
            [[0.9, 0.1], [0.7, 0.3], [0.2, 0.8]]
        ),
        "observed_choice_index": np.asarray([0, 0, 1]),
        "valid_trial_mask": np.asarray([True, True, True]),
        "loss_values": {"choice_brier": 0.25},
    }
    runs = [
        SingleRunResult(
            params={},
            mean_error=0.25,
            metrics_by_mode={"prior_t": metrics},
            selection_prediction_mode="prior_t",
            loss_metric="choice_brier",
            loss_delta=None,
        )
    ]

    summary = compute_simulation_statistics(
        runs,
        selection_prediction_mode="prior_t",
        config={"history_max_lag": 1, "min_switch_trials": 1},
    )

    assert legacy_compute_simulation_statistics is compute_simulation_statistics
    assert np.isclose(summary["loss"]["choice_brier"]["mean"], 0.25)
    assert "accuracy_shape" in summary["scores"]


def test_group_comparison_uses_paired_units_and_explicit_direction():
    summary = paired_metric_summary(
        candidate=np.asarray([0.8, 1.5, np.nan, 2.0]),
        reference=np.asarray([1.0, 1.0, 3.0, 2.5]),
        lower_is_better=True,
        bootstrap_repeats=500,
        seed=17,
    )

    assert summary["difference_direction"] == "candidate_minus_reference"
    assert summary["n_pairs"] == 3
    assert np.isclose(summary["mean_difference"], (-0.2 + 0.5 - 0.5) / 3.0)
    assert summary["candidate_win_count"] == 2
    assert np.isclose(summary["candidate_win_fraction"], 2.0 / 3.0)

    adjusted = benjamini_hochberg([0.01, 0.02, 0.20, np.nan])
    assert np.allclose(adjusted[:3], [0.03, 0.03, 0.20])
    assert np.isnan(adjusted[3])


def test_accuracy_curve_berhu_and_every_configured_loss_live_in_metrics():
    metrics = build_prediction_metric_bundle(
        np.asarray(
            [
                [0.5, 0.5],
                [0.8, 0.2],
                [0.3, 0.7],
                [0.6, 0.4],
            ]
        ),
        choices=np.asarray([1, 1, 1, 2]),
        feedback=np.asarray([0.0, 1.0, 0.0, 0.0]),
        categories=np.asarray([1, 1, 2, 1]),
        target_probabilities=np.asarray(
            [
                [0.5, 0.5],
                [0.9, 0.1],
                [0.2, 0.8],
                [0.7, 0.3],
            ]
        ),
        window_size=2,
    )
    metrics["sliding_true_acc"] = np.asarray([0.0, 0.5, 1.0])
    metrics["sliding_pred_acc"] = np.asarray([0.1, 0.9, 0.2])

    expected_berhu = np.mean([0.1, 0.5, 1.7])
    assert np.isclose(accuracy_curve_berhu(metrics, delta=0.2), expected_berhu)
    assert np.isclose(accuracy_berhu(metrics, delta=0.2), expected_berhu)
    assert np.isclose(
        SharedAccuracyCurveBerHuLoss(0.2).compute(metrics), expected_berhu
    )
    assert OptimizerAccuracyCurveBerHuLoss is SharedAccuracyCurveBerHuLoss

    all_losses = compute_loss_values(metrics, loss_delta=0.2)
    assert set(all_losses) == set(LOSS_METRIC_CHOICES)
    attached = attach_loss_metrics(
        metrics,
        loss_metric="accuracy_curve_berhu",
        loss_delta=0.2,
    )
    assert np.isclose(attached["objective_error"], expected_berhu)
    assert attached["loss_metric"] == "accuracy_curve_berhu"


def test_optimizer_defines_no_local_loss_strategy_implementations():
    source_path = Path(
        "src/Bayesian_state/optimization/optimizer_common.py"
    )
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    local_classes = {
        node.name for node in module.body if isinstance(node, ast.ClassDef)
    }
    local_functions = {
        node.name for node in module.body if isinstance(node, ast.FunctionDef)
    }

    assert not {
        "LossStrategy",
        "AccuracyCurveMAELoss",
        "AccuracyCurveMSELoss",
        "AccuracyCurveFamilyMSELoss",
        "AccuracyCurveBerHuLoss",
        "AccuracyBrierLoss",
        "AccuracyFamilyBrierLoss",
        "AccuracyNLLLoss",
        "ChoiceBrierLoss",
        "ChoiceNLLLoss",
        "WrongChoiceNLLLoss",
        "ConditionalWrongChoiceNLLLoss",
        "TargetProbBrierLoss",
    } & local_classes
    assert "build_loss_strategy" not in local_functions
    assert "compute_loss_values" not in local_functions


def test_simulation_ownership_and_compatibility_facades_are_enforced():
    simulation_modules = {
        path.name
        for path in Path("src/Bayesian_state/simulation").glob("*.py")
        if path.name != "__init__.py"
    }
    assert simulation_modules == {
        "repeated_simulation.py",
        "simulation_config.py",
        "state_model_execution.py",
        "autonomous_model_execution.py",
    }

    facade_paths = (
        Path("src/Bayesian_state/optimization/optimizer_common.py"),
        Path("src/Bayesian_state/optimization/optimizer_simulation.py"),
        Path("src/Bayesian_state/optimization/optimization_config.py"),
        Path("src/Bayesian_state/utils/simulation_statistics.py"),
    )
    for source_path in facade_paths:
        module = ast.parse(source_path.read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            for node in module.body
        ), source_path

    checked_paths = [Path("src/Bayesian_state/run_simulation.py")]
    checked_paths.extend(Path("src/Bayesian_state/simulation").glob("*.py"))
    for source_path in checked_paths:
        module = ast.parse(source_path.read_text(encoding="utf-8"))
        imported_modules = {
            node.module or ""
            for node in ast.walk(module)
            if isinstance(node, ast.ImportFrom)
        }
        assert not any("Bayesian_state.optimization" in name for name in imported_modules)

    for source_path in Path("src/Bayesian_state").rglob("*.py"):
        module = ast.parse(source_path.read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, ast.ImportFrom)
            and any(alias.name == "*" for alias in node.names)
            for node in ast.walk(module)
        ), source_path

    from src.Bayesian_state.simulation.state_model_execution import (
        TrialArrays as CanonicalTrialArrays,
    )
    from src.Bayesian_state.optimization.optimizer_common import TrialArrays as LegacyTrialArrays
    from src.Bayesian_state.optimization.optimizer_simulation import (
        StateModelSimulationRunner as LegacyRunner,
    )
    from src.Bayesian_state.simulation.repeated_simulation import (
        StateModelSimulationRunner,
    )

    assert LegacyTrialArrays is CanonicalTrialArrays
    assert LegacyRunner is StateModelSimulationRunner


def test_model_eval_metric_methods_are_compatibility_wrappers():
    info = {
        "true_acc": np.asarray([0.0, 1.0, 0.0, 1.0]),
        "pred_acc": np.asarray([np.nan, 0.8, 0.3, 0.7]),
        "pred_category_probs": np.asarray(
            [[np.nan, np.nan], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]]
        ),
        "observed_choice_index": np.asarray([-1, 0, 0, 1]),
        "valid_trial_mask": np.asarray([False, True, True, True]),
        "window_size": 2,
    }
    evaluator = ModelEval()

    shared_accuracy = accuracy_metrics_from_info(info)
    wrapped_accuracy = evaluator.compute_accuracy_metrics(info)
    for key in shared_accuracy:
        if isinstance(shared_accuracy[key], np.ndarray):
            assert np.allclose(
                wrapped_accuracy[key], shared_accuracy[key], equal_nan=True
            )
        else:
            assert wrapped_accuracy[key] == shared_accuracy[key]

    shared_brier = choice_brier_curve_metrics_from_info(info)
    wrapped_brier = evaluator.compute_choice_brier_metrics(info)
    assert np.allclose(
        wrapped_brier["choice_brier"], shared_brier["choice_brier"], equal_nan=True
    )
    assert np.allclose(
        wrapped_brier["sliding_choice_brier"],
        shared_brier["sliding_choice_brier"],
        equal_nan=True,
    )


def test_metrics_and_model_evaluation_module_boundaries_are_explicit():
    metric_modules = {
        path.name
        for path in Path("src/Bayesian_state/metrics").glob("*.py")
        if path.name != "__init__.py"
    }
    assert metric_modules == {
        "_numeric.py",
        "behavior_metrics.py",
        "group_statistics.py",
        "losses.py",
        "prediction_metrics.py",
        "trajectory_selection.py",
        "trajectory_statistics.py",
        "trial_metrics.py",
    }

    general_module = ast.parse(
        Path("src/Bayesian_state/model_evaluation/model_evaluation.py").read_text(
            encoding="utf-8"
        )
    )
    transition_module = ast.parse(
        Path(
            "src/Bayesian_state/model_evaluation/transition_evaluation.py"
        ).read_text(encoding="utf-8")
    )
    general_class = next(
        node
        for node in general_module.body
        if isinstance(node, ast.ClassDef) and node.name == "ModelEval"
    )
    transition_class = next(
        node
        for node in transition_module.body
        if isinstance(node, ast.ClassDef) and node.name == "TransitionEvaluationMixin"
    )
    transition_methods = {
        "plot_dynamic_strategy_profile",
        "plot_hypothesis_active_set_counts",
        "plot_strategy_amount",
        "plot_strategy_amount_details",
    }
    assert not transition_methods & {
        node.name for node in general_class.body if isinstance(node, ast.FunctionDef)
    }
    assert transition_methods <= {
        node.name for node in transition_class.body if isinstance(node, ast.FunctionDef)
    }


def test_transition_capabilities_follow_log_fields_not_model_names():
    evaluator = ModelEval()

    assert evaluator.transition_capabilities({"model": "dynamic_discrete"}) == set()
    assert evaluator.transition_capabilities(
        {
            "model": "unrelated_name",
            "strategy_counts_log": [
                {
                    "state_probabilities": {"stable": 0.7, "aggressive": 0.3},
                    "active_total": 12,
                }
            ],
        }
    ) == {"dynamic_discrete", "active_set"}
    assert evaluator.transition_capabilities(
        {
            "strategy_counts_log": [
                {"predictive_m": 0.2, "predictive_g": 0.4}
            ]
        }
    ) == {"dynamic_continuous"}

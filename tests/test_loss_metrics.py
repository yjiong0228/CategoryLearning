from __future__ import annotations

import numpy as np

from src.Bayesian_state.utils.optimizer_common import (
    SingleRunResult,
    build_loss_strategy,
    compute_prediction_metrics,
    compute_loss_values,
    exponential_smooth_curve,
    _choice_readout_weights,
)
from src.Bayesian_state.utils.optimizer_simulation import aggregate_simulation_runs
from src.Bayesian_state.utils.optimization_config import resolve_loss_delta


class _BetaSensitivePartition:
    n_cats = 2

    def get_category_probabilities(self, hypo, data, beta, distance_mode="prototype"):
        if float(beta) >= 10.0:
            return np.asarray([[0.9], [0.1]], dtype=float)
        return np.asarray([[0.5], [0.5]], dtype=float)


class _FakeEngine:
    def __init__(self):
        self.beta = np.asarray([0.1], dtype=float)
        self.distance_mode = "prototype"


class _FakeModel:
    def __init__(self):
        self.hypotheses_set = [0]
        self.partition_model = _BetaSensitivePartition()
        self.engine = _FakeEngine()


def test_prediction_metrics_uses_trial_beta_log_when_available() -> None:
    model = _FakeModel()
    post = np.asarray([[1.0], [1.0], [1.0]], dtype=float)
    prior = np.asarray([[1.0], [1.0], [1.0]], dtype=float)
    step_log = [
        {"perceived_stimulus": np.asarray([0.0], dtype=float)},
        {"perceived_stimulus": np.asarray([0.0], dtype=float)},
        {"perceived_stimulus": np.asarray([0.0], dtype=float)},
    ]
    stimulus = np.zeros((3, 1), dtype=float)
    choices = np.asarray([1, 1, 1], dtype=int)
    feedback = np.asarray([1.0, 1.0, 1.0], dtype=float)
    categories = np.asarray([1, 1, 1], dtype=int)
    beta_log = np.asarray([[0.1], [15.0], [15.0]], dtype=float)

    with_log = compute_prediction_metrics(
        model,
        post,
        prior,
        step_log,
        stimulus,
        choices,
        feedback,
        categories,
        None,
        window_size=1,
        prediction_mode="prior_t",
        loss_metric="choice_brier",
        choice_readout_config={"method": "map_hypothesis"},
        beta_log=beta_log,
    )["prior_t"]
    without_log = compute_prediction_metrics(
        model,
        post,
        prior,
        step_log,
        stimulus,
        choices,
        feedback,
        categories,
        None,
        window_size=1,
        prediction_mode="prior_t",
        loss_metric="choice_brier",
        choice_readout_config={"method": "map_hypothesis"},
    )["prior_t"]

    assert np.isclose(with_log["pred_acc"][1], 0.9)
    assert np.isclose(without_log["pred_acc"][1], 0.5)


def test_berhu_numeric_piecewise() -> None:
    strategy = build_loss_strategy("accuracy_curve_berhu", loss_delta=0.1)
    metrics = {
        "sliding_true_acc": np.asarray([0.0, 0.0], dtype=float),
        "sliding_pred_acc": np.asarray([0.05, 0.2], dtype=float),
    }
    got = strategy.compute(metrics)
    expected = (0.05 + ((0.2 ** 2 + 0.1 ** 2) / (2.0 * 0.1))) / 2.0
    assert np.isclose(got, expected)


def test_berhu_boundary_is_continuous() -> None:
    strategy = build_loss_strategy("accuracy_curve_berhu", loss_delta=0.1)
    metrics = {
        "sliding_true_acc": np.asarray([0.0], dtype=float),
        "sliding_pred_acc": np.asarray([0.1], dtype=float),
    }
    got = strategy.compute(metrics)
    assert np.isclose(got, 0.1)


def test_berhu_missing_delta_raises() -> None:
    try:
        _ = build_loss_strategy("accuracy_curve_berhu")
        assert False, "Expected ValueError for missing loss_delta with berhu"
    except ValueError as e:
        assert "loss_delta" in str(e)


def test_resolve_loss_delta_requires_positive_for_berhu() -> None:
    assert resolve_loss_delta({"loss_delta": 0.05}, "accuracy_curve_berhu") == 0.05
    try:
        _ = resolve_loss_delta({}, "accuracy_curve_berhu")
        assert False, "Expected ValueError for missing loss_delta"
    except ValueError as e:
        assert "loss_delta" in str(e)
    try:
        _ = resolve_loss_delta({"loss_delta": 0.0}, "accuracy_curve_berhu")
        assert False, "Expected ValueError for non-positive loss_delta"
    except ValueError as e:
        assert "loss_delta" in str(e)


def test_resolve_loss_delta_ignored_for_other_losses() -> None:
    assert resolve_loss_delta({}, "accuracy_curve_mse") is None
    assert resolve_loss_delta({"loss_delta": 0.5}, "accuracy_nll") is None


def test_exponential_smooth_curve_matches_manual_formula_and_skips_nan() -> None:
    got = exponential_smooth_curve(
        np.asarray([1.0, 0.0, np.nan, 1.0], dtype=float),
        alpha=0.5,
        init_value=0.25,
    )
    expected = np.asarray([0.625, 0.3125, 0.3125, 0.65625], dtype=float)
    assert np.allclose(got, expected)
    assert np.all((got >= 0.0) & (got <= 1.0))


def test_exponential_smooth_curve_rejects_invalid_alpha() -> None:
    try:
        exponential_smooth_curve([1.0], alpha=0.0, init_value=0.5)
        assert False, "Expected ValueError for invalid alpha"
    except ValueError as e:
        assert "alpha" in str(e)


def test_choice_brier_is_recorded_even_when_not_objective() -> None:
    metrics = {
        "sliding_true_acc": np.asarray([0.5], dtype=float),
        "sliding_pred_acc": np.asarray([0.6], dtype=float),
        "sliding_true_family_acc": np.asarray([np.nan], dtype=float),
        "sliding_pred_family_acc": np.asarray([np.nan], dtype=float),
        "pred_category_probs": np.asarray(
            [
                [np.nan, np.nan],
                [0.8, 0.2],
                [0.3, 0.7],
            ],
            dtype=float,
        ),
        "observed_choice_index": np.asarray([-1, 0, 1], dtype=int),
        "true_category_index": np.asarray([-1, 0, 1], dtype=int),
        "true_acc": np.asarray([np.nan, 1.0, 0.0], dtype=float),
        "pred_acc": np.asarray([np.nan, 0.8, 0.7], dtype=float),
        "true_family_acc": np.asarray([np.nan, np.nan, np.nan], dtype=float),
        "pred_family_acc": np.asarray([np.nan, np.nan, np.nan], dtype=float),
        "target_probs": np.full((3, 2), np.nan, dtype=float),
        "valid_trial_mask": np.asarray([False, True, True], dtype=bool),
    }

    loss_values = compute_loss_values(metrics)
    assert np.isclose(loss_values["choice_brier"], 0.13)

    run = SingleRunResult(
        params={},
        mean_error=0.01,
        metrics_by_mode={
            "prior_t": {
                **metrics,
                "loss_metric": "accuracy_curve_mse",
                "mean_error": 0.01,
                "loss_values": loss_values,
            }
        },
        selection_prediction_mode="prior_t",
        loss_metric="accuracy_curve_mse",
        loss_delta=None,
    )
    result = aggregate_simulation_runs(
        [run, run],
        params={},
        subject_id=1,
        condition=1,
        window_size=2,
        selection_prediction_mode="prior_t",
        simulation_repeats=2,
        simulation_point_seed=123,
        keep_logs=False,
    )

    loss_summary = result.statistics_summary["loss"]
    assert np.isclose(loss_summary["choice_brier"]["mean"], 0.13)
    assert loss_summary["choice_brier"]["count"] == 2


def test_marginal_prediction_scores_average_probabilities_across_runs() -> None:
    common = {
        "observed_choice_index": np.asarray([-1, 0, 1], dtype=int),
        "valid_trial_mask": np.asarray([False, True, True], dtype=bool),
        "sliding_true_acc": np.asarray([0.5, 0.5], dtype=float),
        "loss_metric": "choice_brier",
        "mean_error": 0.2,
        "loss_values": {"choice_brier": 0.2},
    }
    run_metrics = [
        {
            **common,
            "pred_category_probs": np.asarray(
                [[np.nan, np.nan], [0.9, 0.1], [0.2, 0.8]],
                dtype=float,
            ),
            "sliding_pred_acc": np.asarray([0.2, 0.8], dtype=float),
        },
        {
            **common,
            "pred_category_probs": np.asarray(
                [[np.nan, np.nan], [0.5, 0.5], [0.6, 0.4]],
                dtype=float,
            ),
            "sliding_pred_acc": np.asarray([0.4, 0.6], dtype=float),
        },
    ]
    runs = [
        SingleRunResult(
            params={},
            mean_error=0.2,
            metrics_by_mode={"prior_t": metrics},
            selection_prediction_mode="prior_t",
            loss_metric="choice_brier",
            loss_delta=None,
        )
        for metrics in run_metrics
    ]

    result = aggregate_simulation_runs(
        runs,
        params={},
        subject_id=1,
        condition=1,
        window_size=2,
        selection_prediction_mode="prior_t",
        simulation_repeats=2,
        simulation_point_seed=123,
        keep_logs=False,
    )

    marginal = result.statistics_summary["marginal_prediction"]
    assert np.isclose(marginal["choice_brier"], 0.25)
    assert np.isclose(marginal["choice_nll"], -0.5 * (np.log(0.7) + np.log(0.6)))
    assert np.isclose(marginal["trajectory_crps"], 0.15)
    assert np.isclose(marginal["trajectory_median_mae"], 0.2)


def test_loss_summary_records_best10_and_best25_lower_tail_means() -> None:
    runs = []
    for value in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
        runs.append(
            SingleRunResult(
                params={},
                mean_error=value,
                metrics_by_mode={
                    "prior_t": {
                        "loss_metric": "choice_brier",
                        "mean_error": value,
                        "loss_values": {
                            "choice_brier": value,
                            "accuracy_curve_berhu": value * 2.0,
                        },
                    }
                },
                selection_prediction_mode="prior_t",
                loss_metric="choice_brier",
                loss_delta=None,
            )
        )

    result = aggregate_simulation_runs(
        runs,
        params={},
        subject_id=1,
        condition=1,
        window_size=2,
        selection_prediction_mode="prior_t",
        simulation_repeats=len(runs),
        simulation_point_seed=123,
        keep_logs=False,
    )

    choice_summary = result.statistics_summary["loss"]["choice_brier"]
    berhu_summary = result.statistics_summary["loss"]["accuracy_curve_berhu"]

    assert np.isclose(choice_summary["best10_mean"], 0.10)
    assert choice_summary["best10_count"] == 1
    assert np.isclose(choice_summary["best25_mean"], 0.15)
    assert np.isclose(choice_summary["best25-mean"], 0.15)
    assert choice_summary["best25_count"] == 2
    assert np.isclose(berhu_summary["best25_mean"], 0.30)


def test_choice_readout_map_and_sticky_behaviors() -> None:
    distribution = np.asarray([0.2, 0.8], dtype=float)

    map_weights, map_log = _choice_readout_weights(
        distribution,
        trial_idx=1,
        feedback=np.asarray([1.0], dtype=float),
        config={"method": "map_hypothesis"},
        rng=np.random.default_rng(3),
        sticky_state={},
    )
    assert map_weights.tolist() == [0.0, 1.0]
    assert map_log["selected_arg"] == 1

    sticky_state = {"selected_arg": 0}
    sticky_weights, sticky_log = _choice_readout_weights(
        np.asarray([0.0, 1.0], dtype=float),
        trial_idx=1,
        feedback=np.asarray([1.0], dtype=float),
        config={"method": "sticky_sample", "switch_probability": 0.0},
        rng=np.random.default_rng(3),
        sticky_state=sticky_state,
    )
    assert sticky_weights.tolist() == [0.0, 1.0]
    assert sticky_log["switched"] is True
    assert sticky_state["selected_arg"] == 1


def test_stubborn_readout_lowers_post_error_switch_probability() -> None:
    distribution = np.asarray([0.8, 0.2], dtype=float)
    sticky_state = {"selected_arg": 0}

    _, log = _choice_readout_weights(
        distribution,
        trial_idx=1,
        feedback=np.asarray([0.0], dtype=float),
        config={
            "method": "stubborn_sticky",
            "switch_probability": 0.5,
            "post_error_switch_delta": -0.3,
        },
        rng=np.random.default_rng(3),
        sticky_state=sticky_state,
    )

    assert np.isclose(log["switch_probability"], 0.2)

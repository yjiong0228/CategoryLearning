from __future__ import annotations

import numpy as np

from src.Bayesian_state.utils.optimizer_common import (
    SingleRunResult,
    build_loss_strategy,
    compute_loss_values,
)
from src.Bayesian_state.utils.optimizer_simulation import aggregate_simulation_runs
from src.Bayesian_state.utils.optimization_config import resolve_loss_delta


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

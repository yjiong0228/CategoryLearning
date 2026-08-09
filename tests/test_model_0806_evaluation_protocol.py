from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.Bayesian_state.model_evaluation.model_evaluation import ModelEval
from src.Bayesian_state.run_model_evaluation import (
    load_simulation_results,
    run_basic_plots,
)
from src.Bayesian_state.simulation.state_model_execution import (
    compute_metrics_from_category_probabilities,
)
from src.Bayesian_state.simulation.simulation_config import (
    EVALUATION_ROLE_OPTIMIZATION,
    EVALUATION_ROLE_SIMULATION,
    resolve_evaluation_score_mask,
)


def _sequential_protocol() -> dict[str, object]:
    return {
        "mode": "sequential_holdout",
        "train_fraction": 0.5,
        "min_train_trials": 2,
        "min_evaluation_trials": 2,
        "optimization_partition": "train",
        "simulation_partition": "evaluation",
    }


def test_sequential_holdout_uses_disjoint_role_specific_score_masks() -> None:
    train_mask, train_context = resolve_evaluation_score_mask(
        10,
        _sequential_protocol(),
        role=EVALUATION_ROLE_OPTIMIZATION,
    )
    evaluation_mask, evaluation_context = resolve_evaluation_score_mask(
        10,
        _sequential_protocol(),
        role=EVALUATION_ROLE_SIMULATION,
    )

    np.testing.assert_array_equal(
        train_mask,
        np.asarray([True] * 5 + [False] * 5),
    )
    np.testing.assert_array_equal(
        evaluation_mask,
        np.asarray([False] * 5 + [True] * 5),
    )
    assert not np.any(train_mask & evaluation_mask)
    assert np.all(train_mask | evaluation_mask)
    assert train_context["partition"] == "train"
    assert evaluation_context["partition"] == "evaluation"
    assert train_context["split_index"] == evaluation_context["split_index"] == 5
    assert train_context["score_trial_count"] == evaluation_context["score_trial_count"] == 5


def test_sequential_holdout_rejects_ambiguous_or_too_small_splits() -> None:
    with pytest.raises(ValueError, match="only one"):
        resolve_evaluation_score_mask(
            10,
            {"train_trials": 5, "train_fraction": 0.5},
            role=EVALUATION_ROLE_OPTIMIZATION,
        )
    with pytest.raises(ValueError, match="minimum partition sizes"):
        resolve_evaluation_score_mask(
            10,
            {
                "train_trials": 9,
                "min_train_trials": 2,
                "min_evaluation_trials": 2,
            },
            role=EVALUATION_ROLE_SIMULATION,
        )


def test_excluded_holdout_trials_do_not_change_optimization_loss() -> None:
    train_mask, _ = resolve_evaluation_score_mask(
        10,
        _sequential_protocol(),
        role=EVALUATION_ROLE_OPTIMIZATION,
    )
    evaluation_mask, _ = resolve_evaluation_score_mask(
        10,
        _sequential_protocol(),
        role=EVALUATION_ROLE_SIMULATION,
    )
    baseline = np.tile(np.asarray([0.9, 0.1]), (10, 1))
    changed_holdout = baseline.copy()
    changed_holdout[5:] = np.asarray([0.1, 0.9])
    common = {
        "choices": np.ones(10, dtype=int),
        "feedback": np.ones(10, dtype=float),
        "categories": np.ones(10, dtype=int),
        "target_probs": None,
        "window_size": 2,
        "loss_metric": "choice_brier",
    }

    baseline_train = compute_metrics_from_category_probabilities(
        baseline,
        score_trial_mask=train_mask,
        **common,
    )
    changed_train = compute_metrics_from_category_probabilities(
        changed_holdout,
        score_trial_mask=train_mask,
        **common,
    )
    changed_evaluation = compute_metrics_from_category_probabilities(
        changed_holdout,
        score_trial_mask=evaluation_mask,
        **common,
    )

    assert baseline_train["mean_error"] == pytest.approx(changed_train["mean_error"])
    assert changed_evaluation["mean_error"] > changed_train["mean_error"]
    np.testing.assert_array_equal(
        changed_train["valid_trial_mask"],
        train_mask & (np.arange(10) > 0),
    )


def _write_particle_subject(input_dir: Path) -> None:
    subject_dir = input_dir / "subjects"
    subject_dir.mkdir(parents=True)
    transition_rate = [0.15, 0.20, 0.25, 0.30]
    search_range = [0.35, 0.40, 0.45, 0.50]
    payload = {
        "result_type": "simulation",
        "subject_id": 105,
        "condition": 1,
        "simulation": {"window_size": 2, "simulation_repeats": 1},
        "selection": {
            "selection_prediction_mode": "prior_t",
            "prediction_mode": "prior_t",
            "loss_metric": "choice_brier",
            "selection_meta": {
                "score_context": {
                    "mode": "sequential_holdout",
                    "partition": "evaluation",
                    "split_index": 2,
                    "score_trial_count": 2,
                }
            },
        },
        "representative_run": {
            "metrics_by_mode": {
                "prior_t": {
                    "mean_error": 0.24,
                    "loss_metric": "choice_brier",
                    "particle_count": 4,
                    "true_acc": [1.0, 0.0, 1.0, 1.0],
                    "pred_acc": [0.5, 0.4, 0.7, 0.8],
                    "sliding_true_acc": [float("nan"), 2.0 / 3.0],
                    "sliding_pred_acc": [float("nan"), 0.75],
                    "sliding_pred_acc_std": [float("nan"), 0.10],
                    "exp_true_acc": [0.5, 0.4, 0.6, 0.7],
                    "exp_pred_acc": [0.5, 0.46, 0.56, 0.66],
                    "observed_choice_index": [0, 1, 0, 0],
                    "pred_category_probs": [
                        [0.5, 0.5],
                        [0.6, 0.4],
                        [0.7, 0.3],
                        [0.8, 0.2],
                    ],
                    "valid_trial_mask": [False, False, True, True],
                    "choice_brier_by_trial": [float("nan"), float("nan"), 0.18, 0.08],
                }
            },
            "state_log": {
                "marginal_prior": [
                    [0.50, 0.30, 0.20],
                    [0.45, 0.35, 0.20],
                    [0.30, 0.45, 0.25],
                    [0.20, 0.55, 0.25],
                ],
                "marginal_active_probability": [
                    [1.0, 0.5, 0.0],
                    [0.9, 0.6, 0.1],
                    [0.7, 0.8, 0.2],
                    [0.5, 0.9, 0.4],
                ],
                "transition_rate": transition_rate,
                "search_range": search_range,
                "replacement_count": [0.0, 1.0, 1.0, 2.0],
                "replacement_fraction": [0.0, 0.25, 0.25, 0.50],
                "removed_mass": [0.0, 0.1, 0.2, 0.3],
                "newcomer_distance": [0.0, 0.2, 0.3, 0.4],
                "feedback_surprise": [0.2, 0.8, 0.5, 1.1],
                "feedback_uncertainty": [0.7, 0.6, 0.5, 0.4],
                "pre_choice_ess": [4.0, 3.2, 2.4, 3.8],
                "post_choice_ess": [3.5, 2.8, 2.0, 3.4],
                "resampled": [False, False, True, False],
            },
            "trial_events": [],
            "transition_counts": [
                {
                    "predictive_m": transition_rate[index],
                    "predictive_g": search_range[index],
                    "active_total": 2.0,
                    "strategies": [],
                }
                for index in range(4)
            ],
        },
    }
    (subject_dir / "subject_105.json").write_text(
        json.dumps(payload, allow_nan=True),
        encoding="utf-8",
    )


def test_particle_result_adapter_preserves_marginal_state_semantics(tmp_path: Path) -> None:
    _write_particle_subject(tmp_path)
    info = load_simulation_results(tmp_path)[105]

    assert info["state_distribution_kind"] == "particle_marginal"
    assert info["posterior_log"] is None
    assert info["prior_log"] == info["marginal_prior_log"]
    assert np.asarray(info["prior_log"]).shape == (4, 3)
    assert np.asarray(info["marginal_active_probability"]).shape == (4, 3)
    assert info["transition_rate"] == [0.15, 0.20, 0.25, 0.30]
    assert info["score_context"]["partition"] == "evaluation"


def test_dynamic_continuous_particle_plots_and_basic_dispatch(tmp_path: Path, monkeypatch) -> None:
    _write_particle_subject(tmp_path)
    results = load_simulation_results(tmp_path)
    evaluator = ModelEval()

    plot_methods = (
        (evaluator.plot_dynamic_continuous_controls, "controls.png"),
        (evaluator.plot_dynamic_continuous_signals, "signals.png"),
        (evaluator.plot_particle_filter_ess, "ess.png"),
        (evaluator.plot_marginal_active_probabilities, "active.png"),
    )
    for method, filename in plot_methods:
        output = tmp_path / filename
        method(results, save_path=output)
        assert output.is_file()
        plt.close("all")

    called: list[str] = []

    def fake_plot(*args, save_path=None, **kwargs):
        del args, kwargs
        called.append(Path(save_path).name)

    for name in (
        "plot_accuracy_comparison",
        "plot_exponential_accuracy_comparison",
        "plot_choice_brier",
        "plot_prior_probabilities",
        "plot_dynamic_continuous_controls",
        "plot_dynamic_continuous_signals",
        "plot_particle_filter_ess",
        "plot_marginal_active_probabilities",
        "plot_hypothesis_active_set_counts",
    ):
        monkeypatch.setattr(evaluator, name, fake_plot)

    records: list[dict[str, object]] = []
    run_basic_plots(
        evaluator,
        results,
        tmp_path / "evaluation",
        subjects=None,
        window_size=2,
        exp_accuracy_alpha=None,
        records=records,
        posterior_limit=False,
    )

    statuses = {record["name"]: record["status"] for record in records}
    assert statuses["posterior_probabilities"] == "not_applicable"
    assert statuses["prior_probabilities"] == "ok"
    assert statuses["dynamic_continuous_controls"] == "ok"
    assert statuses["dynamic_continuous_signals"] == "ok"
    assert statuses["particle_filter_ess"] == "ok"
    assert statuses["marginal_active_probabilities"] == "ok"
    assert {
        "dynamic_continuous_controls.png",
        "dynamic_continuous_signals.png",
        "particle_filter_ess.png",
        "marginal_active_probabilities.png",
    }.issubset(called)

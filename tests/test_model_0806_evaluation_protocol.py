from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.Bayesian_state.model_evaluation.model_evaluation import ModelEval
from src.Bayesian_state.model_evaluation.particle_filter_strategy_audit import (
    _disable_strategy_controllers,
    _event_indices,
    _mean_matched_controls,
    _rolling_probability_runs,
)
from src.Bayesian_state.model_evaluation.particle_filter_choice_transmission_audit import (
    _correct_probabilities,
    _engine_with_strategy_confidence_gain,
    _gain_screen_summary_rows,
    _low_entry_delta,
    _performance_phase,
    _posterior_medoid_index,
    _save_error_transmission_figure,
    _save_figure,
    _save_gain_screen_figure,
    _strategy_confidence_gain_values,
    _weighted_path_quantiles,
)
from src.Bayesian_state.model_evaluation.particle_filter_residual_diagnostics import (
    _plot_diagnostics,
)
import src.Bayesian_state.run_model_evaluation as evaluation_cli
from src.Bayesian_state.run_model_evaluation import (
    load_simulation_results,
    run_basic_plots,
    run_behavior_ppc_plots,
)
from src.Bayesian_state.simulation.state_model_execution import (
    SingleRunResult,
    compute_metrics_from_category_probabilities,
)
from src.Bayesian_state.simulation.repeated_simulation import aggregate_simulation_runs
from src.Bayesian_state.simulation.simulation_config import (
    EVALUATION_ROLE_OPTIMIZATION,
    EVALUATION_ROLE_SIMULATION,
    resolve_evaluation_score_mask,
)
from src.Bayesian_state.utils.stream import StreamList


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


def test_strategy_audit_static_counterfactuals_are_mean_matched_and_fixed() -> None:
    info = {
        "best_params": {
            "engine.modules.hypo_transitions_mod.kwargs.capacity": 3,
        },
        "predictive_strategy_exploit": [1.0, 0.70, 0.70],
        "predictive_strategy_local_explore": [0.0, 0.20, 0.20],
        "predictive_strategy_global_explore": [0.0, 0.10, 0.10],
    }
    matched_m, matched_g, matched_exploration = _mean_matched_controls(info)
    assert matched_exploration == pytest.approx(0.30)
    assert matched_g == pytest.approx(1.0 / 3.0)
    assert 1.0 - (1.0 - matched_m) ** 3 == pytest.approx(0.30)

    engine = {
        "modules": {
            "hypo_transitions_mod": {
                "class": "dynamic",
                "kwargs": {
                    "m": 0.15,
                    "g": 0.35,
                    "rate_controller": {
                        "m_phi": 0.5,
                        "m_beta_surprise": 0.8,
                    },
                    "range_controller": {"g_beta_uncertainty": 0.4},
                },
            }
        }
    }
    static = _disable_strategy_controllers(engine, m=matched_m, g=matched_g)
    transition = static["modules"]["hypo_transitions_mod"]
    assert transition["class"].endswith("StaticWorkspaceHypothesisTransitionModule")
    kwargs = transition["kwargs"]
    assert kwargs["m"] == pytest.approx(matched_m)
    assert kwargs["g"] == pytest.approx(matched_g)
    assert kwargs["rate_controller"]["m_beta_surprise"] == 0.0
    assert kwargs["range_controller"]["g_beta_uncertainty"] == 0.0


def test_strategy_audit_events_are_causal_and_statistics_can_be_skipped() -> None:
    true_accuracy = np.asarray([1, 0, 0, 1, 1, 0, 0, 0], dtype=float)
    causal_accuracy = np.asarray([np.nan, np.nan, 0.5, 0.4, 0.7, 0.6, 0.5, 0.3])
    events = _event_indices(
        true_accuracy,
        causal_accuracy,
        low_threshold=0.60,
    )
    assert events["error_streak"] == [3, 7]
    assert events["low_performance_entry"] == [2, 5]
    rolling = _rolling_probability_runs(
        np.asarray([[0.0, 0.2, 0.4, 0.6], [1.0, 0.8, 0.6, 0.4]]),
        window_size=2,
    )
    np.testing.assert_allclose(rolling, [[0.3, 0.5], [0.7, 0.5]])

    runs = [
        SingleRunResult(
            params={},
            mean_error=0.2 + index * 0.1,
            metrics_by_mode={"prior_t": {}},
            selection_prediction_mode="prior_t",
            loss_metric="choice_nll",
            loss_delta=None,
            trajectory_seed=10 + index,
        )
        for index in range(2)
    ]
    result = aggregate_simulation_runs(
        runs,
        params={},
        subject_id=1,
        condition=1,
        window_size=2,
        selection_prediction_mode="prior_t",
        simulation_repeats=2,
        simulation_point_seed=None,
        keep_logs=True,
        compute_statistics=False,
    )
    assert result.statistics_summary == {}
    assert [run["trajectory_seed"] for run in result.raw_runs] == [10, 11]


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
        "best_params": {
            "engine.modules.hypo_transitions_mod.kwargs.capacity": 4,
        },
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
                "predictive_misconception_capture_hold_probability": [
                    0.0, 0.2, 0.8, 0.4
                ],
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


def _add_particle_run_stream(input_dir: Path) -> None:
    subject_path = input_dir / "subjects" / "subject_105.json"
    payload = json.loads(subject_path.read_text(encoding="utf-8"))
    metrics = payload["representative_run"]["metrics_by_mode"]["prior_t"]
    metrics["sliding_true_acc"] = [float("nan"), 1.0]
    metrics["score_trial_mask"] = [False, False, True, True]
    cache_dir = input_dir / "cache"
    cache_dir.mkdir(parents=True)
    stream_path = cache_dir / "subject_105_raw_runs.gz"
    stream = StreamList(str(stream_path), 0)
    stream.append(
        {
            "run_index": 0,
            "mean_error": 0.24,
            "selection_prediction_mode": "prior_t",
            "metrics_by_mode": {"prior_t": metrics},
        }
    )
    payload["raw_runs_ref"] = {
        "path": "../cache/subject_105_raw_runs.gz",
        "count": 1,
    }
    subject_path.write_text(
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
    assert info["predictive_misconception_capture_hold_probability"] == [
        0.0, 0.2, 0.8, 0.4
    ]
    assert info["score_context"]["partition"] == "evaluation"


def test_dynamic_continuous_particle_plots_and_basic_dispatch(tmp_path: Path, monkeypatch) -> None:
    _write_particle_subject(tmp_path)
    results = load_simulation_results(tmp_path)
    evaluator = ModelEval()

    plot_methods = (
        (evaluator.plot_particle_filter_dynamic_strategy_profile, "strategy.png"),
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
        "plot_particle_filter_dynamic_strategy_profile",
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
    assert statuses["dynamic_strategy_profile"] == "ok"
    assert statuses["dynamic_continuous_signals"] == "ok"
    assert statuses["particle_filter_ess"] == "ok"
    assert statuses["marginal_active_probabilities"] == "ok"
    assert {
        "dynamic_strategy_profile.png",
        "dynamic_continuous_signals.png",
        "particle_filter_ess.png",
        "marginal_active_probabilities.png",
    }.issubset(called)


def test_particle_continuous_strategy_profile_prefers_exact_pre_choice_fields():
    evaluator = ModelEval()
    info = {
        "subject_id": 7,
        "condition": 1,
        "window_size": 2,
        "true_acc": [1.0, 0.0, 1.0, 1.0],
        "best_params": {
            "engine.modules.hypo_transitions_mod.kwargs.capacity": 3,
        },
        "predictive_strategy_exploit": [1.0, 0.7, 0.6, 0.8],
        "predictive_strategy_local_explore": [0.0, 0.2, 0.1, 0.1],
        "predictive_strategy_global_explore": [0.0, 0.1, 0.3, 0.1],
        "predictive_swap_probability": [0.0, 0.3, 0.4, 0.2],
        "predictive_swap_event_probability": [0.0, 0.25, 0.5, 0.25],
        "predictive_newcomer_distance": [0.0, 0.05, 0.2, 0.1],
        "predictive_misconception_capture_hold_probability": [0.0, 0.2, 0.8, 0.4],
        "predictive_misconception_capture_switch_event_probability": [0.0, 0.2, 0.1, 0.0],
        "predictive_misconception_capture_eligible_probability": [0.0, 0.3, 0.9, 0.5],
    }

    profile = evaluator._particle_continuous_strategy_data(
        info,
        window_size=2,
    )

    assert profile["source_semantics"] == "pre_choice_particle_marginal"
    np.testing.assert_allclose(profile["strategy"].sum(axis=1), 1.0)
    np.testing.assert_allclose(profile["strategy"][0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(
        profile["conditional_newcomer_distance"][1:],
        [0.2, 0.4, 0.4],
    )
    assert profile["capture_diagnostics_available"] is True
    assert profile["low_capture_hold"] == pytest.approx(0.6)


def test_particle_filter_accuracy_band_uses_behavioral_draws(tmp_path: Path) -> None:
    _write_particle_subject(tmp_path)
    _add_particle_run_stream(tmp_path)
    evaluator = ModelEval()
    output = tmp_path / "evaluation" / "basic" / "accuracy_band.png"
    summary_path = tmp_path / "evaluation" / "basic" / "accuracy_band_summary.csv"

    summary = evaluator.plot_particle_filter_accuracy_band_group(
        tmp_path,
        save_path=output,
        summary_path=summary_path,
        eval_prediction_mode="prior_t",
        n_draws=1000,
        seed=23,
    )

    assert output.is_file()
    assert summary_path.is_file()
    assert summary.loc[0, "band_type"] == "observed_history_conditional_behavioral"
    assert summary.loc[0, "n_pf_runs"] == 1
    assert summary.loc[0, "n_behavioral_draws"] == 1000
    assert summary.loc[0, "mean_width_90"] > 0.0


def test_behavior_ppc_dispatches_one_backend_specific_accuracy_band(
    tmp_path: Path,
    monkeypatch,
) -> None:
    evaluator = ModelEval()
    calls: list[str] = []

    def fake_band(*args, **kwargs):
        del args, kwargs
        calls.append("particle_filter_accuracy_band")

    def fake_behavior(*args, **kwargs):
        del args, kwargs
        calls.append("behavior_ppc")

    def fake_residual(*args, **kwargs):
        del args, kwargs
        calls.append("sequential_residuals")

    monkeypatch.setattr(
        evaluator,
        "plot_particle_filter_accuracy_band_group",
        fake_band,
    )
    monkeypatch.setattr(evaluator, "save_behavior_ppc_outputs", fake_behavior)
    monkeypatch.setattr(
        evaluation_cli,
        "run_particle_filter_residual_diagnostics",
        fake_residual,
    )
    records: list[dict[str, object]] = []
    run_behavior_ppc_plots(
        evaluator=evaluator,
        results={105: {"state_distribution_kind": "particle_marginal"}},
        input_dir=tmp_path,
        output_dir=tmp_path / "evaluation",
        records=records,
        subjects=None,
        eval_prediction_mode="prior_t",
        max_runs_per_subject=None,
        accuracy_band_draws=1000,
        accuracy_band_seed=23,
    )

    assert calls == [
        "particle_filter_accuracy_band",
        "behavior_ppc",
        "sequential_residuals",
    ]
    assert [record["name"] for record in records] == [
        "particle_filter_accuracy_band",
        "behavior_ppc",
        "particle_filter_sequential_residual_diagnostics",
    ]


def test_particle_filter_residual_figure_is_png_only(tmp_path: Path) -> None:
    trials = pd.DataFrame(
        {
            "subject_id": [103] * 12,
            "trial": np.arange(1, 13),
            "rolling_accuracy_residual_z": np.linspace(-1.0, 1.0, 12),
        }
    )
    lags = pd.DataFrame(
        {
            "subject_id": [103] * 4,
            "residual_type": [
                "accuracy",
                "accuracy",
                "choice_label",
                "choice_label",
            ],
            "lag": [1, 2, 1, 2],
            "z": [0.2, 2.2, -0.1, 0.3],
            "p": [0.8, 0.02, 0.9, 0.7],
        }
    )
    summary = pd.DataFrame(
        {
            "subject_id": [103],
            "max_lag": [2],
            "intercept_minus_baseline_nll": [0.01],
            "state_minus_intercept_nll": [-0.02],
        }
    )
    output = tmp_path / "residuals.png"
    _plot_diagnostics(trials, lags, summary, output)
    assert output.is_file()
    assert sorted(path.suffix for path in tmp_path.iterdir()) == [".png"]


def test_choice_transmission_helpers_align_categories_and_event_windows() -> None:
    probabilities = np.asarray(
        [
            [[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.1, 0.9]],
            [[0.7, 0.3], [0.4, 0.6], [0.5, 0.5], [0.2, 0.8]],
        ]
    )
    correct = _correct_probabilities(probabilities, np.asarray([0, 1, 0, 1]))
    np.testing.assert_allclose(
        correct,
        [[0.8, 0.7, 0.6, 0.9], [0.7, 0.6, 0.5, 0.8]],
    )

    values = np.asarray([0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.5, 0.6])
    delta, count = _low_entry_delta(
        values,
        [3],
        pre_offsets=(-3, -2, -1),
        post_offsets=(0, 1, 2),
    )
    assert count == 1
    assert delta == pytest.approx(
        np.mean([0.4, 0.3, 0.2]) - np.mean([0.8, 0.7, 0.6])
    )


def test_ancestral_path_helpers_use_posterior_weights() -> None:
    values = np.asarray(
        [
            [0.1, 0.2],
            [0.5, 0.6],
            [0.9, 1.0],
        ]
    )
    weights = np.asarray([0.2, 0.6, 0.2])
    quantiles = _weighted_path_quantiles(values, weights)
    np.testing.assert_allclose(quantiles[:, 0], [0.1, 0.5, 0.9])
    np.testing.assert_allclose(quantiles[:, 1], [0.2, 0.6, 1.0])

    strategy = np.stack([values, 1.0 - values], axis=2)
    medoid, scores = _posterior_medoid_index(strategy, weights)
    assert medoid == 1
    assert scores[medoid] == pytest.approx(np.min(scores))


def test_gain_screen_uses_disabled_ablation_and_causal_deep_valleys(
    tmp_path: Path,
) -> None:
    assert _strategy_confidence_gain_values([3, 0, 2, 1]) == (
        0.0,
        1.0,
        2.0,
        3.0,
    )
    with pytest.raises(ValueError, match="gain=0"):
        _strategy_confidence_gain_values([1, 2, 3])

    engine = {
        "choice_readout": {
            "kwargs": {
                "method": "sharpened_expectation",
                "strategy_confidence_gain": 2.0,
            }
        }
    }
    changed = _engine_with_strategy_confidence_gain(engine, 3.0)
    assert changed["choice_readout"]["kwargs"]["strategy_confidence_gain"] == 3.0
    assert engine["choice_readout"]["kwargs"]["strategy_confidence_gain"] == 2.0

    profile = {
        "subject_id": 103,
        "n_trials": 5,
        "n_seeds": 4,
        "score_mask": np.ones(5, dtype=bool),
        "observed_choice_index": np.asarray([0, 1, 0, 1, 0]),
        "phase": np.asarray(
            ["warmup", "mastery", "low", "low", "middle_recovery"],
            dtype=object,
        ),
        "causal_accuracy": np.asarray([np.nan, 0.9, 0.35, 0.30, 0.7]),
        "mean_category": {
            "current_marginal": np.asarray(
                [
                    [0.5, 0.5],
                    [0.2, 0.8],
                    [0.4, 0.6],
                    [0.6, 0.4],
                    [0.7, 0.3],
                ]
            )
        },
        "expected_correct": {
            "current_marginal": np.asarray([0.5, 0.8, 0.4, 0.4, 0.7])
        },
        "true_accuracy": np.asarray([1.0, 1.0, 0.0, 0.0, 1.0]),
        "choice_confidence_signal": np.asarray([0.0, 0.8, 0.0, 0.0, 0.2]),
        "strategy_choice_precision": np.asarray([1.0, 2.6, 1.0, 1.0, 1.4]),
        "exploration": np.asarray([0.3, 0.1, 0.7, 0.7, 0.3]),
        "failure_pressure": np.asarray([0.4, 0.1, 0.9, 0.9, 0.4]),
        "mastery_evidence": np.asarray([0.5, 0.9, 0.2, 0.2, 0.6]),
    }
    summary = pd.DataFrame(
        _gain_screen_summary_rows(
            profile,
            gain=2.0,
            deep_valley_threshold=0.40,
        )
    )
    deep = summary[summary["stratum"] == "deep_valley"].iloc[0]
    assert deep["n_trials"] == 2
    assert deep["choice_nll"] == pytest.approx(-np.log(0.4))
    assert deep["mean_confidence_signal"] == pytest.approx(0.0)

    figure_rows: list[dict[str, object]] = []
    for subject_id in (103, 120):
        for gain in (0.0, 1.0, 2.0, 3.0):
            for stratum in ("overall", "low", "middle_recovery", "mastery", "deep_valley"):
                improvement = (
                    0.02 * gain
                    if stratum in {"overall", "middle_recovery", "mastery"}
                    else 0.0
                )
                figure_rows.append(
                    {
                        "subject_id": subject_id,
                        "strategy_confidence_gain": gain,
                        "stratum": stratum,
                        "choice_nll": 0.65 - improvement,
                        "choice_nll_improvement_from_gain0": improvement,
                        "deep_valley_threshold": 0.40,
                        "n_common_seeds": 4,
                    }
                )
    output = _save_gain_screen_figure(
        pd.DataFrame(figure_rows),
        tmp_path / "strategy_confidence_gain_screen",
    )
    assert output == tmp_path / "strategy_confidence_gain_screen.png"
    assert output.exists()
    assert sorted(path.suffix for path in tmp_path.iterdir()) == [".png"]


def test_choice_transmission_figure_export_is_png_only(tmp_path: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0], [0.0, 1.0])
    output = _save_figure(fig, tmp_path / "audit_figure")
    assert output == tmp_path / "audit_figure.png"
    assert output.exists()
    assert sorted(path.suffix for path in tmp_path.iterdir()) == [".png"]


def test_error_transmission_phases_are_causal_and_export_png_only(
    tmp_path: Path,
) -> None:
    phases = _performance_phase(
        np.asarray([np.nan, 0.55, 0.70, 0.85]),
        low_threshold=0.60,
    )
    assert phases.tolist() == [
        "warmup",
        "low",
        "middle_recovery",
        "mastery",
    ]

    phase_rows = []
    for index, phase in enumerate(phases):
        phase_rows.append(
            {
                "subject_id": 103,
                "performance_phase": phase,
                "correct_predicting_rule_available": 0.9,
                "belief_mass_on_correct_predicting_rules": 0.5 + 0.05 * index,
                "belief_only_correct_probability": 0.55 + 0.05 * index,
                "sharpened_correct_probability": 0.56 + 0.05 * index,
                "strategy_confidence_correct_probability": (
                    0.57 + 0.05 * index
                ),
                "final_correct_probability": 0.55 + 0.05 * index,
                "observed_correct": 0.50 + 0.10 * index,
                "strategy_explore": 0.7 - 0.10 * index,
                "failure_pressure": 0.8 - 0.15 * index,
                "mastery_evidence": 0.1 + 0.20 * index,
            }
        )
    output = _save_error_transmission_figure(
        pd.DataFrame(phase_rows),
        tmp_path / "error_transmission_layers",
    )
    assert output == tmp_path / "error_transmission_layers.png"
    assert output.exists()
    assert sorted(path.suffix for path in tmp_path.iterdir()) == [".png"]

    persistent_rows = pd.DataFrame(phase_rows).assign(
        persistent_execution_enabled=True,
        persistent_execution_correct_probability=lambda frame: (
            frame["strategy_confidence_correct_probability"] - 0.03
        ),
        execution_switch_probability=[0.08, 0.10, 0.04, 0.01],
    )
    persistent_dir = tmp_path / "persistent"
    persistent_output = _save_error_transmission_figure(
        persistent_rows,
        persistent_dir / "error_transmission_layers",
    )
    assert persistent_output == persistent_dir / "error_transmission_layers.png"
    assert persistent_output.exists()
    assert sorted(path.suffix for path in persistent_dir.iterdir()) == [".png"]

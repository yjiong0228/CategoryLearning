"""Autonomous shape summaries preserve partial reward but plot species accuracy."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml

from src.Bayesian_state.evaluation.autonomous_trajectories import (
    AutonomousEnsemble,
    AutonomousEvaluationSpec,
    _trajectory_summary_frame,
    generate_autonomous_ensemble,
    plot_autonomous_trajectory_ensemble,
    summarize_trajectory_shapes,
)
from src.Bayesian_state.simulation.data import TrialArrays
from src.Bayesian_state.evaluation.internal_cognitive_trajectories import (
    _save_fixed_path_model_human_figure,
    summarize_fixed_path_fit,
)


def test_partial_rewards_do_not_count_as_half_a_species_success_in_curves_or_summaries():
    rewards = np.asarray([[1, 0.5, 0, 0.5, 1, 0], [0.5, 1, 1, 1, 0.5, 0]])
    ensemble = AutonomousEnsemble(
        choices=np.asarray([[1, 2, 1, 3, 2, 4], [2, 1, 3, 4, 1, 3]]),
        feedback=rewards,
        expected_correct_probability=np.full((2, 6), 0.25),
        trajectory_seeds=np.asarray([1, 2]),
    )
    summary = summarize_trajectory_shapes(
        ensemble,
        observed_feedback=rewards[0],
        window_size=1,
        mastery_sustain_windows=1,
        final_block_trials=3,
        max_clusters=2,
    )

    np.testing.assert_array_equal(summary.rolling_accuracy[0], [0, 0, 0, 1, 0])
    np.testing.assert_array_equal(summary.observed_rolling_accuracy, [0, 0, 0, 1, 0])
    np.testing.assert_allclose(summary.overall_accuracy, [1 / 3, 1 / 2])
    np.testing.assert_allclose(summary.final_block_accuracy, [1 / 3, 1 / 3])
    np.testing.assert_array_equal(ensemble.feedback, rewards)

    frame = _trajectory_summary_frame(
        ensemble, summary, categories=np.asarray([1, 1, 3, 4, 2, 1]), n_categories=4
    )
    np.testing.assert_allclose(frame["species_accuracy"], [1 / 3, 1 / 2])
    np.testing.assert_allclose(frame["family_accuracy"], [2 / 3, 5 / 6])
    np.testing.assert_allclose(frame["mean_reward"], [1 / 2, 2 / 3])


def test_real_autonomous_ensemble_retains_half_rewards_in_float_arrays(tmp_path):
    config_path = Path(__file__).resolve().parents[3] / "configs/exp123/model_struct/pmh_model_cond3_0826.yaml"
    engine_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    engine_config["output_noise"]["kwargs"]["base_lapse"] = 1.0
    engine_config["modules"]["perception_mod"]["kwargs"] = {"mean": [0.0] * 4, "std": [0.0] * 4}
    categories = np.tile([1, 2, 3, 4], 3)
    arrays = TrialArrays(
        stimulus=np.tile([0.2, 0.8, 0.3, 0.7], (12, 1)),
        choices=np.ones(12, dtype=int),
        feedback=np.zeros(12),
        categories=categories,
    )
    spec = AutonomousEvaluationSpec(
        config_path=config_path,
        label="condition3 test",
        subject_id=301,
        condition=3,
        window_size=1,
        engine_config=engine_config,
        fixed_hyperparams={},
        arrays=arrays,
        processed_data_dir=tmp_path,
        dataset_paths={},
        config_sha256="test",
        engine_config_sha256="test",
    )
    ensemble = generate_autonomous_ensemble(spec, rollout_count=2, analysis_seed=915, n_jobs=1)
    table = np.asarray([[1, 0.5, 0, 0], [0.5, 1, 0, 0], [0, 0, 1, 0.5], [0, 0, 0.5, 1]])
    expected = table[categories - 1, ensemble.choices - 1]
    assert set(expected.reshape(-1)) == {0, 0.5, 1}
    np.testing.assert_array_equal(ensemble.feedback, expected)
    assert np.issubdtype(ensemble.feedback.dtype, np.floating)


def test_fixed_cognitive_path_condition3_compares_species_accuracy():
    fit = summarize_fixed_path_fit(
        observed_feedback=np.asarray([1.0, 0.5, 0.0, 0.5]),
        selected_correct_probability=np.asarray([0.8, 0.4, 0.6, 0.9]),
        selected_observed_choice_probability=np.asarray([0.8, 0.6, 0.6, 0.9]),
        window_size=2,
        condition=3,
    )
    np.testing.assert_allclose(fit.rolling_human_accuracy[1:], [0.5, 0.0, 0.0])
    assert fit.empirical_accuracy == 0.25
    assert fit.model_expected_accuracy == pytest.approx(0.675)


def test_condition3_accuracy_figures_use_species_chance_and_mark_partial_errors(tmp_path, monkeypatch):
    rewards = np.asarray([[1, 0.5, 0, 0.5], [0.5, 1, 1, 0]])
    ensemble = AutonomousEnsemble(
        choices=np.ones((2, 4), dtype=int),
        feedback=rewards,
        expected_correct_probability=np.full((2, 4), 0.25),
        trajectory_seeds=np.asarray([1, 2]),
    )
    summary = summarize_trajectory_shapes(
        ensemble, observed_feedback=rewards[0], window_size=1,
        mastery_sustain_windows=1, final_block_trials=2, max_clusters=2,
    )
    spec = SimpleNamespace(
        condition=3, subject_id=301, label="test", window_size=1,
        arrays=SimpleNamespace(stimulus=np.zeros((4, 4))),
    )
    close = plt.close
    monkeypatch.setattr(plt, "close", lambda *_: None)
    try:
        plot_autonomous_trajectory_ensemble(
            spec, ensemble, summary, save_path=tmp_path / "autonomous.png"
        )
        figure = plt.gcf()
        for axis in figure.axes[:2]:
            baseline = [line for line in axis.lines if line.get_linestyle() == ":"]
            assert len(baseline) == 1
            np.testing.assert_array_equal(baseline[0].get_ydata(), [0.25, 0.25])

        fit = summarize_fixed_path_fit(
            observed_feedback=rewards[0], selected_correct_probability=np.full(4, 0.25),
            selected_observed_choice_probability=np.full(4, 0.25), window_size=1, condition=3,
        )
        _save_fixed_path_model_human_figure(
            trial=np.arange(1, 5), observed_feedback=rewards[0], fit=fit,
            subject_id=301, window_size=1, output_path=tmp_path / "fixed.png", condition=3,
        )
        axis = plt.gcf().axes[0]
        baseline = [line for line in axis.lines if line.get_linestyle() == ":"]
        np.testing.assert_array_equal(baseline[0].get_ydata(), [0.25, 0.25])
        np.testing.assert_array_equal(axis.collections[0].get_offsets()[:, 0], [2, 3, 4])
    finally:
        close("all")

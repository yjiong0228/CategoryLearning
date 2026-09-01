import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.Bayesian_state.evaluation import internal_cognitive_trajectories as ict
from src.Bayesian_state.evaluation.internal_cognitive_trajectories import (
    CognitivePathEnsemble,
    generate_filter_seeds,
    summarize_cognitive_paths,
)


def _toy_ensemble() -> CognitivePathEnsemble:
    n_runs, n_particles, n_trials, n_hypotheses = 4, 5, 10, 3
    n_paths = n_runs * n_particles
    seed_index = np.repeat(np.arange(n_runs), n_particles)
    particle_indices = np.tile(
        np.arange(n_particles)[:, None], (n_runs, n_trials)
    )
    executed = np.zeros((n_paths, n_trials), dtype=int)
    executed[n_paths // 2 :, 4:] = 1
    prior = np.full((n_paths, n_trials, n_hypotheses), 0.05, dtype=float)
    prior[:, :, 0] = 0.90
    prior[n_paths // 2 :, 4:, 0] = 0.10
    prior[n_paths // 2 :, 4:, 1] = 0.85
    prior /= prior.sum(axis=2, keepdims=True)
    active = prior > 0.04
    swap = np.zeros((n_paths, n_trials), dtype=float)
    swap[n_paths // 2 :, 4] = 1.0
    correct = np.where(executed == 0, 0.8, 0.6)
    paths = {
        "correct_probability": correct,
        "observed_choice_probability": correct.copy(),
        "strategy_exploit": 1.0 - 0.2 * np.ones_like(correct),
        "strategy_local_explore": 0.15 * np.ones_like(correct),
        "strategy_global_explore": 0.05 * np.ones_like(correct),
        "swap_event": swap,
        "replacement_fraction": swap.copy(),
        "transition_rate": 0.1 * np.ones_like(correct),
        "search_range": 0.2 * np.ones_like(correct),
        "failure_pressure": 0.3 * np.ones_like(correct),
        "mastery_evidence": 0.7 * np.ones_like(correct),
        "hypothesis_prior": prior,
        "hypothesis_posterior": prior.copy(),
        "active_hypothesis_mask": active,
        "executed_hypothesis": executed,
        "execution_switch_event": swap.copy(),
        "execution_dwell_trials": np.ones_like(correct),
        "executed_beta": 5.0 * np.ones_like(correct),
    }
    spec = SimpleNamespace(arrays=SimpleNamespace(choices=np.ones(n_trials)))
    return CognitivePathEnsemble(
        spec=spec,
        filter_seeds=np.arange(n_runs, dtype=np.uint32),
        particle_count=n_particles,
        weights=np.full(n_paths, 1.0 / n_paths),
        seed_index=seed_index,
        terminal_particle=np.tile(np.arange(n_particles), n_runs),
        particle_indices=particle_indices,
        paths=paths,
        marginal_choice_probability=np.full((n_trials, 2), 0.5),
        online_hypothesis_prior=np.mean(prior, axis=0),
        online_active_probability=np.mean(active, axis=0),
        online_executed_probability=np.mean(
            np.eye(n_hypotheses)[executed], axis=0
        ),
        online_swap_probability=np.mean(swap, axis=0),
        pre_choice_ess=np.full((n_runs, n_trials), n_particles),
        post_choice_ess=np.full((n_runs, n_trials), n_particles),
        resampled=np.zeros((n_runs, n_trials), dtype=bool),
    )


def _write_toy_cognitive_artifacts(
    input_dir,
    *,
    sample_source_index: np.ndarray | None = None,
    source_seed_index: np.ndarray | None = None,
    trial: np.ndarray | None = None,
    manifest_trial_count: int = 4,
) -> None:
    input_dir.mkdir()
    trial_values = (
        np.asarray([1, 2, 3, 4], dtype=int)
        if trial is None
        else np.asarray(trial)
    )
    pd.DataFrame(
        {
            "trial": trial_values,
            "observed_feedback": [1.0, 0.0, 1.0, 1.0],
        }
    ).to_csv(input_dir / "internal_cognitive_trial_summary.csv", index=False)
    np.savez_compressed(
        input_dir / "internal_cognitive_path_samples.npz",
        path_observed_choice_probability=np.asarray(
            [[0.9, 0.9, 0.1, 0.1], [0.6, 0.6, 0.6, 0.6]]
        ),
        path_correct_probability=np.asarray(
            [[0.9, 0.9, 0.1, 0.1], [0.7, 0.4, 0.6, 0.8]]
        ),
        sampled_source_seed_index=(
            np.asarray([4, 9])
            if source_seed_index is None
            else np.asarray(source_seed_index)
        ),
        sample_source_index=(
            np.asarray([101, 205])
            if sample_source_index is None
            else np.asarray(sample_source_index)
        ),
        sampled_source_terminal_particle=np.asarray([12, 27]),
        cluster_labels=np.asarray([0, 1]),
    )
    (input_dir / "analysis_manifest.json").write_text(
        json.dumps(
            {
                "subject_id": 101,
                    "trial_count": int(manifest_trial_count),
                    "raw_terminal_path_count": 300,
                    "equal_weight_draw_count": 2,
            }
        ),
        encoding="utf-8",
    )


def test_filter_seeds_are_deterministic_and_unique():
    first = generate_filter_seeds(
        analysis_seed=20260831, subject_id=101, seed_count=8
    )
    second = generate_filter_seeds(
        analysis_seed=20260831, subject_id=101, seed_count=8
    )
    np.testing.assert_array_equal(first, second)
    assert np.unique(first).size == 8


def test_cognitive_path_summary_keeps_complete_equal_weight_draws():
    ensemble = _toy_ensemble()
    summary = summarize_cognitive_paths(
        ensemble,
        draw_count=20,
        analysis_seed=20260831,
    )
    assert summary.sample_source_index.shape == (20,)
    assert summary.sampled_paths["hypothesis_prior"].shape == (20, 10, 3)
    assert summary.sampled_paths["executed_hypothesis"].shape == (20, 10)
    assert 1 <= summary.cluster_count <= 4
    assert np.isclose(np.sum(summary.cluster_shares), 1.0)
    assert summary.ancestor_unique_count.shape == (10,)
    assert summary.ancestor_effective_count.shape == (10,)
    assert np.all(summary.ancestor_effective_count > 0.0)


def test_best_complete_path_is_selected_once_by_full_sequence_likelihood():
    observed_choice_probability = np.asarray(
        [
            [0.90, 0.90, 0.10],
            [0.60, 0.60, 0.60],
        ],
        dtype=float,
    )

    selected = ict.select_best_complete_path(observed_choice_probability)

    assert selected.path_index == 1
    np.testing.assert_allclose(
        selected.sequence_nll,
        -3.0 * np.log(0.60),
    )


@pytest.mark.parametrize("invalid_value", [np.nan, -0.01, 1.01])
def test_best_complete_path_rejects_invalid_choice_probabilities(invalid_value):
    probability = np.asarray([[0.8, 0.7], [0.6, 0.5]], dtype=float)
    probability[0, 1] = invalid_value

    with pytest.raises(ValueError, match=r"finite probabilities in \[0, 1\]"):
        ict.select_best_complete_path(probability)


def test_fixed_path_fit_compares_one_model_curve_with_human_accuracy():
    fit = ict.summarize_fixed_path_fit(
        observed_feedback=np.asarray([1.0, 0.0, 1.0, 1.0]),
        selected_correct_probability=np.asarray([0.8, 0.4, 0.6, 0.9]),
        selected_observed_choice_probability=np.asarray([0.8, 0.6, 0.6, 0.9]),
        window_size=2,
    )

    np.testing.assert_allclose(
        fit.rolling_human_accuracy[1:],
        [0.5, 0.5, 1.0],
    )
    np.testing.assert_allclose(
        fit.rolling_model_correct_probability[1:],
        [0.6, 0.5, 0.75],
    )
    np.testing.assert_allclose(fit.empirical_accuracy, 0.75)
    np.testing.assert_allclose(fit.model_expected_accuracy, 0.675)
    np.testing.assert_allclose(
        fit.rolling_rmse,
        np.sqrt((0.1**2 + 0.0**2 + 0.25**2) / 3.0),
    )


def test_fixed_path_fit_rejects_nonbinary_human_feedback():
    with pytest.raises(ValueError, match="binary"):
        ict.summarize_fixed_path_fit(
            observed_feedback=np.asarray([1.0, 0.5, 0.0]),
            selected_correct_probability=np.asarray([0.8, 0.6, 0.4]),
            selected_observed_choice_probability=np.asarray([0.8, 0.6, 0.6]),
            window_size=2,
        )


def test_fixed_path_fit_rejects_invalid_model_probabilities():
    with pytest.raises(ValueError, match=r"finite probabilities in \[0, 1\]"):
        ict.summarize_fixed_path_fit(
            observed_feedback=np.asarray([1.0, 0.0, 1.0]),
            selected_correct_probability=np.asarray([0.8, np.nan, 0.6]),
            selected_observed_choice_probability=np.asarray([0.8, 0.6, 0.6]),
            window_size=2,
        )


def test_model_human_artifact_uses_one_selected_path_for_every_trial(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_toy_cognitive_artifacts(input_dir)

    outputs = ict.render_best_complete_path_model_human_from_artifacts(
        input_dir=input_dir,
        output_dir=output_dir,
        window_size=2,
    )

    assert outputs["figure"].is_file()
    source = pd.read_csv(outputs["trial_source"])
    assert source["selected_path_index"].nunique() == 1
    assert int(source["selected_path_index"].iloc[0]) == 1
    np.testing.assert_allclose(
        source["model_correct_probability"],
        [0.7, 0.4, 0.6, 0.8],
    )
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["selection"]["candidate_complete_path_count"] == 2
    assert manifest["selection"]["selected_path_index"] == 1
    assert manifest["selection"]["selected_sample_source_index"] == 205
    assert manifest["trialwise_particle_reweighting"] is False
    assert manifest["path_switching_after_selection"] is False

    summary = pd.read_csv(outputs["summary"]).iloc[0]
    assert int(summary["selected_path_index"]) == 1
    assert int(summary["selected_sample_source_index"]) == 205
    np.testing.assert_allclose(summary["sequence_nll"], -4.0 * np.log(0.6))
    np.testing.assert_allclose(summary["mean_trial_nll"], -np.log(0.6))
    np.testing.assert_allclose(summary["empirical_accuracy"], 0.75)
    np.testing.assert_allclose(summary["model_expected_accuracy"], 0.625)
    np.testing.assert_allclose(
        summary["rolling_rmse"],
        np.sqrt((0.05**2 + 0.0**2 + 0.30**2) / 3.0),
    )
    assert np.isfinite(float(summary["rolling_correlation"]))


def test_model_human_artifact_rejects_an_existing_empty_output_directory(
    tmp_path,
):
    output_dir = tmp_path / "already_exists"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="new output directory"):
        ict.render_best_complete_path_model_human_from_artifacts(
            input_dir=tmp_path / "unused",
            output_dir=output_dir,
            window_size=2,
        )


def test_model_human_artifact_rejects_misaligned_path_provenance(tmp_path):
    input_dir = tmp_path / "input"
    _write_toy_cognitive_artifacts(
        input_dir,
        source_seed_index=np.asarray([4]),
    )

    with pytest.raises(ValueError, match="candidate path count"):
        ict.render_best_complete_path_model_human_from_artifacts(
            input_dir=input_dir,
            output_dir=tmp_path / "output",
            window_size=2,
        )


@pytest.mark.parametrize("invalid_index", [-1, 300])
def test_model_human_artifact_rejects_invalid_raw_path_index(
    tmp_path, invalid_index
):
    input_dir = tmp_path / "input"
    _write_toy_cognitive_artifacts(
        input_dir,
        sample_source_index=np.asarray([101, invalid_index]),
    )

    with pytest.raises(ValueError, match="raw terminal path range"):
        ict.render_best_complete_path_model_human_from_artifacts(
            input_dir=input_dir,
            output_dir=tmp_path / "output",
            window_size=2,
        )


def test_model_human_artifact_requires_contiguous_trial_order(tmp_path):
    input_dir = tmp_path / "input"
    _write_toy_cognitive_artifacts(
        input_dir,
        trial=np.asarray([1, 3, 2, 4]),
    )

    with pytest.raises(ValueError, match="contiguous"):
        ict.render_best_complete_path_model_human_from_artifacts(
            input_dir=input_dir,
            output_dir=tmp_path / "output",
            window_size=2,
        )


def test_model_human_artifact_rejects_noninteger_trial_values(tmp_path):
    input_dir = tmp_path / "input"
    _write_toy_cognitive_artifacts(
        input_dir,
        trial=np.asarray([1.0, 2.5, 3.0, 4.0]),
    )

    with pytest.raises(ValueError, match="finite integer"):
        ict.render_best_complete_path_model_human_from_artifacts(
            input_dir=input_dir,
            output_dir=tmp_path / "output",
            window_size=2,
        )


def test_model_human_artifact_checks_manifest_trial_count(tmp_path):
    input_dir = tmp_path / "input"
    _write_toy_cognitive_artifacts(input_dir, manifest_trial_count=5)

    with pytest.raises(ValueError, match="manifest trial_count"):
        ict.render_best_complete_path_model_human_from_artifacts(
            input_dir=input_dir,
            output_dir=tmp_path / "output",
            window_size=2,
        )

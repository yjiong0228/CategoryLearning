"""Task rewards and reported accuracy have different meanings in condition 3."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.Bayesian_state.simulation.autonomous import run_autonomous_category_learning


def _condition3_config():
    config_path = Path(__file__).resolve().parents[2] / "configs/exp123/model_struct/pmh_model_cond3_0826.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["modules"]["perception_mod"]["kwargs"] = {"mean": [0.0] * 4, "std": [0.0] * 4}
    return config


def test_condition3_environment_scores_all_response_category_pairs():
    from src.Bayesian_state.metrics.task import category_learning_feedback

    expected = np.asarray(
        [[1, 0.5, 0, 0], [0.5, 1, 0, 0], [0, 0, 1, 0.5], [0, 0, 0.5, 1]]
    )
    actual = [
        [category_learning_feedback(choice, category, condition=3) for choice in range(1, 5)]
        for category in range(1, 5)
    ]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("condition, n_categories", [(1, 2), (2, 4)])
def test_binary_environment_retains_exact_match_feedback(condition, n_categories):
    from src.Bayesian_state.metrics.task import category_learning_feedback

    actual = [
        [category_learning_feedback(choice, category, condition=condition) for choice in range(1, n_categories + 1)]
        for category in range(1, n_categories + 1)
    ]
    np.testing.assert_array_equal(actual, np.eye(n_categories))


def test_partial_rewards_are_distinct_from_species_and_family_accuracy():
    from src.Bayesian_state.metrics.task import category_learning_metrics

    actual = category_learning_metrics(
        choices=[1, 2, 4, 3],
        categories=[1, 1, 1, 4],
        feedback=[1.0, 0.5, 0.0, 0.5],
        n_categories=4,
    )
    assert actual == {"species_accuracy": 0.25, "family_accuracy": 0.75, "mean_reward": 0.5}


def test_condition2_family_accuracy_cannot_be_reconstructed_from_binary_rewards():
    from src.Bayesian_state.metrics.task import category_learning_metrics

    actual = category_learning_metrics(
        choices=[1, 2, 4, 3],
        categories=[1, 1, 1, 4],
        feedback=[1.0, 0.0, 0.0, 0.0],
        n_categories=4,
    )
    assert actual == {"species_accuracy": 0.25, "family_accuracy": 0.75, "mean_reward": 0.25}


def test_task_metrics_reject_misaligned_arrays():
    from src.Bayesian_state.metrics.task import category_learning_metrics

    with pytest.raises(ValueError, match="align"):
        category_learning_metrics(choices=[1, 2], categories=[1], feedback=[1, 0], n_categories=4)


def test_real_autonomous_path_preserves_graded_rewards_and_seeded_choices():
    stimulus = np.tile([0.2, 0.8, 0.3, 0.7], (12, 1))
    categories = np.tile([1, 2, 3, 4], 3)
    kwargs = dict(
        engine_config=_condition3_config(),
        subject_id=301,
        condition=3,
        stimulus=stimulus,
        categories=categories,
        trajectory_seed=915,
        output_noise_config={"enabled": True, "base_lapse": 1.0, "lapse_target": "uniform"},
    )
    first = run_autonomous_category_learning(**kwargs)
    second = run_autonomous_category_learning(**kwargs)
    table = np.asarray([[1, 0.5, 0, 0], [0.5, 1, 0, 0], [0, 0, 1, 0.5], [0, 0, 0.5, 1]])
    expected = table[categories - 1, first.trajectory.choices - 1]
    assert set(expected) == {0, 0.5, 1}
    np.testing.assert_array_equal(first.trajectory.feedback, expected)
    np.testing.assert_array_equal(first.trajectory.choices, second.trajectory.choices)
    np.testing.assert_array_equal(first.trajectory.feedback, second.trajectory.feedback)
    np.testing.assert_allclose(first.trajectory.posterior, second.trajectory.posterior)
    assert first.metrics == {
        "species_accuracy": np.mean(expected == 1),
        "family_accuracy": np.mean(expected > 0),
        "mean_reward": np.mean(expected),
    }
    changed_targets = dict(kwargs, categories=np.ones(12, dtype=int))
    changed = run_autonomous_category_learning(**changed_targets)
    np.testing.assert_array_equal(first.trajectory.cognitive_probabilities[0], changed.trajectory.cognitive_probabilities[0])

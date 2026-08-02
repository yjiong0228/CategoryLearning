from __future__ import annotations

import numpy as np
import pandas as pd

from src.Bayesian_state.utils.unified_newplan import (
    PerceptionSpec,
    build_partition,
    encode_partition_regions,
    feedback_compatible_categories,
    integrated_rule_probabilities,
    nr2_dynamic_readout_predictions,
    partition_prior,
    rule_predictions,
    sobol_noise,
    temporal_holdout_mask,
)


def test_feedback_compatible_categories_cover_exact_plan_events() -> None:
    np.testing.assert_array_equal(feedback_compatible_categories(1, 0, 1), [0])
    np.testing.assert_array_equal(feedback_compatible_categories(1, 0, 0), [1])
    np.testing.assert_array_equal(feedback_compatible_categories(2, 2, 0), [0, 1, 3])
    np.testing.assert_array_equal(feedback_compatible_categories(3, 0, 0.5), [1])
    np.testing.assert_array_equal(feedback_compatible_categories(3, 2, 0.5), [3])
    np.testing.assert_array_equal(feedback_compatible_categories(3, 1, 0), [2, 3])


def test_integrated_zero_noise_matches_partition_regions_and_normalizes() -> None:
    for condition, target in ((1, 0), (2, 42), (3, 42)):
        partition = build_partition(condition)
        points = partition.prototypes[target, 0].copy()
        q_values = integrated_rule_probabilities(
            points,
            np.zeros((8, 4), dtype=float),
            encode_partition_regions(partition),
        )
        np.testing.assert_allclose(q_values.sum(axis=2), 1.0, atol=1e-7)
        for category in range(partition.n_cats):
            assert q_values[category, target, category] > 0.999


def test_sobol_noise_is_reproducible_and_nested() -> None:
    spec = PerceptionSpec("uniform", np.zeros(4), np.array([0.1, 0.2, 0.3, 0.4]))
    noise_128 = sobol_noise(spec, 128, 123)
    noise_256 = sobol_noise(spec, 256, 123)
    np.testing.assert_allclose(noise_128, noise_256[:128])
    np.testing.assert_allclose(noise_128, sobol_noise(spec, 128, 123))


def test_family_prior_equalizes_total_split_family_mass() -> None:
    for condition in (1, 2):
        partition = build_partition(condition)
        prior = partition_prior(partition, "uniform_family")
        families = np.array([split.type for split in partition.splits], dtype=object)
        masses = [prior[families == family].sum() for family in sorted(set(families))]
        np.testing.assert_allclose(masses, np.repeat(masses[0], len(masses)))
        assert np.isclose(prior.sum(), 1.0)


def test_rule_prediction_uses_feedback_only_after_current_choice() -> None:
    partition = build_partition(1)
    n_trials = 3
    q_values = np.full((n_trials, partition.length, 2), 0.5, dtype=float)
    q_values[:, 0, :] = np.array([0.95, 0.05])
    choices = np.array([0, 0, 0])
    feedback_a = np.array([1.0, 1.0, 1.0])
    feedback_b = np.array([0.0, 1.0, 1.0])
    a = rule_predictions(q_values, choices, feedback_a, 1)
    b = rule_predictions(q_values, choices, feedback_b, 1)
    np.testing.assert_allclose(a.probabilities[0], b.probabilities[0])
    assert not np.allclose(a.probabilities[1], b.probabilities[1])
    np.testing.assert_allclose(a.probabilities.sum(axis=1), 1.0)
    np.testing.assert_allclose(b.probabilities.sum(axis=1), 1.0)


def test_r0_is_exact_lambda_one_kappa_one_boundary() -> None:
    partition = build_partition(1)
    rng = np.random.default_rng(10)
    raw = rng.random((5, partition.length, 2))
    q_values = raw / raw.sum(axis=2, keepdims=True)
    choices = np.array([0, 1, 1, 0, 1])
    feedback = np.array([1, 0, 1, 0, 1], dtype=float)
    first = rule_predictions(q_values, choices, feedback, 1)
    second = rule_predictions(
        q_values, choices, feedback, 1, retention=1.0, sensitivity=1.0
    )
    np.testing.assert_allclose(first.probabilities, second.probabilities)


def test_dynamic_nr2_predicts_before_using_current_feedback_target() -> None:
    stimuli = np.asarray([[0.2, 0.4], [0.8, 0.6], [0.1, 0.9]], dtype=float)
    targets_a = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    targets_b = targets_a.copy()
    targets_b[0] = [0.0, 1.0]
    practice = np.linspace(0.0, 1.0, len(stimuli))
    first = nr2_dynamic_readout_predictions(
        stimuli, targets_a, 0.2, np.log(0.5), 1.0, practice
    )
    second = nr2_dynamic_readout_predictions(
        stimuli, targets_b, 0.2, np.log(0.5), 1.0, practice
    )
    np.testing.assert_allclose(first[0], second[0])
    assert not np.allclose(first[1], second[1])
    np.testing.assert_allclose(first.sum(axis=1), 1.0)


def test_temporal_holdout_uses_last_block_or_final_quarter() -> None:
    frame = pd.DataFrame(
        {
            "iSession": [1] * 4 + [2] * 3,
            "iBlock": [1] * 4 + [1] * 3,
            "iTrial": np.arange(7),
        }
    )
    mask, metadata = temporal_holdout_mask(frame)
    np.testing.assert_array_equal(mask, [False] * 4 + [True] * 3)
    assert metadata["method"] == "last_complete_block"

    one_block = frame.iloc[:4].copy()
    mask, metadata = temporal_holdout_mask(one_block)
    np.testing.assert_array_equal(mask, [False, False, False, True])
    assert metadata["method"] == "last_25_percent"

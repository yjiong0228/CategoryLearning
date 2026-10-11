"""Report-distribution comparisons retain time windows and observed-only codes."""
from __future__ import annotations

import numpy as np
import pytest

from src.Bayesian_state.metrics.oral_distribution_alignment import (
    distribution_overlap,
    js_similarity,
    rolling_report_distributions,
)


def test_identical_disjoint_and_partly_shared_distributions():
    predicted = np.array([[.3, .7, 0.], [1., 0., 0.], [.6, .4, 0.]])
    observed = np.array([[.3, .7, 0.], [0., 0., 1.], [.2, .3, .5]])
    np.testing.assert_allclose(distribution_overlap(predicted, observed), [1., 0., .5])
    np.testing.assert_allclose(
        distribution_overlap(predicted, observed),
        1. - .5 * np.abs(predicted - observed).sum(axis=1),
    )
    similarities = js_similarity(predicted, observed)
    np.testing.assert_allclose(similarities[:2], [1., 0.])
    # Analytic KL terms to midpoint [.4, .35, .25], with 0*log(0) omitted.
    divergence = .5 * (
        .6 * np.log2(.6 / .4) + .4 * np.log2(.4 / .35)
        + .2 * np.log2(.2 / .4) + .3 * np.log2(.3 / .35) + .5 * np.log2(.5 / .25)
    )
    assert similarities[2] == pytest.approx(1. - divergence)
    assert isinstance(distribution_overlap(predicted[0], observed[0]), float)
    assert isinstance(js_similarity(predicted[0], observed[0]), float)


def test_js_is_zero_safe_including_smallest_positive_mass():
    tiny = np.nextafter(0., 1.)
    predicted = np.array([[1., 0., 0.], [1., tiny, 0.]])
    observed = np.array([[0., 1., 0.], [1., 0., tiny]])
    with np.errstate(all="raise"):
        np.testing.assert_allclose(js_similarity(predicted, observed), [0., 1.])
    np.testing.assert_allclose(js_similarity(predicted, observed), js_similarity(observed, predicted))


def test_rolling_comparison_uses_same_scored_trials_and_actual_time_window():
    predicted = np.array([
        [1., 0.], [.8, .2], [0., 1.], [.2, .8], [.9, .1], [.4, .6],
    ])
    # Trial 0 is encoded but deliberately unscored; 2 and 4 are unencoded.
    indices = np.array([0, 0, -1, 1, -1, 1])
    mask = np.array([False, True, False, True, False, True])
    before = predicted.copy()
    result = rolling_report_distributions(predicted, indices, mask, window=3, min_reports=2)
    np.testing.assert_array_equal(result["n_reports"], [0, 1, 1, 2, 1, 2])
    np.testing.assert_array_equal(result["valid"], [False, False, False, True, False, True])
    np.testing.assert_allclose(result["model"][[3, 5]], [[.5, .5], [.3, .7]])
    np.testing.assert_allclose(result["observed"][[3, 5]], [[.5, .5], [0., 1.]])
    np.testing.assert_allclose(result["overlap"][[3, 5]], [1., .7])
    for key in ("model", "observed", "overlap", "js_similarity"):
        assert np.isnan(result[key][~result["valid"]]).all()
    np.testing.assert_array_equal(predicted, before)


def test_unencoded_current_trial_can_have_a_valid_prior_window():
    result = rolling_report_distributions(
        [[.8, .2], [.2, .8], [1., 0.]], [0, 1, -1], [True, True, False],
        window=3, min_reports=2,
    )
    assert result["valid"][2]
    np.testing.assert_allclose(result["model"][2], [.5, .5])
    np.testing.assert_allclose(result["observed"][2], [.5, .5])
    assert result["overlap"][2] == 1.


def test_observed_only_code_keeps_zero_model_column_and_empirical_denominator():
    predicted = np.array([[.75, .25, 0.]] * 4)
    result = rolling_report_distributions(
        predicted, [0, 2, 2, -1], [True, True, True, False], window=4, min_reports=3,
    )
    np.testing.assert_allclose(result["model"][3], [.75, .25, 0.])
    np.testing.assert_allclose(result["observed"][3], [1. / 3., 0., 2. / 3.])
    assert result["n_reports"][3] == 3
    assert result["overlap"][3] == pytest.approx(1. / 3.)
    assert result["js_similarity"][3] == pytest.approx(
        js_similarity([.75, .25, 0.], [1. / 3., 0., 2. / 3.])
    )


def test_vocabulary_label_permutation_preserves_distribution_and_rolling_metrics():
    predicted = np.array([[.75, .25, 0.], [.1, .9, 0.], [.3, .7, 0.], [.5, .5, 0.]])
    indices = np.array([0, 1, 2, -1])
    mask = np.array([True, True, True, False])
    order = np.array([2, 0, 1])
    inverse = np.argsort(order)
    permuted_indices = indices.copy()
    permuted_indices[mask] = inverse[indices[mask]]
    original = rolling_report_distributions(predicted, indices, mask, window=3, min_reports=2)
    permuted = rolling_report_distributions(
        predicted[:, order], permuted_indices, mask, window=3, min_reports=2,
    )
    for key in ("model", "observed"):
        np.testing.assert_allclose(original[key][:, order], permuted[key], equal_nan=True)
    for key in ("n_reports", "valid", "overlap", "js_similarity"):
        np.testing.assert_allclose(original[key], permuted[key], equal_nan=True)


def test_empty_trial_sequence_has_correct_shapes():
    result = rolling_report_distributions(np.empty((0, 2)), np.array([], dtype=int),
                                         np.array([], dtype=bool))
    for key in ("model", "observed"):
        assert result[key].shape == (0, 2)
    for key in ("n_reports", "valid", "overlap", "js_similarity"):
        assert result[key].shape == (0,)


def test_all_unencoded_reports_keep_counts_zero_and_distributions_missing():
    result = rolling_report_distributions(
        np.full((4, 2), .5), [-1] * 4, [False] * 4, window=3, min_reports=2,
    )
    np.testing.assert_array_equal(result["n_reports"], [0, 0, 0, 0])
    assert not result["valid"].any()
    for key in ("model", "observed", "overlap", "js_similarity"):
        assert np.isnan(result[key]).all()


def test_aggregate_frequency_agreement_can_equal_one_with_reversed_report_order():
    # Both reports are individually assigned zero probability, yet the two
    # whole-window distributions each allocate half their mass to each code.
    result = rolling_report_distributions(
        [[1., 0.], [0., 1.]], [1, 0], [True, True], window=2, min_reports=2,
    )
    np.testing.assert_allclose(result["model"][1], [.5, .5])
    np.testing.assert_allclose(result["observed"][1], [.5, .5])
    assert result["overlap"][1] == 1.
    assert result["js_similarity"][1] == 1.


@pytest.mark.parametrize("p,q", [
    ([.5, .5], [.5, .4]),
    ([.5, .5], [[.5, .5]]),
    ([1.1, -.1], [.5, .5]),
    ([np.nan, .5], [.5, .5]),
    ([np.inf, 0.], [1., 0.]),
    ([.5 + .1j, .5], [.5, .5]),
    ([], []),
    (1., 1.),
])
def test_invalid_distributions_are_rejected(p, q):
    for metric in (distribution_overlap, js_similarity):
        with pytest.raises(ValueError):
            metric(p, q)


@pytest.mark.parametrize("replacement", [
    {"probabilities": [[.8, .3], [.5, .5]]},
    {"probabilities": [[.8, .2], [np.nan, np.nan]]},
    {"probabilities": [[.8, .2], [1.1, -.1]]},
    {"probabilities": [.8, .2]},
    {"probabilities": np.empty((2, 0))},
    {"observed_indices": [0]},
    {"observed_indices": [0, .5]},
    {"observed_indices": [0, np.nan]},
    {"observed_indices": [0, -2]},
    {"observed_indices": [0, 2]},
    {"observed_indices": [False, True]},
    {"score_mask": [1, 0]},
    {"score_mask": [True]},
    {"score_mask": [True, True]},
    {"window": 0},
    {"window": True},
    {"window": 3.5},
    {"min_reports": 0},
    {"min_reports": True},
    {"min_reports": 3},
])
def test_invalid_rolling_input_contract_is_rejected(replacement):
    values = dict(probabilities=[[.8, .2], [.5, .5]], observed_indices=[0, -1],
                  score_mask=[True, False], window=2, min_reports=1)
    values.update(replacement)
    with pytest.raises(ValueError):
        rolling_report_distributions(**values)

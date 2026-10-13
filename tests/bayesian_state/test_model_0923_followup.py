"""Protect score aggregation, common-seed comparisons and feedback semantics."""
import numpy as np
import pytest

from src.Bayesian_state.evaluation.model_0923_followup import compare_finalists, learning_summary


def test_finalist_scores_average_probabilities_and_exclude_initialization():
    values = np.array([
        [[[.01, .99], [.2, .8]], [[.99, .01], [.8, .2]]],
        [[[.99, .01], [.45, .55]], [[.99, .01], [.45, .55]]],
    ])
    rows, summary = compare_finalists(values, np.array([1, 1]), 0)
    assert rows[0]['fresh_mean_probability_nll'] == pytest.approx(-np.log(.5))
    assert rows[1]['fresh_mean_probability_nll'] == pytest.approx(-np.log(.45))
    assert summary['selected_in_fresh_minimum']
    # Mean single-seed NLL is a different objective and would favor candidate 1.
    assert np.mean(-np.log([.2, .8])) > -np.log(.45)


def test_common_seed_jackknife_cancels_shared_noise():
    base = np.array([.2, .4, .6, .8])
    values = np.zeros((2, 4, 2, 2))
    values[0, :, :, 0] = base[:, None]
    values[1, :, :, 0] = base[:, None] / 2
    values[:, :, :, 1] = 1 - values[:, :, :, 0]
    rows, _ = compare_finalists(values, np.array([1, 1]), 0)
    assert rows[1]['fresh_gap_to_selected'] == pytest.approx(np.log(2))
    assert rows[1]['paired_seed_jackknife_mc_se'] < 1e-14
    assert rows[0]['identical_all_seed_probabilities_to_selected']


def test_half_feedback_is_not_exact_correct_and_initialization_is_excluded():
    observed = np.array([1, .5, 0, 1])
    simulated = np.array([[1, .5, 0, 1], [0, 1, .5, 0]])
    expected = np.full_like(simulated, .4)
    rows, summary = learning_summary(observed, simulated, expected, np.full(4, .6), 2, 2, .8)
    assert summary['overall_observed_accuracy'] == pytest.approx(1 / 3)
    assert summary['overall_autonomous_accuracy_median'] == pytest.approx(1 / 3)
    assert summary['late_scored_trials'] == 2
    assert summary['late_autonomous_expected_median'] == pytest.approx(.4)
    assert sum(row['scored_trials'] for row in rows) == 3
    assert rows[0]['first_valid_trial'] == 2
    assert rows[-1]['last_valid_trial'] == 4


def test_invalid_probability_inputs_are_rejected():
    values = np.full((2, 2, 3, 2), .6)
    with pytest.raises(ValueError, match='sum to one'):
        compare_finalists(values, np.ones(3, dtype=int), 0)

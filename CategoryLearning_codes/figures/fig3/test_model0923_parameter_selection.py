"""Scientific checks for the cumulative likelihood explanation."""
import numpy as np
import pytest

from CategoryLearning_codes.figures.fig3.model0923_parameter_selection import cumulative_advantage
from src.Bayesian_state.evaluation.model_0923_h_validation import nll


def test_score_accumulation_pools_probabilities_before_log():
    reference = np.array([[.1, .9, .5], [.9, .7, .7]])
    alternative = np.array([[.4, .6, .4], [.4, .6, .4]])
    expected = np.log(np.array([.5, .8, .6]) / np.array([.4, .6, .4])).cumsum()
    actual = cumulative_advantage(reference, alternative)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(actual[-1], 3*(nll(alternative)-nll(reference)))
    # Averaging filter log probabilities would reverse the first-trial result.
    assert actual[0] > 0
    assert np.mean(np.log(reference[:, 0]/alternative[:, 0])) < 0


def test_same_predictions_have_zero_advantage_including_zero_probability():
    predictions = np.array([[0., .5], [0., 1.]])
    np.testing.assert_array_equal(cumulative_advantage(predictions, predictions), [0., 0.])


@pytest.mark.parametrize('invalid', [np.array([[np.nan]]), np.array([[-.1]]),
                                    np.array([[1.1]]), np.ones((2, 1)), np.array([.5])])
def test_invalid_or_unaligned_probabilities_fail(invalid):
    with pytest.raises(ValueError):
        cumulative_advantage(np.array([[.5]]), invalid)

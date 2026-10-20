import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis.model_0923_cue_model_comparison import pair_values, path_ratio


def test_recurrence_conditions_on_source_without_selecting_target():
    probabilities = np.full((6, 4), .25)
    choices = np.array([2, 2, 1, 2, 3, 1])
    result = pair_values(probabilities, choices, np.array([0, 2, 4]), np.array([1, 3, 5]), np.array([2, 2, 3]))
    np.testing.assert_array_equal(result['source_match'], [True, False, True])
    np.testing.assert_array_equal(result['target_match'], [True, True, False])
    np.testing.assert_array_equal(result['joint_match'], [True, False, False])
    np.testing.assert_array_equal(result['matched_target_expected'], [.25, 0., .25])
    with pytest.raises(ValueError):
        pair_values(probabilities, choices, np.array([2]), np.array([2]), np.array([2]))


def test_ratio_pools_events_and_preserves_zero_opportunity_paths():
    value, error = path_ratio(np.array([0, 1, 2]), np.array([0, 1, 4]))
    assert value == .6
    deleted = np.array([.6, .5, 1.])
    assert error == pytest.approx(np.sqrt(2/3*np.sum((deleted-deleted.mean())**2)))
    value, error = path_ratio(np.array([0, 0]), np.array([0, 0]))
    assert np.isnan(value) and np.isnan(error)
    value, error = path_ratio(np.array([0, 1]), np.array([0, 1]))
    assert value == 1. and np.isnan(error)
    with pytest.raises(ValueError):
        path_ratio(np.array([2]), np.array([1]))

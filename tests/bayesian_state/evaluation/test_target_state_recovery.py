"""Check membership and contribution semantics with explicit particle states."""
import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis.model_0923_target_state_recovery import TargetCollector


def snapshot(weights):
    return dict(weights=np.array(weights), active=np.array([[1, 0], [1, 1]]),
        choice_weights=np.array([[1., 0.], [.2, .8]]), beta=np.array([[4., 99.], [5., 7.]]))


def test_target_values_keep_membership_and_raw_contribution_separate():
    collector = TargetCollector(1, 1)
    first, second = snapshot([.75, .25]), snapshot([.4, .6])
    collector(0, 'predictive', first)
    collector(0, 'filtered', second)
    first['weights'][:] = 0
    arrays = collector.arrays()
    np.testing.assert_array_equal(arrays['predictive_weights'], [[.75, .25]])
    np.testing.assert_array_equal(arrays['filtered_present'], [[False, True]])
    np.testing.assert_array_equal(arrays['filtered_contribution'], [[0., .8]])
    np.testing.assert_array_equal(arrays['filtered_active_beta'], [[0., 7.]])
    assert (arrays['filtered_weights'] * arrays['filtered_present']).sum() == .6
    assert (arrays['filtered_weights'] * arrays['filtered_contribution']).sum() == .48


def test_inactive_rule_cannot_contribute_to_choice():
    value = snapshot([.5, .5])
    value['choice_weights'][0] = [.5, .5]
    with pytest.raises(AssertionError):
        TargetCollector(1, 1)(0, 'predictive', value)


def test_missing_phase_or_reordered_trials_rejected():
    collector = TargetCollector(1, 2)
    with pytest.raises(ValueError, match='order'):
        collector(1, 'predictive', snapshot([.5, .5]))
    collector(0, 'predictive', snapshot([.5, .5]))
    with pytest.raises(ValueError, match='Incomplete'):
        collector.arrays()


@pytest.mark.parametrize('weights', [[0., 0.], [-.1, 1.1], [float('nan'), 1.]])
def test_invalid_weights_rejected(weights):
    with pytest.raises((ValueError, AssertionError)):
        TargetCollector(1, 1)(0, 'predictive', snapshot(weights))

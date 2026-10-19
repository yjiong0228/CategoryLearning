"""Analytical weighted distributions separate presence and conditional precision."""
import numpy as np

from src.Bayesian_state.evaluation.model_0923_target_state_recovery import summaries, weighted_summary


def test_quantiles_ignore_zero_mass_values_and_preserve_empirical_atoms():
    values = np.array([[[0., 2., 10.], [4., 8., 9.]]])
    weights = np.array([[[0., .9, .1], [0., 0., 0.]]])
    result = weighted_summary(values, weights)
    np.testing.assert_allclose(result['mean'][0], 2.8)
    assert result['q05'][0] == 2 and result['q50'][0] == 2 and result['q95'][0] == 10
    assert all(np.isnan(value[1]) for value in result.values())


def test_conditional_precision_does_not_average_absence_as_zero_precision():
    arrays = dict(filtered_weights=np.array([[[.75, .25], [1., 0.]]]),
        filtered_present=np.array([[[0, 1], [0, 1]]]),
        filtered_contribution=np.array([[[0., .8], [0., .7]]]),
        filtered_active_beta=np.array([[[0., 7.], [0., 9.]]]))
    truth = dict(present=np.array([1, 1]), contribution=np.array([.2, .3]), beta=np.array([5., 8.]))
    curves, metrics = summaries(arrays, truth, 'filtered')
    assert curves['conditional_beta_mean'][0] == 7
    assert np.isnan(curves['conditional_beta_mean'][1])
    assert metrics['conditional_beta_rmse'] == 2
    assert metrics['conditional_beta_scored_trials'] == 1
    assert metrics['conditional_beta_undefined_when_truth_present'] == 1
    np.testing.assert_allclose(metrics['membership_brier'], (.75**2 + 1)/2)

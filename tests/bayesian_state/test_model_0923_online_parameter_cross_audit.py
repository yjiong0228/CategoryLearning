"""Analytic checks for an implementation-independent cross-result audit."""
import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis.model_0923_online_parameter_cross_audit import (
    distribution, loss, mc_error,
)


def test_marginal_probability_is_averaged_before_logarithm():
    np.testing.assert_allclose(loss([[.1, .25], [.4, .25]]), np.log(4))
    assert np.isfinite(loss([[0.], [0.]])).all()


def test_point_mass_at_observation_has_zero_loss_and_full_coverage():
    scores = distribution([[0., .5, 1.], [0., .5, 1.]], [0., .5, 1.])
    assert scores == dict(energy=0., crps=0., interval_score=0.,
                         interval_coverage=1., interval_width=0., realized_mean_mae=0.)


def test_two_point_distribution_has_known_scores():
    scores = distribution([[0.], [1.]], [0.])
    assert scores == pytest.approx(dict(energy=.25, crps=.25, interval_score=1.8,
                                       interval_coverage=0., interval_width=.8, realized_mean_mae=.5))


def test_jackknife_scale_uses_complete_deletion_estimates():
    assert mc_error([1., 2., 3.]) == pytest.approx(np.sqrt(4 / 3))
    assert mc_error([.2, .2, .2]) == pytest.approx(0.)


@pytest.mark.parametrize('p', [[], [[]], [.2, .4], [[np.nan]], [[np.inf]], [[-.01]], [[1.01]]])
def test_invalid_probability_arrays_are_rejected(p):
    with pytest.raises(ValueError):
        loss(p)


@pytest.mark.parametrize('x,y', [([], []), ([[]], []), ([0., 1.], [0., 1.]),
                                ([[0., 1.]], [0.]), ([[np.nan]], [0.]), ([[1.]], [np.inf])])
def test_invalid_distribution_arrays_are_rejected(x, y):
    with pytest.raises(ValueError):
        distribution(x, y)


@pytest.mark.parametrize('x', [[], [1.], [[1., 2.]], [0., np.nan]])
def test_invalid_deletion_estimates_are_rejected(x):
    with pytest.raises(ValueError):
        mc_error(x)

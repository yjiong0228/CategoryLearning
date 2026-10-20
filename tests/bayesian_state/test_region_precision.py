import numpy as np
import pytest
from scipy.special import softmax

from src.Bayesian_state.hypothesis_space.observation_model.region_precision import (
    region_response_feedback_derivatives,
)


@pytest.mark.parametrize('feedback', [0., 1.])
@pytest.mark.parametrize('beta', [.1, 3., 25.])
def test_precision_derivative_matches_independent_finite_difference(feedback, beta):
    distances = np.array([[.1, .7, .2, .5], [.8, .3, .6, .1]])
    logits = np.random.default_rng(29).normal(size=(2, 4, 4))
    association = softmax(logits, axis=-1)
    q = softmax(-beta * distances, axis=-1)
    log_event, _, gradient = region_response_feedback_derivatives(q, logits, distances, 2, feedback)

    def objective(precision):
        p = np.einsum('hg,hgc->hc', softmax(-precision * distances, axis=-1), association)
        return np.log(p[:, 1] if feedback else p[:, [0, 2, 3]].sum(axis=-1))

    step = 1e-5
    np.testing.assert_allclose(log_event, objective(beta), atol=1e-12, rtol=0)
    np.testing.assert_allclose(gradient, (objective(beta+step)-objective(beta-step))/(2*step),
                               atol=3e-10, rtol=0)


@pytest.mark.parametrize('feedback', [0., 1.])
def test_uniform_associations_provide_no_evidence_for_precision(feedback):
    d = np.array([[.1, .7, .2, .5], [.8, .3, .6, .1]])
    log_event, association_gradient, precision = region_response_feedback_derivatives(
        softmax(-12*d, axis=-1), np.zeros((2, 4, 4)), d, 3, feedback)
    np.testing.assert_allclose(log_event, np.log(.25 if feedback else .75), atol=1e-14, rtol=0)
    np.testing.assert_allclose(precision, 0, atol=1e-14, rtol=0)
    assert np.any(association_gradient != 0)


def test_identity_limit_matches_fixed_label_gradient_and_region_relabeling():
    d = np.array([.1, .7, .2, .5]); q = softmax(-5*d)
    logits = np.eye(4)*1000
    for feedback in (0., 1.):
        a = region_response_feedback_derivatives(q, logits, d, 2, feedback)
        allowed = np.array([False, True, False, False])
        if feedback == 0.: allowed = ~allowed
        conditional = q*allowed; conditional /= conditional.sum()
        np.testing.assert_allclose(a[2], q @ d - conditional @ d, atol=1e-12, rtol=0)
        order = np.array([2, 0, 3, 1])
        b = region_response_feedback_derivatives(q[order], logits[order], d[order], 2, feedback)
        np.testing.assert_allclose(a[0], b[0], atol=1e-12, rtol=0)
        np.testing.assert_allclose(a[2], b[2], atol=1e-12, rtol=0)
        np.testing.assert_allclose(a[1][order], b[1], atol=1e-12, rtol=0)


def test_very_unlikely_feedback_and_zero_probability_regions_remain_finite():
    logits = np.tile([1000., -1000., -1000., -1000.], (3, 1))
    e, a, b = region_response_feedback_derivatives(
        np.array([.8, 0., .2]), logits, np.array([.1, 1., .4]), 1, 0.)
    np.testing.assert_allclose(e, -2000+np.log(3), atol=1e-10, rtol=0)
    np.testing.assert_array_equal(a[1], 0.)
    np.testing.assert_allclose(b, 0, atol=1e-12, rtol=0)
    assert np.isfinite(a).all()


@pytest.mark.parametrize('distances', [[.1], [.1, -.2], [.1, np.nan]])
def test_invalid_distances_rejected(distances):
    with pytest.raises(ValueError, match='Distances'):
        region_response_feedback_derivatives(np.array([.4, .6]), np.zeros((2, 4)),
                                               np.array(distances), 1, 0.)

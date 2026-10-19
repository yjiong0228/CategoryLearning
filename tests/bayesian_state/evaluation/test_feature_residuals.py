"""Prefix-only estimation and interpretable synthetic residual signals."""
import numpy as np
import pytest
from scipy.special import expit

from src.Bayesian_state.evaluation.particle_filter.feature_residuals import fit_prefix_feature_probe


def fixture():
    rng = np.random.default_rng(74018)
    x = rng.normal(size=(600, 3))
    p = expit(.7 * x[:, 0])
    y = rng.binomial(1, expit(.7 * x[:, 0] + 1.5 * x[:, 1])).astype(float)
    return p, y, x


def test_later_outcomes_do_not_change_fit_or_predictions():
    p, y, x = fixture()
    first = fit_prefix_feature_probe(p, y, x, training_trials=400)
    y[400:] = 1 - y[400:]
    second = fit_prefix_feature_probe(p, y, x, training_trials=400)
    np.testing.assert_array_equal(first.coefficient, second.coefficient)
    np.testing.assert_array_equal(first.probability, second.probability)


def test_later_features_do_not_change_scaling_or_fitted_coefficients():
    p, y, x = fixture()
    first = fit_prefix_feature_probe(p, y, x, training_trials=400)
    x[400:] += 20.
    second = fit_prefix_feature_probe(p, y, x, training_trials=400)
    np.testing.assert_array_equal(first.feature_mean, second.feature_mean)
    np.testing.assert_array_equal(first.feature_scale, second.feature_scale)
    np.testing.assert_array_equal(first.coefficient, second.coefficient)
    np.testing.assert_array_equal(first.probability[:400], second.probability[:400])


def test_predictable_omitted_feature_improves_later_choice_probability():
    p, y, x = fixture()
    base = fit_prefix_feature_probe(p, y, x[:, :1], training_trials=400)
    extra = fit_prefix_feature_probe(p, y, x, training_trials=400)
    assert extra.coefficient[2] > .8
    nll = lambda q: -np.mean(y[400:] * np.log(q[400:]) + (1-y[400:]) * np.log1p(-q[400:]))
    assert nll(extra.probability) < nll(base.probability) - .1
    assert extra.gradient_max_abs <= 1e-4


def test_constant_feature_has_zero_coefficient_and_inputs_remain_unchanged():
    p, y, x = fixture()
    x[:, 2] = 4.
    copies = [a.copy() for a in (p, y, x)]
    fit = fit_prefix_feature_probe(p, y, x, training_trials=400)
    assert fit.coefficient[3] == 0 and fit.feature_scale[2] == 1
    for old, new in zip(copies, (p, y, x)):
        np.testing.assert_array_equal(old, new)


@pytest.mark.parametrize('case', ['probability', 'outcome', 'features', 'prefix', 'ridge'])
def test_invalid_inputs_are_rejected(case):
    p, y, x = fixture()
    cut, ridge = 400, 1.
    if case == 'probability': p[0] = 0.
    if case == 'outcome': y[0] = .5
    if case == 'features': x[0, 0] = np.nan
    if case == 'prefix': cut = 600
    if case == 'ridge': ridge = -1.
    with pytest.raises(ValueError):
        fit_prefix_feature_probe(p, y, x, training_trials=cut, ridge=ridge)

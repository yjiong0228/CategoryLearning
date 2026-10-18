"""Trial timing and matched-arm checks for the behavior-only diagnostic."""
from copy import deepcopy

import numpy as np
import pytest
import yaml
from pathlib import Path

from src.Bayesian_state.workflows.analysis.model_0923_association_diagnostic import replay, validate_settings


def test_current_response_and_future_feedback_do_not_enter_current_prediction():
    rng = np.random.default_rng(86); q = rng.dirichlet(np.ones(4), size=(8, 3))
    y = np.array([1, 2, 3, 4, 1, 2, 3, 4]); f = np.array([1., 0., 1., 0., 1., 0., 1., 0.])
    arm = dict(learn_associations=True, association_retention='stable')
    a = replay(q, y, f, gamma=.8, arm=arm, condition=2, trace=True)
    changed_y = y.copy(); changed_y[4:] = 4
    changed_f = f.copy(); changed_f[4:] = 0.
    b = replay(q, changed_y, changed_f, gamma=.8, arm=arm, condition=2, trace=True)
    np.testing.assert_array_equal(a['probabilities'][:5], b['probabilities'][:5])
    assert not np.allclose(a['probabilities'][5:], b['probabilities'][5:])
    np.testing.assert_allclose(a['probabilities'].sum(axis=1), 1.)
    np.testing.assert_array_equal(a['observed_probability'], a['probabilities'][np.arange(8), y-1])


def test_retention_modes_coincide_at_unit_gamma():
    q = np.random.default_rng(56).dirichlet(np.ones(4), size=(5, 2))
    y = np.array([1, 2, 3, 4, 1]); f = np.array([1., .5, 0., 1., .5])
    for learn in (False, True):
        outputs = [replay(q, y, f, gamma=1., condition=3,
                   arm=dict(learn_associations=learn, association_retention=mode)) for mode in ('shared', 'stable')]
        np.testing.assert_array_equal(outputs[0]['probabilities'], outputs[1]['probabilities'])


def test_fixed_binary_arms_coincide_for_every_retention():
    q = np.random.default_rng(47).dirichlet(np.ones(4), size=(5, 2))
    for gamma in (0., .3, .8, 1.):
        outputs = [replay(q, np.ones(5), np.zeros(5), gamma=gamma, condition=2,
                   arm=dict(learn_associations=False, association_retention=mode)) for mode in ('shared', 'stable')]
        np.testing.assert_array_equal(outputs[0]['probabilities'], outputs[1]['probabilities'])


def test_config_requires_matched_factorial_and_valid_grid():
    settings = yaml.safe_load(Path('configs/shared/model0923_association_diagnostic.yaml').read_text())
    validate_settings(settings)
    for key, value in [('rule_retention', [1.1]), ('beta', [1., 1.]), ('subjects', [207, 207]), ('parallel_budget', 129)]:
        bad = deepcopy(settings); bad[key] = value
        with pytest.raises(ValueError): validate_settings(bad)
    bad = deepcopy(settings); bad['arms'].pop('FIXED_STABLE')
    with pytest.raises(ValueError): validate_settings(bad)


def test_invalid_data_rejected_before_scoring():
    arm = dict(learn_associations=True, association_retention='stable')
    with pytest.raises(ValueError): replay(np.full((2, 1, 4), .25), np.array([1, 5]), np.ones(2), gamma=1., arm=arm, condition=2)
    with pytest.raises(ValueError): replay(np.full((2, 1, 4), .25), np.ones(1), np.ones(2), gamma=1., arm=arm, condition=2)

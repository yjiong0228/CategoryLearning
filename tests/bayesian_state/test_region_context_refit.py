import numpy as np
import pytest

from src.Bayesian_state.model.region_context_evidence import ContextCatalogueRegionLearner
from src.Bayesian_state.workflows.analysis.model_0923_region_context_refit import replay


def test_replay_retains_prefeedback_states_and_exact_prefix_after_later_extension():
    q = np.random.default_rng(23).dirichlet(np.ones(4), size=(12, 3))
    y = np.tile(np.arange(1, 5), 3); f = np.tile([0., 1., 1.], 4)
    short = replay(q[:8], y[:8], f[:8], gamma=.8, rate=.5)
    full = replay(q, y, f, gamma=.8, rate=.5, trace=True)
    np.testing.assert_array_equal(short['probabilities'], full['probabilities'][:8])
    model = ContextCatalogueRegionLearner(3, rule_retention=.8, association_learning_rate=.5, mode='LOCAL')
    for t in range(len(q)):
        np.testing.assert_array_equal(full['associations'][t], model.associations)
        np.testing.assert_array_equal(full['rule_weights'][t], model.rule_weights(q[t]))
        np.testing.assert_array_equal(full['probabilities'][t], model.predict(q[t]))
        model.update(q[t], int(y[t]), float(f[t]))
    altered = f.copy(); altered[-1] = 1-altered[-1]
    np.testing.assert_array_equal(replay(q, y, altered, gamma=.8, rate=.5)['probabilities'], full['probabilities'])


def test_replay_rejects_misaligned_or_invalid_observations():
    q = np.ones((3, 2, 4)) / 4
    for y, f in [([1, 2], [0, 1, 0]), ([1, 2.5, 3], [0, 1, 0]), ([1, 2, 3], [0, .5, 0])]:
        with pytest.raises(ValueError): replay(q, y, f, gamma=.8, rate=.5)

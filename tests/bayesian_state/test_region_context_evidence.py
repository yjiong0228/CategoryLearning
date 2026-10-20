import numpy as np
import pytest

from src.Bayesian_state.model.region_catalogue_learner import FixedCatalogueRegionLearner
from src.Bayesian_state.model.region_context_evidence import (
    ContextCatalogueRegionLearner, RegionContextEvidence,
)


def test_context_scores_match_explicit_discounted_similarity_history():
    rng = np.random.default_rng(923)
    q = rng.dirichlet(np.ones(4), size=(10, 3))
    evidence = rng.normal(size=(10, 3)); gamma = .8
    memory = RegionContextEvidence(3, retention=gamma)
    for t in range(10):
        v = q[t]/np.linalg.norm(q[t], axis=-1, keepdims=True)
        expected = np.zeros(3)
        for s in range(t):
            old = q[s]/np.linalg.norm(q[s], axis=-1, keepdims=True)
            expected += gamma**(t-1-s)*(v*old).sum(-1)*evidence[s]
        np.testing.assert_allclose(memory.scores(q[t]), expected, atol=1e-14, rtol=0)
        memory.update(q[t], evidence[t])
    detached = memory.evidence; detached[:] = 999
    assert not np.any(memory.evidence == 999)


def test_local_specialists_remain_distinguishable_after_opposite_context_feedback():
    memory = RegionContextEvidence(2, retention=1.)
    a = np.repeat(np.eye(4)[0][None], 2, axis=0)
    b = np.repeat(np.eye(4)[1][None], 2, axis=0)
    memory.update(a, np.array([1., -1.]))
    memory.update(b, np.array([-1., 1.]))
    np.testing.assert_array_equal(memory.scores(a), [1., -1.])
    np.testing.assert_array_equal(memory.scores(b), [-1., 1.])
    # Summing globally would give zero to both rules.
    np.testing.assert_array_equal(memory.evidence.sum(-1), [0., 0.])


@pytest.mark.parametrize('gamma', [0., .8, 1.])
def test_identical_region_activation_reduces_to_global_history(gamma):
    q = np.array([[.1, .2, .3, .4], [.6, .1, .2, .1]])
    global_model = ContextCatalogueRegionLearner(2, rule_retention=gamma, association_learning_rate=.5, mode='GLOBAL')
    local_model = ContextCatalogueRegionLearner(2, rule_retention=gamma, association_learning_rate=.5, mode='LOCAL')
    for t in range(20):
        np.testing.assert_allclose(local_model.predict(q), global_model.predict(q), atol=2e-15, rtol=0)
        np.testing.assert_allclose(local_model.rule_weights(q), global_model.rule_weights(q), atol=3e-15, rtol=0)
        for model in (global_model, local_model):model.update(q, t % 4+1, float(t % 3==0))
        np.testing.assert_array_equal(local_model.associations, global_model.associations)


def test_global_reproduces_maintained_learner_and_local_keeps_same_association_updates():
    baseline = FixedCatalogueRegionLearner(3, rule_retention=.8, association_learning_rate=.5)
    models = [ContextCatalogueRegionLearner(3, rule_retention=.8, association_learning_rate=.5, mode=m)
              for m in ('GLOBAL', 'LOCAL')]
    q = np.random.default_rng(23).dirichlet(np.ones(4), size=(25, 3))
    for t, x in enumerate(q):
        np.testing.assert_array_equal(models[0].predict(x), baseline.predict(x))
        expected_associations = baseline.associations
        for model in models:np.testing.assert_array_equal(model.associations, expected_associations)
        y, f = t % 4+1, float(t % 3==0)
        baseline.update(x, y, f)
        for model in models:model.update(x, y, f)


def test_last_feedback_cannot_change_earlier_predictions_and_zero_rate_is_neutral():
    q = np.random.default_rng(33).dirichlet(np.ones(4), size=(8, 2))
    paths = []
    for last in (0., 1.):
        model = ContextCatalogueRegionLearner(2, rule_retention=.8, association_learning_rate=.5, mode='LOCAL')
        predictions=[]
        for t, x in enumerate(q):
            predictions.append(model.predict(x));model.update(x, 1, last if t==7 else 1.)
        paths.append(predictions)
    np.testing.assert_array_equal(*paths)
    neutral = ContextCatalogueRegionLearner(2, rule_retention=.8, association_learning_rate=0., mode='LOCAL')
    for x in q:
        np.testing.assert_allclose(neutral.predict(x), .25, atol=1e-15, rtol=0)
        neutral.update(x, 2, 0.)
        np.testing.assert_array_equal(neutral.rule_weights(x), [.5, .5])

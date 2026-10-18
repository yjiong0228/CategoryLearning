"""Independent checks of the controlled fixed-catalogue association learner."""
from itertools import permutations
from types import SimpleNamespace

import numpy as np
import pytest

from src.Bayesian_state.model.association_learner import FixedCatalogueAssociationLearner
from src.Bayesian_state.model.modules.pairing_memory import HierarchicalPairingMemoryModule
from src.Bayesian_state.hypothesis_space.observation_model.pairing import PAIRING_SIBLINGS, pairing_feedback_kernel
from src.Bayesian_state.hypothesis_space.observation_model.response_association import update_associations


@pytest.mark.parametrize('hierarchical', [False, True])
@pytest.mark.parametrize('learn', [False, True])
def test_sequence_matches_independently_enumerated_joint_bayes(hierarchical, learn):
    model = FixedCatalogueAssociationLearner(2, rule_retention=1., association_retention=1.,
                                            learn_associations=learn, hierarchical=hierarchical)
    mappings = list(permutations(range(4))); nz = 3 if hierarchical else 1
    joint = np.full((2, 24, nz), 1/(2*24*nz)) if learn else np.zeros((2, 24, nz))
    if not learn: joint[:, 0] = 1/(2*nz)
    rng = np.random.default_rng(452)
    for trial in range(12):
        q = rng.dirichlet(np.ones(4), size=2); choice = trial % 4
        feedback = (0., .5, 1.)[trial % 3] if hierarchical else float(trial % 2)
        expected = np.zeros(4); event = np.zeros_like(joint)
        for h in range(2):
            for a, mapping in enumerate(mappings):
                for z in range(nz):
                    for region, response in enumerate(mapping):
                        expected[response] += joint[h, a, z]*q[h, region]
                        actual = 1. if response == choice else (
                            .5 if hierarchical and response == PAIRING_SIBLINGS[z, choice] else 0.)
                        if actual == feedback: event[h, a, z] += q[h, region]
        np.testing.assert_allclose(model.predict(q), expected, rtol=1e-12, atol=1e-15)
        joint *= event; joint /= joint.sum()
        model.update(q, choice+1, feedback)
        np.testing.assert_allclose(model.rule_weights[:, None, None]*model.association_weights,
                                   joint, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize('hierarchical', [False, True])
def test_fixed_labels_remain_fixed_at_zero_retention(hierarchical):
    model = FixedCatalogueAssociationLearner(1, rule_retention=0., association_retention=0.,
                                            learn_associations=False, hierarchical=hierarchical)
    q = np.array([[.7, .1, .05, .15]])
    for choice in range(1, 5):
        np.testing.assert_allclose(model.predict(q), q[0])
        model.update(q, choice, 0.)
        assert np.count_nonzero(model.association_weights[:, 1:]) == 0


def test_stable_association_survives_memoryless_rule_competition():
    model = FixedCatalogueAssociationLearner(1, rule_retention=0., association_retention=1.,
                                            learn_associations=True, hierarchical=False)
    mapping = [3, 0, 2, 1]
    for g in range(3): model.update(np.eye(4)[g:g+1], mapping[g]+1, 1.)
    before = model.predict(np.eye(4)[3:4])
    np.testing.assert_array_equal(before, np.eye(4)[mapping[3]])
    # Accessors return independent snapshots for downstream diagnostic code.
    rules = model.rule_weights; weights = model.association_weights
    rules[:] = 0.; weights[:] = 0.
    np.testing.assert_array_equal(model.predict(np.eye(4)[3:4]), before)


def test_no_information_feedback_preserves_geometry_mass_with_different_association_entropy():
    model = FixedCatalogueAssociationLearner(2, rule_retention=1., association_retention=.5,
                                            learn_associations=True, hierarchical=False)
    model.update(np.array([[.7, .1, .1, .1], [.25, .25, .25, .25]]), 1, 1.)
    before = model.rule_weights
    model.update(np.full((2, 4), .25), 1, 1.)
    np.testing.assert_allclose(model.rule_weights, before)


def test_bad_association_support_rejected():
    w = np.full((1, 24, 1), 1/24)
    for bad in [np.zeros(24, dtype=bool), np.ones(24), np.ones(23, dtype=bool), np.arange(24)==0]:
        with pytest.raises(ValueError):
            update_associations(w, np.ones_like(w), retention=0., association_mask=bad)


def test_impossible_feedback_does_not_partially_commit_state():
    model = FixedCatalogueAssociationLearner(1, rule_retention=1., association_retention=1.,
                                            learn_associations=False, hierarchical=False)
    before = model.association_weights
    with pytest.raises(ValueError): model.update(np.array([[1., 0., 0., 0.]]), 1, 0.)
    np.testing.assert_array_equal(model.association_weights, before)
    np.testing.assert_array_equal(model.rule_weights, [1.])


@pytest.mark.parametrize('gamma', [0., .3, .8, .99, 1.])
def test_fixed_shared_agrees_with_existing_factorized_pairing_memory(gamma):
    model = FixedCatalogueAssociationLearner(3, rule_retention=gamma, association_retention=gamma,
                                            learn_associations=False, hierarchical=True)
    engine = SimpleNamespace(prior=np.full(3, 1/3), hypotheses_mask=np.ones(3))
    existing = HierarchicalPairingMemoryModule(engine, gamma=gamma, forgetting_mode='factorized')
    rng = np.random.default_rng(236)
    for t in range(16):
        q = rng.dirichlet(np.ones(4), size=3); choice = t % 4 + 1; feedback = (0., .5, 1.)[t % 3]
        existing.prepare_feedback(pairing_feedback_kernel(q, choice, feedback)); existing.process()
        model.update(q, choice, feedback)
        np.testing.assert_allclose(model.rule_weights, engine.posterior, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(model.association_weights[:, 0], existing.conditional_pairing(), rtol=1e-12, atol=1e-15)
        engine.prior = engine.posterior.copy()

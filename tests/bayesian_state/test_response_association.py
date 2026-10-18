"""Independent enumeration checks for the experimental association factor."""
from itertools import permutations

import numpy as np
import pytest

from src.Bayesian_state.hypothesis_space.observation_model.pairing import (
    PAIRING_SIBLINGS, pairing_feedback_kernel,
)
from src.Bayesian_state.hypothesis_space.observation_model.response_association import (
    ASSOCIATIONS, associated_probabilities, association_choice_probabilities,
    association_feedback_kernel, update_associations,
)


def test_every_bijection_and_prediction_has_independent_enumeration():
    p = np.array([[.1, .2, .3, .4], [.7, .1, .05, .15]])
    mapped = associated_probabilities(p)
    assert len({tuple(x) for x in ASSOCIATIONS}) == 24
    assert not ASSOCIATIONS.flags.writeable
    for i, mapping in enumerate(permutations(range(4))):
        expected = np.empty_like(p)
        for geometry, response in enumerate(mapping):
            expected[:, response] = p[:, geometry]
        np.testing.assert_array_equal(mapped[:, i], expected)
    np.testing.assert_allclose(mapped.sum(axis=-1), 1.)


@pytest.mark.parametrize('pairings', [1, 3])
def test_uniform_associations_are_response_symmetric_without_rule_mass_change(pairings):
    p = np.array([[1., 0., 0., 0.], [.1, .2, .3, .4]])
    weights = np.full((2, 24, pairings), 1/(24*pairings))
    np.testing.assert_allclose(association_choice_probabilities(p, weights), .25)
    kernel = association_feedback_kernel(p, 2, 1., hierarchical=pairings == 3)
    posterior, evidence = update_associations(weights, kernel, retention=1.)
    np.testing.assert_allclose(evidence, .25)
    prior = np.array([.2, .8]); updated = prior * evidence; updated /= updated.sum()
    np.testing.assert_allclose(updated, prior)
    np.testing.assert_allclose(posterior.sum(axis=(1, 2)), 1.)


def test_known_identity_matches_existing_pairing_feedback_for_every_event():
    p = np.array([[.1, .2, .3, .4]])
    weights = np.zeros((1, 24, 3)); weights[:, 0, :] = [0.2, .3, .5]
    np.testing.assert_array_equal(association_choice_probabilities(p, weights), p)
    for y in range(1, 5):
        for feedback in (0., .5, 1.):
            kernel = association_feedback_kernel(p, y, feedback, hierarchical=True)
            np.testing.assert_array_equal(kernel[:, 0], pairing_feedback_kernel(p, y, feedback))
            _, evidence = update_associations(weights, kernel, retention=1.)
            expected = (pairing_feedback_kernel(p, y, feedback) * weights[:, 0]).sum(axis=1)
            np.testing.assert_allclose(evidence, expected)


def test_partial_feedback_and_all_mapped_events_partition_probability():
    p = np.array([[.05, .15, .3, .5]])
    for y in range(1, 5):
        kernels = [association_feedback_kernel(p, y, f, hierarchical=True) for f in (0., .5, 1.)]
        np.testing.assert_allclose(sum(kernels), 1.)
        for i, mapping in enumerate(permutations(range(4))):
            for pairing in range(3):
                sibling = PAIRING_SIBLINGS[pairing, y-1]
                region = mapping.index(sibling)
                assert kernels[1][0, i, pairing] == p[0, region]
        binary = [association_feedback_kernel(p, y, f) for f in (0., 1.)]
        np.testing.assert_allclose(sum(binary), 1.)


def test_update_agrees_with_joint_bayes_and_does_not_mutate_inputs():
    rng = np.random.default_rng(642); weights = rng.uniform(size=(2, 24, 3))
    weights /= weights.sum(axis=(1, 2), keepdims=True)
    p = np.array([[.1, .2, .3, .4], [.7, .1, .1, .1]])
    kernel = association_feedback_kernel(p, 2, .5, hierarchical=True)
    before = weights.copy(); saved_kernel = kernel.copy()
    posterior, evidence = update_associations(weights, kernel, retention=1.)
    prior = np.array([.3, .7]); joint = prior[:, None, None] * weights * kernel
    joint /= joint.sum()
    marginal = prior * evidence; marginal /= marginal.sum()
    np.testing.assert_allclose(posterior*marginal[:, None, None], joint)
    np.testing.assert_array_equal(weights, before); np.testing.assert_array_equal(kernel, saved_kernel)


def test_feedback_learns_one_mapping_sequentially_with_uniform_initial_prediction():
    mapping = (3, 0, 2, 1)
    identity = next(i for i, row in enumerate(ASSOCIATIONS) if tuple(row) == mapping)
    weights = np.full((1, 24, 1), 1/24); evidences = []
    for geometry in range(4):
        p = np.eye(4)[geometry:geometry+1]
        before = association_choice_probabilities(p, weights)[0]
        assert before[mapping[geometry]] == pytest.approx(1/(4-geometry))
        kernel = association_feedback_kernel(p, mapping[geometry]+1, 1.)
        weights, evidence = update_associations(weights, kernel, retention=1.)
        evidences.append(evidence[0])
        assert np.count_nonzero(weights) == (6, 2, 1, 1)[geometry]
    assert weights[0, identity, 0] == 1.
    assert np.prod(evidences) == pytest.approx(1/24)


def test_conditional_forgetting_preserves_geometry_evidence_and_zero_kernel_is_finite():
    weights = np.zeros((2, 24, 1)); weights[0, 0] = 1.; weights[1] = 1/24
    posterior, evidence = update_associations(weights, np.ones_like(weights), retention=.5)
    np.testing.assert_allclose(evidence, 1.)
    np.testing.assert_allclose(posterior.sum(axis=(1, 2)), 1.)
    reset, _ = update_associations(weights, np.ones_like(weights), retention=0.)
    np.testing.assert_allclose(reset, 1/24)
    finite, impossible = update_associations(weights, np.zeros_like(weights), retention=1.)
    np.testing.assert_array_equal(impossible, [0., 0.]); np.testing.assert_array_equal(finite, weights)


def test_tiny_complement_probability_survives():
    p = np.array([[1., 1e-30, 2e-30, 3e-30]])
    kernel = association_feedback_kernel(p, 1, 0.)
    assert kernel[0, 0, 0] == pytest.approx(6e-30, rel=1e-12, abs=0.)


@pytest.mark.parametrize('bad', [np.ones((1, 3))/3, np.array([[.5, .5, .2, -.2]]),
                                np.array([[.5, .5, .5, .5]]), np.array([[np.nan, 0., 0., 1.]])])
def test_invalid_geometry_probabilities_rejected(bad):
    with pytest.raises(ValueError): associated_probabilities(bad)


def test_bad_feedback_weights_and_retention_are_rejected():
    p = np.full((1, 4), .25); w = np.full((1, 24, 1), 1/24)
    with pytest.raises(ValueError): association_feedback_kernel(p, 1, .5)
    with pytest.raises(ValueError): association_choice_probabilities(p, w*.5)
    with pytest.raises(ValueError): update_associations(w, np.ones_like(w), retention=1.1)
    with pytest.raises(ValueError): update_associations(w, np.ones((1, 24, 3)), retention=1.)

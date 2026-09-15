"""Independent algebraic checks for the Model 0826 condition 3 extension."""

from itertools import permutations
from types import SimpleNamespace

import numpy as np
import pytest

from src.Bayesian_state.hypothesis_space.observation_model.pairing import (
    PAIRING_SIBLINGS, pairing_feedback_kernel,
)
from src.Bayesian_state.model.modules.pairing_memory import (
    HierarchicalPairingMemoryModule, transport_joint_belief,
)
from src.Bayesian_state.model.modules.hypothesis_transition.contracts import HypothesisSelection


def memory(joint, gamma=1.0):
    joint = np.asarray(joint, dtype=float)
    engine = SimpleNamespace(prior=joint.sum(axis=1), set_size=len(joint),
                             hypotheses_mask=(joint.sum(axis=1) > 0).astype(float))
    mod = HierarchicalPairingMemoryModule(engine, gamma=gamma)
    mod.joint = joint.copy()
    return mod


def test_ternary_kernel_has_disjoint_events_and_no_truth_pair_prior():
    p = np.array([[.1, .2, .3, .4], [.7, .1, .15, .05]])
    kernels = [pairing_feedback_kernel(p, 1, f) for f in (1., .5, 0.)]
    np.testing.assert_allclose(sum(kernels), 1.)
    np.testing.assert_allclose(kernels[1], p[:, [1, 2, 3]])
    np.testing.assert_allclose(kernels[2], [[.7, .6, .5], [.2, .15, .25]])
    np.testing.assert_allclose(kernels[1].mean(axis=1), (1-p[:, 0])/3)
    np.testing.assert_allclose(kernels[2].mean(axis=1), 2*(1-p[:, 0])/3)


def test_small_cross_family_mass_is_not_lost_to_subtraction():
    # Softmax may round the largest probability to 1 while retaining tiny
    # alternatives. The event still has positive mass in those alternatives.
    p = np.array([[1., 1e-20, 2e-20, 3e-20]])
    np.testing.assert_allclose(pairing_feedback_kernel(p, 1, 0.),
                               [[5e-20, 4e-20, 3e-20]], rtol=1e-12, atol=0.)


def test_kernel_is_equivariant_under_all_response_relabellings():
    p = np.array([[.11, .22, .28, .39]])
    # Relabel emissions, response and the three pairings together.
    for raw in permutations(range(4)):
        perm = np.asarray(raw)
        inverse = np.argsort(perm)
        pair_order = []
        for row in PAIRING_SIBLINGS:
            relabelled = inverse[row[perm]]
            pair_order.append(next(i for i, candidate in enumerate(PAIRING_SIBLINGS)
                                   if np.array_equal(candidate, relabelled)))
        for f in (0., .5, 1.):
            old = pairing_feedback_kernel(p, 2, f)
            new = pairing_feedback_kernel(p[:, perm], int(inverse[1])+1, f)
            np.testing.assert_allclose(old, new[:, pair_order])


@pytest.mark.parametrize('gamma', [0., .6, 1.])
def test_joint_tempering_and_single_feedback_update(gamma):
    prior = np.array([[.24, .12, .04], [.06, .18, .36], [0., 0., 0.]])
    mod = memory(prior, gamma)
    kernel = np.array([[.1, .2, .7], [.5, .3, .2], [.3, .4, .3]])
    powered = np.zeros_like(prior)
    powered[:2] = prior[:2]**gamma
    powered /= powered.sum()
    expected = powered*kernel
    expected /= expected.sum()
    evidence = (powered[:2]/powered[:2].sum(axis=1, keepdims=True)*kernel[:2]).sum(axis=1)
    mod.prepare_feedback(kernel)
    np.testing.assert_allclose(mod.feedback_evidence[:2], evidence)
    mod.process()
    np.testing.assert_allclose(mod.joint, expected)
    np.testing.assert_allclose(mod.engine.posterior, expected.sum(axis=1))
    np.testing.assert_allclose(mod.feedback_evidence[:2], evidence)


def test_snapshot_is_independent_and_restores_feedback_cache():
    mod = memory([[.2, .1, .2], [.1, .3, .1]], .8)
    mod.prepare_feedback(np.array([[.1, .4, .2], [.5, .2, .1]]))
    snapshot = mod.state_dict()
    other = memory([[1/6]*3]*2, .8)
    other.load_state_dict(snapshot)
    other.process()
    assert not np.array_equal(other.joint, mod.joint)
    np.testing.assert_array_equal(snapshot['joint'], mod.joint)
    mod.process()
    np.testing.assert_array_equal(other.joint, mod.joint)


@pytest.mark.parametrize('field,value', [
    ('pending_joint', np.full((2, 3), np.nan)),
    ('pending_joint', np.ones((2, 2))/4),
    ('pending_joint', np.ones((2, 3))),
    ('feedback_evidence', np.array([np.nan, .3])),
    ('feedback_evidence', np.array([.2])),
    ('feedback_evidence', np.array([1.2, .3])),
])
def test_snapshot_rejects_invalid_auxiliary_state_without_mutation(field, value):
    mod = memory([[.2, .1, .2], [.1, .3, .1]])
    before = mod.joint.copy()
    snapshot = mod.state_dict()
    snapshot[field] = value
    with pytest.raises(ValueError):
        mod.load_state_dict(snapshot)
    np.testing.assert_array_equal(mod.joint, before)


@pytest.mark.parametrize('method', ['similarity_transport', 'mass_preserving_similarity_transport', 'pairwise_mass_transfer'])
@pytest.mark.parametrize('after,pairs', [([0, 1], []), ([0, 2], [(1, 2)]), ([2, 3], [(0, 2), (1, 3)])])
def test_transport_retains_pair_knowledge_and_scalar_marginal(method, after, pairs):
    joint = np.array([[.32, .04, .04], [.06, .12, .42], [0, 0, 0], [0, 0, 0]])
    kernel = np.array([[0, .2, .3, .5], [.4, 0, .4, .2], [.2, .5, 0, .3], [.1, .3, .6, 0]])
    base = np.array([.1, .2, .3, .4])
    selection = HypothesisSelection.from_active_sets([0, 1], after, replacement_pairs=pairs)
    out, fallback = transport_joint_belief(joint, selection, method=method,
                                         local_kernel=kernel, base_prior=base, global_fraction=.3)
    # Independent scalar projection, preserving the existing prior definition.
    pi = joint.sum(axis=1)
    expected = np.zeros(4)
    if not pairs:
        expected = pi
    elif method == 'pairwise_mass_transfer':
        expected = pi.copy()
        for old, new in pairs:
            expected[new], expected[old] = expected[old], 0.
    else:
        target = np.asarray(after if method == 'similarity_transport' else selection.newcomers)
        local = (pi @ kernel)[target]
        global_ = base[target]
        projected = .7*local/local.sum() + .3*global_/global_.sum()
        survivors = selection.survivors
        if method == 'similarity_transport':
            fraction = len(pairs)/2
            if len(survivors):
                expected[survivors] = (1-fraction)*pi[survivors]/pi[survivors].sum()
            expected[target] += fraction*projected
        else:
            expected[survivors] = pi[survivors]
            expected[target] = pi[selection.dropped].sum()*projected
    np.testing.assert_allclose(out.sum(axis=1), expected)
    assert not fallback
    if pairs:
        assert not np.allclose(out.sum(axis=0), 1/3)


def test_transport_zero_local_mass_falls_back_to_global_pair_belief():
    joint = np.array([[.7, .2, .1], [0, 0, 0]])
    selection = HypothesisSelection.from_active_sets([0], [1], replacement_pairs=[(0, 1)])
    out, fallback = transport_joint_belief(joint, selection, method='similarity_transport',
                                         local_kernel=np.eye(2), base_prior=np.ones(2)/2,
                                         global_fraction=0.)
    assert fallback
    np.testing.assert_allclose(out[1], [.7, .2, .1])


@pytest.mark.parametrize('kwargs', [{'w0': .1}, {'feedback_gain': .5}, {'gamma': np.nan}])
def test_reject_unsupported_memory_assumptions(kwargs):
    with pytest.raises(ValueError):
        HierarchicalPairingMemoryModule(SimpleNamespace(prior=np.ones(2)/2, set_size=2), **kwargs)

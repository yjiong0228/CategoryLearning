from copy import deepcopy

import numpy as np
import pytest
from scipy.special import softmax

from src.Bayesian_state.model.active_rule_associations import ActiveRuleAssociations
from src.Bayesian_state.model.region_response_learner import RegionResponseLearner


def test_only_active_rules_learn_and_reentry_restores_last_state():
    bank = ActiveRuleAssociations(5, 4, 4, learning_rate=.5)
    references = [RegionResponseLearner(4, 4, learning_rate=.5) for _ in range(5)]
    schedule = [[2, 0], [2, 0], [1, 2], [1, 3], [3, 1], [0, 2]]
    rng = np.random.default_rng(35)
    counts = np.zeros(5, dtype=int)
    for trial, active in enumerate(schedule):
        geometry = rng.dirichlet(np.ones(4), len(active))
        choice, feedback = trial % 4 + 1, float(trial % 2)
        before = bank.associations(range(5))
        predicted = bank.predict(active, geometry)
        evidence = bank.evaluate_feedback(active, geometry, choice, feedback)
        np.testing.assert_array_equal(bank.associations(range(5)), before)
        event = predicted[:, choice-1] if feedback else np.delete(predicted, choice-1, axis=1).sum(1)
        np.testing.assert_allclose(evidence.log_event, np.log(event), atol=1e-12, rtol=0)
        bank.commit_feedback(evidence)
        counts[active] += 1
        for row, rule in enumerate(active):
            references[rule].update(geometry[row], choice, feedback)
        expected = np.stack([r.associations for r in references])
        np.testing.assert_allclose(bank.associations(range(5)), expected, atol=1e-12, rtol=0)
        absent = np.setdiff1d(np.arange(5), active)
        np.testing.assert_array_equal(bank.associations(absent), before[absent])
    np.testing.assert_array_equal(bank.state_dict()['feedback_counts'], counts)
    np.testing.assert_array_equal(bank.associations([4]), .25)


def test_belief_and_precision_use_old_associations_before_commit():
    bank = ActiveRuleAssociations(3, 4, 4, learning_rate=2.)
    ids = [2, 0]; d = np.array([[.1, .7, .2, .5], [.8, .3, .6, .1]])
    q = softmax(-5*d, axis=-1)
    bank.commit_feedback(bank.evaluate_feedback(ids, q, 2, 1.))
    before = bank.associations(ids); prior = np.array([.3, .7])
    p = np.einsum('hg,hgc->hc', q, before)
    evidence = bank.evaluate_feedback(ids, q, 3, 0., distances=d)
    direct = np.delete(p, 2, axis=1).sum(axis=1)
    expected_belief = prior**.8 * direct; expected_belief /= expected_belief.sum()
    belief = prior**.8 * np.exp(evidence.log_event); belief /= belief.sum()
    np.testing.assert_allclose(belief, expected_belief, atol=1e-12, rtol=0)
    compatibility = before[:, :, [0, 1, 3]].sum(-1)
    conditioned = q*compatibility; conditioned /= conditioned.sum(-1, keepdims=True)
    np.testing.assert_allclose(evidence.precision_gradient,
                               np.sum(q*d, -1)-np.sum(conditioned*d, -1), atol=1e-12, rtol=0)
    bank.commit_feedback(evidence)
    assert np.max(np.abs(bank.associations(ids)-before)) > .001


@pytest.mark.parametrize('rate', [0., .5])
def test_repeated_and_superseded_commits_are_rejected(rate):
    bank = ActiveRuleAssociations(2, 2, 2, learning_rate=rate)
    first = bank.evaluate_feedback([0], [[.8, .2]], 1, 1.)
    stale = bank.evaluate_feedback([0], [[.8, .2]], 2, 0.)
    bank.commit_feedback(first)
    for item in (first, stale):
        with pytest.raises(ValueError, match='stale'):
            bank.commit_feedback(item)
    assert bank.state_dict()['feedback_counts'][0] == 1


def test_particle_snapshot_copies_continue_exactly_and_reject_cross_particle_update():
    a = ActiveRuleAssociations(4, 4, 4, learning_rate=.7)
    a.commit_feedback(a.evaluate_feedback([2], [[.2, .1, .3, .4]], 3, 1.))
    state = a.state_dict()
    b = ActiveRuleAssociations(4, 4, 4, learning_rate=.7); b.load_state_dict(state)
    state['logits'][:] = 999; state['feedback_counts'][:] = 999
    for active, choice, feedback in [([0, 3], 2, 0.), ([2, 3], 4, 1.)]:
        q = np.array([[.1, .2, .3, .4], [.4, .3, .2, .1]])
        evaluations = [m.evaluate_feedback(active, q, choice, feedback) for m in (a, b)]
        with pytest.raises(ValueError, match='another'):
            b.commit_feedback(evaluations[0])
        for model, evaluation in zip((a, b), evaluations): model.commit_feedback(evaluation)
        np.testing.assert_array_equal(a.associations(range(4)), b.associations(range(4)))
    b.commit_feedback(b.evaluate_feedback([2], [[.2, .1, .3, .4]], 1, 1.))
    assert not np.array_equal(a.associations([2]), b.associations([2]))


def test_loading_snapshot_invalidates_outstanding_feedback():
    bank = ActiveRuleAssociations(2, 2, 2, learning_rate=1.)
    evidence = bank.evaluate_feedback([0], [[.3, .7]], 1, 1.)
    bank.load_state_dict(bank.state_dict())
    with pytest.raises(ValueError, match='another'): bank.commit_feedback(evidence)


def test_feedback_arrays_are_immutable():
    bank = ActiveRuleAssociations(2, 2, 2, learning_rate=1.)
    evidence = bank.evaluate_feedback([0], [[.3, .7]], 1, 1., distances=np.array([[.3, .2]]))
    for array in (evidence.rules, evidence.log_event, evidence.association_gradient,
                  evidence.precision_gradient):
        with pytest.raises(ValueError): array.setflags(write=True)


@pytest.mark.parametrize('field,value', [
    ('version', 'other'), ('learning_rate', 2.), ('logits', np.full((2, 2, 2), np.nan)),
    ('logits', np.ones((2, 2, 2))), ('feedback_counts', np.array([-1, 0])),
    ('feedback_counts', np.array([1., 0.])), ('feedback_counts', np.zeros(3, dtype=int)),
])
def test_invalid_snapshot_is_rejected_atomically(field, value):
    bank = ActiveRuleAssociations(2, 2, 2, learning_rate=1.)
    state = bank.state_dict(); state[field] = value
    before = deepcopy(bank.state_dict())
    with pytest.raises(ValueError, match='snapshot'): bank.load_state_dict(state)
    after = bank.state_dict()
    np.testing.assert_array_equal(before['logits'], after['logits'])
    np.testing.assert_array_equal(before['feedback_counts'], after['feedback_counts'])


@pytest.mark.parametrize('rules', [[0, 0], [0., 1.], [True], [-1], [2], []])
def test_invalid_rule_identity_is_rejected(rules):
    bank = ActiveRuleAssociations(2, 2, 2, learning_rate=1.)
    with pytest.raises(ValueError, match='rule IDs'): bank.associations(rules)

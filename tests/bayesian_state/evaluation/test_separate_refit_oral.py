"""Preserve actual belief identity, common reports and paired filter units."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.Bayesian_state.workflows.analysis import model_0923_separate_belief_refit_oral as oral


def identities():
    before = dict(scientific={'particles': 512, 'trials': 576}, arm_settings={'precision': 'fast'},
        mechanism={'kind': 'E', 'retain_learned_precision': True, 'online_belief': True},
        archived_pf_sha256='same', task_sha256={'training': 'same'}, oral_manifest_sha256='same',
        model_sources={'original': 'hash'}, candidate={'id': 'old', 'hyperparams': {'gamma': .3}},
        structural_intervention='retained_rule_precision', parameter_estimation_scope='prefix',
        parameter_candidate_policy='projected_common_search_policy_behavior_refit')
    after = deepcopy(before); after['mechanism'].pop('online_belief')
    after['mechanism']['separate_belief'] = True
    after['candidate'] = {'id': 'new', 'hyperparams': {'gamma': .95}}
    after['model_sources']['separate'] = 'new_hash'
    return before, after


def test_permits_declared_memory_change_and_parameter_reestimation_without_mutating_identity():
    before, after = identities(); saved = deepcopy((before, after))
    oral.validate_identities(before, after)
    assert (before, after) == saved
    with pytest.raises(ValueError): oral.comparison.validate_comparison(before, after, 'parameters')


@pytest.mark.parametrize('fault', ['reversed', 'no_online', 'no_separate', 'no_retention', 'full_fit',
                                   'oral_selected', 'budget', 'oral', 'task', 'other_mechanism', 'source'])
def test_rejects_changed_data_use_or_uncontrolled_scientific_setting(fault):
    before, after = identities()
    if fault == 'reversed': before, after = after, before
    elif fault == 'no_online': before['mechanism']['online_belief'] = False
    elif fault == 'no_separate': after['mechanism']['separate_belief'] = False
    elif fault == 'no_retention': after['mechanism']['retain_learned_precision'] = False
    elif fault == 'full_fit': after['parameter_estimation_scope'] = 'full'
    elif fault == 'oral_selected': after['parameter_candidate_policy'] = 'oral_selected'
    elif fault == 'budget': after['scientific']['particles'] = 128
    elif fault == 'oral': after['oral_manifest_sha256'] = 'changed'
    elif fault == 'task': after['task_sha256']['training'] = 'changed'
    elif fault == 'other_mechanism': after['mechanism']['kind'] = 'new'
    elif fault == 'source': after['model_sources']['original'] = 'changed'
    with pytest.raises(ValueError): oral.validate_identities(before, after)


def test_missing_wrapper_requires_its_verified_reference_hash():
    before, after = identities(); del after['model_sources']['original']
    with pytest.raises(ValueError): oral.validate_identities(before, after)
    oral.validate_identities(before, after, executed_sources={'original': 'hash'})
    with pytest.raises(ValueError): oral.validate_identities(before, after, executed_sources={'original': 'wrong'})


def state():
    common = dict(belief_intervention=oral.refit.INTERVENTION, evaluation_use=oral.states.USE)
    job = dict(common, candidate={'hyperparams': {'engine.modules.memory_mod.kwargs.gamma': .95}},
               probe_mechanism='SEPARATE', online_belief=False, model='POOL', reported_arm='SEPARATE_POOL')
    record = dict(common, online_belief=False, separate_belief=True, arm='POOL', proposal_arm='POOL',
        belief_gamma=.95, belief_matches_saved_state=True, belief_updates=28, particles=4, trials=8,
        exact_behavior_replay=True, timing='after_choice_before_current_feedback', data_use='behavior_only; no_oral_input')
    return job, record, deepcopy(common), deepcopy(common)


def test_requires_the_actual_completed_feedback_state():
    oral.check_separate_state(*state())


@pytest.mark.parametrize('index,key,value', [(0, 'probe_mechanism', 'ONLINE'), (0, 'reported_arm', 'POOL'),
    (1, 'separate_belief', False), (1, 'belief_gamma', .3), (1, 'belief_updates', 32),
    (1, 'belief_matches_saved_state', False), (1, 'exact_behavior_replay', False),
    (1, 'timing', 'after_feedback'), (1, 'data_use', 'oral_selected'),
    (2, 'belief_intervention', 'online_retention_evidence'), (3, 'evaluation_use', 'other')])
def test_rejects_mislabelled_or_unverified_separate_state(index, key, value):
    args = state(); args[index][key] = value
    with pytest.raises(ValueError): oral.check_separate_state(*args)


def test_feedback_comparison_keeps_known_shift_and_true_mechanism_labels():
    probability = np.array([[.1, .3, .2, .6], [.9, .7, .8, .4]])
    support = np.array([[.3, .2, .6, .3], [.1, .6, .2, .7]])
    feedback = np.array([0., 1., .5, 1.]); complete = np.array([1, 1, 0, 1], bool)
    trials = pd.DataFrame({key: [str(key)] * 4 for key in oral.comparison.IDENTITY})
    banks = []
    for shift in (0., .1):
        fractions = np.stack((support + shift, np.zeros_like(support), 1 - support - shift), axis=-1)
        joint = np.repeat(fractions[:, :, None, :], 2, axis=2) * probability[:, :, None, None]
        prior = np.broadcast_to([.4, .1, .5], joint.shape).copy()
        estimates = oral.strata.stratified_estimates(joint, probability, prior, complete, feedback, 2)
        banks.append(dict(seeds=np.array([3, 5]), trials=trials, feedback=feedback,
                          categories=np.array([1, 1, 2, 1]), estimates=estimates))
    scores, deleted = oral.feedback_comparison(*banks, 208, 'POOL')
    assert len(scores) == 72 and len(deleted) == 24
    assert all(key.startswith(('online_', 'separate_')) for key in deleted)
    nonempty = scores.complete_reports.gt(0)
    for metric, expected in [('conditioned', .1), ('unweighted', 0.), ('current_choice_gain', .1)]:
        rows = scores[nonempty & scores.metric.eq(metric)]
        np.testing.assert_allclose(rows.support_difference, expected, atol=1e-14, rtol=0)
        np.testing.assert_allclose(rows.undetermined_difference, -expected, atol=1e-14, rtol=0)
        np.testing.assert_allclose(rows.support_difference_mcse, 0., atol=1e-14, rtol=0)
    assert scores.loc[~nonempty].filter(regex='difference').isna().all().all()
    banks[1]['seeds'] = banks[1]['seeds'][::-1]
    with pytest.raises(AssertionError): oral.feedback_comparison(*banks, 208, 'POOL')

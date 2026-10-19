"""State provenance and oral contrasts must preserve paired parameter points."""
from copy import deepcopy

import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis.model_0923_separate_belief_states import (
    CELLS, check_identity, paired_contrasts)


def execution(cell):
    mechanism, parameter = cell.split('_')
    point = {'engine.modules.memory_mod.kwargs.gamma': 1.,
             'engine.modules.hypo_transitions_mod.kwargs.intervention.decay': .3}
    job = dict(probe_mechanism=mechanism, candidate={'id': parameter, 'hyperparams': point},
        model='SEARCH', phase='evaluate', retain_precision=True, particles=128, seeds=[47],
        online_belief=mechanism == 'ONLINE', structural_intervention='retained_rule_precision',
        belief_intervention='online_retention_evidence' if mechanism == 'ONLINE' else 'separate_gamma_belief_evidence')
    record = {key: deepcopy(job[key]) for key in ('candidate', 'model', 'seeds', 'particles')}
    record['trials'] = 8
    precision = dict(retained_precision=True, structural_intervention='retained_rule_precision', transitions=1024)
    common = dict(enabled=True, belief_intervention=job['belief_intervention'])
    belief = (dict(common, exact_transition_timing=True, evidence_matches_retained_state=True, reconstructions=896)
              if mechanism == 'ONLINE' else dict(common, completed_feedback_timing_exact=True,
                  belief_matches_saved_state=True, gamma=1., delta=[.3], updates=896))
    return cell, job, deepcopy(job), record, precision, belief


@pytest.mark.parametrize('cell', CELLS)
def test_accepts_all_six_executed_cells(cell):
    check_identity(*execution(cell))


@pytest.mark.parametrize('change', ['cell', 'candidate', 'seeds', 'particles', 'mechanism', 'timing',
                                  'count', 'gamma', 'delta', 'retention', 'evidence'])
def test_rejects_changed_separate_filter(change):
    args = list(execution('SEPARATE_NEAR'))
    cell, planned, actual, record, precision, belief = args
    if change == 'cell': args[0] = 'UNKNOWN'
    elif change == 'candidate': record['candidate']['id'] = 'changed'
    elif change == 'seeds': record['seeds'] = [48]
    elif change == 'particles': record['particles'] = 512
    elif change == 'mechanism': actual['online_belief'] = True
    elif change == 'timing': belief['completed_feedback_timing_exact'] = False
    elif change == 'count': belief['updates'] = 1024
    elif change == 'gamma': belief['gamma'] = .3
    elif change == 'delta': belief['delta'] = [1.]
    elif change == 'retention': precision['retained_precision'] = False
    elif change == 'evidence': belief['belief_matches_saved_state'] = False
    with pytest.raises(ValueError): check_identity(*args)


def test_rejects_separate_proof_on_online_filter():
    args = list(execution('ONLINE_ONLINE')); args[-1] = execution('SEPARATE_ONLINE')[-1]
    with pytest.raises(ValueError): check_identity(*args)


def estimates():
    result = {}
    for i, parameter in enumerate(('RETAIN', 'ONLINE', 'NEAR')):
        old = np.array([[.2, .1, .7], [.4, .2, .4]])
        difference = (i + 1) * np.array([[.01, -.005, -.005], [.03, -.01, -.02]])
        for mechanism, values in [('ONLINE', old), ('SEPARATE', old + difference)]:
            result[208, mechanism + '_' + parameter] = {
                ('test', 'functional', 'complete'): dict(seeds=np.array([47, 48]),
                    values=values, point=values.mean(0))}
    return result


def test_oral_differences_pair_same_parameter_and_full_filter_deletions():
    table = paired_contrasts(estimates()).set_index('parameter')
    for i, parameter in enumerate(('RETAIN', 'ONLINE', 'NEAR')):
        row = table.loc[parameter]
        assert row.reference == 'ONLINE_' + parameter
        np.testing.assert_allclose(row.support_difference, .02 * (i + 1), atol=1e-14)
        np.testing.assert_allclose(row.support_difference_mcse, .01 * (i + 1), atol=1e-14)
        assert abs(row.support_difference + row.conflict_difference + row.undetermined_difference) < 1e-14


def test_oral_difference_rejects_unmatched_filter_seeds():
    values = estimates()
    values[208, 'SEPARATE_NEAR']['test', 'functional', 'complete']['seeds'] = np.array([49, 50])
    with pytest.raises(AssertionError): paired_contrasts(values)

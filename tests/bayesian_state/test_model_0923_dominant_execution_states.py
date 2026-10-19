"""Execution-state provenance and paired oral contrasts retain their identities."""
from copy import deepcopy

import numpy as np
import pytest

from src.Bayesian_state.evaluation.model_0923_separate_belief_refit import GAMMA, PREFIX, TRANSFER
from src.Bayesian_state.workflows.analysis.model_0923_dominant_execution_states import (
    CELLS, POLICY, check_identity, paired_contrasts)


def execution(cell):
    policy, arm = cell.split('_')
    point = {GAMMA: .8, PREFIX + 'intervention.decay': .3, TRANSFER: .01}
    job = dict(probe_mechanism='SEPARATE', candidate=dict(id=arm, hyperparams=point),
               model=arm, phase='evaluate', retain_precision=True, particles=4, seeds=[47],
               online_belief=False, structural_intervention='retained_rule_precision',
               belief_intervention='separate_gamma_belief_evidence', fixed_coordinates={TRANSFER: .01},
               execution_policy=policy)
    record = {key: deepcopy(job[key]) for key in ('candidate', 'model', 'seeds', 'particles')}
    record['trials'] = 8
    precision = dict(retained_precision=True, structural_intervention='retained_rule_precision', transitions=32)
    belief = dict(enabled=True, belief_intervention=job['belief_intervention'], gamma=.8, delta=[.3], updates=28,
                  completed_feedback_timing_exact=True, belief_matches_saved_state=True)
    selected = policy == 'SELECT'
    proof = dict(enabled=selected, policy=policy, rule_execution_policy=POLICY if selected else 'mixture',
                 uses_existing_dominant=True, task_categories_used_for_selection=False, updates=32 if selected else 0)
    return cell, job, deepcopy(job), record, precision, belief, proof


@pytest.mark.parametrize('cell', CELLS)
def test_all_four_policy_and_proposal_combinations(cell):
    check_identity(*execution(cell))


@pytest.mark.parametrize('index,key,value', [
    (2, 'execution_policy', 'MIX'), (3, 'seeds', [48]), (3, 'particles', 8),
    (4, 'transitions', 28), (5, 'gamma', .3), (5, 'updates', 32),
    (6, 'updates', 28), (6, 'rule_execution_policy', 'mixture'),
    (6, 'task_categories_used_for_selection', True), (6, 'uses_existing_dominant', False),
])
def test_rejects_changed_state_reference(index, key, value):
    args = list(execution('SELECT_POOL'))
    args[index][key] = value
    with pytest.raises(ValueError):
        check_identity(*args)


def estimates():
    result = {}
    for i, arm in enumerate(('POOL', 'SEARCH')):
        old = np.array([[.2, .1, .7], [.4, .2, .4]])
        change = (i + 1) * np.array([[.01, -.005, -.005], [.03, -.01, -.02]])
        for policy, values in (('MIX', old), ('SELECT', old + change)):
            result[208, policy + '_' + arm] = {
                ('test', 'functional', 'complete'): dict(seeds=np.array([47, 48]),
                                                       values=values, point=values.mean(0))}
    return result


def test_pairs_same_parameter_and_complete_filter_deletions():
    table = paired_contrasts(estimates()).set_index('proposal_arm')
    for i, arm in enumerate(('POOL', 'SEARCH')):
        row = table.loc[arm]
        assert row.reference == 'MIX_' + arm and row.arm == 'SELECT_' + arm
        np.testing.assert_allclose(row.support_difference, .02 * (i + 1), atol=1e-14)
        np.testing.assert_allclose(row.support_difference_mcse, .01 * (i + 1), atol=1e-14)
        assert abs(row.support_difference + row.conflict_difference + row.undetermined_difference) < 1e-14


def test_rejects_unmatched_filter_seeds():
    rows = estimates()
    rows[208, 'SELECT_SEARCH']['test', 'functional', 'complete']['seeds'] = np.array([49, 50])
    with pytest.raises(AssertionError):
        paired_contrasts(rows)


def test_rejects_unmatched_report_subsets():
    rows = estimates()
    rows[208, 'SELECT_POOL'] = {}
    with pytest.raises(ValueError, match='subsets'):
        paired_contrasts(rows)

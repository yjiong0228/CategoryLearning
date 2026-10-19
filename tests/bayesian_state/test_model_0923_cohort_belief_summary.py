"""Cohort summary checks actual feedback updates for both belief mechanisms."""
from copy import deepcopy

import pytest

from src.Bayesian_state.evaluation.model_0923_cohort_belief_refit import check_execution
from src.Bayesian_state.evaluation.model_0923_separate_belief_refit import GAMMA, PREFIX, TRANSFER


def records(mechanism, phase):
    intervention = 'online_retention_evidence' if mechanism == 'ONLINE' else 'separate_gamma_belief_evidence'
    fixed = {TRANSFER: .01}
    point = {**fixed, GAMMA: .95, PREFIX + 'intervention.decay': .8}
    job = dict(phase=phase, seeds=[11, 22], particles=4, candidate=dict(hyperparams=point),
        cohort_belief_mechanism=mechanism, probe_mechanism=mechanism, online_belief=mechanism == 'ONLINE',
        belief_intervention=intervention, fixed_coordinates=fixed,
        structural_intervention='retained_rule_precision', retain_precision=True)
    copies = 2 if phase == 'autonomous' else 8
    precision = dict(structural_intervention='retained_rule_precision', retained_precision=True, transitions=copies * 8)
    belief = dict(enabled=True, belief_intervention=intervention, fixed_coordinates=fixed,
        reconstructions=copies * 7, exact_transition_timing=True, evidence_matches_retained_state=True,
        gamma=.95, delta=[.8], updates=copies * 7, completed_feedback_timing_exact=True, belief_matches_saved_state=True)
    return job, dict(trials=8), belief, precision


@pytest.mark.parametrize('mechanism', ['ONLINE', 'SEPARATE'])
@pytest.mark.parametrize('phase', ['training', 'evaluate', 'autonomous'])
def test_counts_completed_feedback_once_per_particle_or_path(mechanism, phase):
    check_execution(*records(mechanism, phase), mechanism)


@pytest.mark.parametrize('mechanism', ['ONLINE', 'SEPARATE'])
@pytest.mark.parametrize('fault', ['identity', 'role', 'precision', 'belief', 'fixed', 'enabled'])
def test_rejects_changed_executed_mechanism_or_feedback_counts(mechanism, fault):
    job, result, belief, precision = deepcopy(records(mechanism, 'training'))
    if fault == 'identity': job['cohort_belief_mechanism'] = 'RETAIN'
    elif fault == 'role': job['phase'] = 'unknown'
    elif fault == 'precision': precision['transitions'] += 1
    elif fault == 'belief': belief['reconstructions'] += 1; belief['updates'] += 1
    elif fault == 'fixed': job['candidate']['hyperparams'][TRANSFER] = .1
    elif fault == 'enabled': belief['enabled'] = False
    with pytest.raises(ValueError):
        check_execution(job, result, belief, precision, mechanism)

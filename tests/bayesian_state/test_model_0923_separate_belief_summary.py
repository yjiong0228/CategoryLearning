"""Preserve scoring budgets and reject false separate-belief execution tags."""
from copy import deepcopy
import json

import numpy as np
import pytest

from src.Bayesian_state.evaluation.model_0923_separate_belief_refit import (
    GAMMA, INTERVENTION, PREFIX, TRANSFER, check_execution, check_training_arrays,
    refit_inactive_channels, training_chunk_budgets)


def test_counts_each_filter_once_with_all_intervention_sidecars(tmp_path):
    chunk = tmp_path / 'subject_208' / 'SEARCH' / 'chunks'; chunk.mkdir(parents=True)
    record = dict(subject=208, model='SEARCH', numeric_pools=[dict(num_threads=1)],
                  particles=128, seeds=[1, 2], trials=384, cpu_seconds=9.)
    (chunk / 'point.json').write_text(json.dumps(record))
    for suffix in ('experiment', 'precision', 'online', 'belief'):
        (chunk / ('point.' + suffix + '.json')).write_text('{}')
    rows = training_chunk_budgets(tmp_path)
    assert len(rows) == 1 and rows[0]['state_updates'] == 128 * 2 * 384
    assert rows[0]['filter_runs'] == 2 and rows[0]['cpu_seconds'] == 9.
    record['numeric_pools'][0]['num_threads'] = 2
    (chunk / 'point.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='Nested numeric threads'):
        training_chunk_budgets(tmp_path)


def execution(phase='training'):
    point = {GAMMA: .95, PREFIX + 'intervention.decay': .3, TRANSFER: .01}
    job = dict(phase=phase, seeds=[1, 2], particles=8, candidate=dict(hyperparams=point),
               belief_intervention=INTERVENTION, probe_mechanism='SEPARATE', online_belief=False,
               fixed_coordinates={TRANSFER: .01}, structural_intervention='retained_rule_precision',
               retain_precision=True)
    copies = 2 if phase == 'autonomous' else 16
    result = dict(trials=10)
    belief = dict(enabled=True, belief_intervention=INTERVENTION, gamma=.95, delta=[.3],
                  completed_feedback_timing_exact=True, belief_matches_saved_state=True, updates=copies * 9)
    precision = dict(retained_precision=True, structural_intervention='retained_rule_precision',
                     transitions=copies * 10)
    return job, result, belief, precision


@pytest.mark.parametrize('phase', ['training', 'evaluate', 'autonomous'])
def test_feedback_counts_match_role(phase):
    check_execution(*execution(phase))


@pytest.mark.parametrize('field,value', [('gamma', .3), ('delta', [.95]), ('updates', 160),
    ('belief_matches_saved_state', False), ('completed_feedback_timing_exact', False),
    ('belief_intervention', 'online_retention_evidence')])
def test_rejects_wrong_evidence_state_and_timing(field, value):
    job, result, belief, precision = execution(); belief[field] = value
    with pytest.raises(ValueError):
        check_execution(job, result, belief, precision)


def test_closed_channels_keep_two_active_memory_rates():
    job, _, _, _ = execution(); point = deepcopy(job['candidate']['hyperparams'])
    point.update({PREFIX + 'exit_rate': 0., PREFIX + 'search_controller.global_search': 1.,
                  PREFIX + 'local_structure_weight': .5})
    mechanism = dict(kind='E', capacity='evidence_mean')
    frozen = dict(settings=dict(belief_intervention=INTERVENTION, mechanism=mechanism,
                                models={'SEARCH': mechanism}),
                  nominees=[dict(subject=208, model='SEARCH', candidate=dict(hyperparams=point))])
    rows = refit_inactive_channels(frozen)
    assert TRANSFER in set(rows.parameter) and PREFIX + 'local_structure_weight' in set(rows.parameter)
    assert GAMMA not in set(rows.parameter) and PREFIX + 'intervention.decay' not in set(rows.parameter)
    assert 'SEPARATE' in rows.loc[rows.parameter.eq(TRANSFER), 'reason'].iloc[0]
    frozen['settings']['belief_intervention'] = 'online_retention_evidence'
    with pytest.raises(ValueError):
        refit_inactive_channels(frozen)


def test_training_schema_preserves_probability_pooling_and_seed_identity():
    job = dict(candidate={'id': 'example'}, model='SEARCH', seeds=[7, 8], particles=128)
    result = dict(job, trials=2, numeric_pools=[dict(num_threads=1)], mean_nll=-np.log(.5))
    arrays = dict(observed_probability=np.array([[.2, .8], [.8, .2]]),
                  correct_probability=np.full((2, 2), .5), seeds=np.array([7, 8]))
    check_training_arrays(job, result, arrays)
    result['mean_nll'] = -np.log(arrays['observed_probability']).mean()
    with pytest.raises(AssertionError):
        check_training_arrays(job, result, arrays)
    result['mean_nll'] = -np.log(.5)
    arrays['seeds'] = np.array([8, 7])
    with pytest.raises(AssertionError):
        check_training_arrays(job, result, arrays)

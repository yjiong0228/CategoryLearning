"""Cross-state replay preserves diagnostic labels and actual interventions."""
from copy import deepcopy

import pytest

from src.Bayesian_state.workflows.analysis.model_0923_online_parameter_cross_states import check_identity


def fixture(cell):
    mechanism, parameter = cell.split('_')
    job = dict(cross_mechanism=mechanism, cross_parameter=parameter, model='SEARCH', phase='evaluate',
               retain_precision=True, seeds=[47], particles=512, candidate={'id': parameter},
               structural_intervention='retained_rule_precision')
    if mechanism == 'ONLINE': job['online_belief'] = True
    record = {k: deepcopy(job[k]) for k in ('candidate', 'model', 'seeds', 'particles')}
    record['trials'] = 8
    precision = dict(retained_precision=True, structural_intervention='retained_rule_precision', transitions=4096)
    online = (dict(enabled=True, exact_transition_timing=True, evidence_matches_retained_state=True,
                   reconstructions=3584) if mechanism == 'ONLINE' else None)
    return cell, job, deepcopy(job), record, precision, online


@pytest.mark.parametrize('cell', ['ONLINE_RETAIN', 'RETAIN_ONLINE', 'ONLINE_NEAR'])
def test_accepts_executed_cross_cells(cell):
    check_identity(*fixture(cell))


@pytest.mark.parametrize('change', ['label', 'nominee', 'seed', 'particles', 'variant', 'timing', 'retention', 'count'])
def test_rejects_changed_execution_or_identity(change):
    cell, planned, actual, record, precision, online = fixture('ONLINE_RETAIN')
    if change == 'label': cell = 'ONLINE_NEAR'
    elif change == 'nominee': record['candidate']['id'] = 'different'
    elif change == 'seed': record['seeds'] = [48]
    elif change == 'particles': record['particles'] = 128
    elif change == 'variant': actual['online_belief'] = False
    elif change == 'timing': online['exact_transition_timing'] = False
    elif change == 'retention': precision['retained_precision'] = False
    elif change == 'count': online['reconstructions'] = 4096
    with pytest.raises(ValueError): check_identity(cell, planned, actual, record, precision, online)


def test_retained_cell_rejects_an_online_execution_sidecar():
    values = list(fixture('RETAIN_ONLINE')); values[-1] = {'enabled': True}
    with pytest.raises(ValueError): check_identity(*values)


def test_original_nominee_is_not_replayed_in_new_cell_adapter():
    values = list(fixture('ONLINE_RETAIN')); values[0] = 'ONLINE_ONLINE'
    with pytest.raises(ValueError): check_identity(*values)

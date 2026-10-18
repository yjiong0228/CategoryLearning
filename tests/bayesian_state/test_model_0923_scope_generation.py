"""Parameter scope labels share paths only for the same scientific point."""
from copy import deepcopy

import pytest

from src.Bayesian_state.optimization.search.adaptive_proposals import candidate
from src.Bayesian_state.workflows.analysis.model_0923_scope_generation import point_bindings, provisional_full_nominees


def inputs():
    point = candidate(dict(value=1), 'fixture')
    original = [dict(subject=308, model=model, candidate=point) for model in ('POOL', 'SEARCH')]
    selected = [dict(row, objective=objective) for row in original for objective in ('prefix', 'full')]
    return selected, original


def test_scope_aliases_share_points_but_mechanisms_remain_distinct():
    selected, original = inputs()
    bindings, points = point_bindings(selected, original)
    assert len(bindings) == 6
    assert len(points) == 2
    assert {key[1] for key in points} == {'POOL', 'SEARCH'}


def test_distinct_full_sequence_point_adds_one_bank():
    selected, original = inputs()
    selected = deepcopy(selected)
    selected[1]['candidate'] = candidate(dict(value=2), 'fixture')
    bindings, points = point_bindings(selected, original)
    assert len(points) == 3
    assert len(bindings) == 6


def test_missing_scope_and_mislabeled_parameter_are_rejected():
    selected, original = inputs()
    with pytest.raises(ValueError, match='Incomplete'):
        point_bindings(selected[:-1], original)
    selected = deepcopy(selected)
    selected[0]['candidate']['hyperparams']['value'] = 3
    with pytest.raises(ValueError, match='identity'):
        point_bindings(selected, original)


def selection_stage():
    a = dict(candidate(dict(value=1), 'fixture'), full_nll=.8, particles=256, seeds=[31, 47])
    b = dict(candidate(dict(value=2), 'fixture'), full_nll=.6, particles=256, seeds=[31, 47])
    return dict(status='complete', stage='selection', points=4, seed_jobs=8,
        groups=[dict(subject=308, model=arm, rows=deepcopy([a,b])) for arm in ('POOL','SEARCH')])


def test_provisional_selection_uses_full_loss_and_keeps_original_comparators():
    selected = provisional_full_nominees(selection_stage(), [308])
    _, original = inputs()
    bindings, points = point_bindings(selected, original, ('full',))
    assert len(bindings) == len(points) == 4
    assert {r['objective'] for r in bindings} == {'full','original'}
    assert all(r['candidate']['hyperparams']['value'] == 2 for r in selected)


def test_provisional_ties_use_stable_parameter_identity():
    stage = selection_stage()
    for group in stage['groups']:
        for row in group['rows']: row['full_nll'] = .6
        group['rows'].reverse()
    selected = provisional_full_nominees(stage, [308])
    expected = min(r['id'] for r in stage['groups'][0]['rows'])
    assert {r['candidate']['id'] for r in selected} == {expected}


@pytest.mark.parametrize('fault', ['status','phase','group','bank','points','seed_jobs','seeds','budget','nan','identity'])
def test_provisional_generation_rejects_incomplete_or_changed_scoring(fault):
    stage = selection_stage()
    if fault == 'status': stage['status'] = 'running'
    elif fault == 'phase': stage['stage'] = 'screen'
    elif fault == 'group': stage['groups'].pop()
    elif fault == 'bank':
        stage['groups'][0]['rows'].pop();stage['points']-=1;stage['seed_jobs']-=2
    elif fault in ('points','seed_jobs'): stage[fault] -= 1
    elif fault == 'seeds': stage['groups'][0]['rows'][0]['seeds'] = [31,31]
    elif fault == 'budget': stage['groups'][0]['rows'][0]['particles'] = 128
    elif fault == 'nan': stage['groups'][0]['rows'][0]['full_nll'] = float('nan')
    elif fault == 'identity': stage['groups'][0]['rows'][0]['hyperparams']['value'] = 3
    with pytest.raises(ValueError): provisional_full_nominees(stage, [308])


def test_different_subjects_can_use_distinct_shared_seed_banks():
    stage = selection_stage();more = deepcopy(stage['groups'])
    for group in more:
        group['subject'] = 208
        for row in group['rows']:row['seeds'] = [53,61]
    stage['groups'] += more;stage['points'] *= 2;stage['seed_jobs'] *= 2
    assert len(provisional_full_nominees(stage,[208,308])) == 4

"""Parameter scope labels share paths only for the same scientific point."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from src.Bayesian_state.optimization.search.adaptive_proposals import candidate
from src.Bayesian_state.workflows.analysis.model_0923_scope_generation import (
    point_bindings, provisional_full_nominees, reference_generation, validate_reused_bank,
)
from src.Bayesian_state.evaluation.model_0923_h_validation import digest


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


@pytest.fixture
def reuse_fixture(tmp_path):
    source = tmp_path/'saved';source.mkdir()
    (source/'source').mkdir()
    engine = source/'source/engine.py';engine.write_text('frozen engine')
    wrapper = source/'source/wrapper.py';wrapper.write_text('earlier generation wrapper')
    task = tmp_path/'task';task.write_text('unchanged task')
    settings = dict(mechanism='E', arms={'POOL':{'capacity':'pooled'}}, features=['neck'])
    frozen = dict(settings=settings, source_sha256={'engine.py':digest(engine)}, input_sha256={str(task):digest(task)})
    point = candidate({'engine.modules.hypo_transitions_mod.kwargs.max_capacity':2},'fixture')
    array = source/'bank.npz';zero=np.zeros((2,2))
    np.savez(array,seeds=[11,17],categories=[1,2],probabilities=np.full((2,2,4),.25),
             choices=[[1,2],[2,1]],feedback=[[1,1],[0,0]],correct_probability=np.full((2,2),.25),
             present=zero,contribution=zero,beta=zero,active_count=np.ones((2,2)),search=zero,
             target_exited=zero,target_capacity_dropped=zero)
    row = dict(subject=308,model='POOL',candidate_id=point['id'],candidate=point,paths=2,trials=2,
               training_trials=1,condition=2,target_rule=42,role='autonomous_from_start',
               target=str(array.with_suffix('')),predictions_sha256=digest(array))
    scope = tmp_path/'scope';reference=tmp_path/'reference'
    protocol=dict(smoke=False,scope=str(scope),reference=str(reference),bindings=[],settings=settings,
                  source_sha256={**frozen['source_sha256'],'wrapper.py':digest(wrapper)},
                  input_sha256=deepcopy(frozen['input_sha256']))
    result=dict(status='complete',smoke=False,bindings=[],rows=[row])
    for name,data in [('protocol',protocol),('generation',result)]:
        (source/(name+'.json')).write_text(json.dumps(data))
    return source,scope,reference,frozen,row


def test_reuse_checks_archived_wrapper_without_requiring_new_wrapper_identity(reuse_fixture):
    source,scope,reference,frozen,row = reuse_fixture
    records,inputs = reference_generation(source,scope,reference,frozen)
    assert list(records)==[(308,'POOL',row['candidate_id'])]
    file=validate_reused_bank(row,row['candidate'],[11,17],row,np.array([1,2]))
    assert file==source/'bank.npz' and str(source/'source/wrapper.py') in inputs


@pytest.mark.parametrize('fault',['scope','incomplete','smoke','mechanism','source','archive','task','duplicate','identity'])
def test_reuse_rejects_changed_provenance(reuse_fixture,fault):
    source,scope,reference,frozen,row = reuse_fixture
    p=json.loads((source/'protocol.json').read_text());g=json.loads((source/'generation.json').read_text())
    if fault=='scope':p['scope']=str(scope/'other')
    elif fault=='incomplete':g['status']='running'
    elif fault=='smoke':g['smoke']=True
    elif fault=='mechanism':p['settings']['mechanism']='J'
    elif fault=='source':p['source_sha256']['engine.py']='changed'
    elif fault=='archive':(source/'source/wrapper.py').write_text('changed')
    elif fault=='task':Path(next(iter(frozen['input_sha256']))).write_text('changed')
    elif fault=='duplicate':g['rows'].append(deepcopy(row))
    else:g['rows'][0]['candidate']['hyperparams']['engine.modules.hypo_transitions_mod.kwargs.max_capacity']=4
    (source/'protocol.json').write_text(json.dumps(p));(source/'generation.json').write_text(json.dumps(g))
    with pytest.raises(ValueError):reference_generation(source,scope,reference,frozen)


@pytest.mark.parametrize('fault',['parameters','count','trials','condition','target','seed_order','categories','array'])
def test_reuse_requires_full_matching_path_bank(reuse_fixture,fault):
    source,_,_,_,row = reuse_fixture
    point=deepcopy(row['candidate']);original=deepcopy(row);seeds=[11,17];categories=np.array([1,2])
    if fault=='parameters':point['hyperparams']['engine.modules.hypo_transitions_mod.kwargs.max_capacity']=4
    elif fault=='count':row['paths']=1
    elif fault=='trials':row['trials']=1
    elif fault=='condition':row['condition']=3
    elif fault=='target':row['target_rule']=43
    elif fault=='seed_order':seeds.reverse()
    elif fault=='categories':categories=categories[::-1]
    else:(source/'bank.npz').write_bytes(b'changed')
    with pytest.raises((ValueError,AssertionError)):
        validate_reused_bank(row,point,seeds,original,categories)

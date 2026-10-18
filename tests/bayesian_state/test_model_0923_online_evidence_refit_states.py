"""Bind ONLINE readouts to actual fitted jobs, before any oral scoring."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from src.Bayesian_state.workflows.analysis import model_0923_online_evidence_refit as refit
from src.Bayesian_state.workflows.analysis import model_0923_online_evidence_refit_states as states
from src.Bayesian_state.utils.seeding import stable_seed


def execution():
    fixed={refit.TRANSFER:.01}
    plan=dict(phase='evaluate',model='POOL',candidate={'hyperparams':fixed})
    actual=dict(plan,structural_intervention='retained_rule_precision',retain_precision=True,
        retention_reference_inputs={},belief_intervention='online_retention_evidence',online_belief=True,
        fixed_coordinates=fixed)
    precision=dict(retained_precision=True,structural_intervention='retained_rule_precision')
    online=dict(enabled=True,belief_intervention='online_retention_evidence',fixed_coordinates=fixed,
        exact_transition_timing=True,evidence_matches_retained_state=True)
    return plan,actual,{},fixed,precision,online


def test_accepts_executed_interventions():
    states.checked_execution(*execution())


@pytest.mark.parametrize('object_index,key,value',[
    (1,'online_belief',False),(1,'retain_precision',False),(1,'fixed_coordinates',{refit.TRANSFER:1.}),
    (1,'retention_reference_inputs',{'changed':'digest'}),(1,'phase','training'),
    (4,'retained_precision',False),(5,'exact_transition_timing',False),
    (5,'evidence_matches_retained_state',False),(5,'enabled',False)])
def test_rejects_mislabelled_or_incomplete_execution(object_index,key,value):
    values=deepcopy(execution());values[object_index][key]=value
    with pytest.raises(ValueError):
        states.checked_execution(*values)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def source_fixture(tmp_path,monkeypatch):
    fit,source,data=tmp_path/'fit',tmp_path/'evaluation',tmp_path/'data'
    folder=data/'subject_208';write(folder/'prepared.json',dict(subject=208))
    fixed={refit.TRANSFER:.01};retention=refit.helpers()
    hashes={name:hashlib.sha256((states.ROOT/name).read_bytes()).hexdigest()
            for name in (states.REFIT,refit.ONLINE,refit.RETENTION,retention.MECHANISM)}
    settings=dict(belief_intervention=refit.INTERVENTION,fixed_coordinates=fixed,
        subjects=[208],arms={'POOL':{'precision':'fast'},'SEARCH':{'precision':'fast'}},
        mechanism={'kind':'E'},protocol='fixture',seed=19,particle_overrides={},budgets={'evaluate':{'seeds':8,'particles':4}})
    nominees=[dict(subject=208,model=arm,candidate=dict(id=arm,hyperparams=fixed))for arm in settings['arms']]
    common=dict(settings=settings,source_sha256=hashes,smoke=False)
    write(fit/'protocol.json',common)
    write(fit/'nominees.json',dict(common,status='frozen_before_test',data=str(data),nominees=nominees))
    nomination=hashlib.sha256((fit/'nominees.json').read_bytes()).hexdigest()
    write(fit/'completion.json',dict(status='complete',test_choices_used=False,smoke=False,nomination_sha256=nomination))
    write(fit/'online_refit_protocol.json',dict(belief_intervention=refit.INTERVENTION,input_sha256={},fixed_coordinates=fixed))
    jobs=[]
    for row in nominees:
        for i in range(8):
            arm=row['model'];name=f'evaluate_S208_{arm}_{arm}_C{i:03d}'
            seed=stable_seed(dict(protocol='fixture',base=19,stage='evaluate',subject=208,repeat=i))
            target=source/'conditional'/name
            plan=dict(name=name,phase='evaluate',model=arm,candidate=row['candidate'],smoke=False,seeds=[seed],
                particles=4,folder=str(folder),arm_settings=settings['arms'][arm],mechanism=settings['mechanism'],
                source_sha256=hashes,target=str(target))
            actual=dict(plan,structural_intervention='retained_rule_precision',retain_precision=True,
                retention_reference_inputs={},belief_intervention=refit.INTERVENTION,online_belief=True,fixed_coordinates=fixed)
            write(source/'jobs'/(name+'.json'),actual)
            write(target.with_suffix('.json'),dict(trials=8))
            write(target.with_suffix('.precision.json'),dict(retained_precision=True,structural_intervention='retained_rule_precision'))
            write(target.with_suffix('.online.json'),dict(enabled=True,belief_intervention=refit.INTERVENTION,
                fixed_coordinates=fixed,exact_transition_timing=True,evidence_matches_retained_state=True,reconstructions=28))
            jobs.append(plan)
    write(source/'protocol.json',dict(fit=str(fit),source_sha256=hashes,nomination_sha256=nomination,jobs=jobs))
    write(source/'completion.json',dict(status='complete',jobs=16,nomination_sha256=nomination))
    monkeypatch.setattr(refit,'reference_inputs',lambda *args:{})
    return dict(subjects=[208],arms=['POOL','SEARCH'],seeds_per_arm=8),source,fit


def test_selects_all_original_seeds_and_declared_smoke_subset(tmp_path,monkeypatch):
    settings,source,_=source_fixture(tmp_path,monkeypatch)
    files,metadata=states.source_jobs(settings,source,False)
    assert len(files)==16 and metadata['parameter_estimation_scope']=='prefix'
    assert len(metadata['selected_parameters'])==2
    assert len(states.source_jobs(settings,source,True)[0])==1


def test_rejects_fit_that_used_later_choices(tmp_path,monkeypatch):
    settings,source,fit=source_fixture(tmp_path,monkeypatch)
    path=fit/'completion.json';value=json.loads(path.read_text());value['test_choices_used']=True;write(path,value)
    with pytest.raises(ValueError,match='training-only'):
        states.source_jobs(settings,source,False)


def test_rejects_reduced_filter_bank(tmp_path,monkeypatch):
    settings,source,_=source_fixture(tmp_path,monkeypatch);settings['seeds_per_arm']=4
    with pytest.raises(ValueError,match='every original'):
        states.source_jobs(settings,source,False)


def test_oral_context_preserves_fixed_coordinate_and_search_policy():
    protocol=dict(parameter_estimation_scope='prefix',evaluation_use=states.USE,
        belief_intervention=refit.INTERVENTION,structural_intervention='retained_rule_precision',
        parameter_candidate_policy='projected_common_search_policy_behavior_refit',online_fit='fit',
        refit_nomination_sha256='digest',fixed_coordinates={refit.TRANSFER:.01},selected_parameters=[{'parameter_id':'fixed'}])
    metadata,note=states.estimation_context(protocol)
    assert metadata['fixed_coordinates']==protocol['fixed_coordinates'] and '投影' in note
    protocol['parameter_estimation_scope']='full'
    with pytest.raises(ValueError):states.estimation_context(protocol)

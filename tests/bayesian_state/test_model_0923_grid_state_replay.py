"""Behavior-only grid selection binds exact saved filter jobs for state replay."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from src.Bayesian_state.workflows.analysis import model_0923_decision_basis_replay as replay
from src.Bayesian_state.workflows.analysis.model_0923_decision_basis_alignment import align_identities, estimation_context


def save(path,value):
    path.write_text(json.dumps(value))


@pytest.fixture
def grid(tmp_path):
    source=tmp_path/'source';reference=tmp_path/'reference';folder=tmp_path/'data'/'subject_208'
    (source/'jobs').mkdir(parents=True);(reference/'jobs').mkdir(parents=True);folder.mkdir(parents=True)
    save(folder/'prepared.json',dict(subject=208))
    plans=[]
    for point,reused,gamma in [('old',True,0.),('new',False,.8)]:
        candidate=dict(id=point,hyperparams={'gamma':gamma})
        job=dict(phase='evaluate',name=f'evaluate_S208_SEARCH_{point}',folder=str(folder),
            runtime='saved_runtime',model='SEARCH',mechanism={'kind':'E'},arm_settings={'precision':'fast'},
            candidate=candidate,seeds=[5,7],particles=512,source_sha256={'core':'abc','grid':'def'},smoke=False)
        chunks=[]
        for index,seed in enumerate(job['seeds']):
            chunk=dict(deepcopy(job),seeds=[seed],name=job['name']+f'_C{index:03d}')
            if reused:chunk['source_sha256']={'core':'abc'}
            save((reference if reused else source)/'jobs'/(chunk['name']+'.json'),chunk)
            if not reused:chunks.append(chunk)
        plans.append(dict(candidate=candidate,job=job,chunks=chunks,reused=reused))
    save(source/'protocol.json',dict(smoke=False,probe='unused',reference=str(reference),plans=plans,
        particles=512,seeds=[5,7]))
    rows=[]
    for phase in ['prefix','full']:
        for point in ['old','new']:
            rows.append(dict(subject=208,model='SEARCH',phase=phase,point=point,anchor=point=='old',particles=512,seeds=2,
                choice_nll=.8 if (phase=='full')==(point=='new') else 1.,choice_nll_mc_se=.001))
    pd.DataFrame(rows).to_csv(source/'scores.csv',index=False)
    save(source/'completion.json',dict(status='complete',smoke=False,subject=208,points=2,
        artifacts_sha256={'scores.csv':replay.digest(source/'scores.csv')}))
    settings=dict(source_kind='capacity_memory_scores',estimation_objective='full',
        subjects=[208],arms=['SEARCH'],seeds_per_arm=2)
    return settings,source


def test_grid_selects_declared_behavior_objective_and_records_provenance(grid):
    settings,source=grid
    files,metadata=replay.source_jobs(settings,source,False)
    assert len(files)==2 and all('_new_' in f.name for f in files)
    assert metadata['selected_parameter_id']=='new'
    assert metadata['selection_score']==.8
    assert metadata['selection_score_sha256']==replay.digest(source/'scores.csv')
    assert metadata['parameter_candidate_policy']=='fixed_capacity_memory_grid'
    assert metadata['evaluation_use']=='development_retrospective_comparison'
    assert not metadata['selected_original_point']
    assert len(replay.source_jobs(settings,source,True)[0])==1


def test_original_grid_winner_reuses_its_original_conditional_jobs(grid):
    settings,source=grid
    files,metadata=replay.source_jobs(dict(settings,estimation_objective='prefix'),source,False)
    assert len(files)==2 and all('_old_' in f.name and f.parent.parent.name=='reference' for f in files)
    assert metadata['selected_original_point']
    assert metadata['reused_reference_filters']


@pytest.mark.parametrize('change',['parameter','seed','particles','subject','mechanism','arm_settings','source'])
def test_reused_reference_must_match_grid_parameter_and_numerical_settings(grid,change):
    settings,source=grid
    settings=dict(settings,estimation_objective='prefix')
    reference=source.parent/'reference'
    file=reference/'jobs'/'evaluate_S208_SEARCH_old_C001.json';job=replay.read(file)
    if change=='parameter':job['candidate']['hyperparams']['gamma']=.2
    elif change=='seed':job['seeds']=[5]
    elif change=='particles':job['particles']=256
    elif change=='subject':save(source.parent/'data'/'subject_208'/'prepared.json',dict(subject=207))
    elif change=='mechanism':job['mechanism']={'kind':'R1'}
    elif change=='arm_settings':job['arm_settings']={'precision':'base'}
    elif change=='source':job['source_sha256']={'core':'changed'}
    save(file,job)
    with pytest.raises(ValueError):replay.source_jobs(settings,source,False)


@pytest.mark.parametrize('change',['incomplete','score_changed','missing_point','duplicate_point','nan',
    'wrong_budget','missing_seed','changed_chunk','smoke','wrong_objective'])
def test_incomplete_or_changed_grid_selection_is_rejected(grid,change):
    settings,source=grid
    done=replay.read(source/'completion.json');scores=pd.read_csv(source/'scores.csv')
    if change=='incomplete':done['status']='running'
    elif change=='score_changed':scores.loc[0,'choice_nll']=9.
    elif change=='missing_point':scores=scores[~(scores.point.eq('new')&scores.phase.eq('full'))]
    elif change=='duplicate_point':scores=pd.concat([scores,scores.iloc[[-1]]])
    elif change=='nan':scores.loc[scores.phase.eq('full'),'choice_nll']=float('nan')
    elif change=='wrong_budget':settings['seeds_per_arm']=3
    elif change=='missing_seed':(source.parent/'reference'/'jobs'/'evaluate_S208_SEARCH_old_C001.json').unlink();settings['estimation_objective']='prefix'
    elif change=='changed_chunk':
        file=source/'jobs'/'evaluate_S208_SEARCH_new_C001.json';job=replay.read(file);job['particles']=256;save(file,job)
    elif change=='smoke':done['smoke']=True
    elif change=='wrong_objective':settings['estimation_objective']='oral'
    scores.to_csv(source/'scores.csv',index=False)
    if change!='score_changed':done['artifacts_sha256']['scores.csv']=replay.digest(source/'scores.csv')
    save(source/'completion.json',done)
    with pytest.raises(ValueError):replay.source_jobs(settings,source,False)


def test_grid_oral_context_describes_its_actual_candidate_policy():
    metadata,note=estimation_context(dict(parameter_estimation_scope='full',
        evaluation_use='development_retrospective_comparison',parameter_candidate_policy='fixed_capacity_memory_grid'))
    assert metadata['parameter_candidate_policy']=='fixed_capacity_memory_grid'
    assert '容量与信念保持率网格' in note and '完整行为序列' in note and '回顾性开发诊断' in note
    assert '同时使用前段与全程目标' not in note


def identity_fixture(tmp_path, *, smoke=True, wrong_split=False, wrong_stimulus=False, complete=False):
    n=12 if complete else 8
    frame=pd.DataFrame(dict(iSub=[208]*12,iSession=[1]*12,iTrial=np.arange(1,13),
        choice=np.arange(12)%4+1,feedback=np.arange(12)%2))
    for i in range(1,5):frame[f'feature{i}']=np.arange(12)/20+i/10
    data=tmp_path/'learning.csv';frame.to_csv(data,index=False)
    # Prepared arrays use the parsed source values, including CSV roundoff.
    frame=pd.read_csv(data)
    stimulus=frame[[f'feature{i}' for i in range(1,5)]].to_numpy()[:n]
    if wrong_stimulus:stimulus[0,0]=.99
    for name,start,stop in [('training.npz',0,4),('sealed_test.npz',4,n)]:
        np.savez(tmp_path/name,stimulus=stimulus[start:stop],choices=frame.choice.to_numpy()[start:stop],
                 feedback=frame.feedback.to_numpy()[start:stop])
    save(tmp_path/'prepared.json',dict(subject=208,smoke=smoke,split='wrong' if wrong_split else 'eight_trial_smoke_fixture',
        trials=n,training_trials=4,paths={'learning_data':str(data)},learning_data_sha256=replay.digest(data),
        array_sha256={name:replay.digest(tmp_path/name) for name in ('training.npz','sealed_test.npz')}))
    return frame[['iSub','iSession','iTrial','choice']],dict(folder=str(tmp_path)),n


@pytest.mark.parametrize('complete',[False,True])
def test_oral_identity_join_accepts_explicit_smoke_and_unchanged_full_sequences(tmp_path,complete):
    oral,job,n=identity_fixture(tmp_path,smoke=not complete,complete=complete)
    joined,cut=align_identities(oral,job,n)
    assert cut==4 and len(joined)==n
    np.testing.assert_array_equal(joined.state_trial,np.arange(n))
    np.testing.assert_array_equal(joined.iTrial,np.arange(1,n+1))


@pytest.mark.parametrize('problem',['unmarked','wrong_split','wrong_stimulus'])
def test_short_oral_fixture_cannot_hide_length_or_stimulus_mismatches(tmp_path,problem):
    oral,job,n=identity_fixture(tmp_path,smoke=problem!='unmarked',
        wrong_split=problem=='wrong_split',wrong_stimulus=problem=='wrong_stimulus')
    with pytest.raises((ValueError,AssertionError)):
        align_identities(oral,job,n)

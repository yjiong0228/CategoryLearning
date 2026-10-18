"""Bind a retained-precision state replay to the actual behavior refit and jobs."""
from copy import deepcopy
import json

import pytest

from src.Bayesian_state.utils.seeding import stable_seed
from src.Bayesian_state.workflows.analysis import model_0923_decision_basis_replay as replay
from src.Bayesian_state.workflows.analysis import model_0923_reentry_precision_refit as refit
from src.Bayesian_state.workflows.analysis.model_0923_decision_basis_alignment import estimation_context


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def retained_evaluation(tmp_path, monkeypatch):
    source, fit, data = (tmp_path/name for name in ('evaluation', 'fit', 'data'))
    folder = data/'subject_208'
    save(folder/'prepared.json', dict(subject=208))
    reference = tmp_path/'reset_reference.json'; save(reference, dict(status='complete'))
    references = {str(reference): replay.digest(reference)}
    monkeypatch.setattr(refit, 'helpers', lambda: None)
    monkeypatch.setattr(refit, 'reference_inputs', lambda helper, settings: deepcopy(references))
    sources = {name: replay.digest(replay.ROOT/name) for name in (replay.REENTRY, replay.REENTRY_REFIT)}
    settings = dict(protocol='fixed_namespace', seed=70, subjects=[208],
        structural_intervention=refit.INTERVENTION, mechanism={'kind':'E'},
        arms={arm:{'precision':'fast'} for arm in ('POOL','SEARCH')},
        budgets={'evaluate':{'particles':512,'seeds':2}}, particle_overrides={})
    rows = [dict(subject=208, model=arm, candidate=dict(id=arm.lower(), hyperparams={'gamma':.8}))
            for arm in settings['arms']]
    frozen = dict(status='frozen_before_test', smoke=False, settings=settings,
                  source_sha256=sources, data=str(data), nominees=rows)
    save(fit/'nominees.json', frozen)
    save(fit/'protocol.json', dict(smoke=False, settings=settings, source_sha256=sources))
    save(fit/'completion.json', dict(status='complete', smoke=False, test_choices_used=False,
                                     nomination_sha256=replay.digest(fit/'nominees.json')))
    save(fit/'retention_protocol.json', dict(structural_intervention=refit.INTERVENTION, input_sha256=references))
    plans = []
    for row in rows:
        arm, candidate = row['model'], row['candidate']
        for i in range(2):
            name = f'evaluate_S208_{arm}_{candidate["id"]}_C{i:03d}'
            plan = dict(phase='evaluate', name=name, folder=str(folder), model=arm, candidate=candidate,
                seeds=[stable_seed(dict(protocol=settings['protocol'],base=settings['seed'],stage='evaluate',subject=208,repeat=i))],
                particles=512, source_sha256=sources, smoke=False, arm_settings=settings['arms'][arm],
                mechanism=settings['mechanism'], runtime='archived', target=str(source/'arrays'/name))
            plans.append(plan)
            save(source/'jobs'/(name+'.json'), dict(deepcopy(plan), structural_intervention=refit.INTERVENTION,
                                                   retain_precision=True, retention_reference_inputs=references))
            save(source/'arrays'/(name+'.precision.json'), dict(structural_intervention=refit.INTERVENTION,
                                                               retained_precision=True, transitions=16, restores=3))
    save(source/'protocol.json', dict(fit=str(fit), nomination_sha256=replay.digest(fit/'nominees.json'),
                                     source_sha256=sources, jobs=plans))
    save(source/'completion.json', dict(status='complete', jobs=4, nomination_sha256=replay.digest(fit/'nominees.json')))
    request = dict(source_kind='reentry_precision_refit', subjects=[208], arms=['POOL','SEARCH'], seeds_per_arm=2)
    return request, source, fit


def test_both_refitted_arms_keep_frozen_points_and_actual_retention(retained_evaluation):
    request, source, fit = retained_evaluation
    files, metadata = replay.source_jobs(request, source, False)
    assert len(files)==4
    assert {replay.read(file)['model'] for file in files}=={'POOL','SEARCH'}
    assert metadata['parameter_estimation_scope']=='prefix'
    assert metadata['parameter_candidate_policy']=='equal_budget_behavior_refit'
    assert metadata['selected_parameters']==[dict(subject=208,arm='POOL',parameter_id='pool'),
                                             dict(subject=208,arm='SEARCH',parameter_id='search')]
    assert str(fit/'nominees.json') in metadata['refit_replay_inputs']
    context, note = estimation_context(metadata)
    assert context['structural_intervention']==refit.INTERVENTION
    assert context['retained_fit']==str(fit)
    assert '重新估参的共同作用' in note and '口述只用于冻结参数后的表征检验' in note
    assert len(replay.source_jobs(request, source, True)[0])==1
    assert len(replay.source_jobs(dict(request,arms=['SEARCH']), source, False)[0])==2


@pytest.mark.parametrize('fault', [
    'unfinished_fit','test_used','smoke_fit','nomination_checksum','settings','source_checksum',
    'intervention','missing_mechanism','reset_inputs','wrong_subject','unsupported_arm','seed_count',
    'duplicate_nominee','missing_nominee','duplicate_plan','job_count','missing_job','wrong_prepared_subject',
    'changed_point','changed_seed','changed_particles','changed_arm_settings','changed_mechanism',
    'disabled_retention','numeric_retention_flag','untagged_job','job_parameter','precision_record'])
def test_refitted_source_rejects_misbound_or_incomplete_science(retained_evaluation, fault):
    request, source, fit = retained_evaluation
    evaluation = replay.read(source/'protocol.json'); completion = replay.read(source/'completion.json')
    frozen = replay.read(fit/'nominees.json'); protocol = replay.read(fit/'protocol.json')
    done = replay.read(fit/'completion.json'); retention = replay.read(fit/'retention_protocol.json')
    plan = evaluation['jobs'][0]; file = source/'jobs'/(plan['name']+'.json'); job = replay.read(file)
    if fault=='unfinished_fit': done['status']='running'
    elif fault=='test_used': done['test_choices_used']=True
    elif fault=='smoke_fit': done['smoke']=protocol['smoke']=frozen['smoke']=True
    elif fault=='nomination_checksum': completion['nomination_sha256']='changed'
    elif fault=='settings': protocol['settings']['seed']+=1
    elif fault=='source_checksum': evaluation['source_sha256'][replay.REENTRY]='changed'
    elif fault=='intervention': retention['structural_intervention']='RESET'
    elif fault=='missing_mechanism':
        for obj in (frozen,protocol,evaluation): obj['source_sha256'].pop(replay.REENTRY)
    elif fault=='reset_inputs': retention['input_sha256']={}
    elif fault=='wrong_subject': request['subjects']=[306]
    elif fault=='unsupported_arm': request['arms']=['BASE']
    elif fault=='seed_count': request['seeds_per_arm']=1
    elif fault=='duplicate_nominee': frozen['nominees'].append(deepcopy(frozen['nominees'][0]))
    elif fault=='missing_nominee': frozen['nominees']=frozen['nominees'][1:]
    elif fault=='duplicate_plan': evaluation['jobs'].append(deepcopy(plan));completion['jobs']+=1
    elif fault=='job_count': completion['jobs']+=1
    elif fault=='missing_job': file.unlink()
    elif fault=='wrong_prepared_subject': save(fit.parent/'data/subject_208/prepared.json',dict(subject=207))
    elif fault=='changed_point': plan['candidate']['hyperparams']['gamma']=.2
    elif fault=='changed_seed': plan['seeds']=[1]
    elif fault=='changed_particles': plan['particles']=256
    elif fault=='changed_arm_settings': plan['arm_settings']['precision']='base'
    elif fault=='changed_mechanism': plan['mechanism']['kind']='R1'
    elif fault=='disabled_retention': job['retain_precision']=False
    elif fault=='numeric_retention_flag': job['retain_precision']=1
    elif fault=='untagged_job': job.pop('structural_intervention')
    elif fault=='job_parameter': job['candidate']['hyperparams']['gamma']=.2
    elif fault=='precision_record': save(source/'arrays'/(plan['name']+'.precision.json'),dict(retained_precision=False))
    if fault.startswith('changed_'):
        job={**deepcopy(plan),**{key:job[key] for key in ('retain_precision','structural_intervention','retention_reference_inputs')}}
    save(fit/'nominees.json',frozen)
    # Preserve the new file's binding so the intended semantic check is exercised.
    done['nomination_sha256']=evaluation['nomination_sha256']=replay.digest(fit/'nominees.json')
    if fault!='nomination_checksum':completion['nomination_sha256']=done['nomination_sha256']
    for path,value in [(fit/'protocol.json',protocol),(fit/'completion.json',done),
                       (fit/'retention_protocol.json',retention),(source/'protocol.json',evaluation),
                       (source/'completion.json',completion)]:save(path,value)
    if fault!='missing_job':save(file,job)
    with pytest.raises(ValueError):replay.source_jobs(request,source,False)


def test_retained_evaluation_cannot_use_unlabelled_default_source(retained_evaluation):
    request,source,_=retained_evaluation;request.pop('source_kind')
    with pytest.raises(ValueError,match='explicit state-replay source_kind'):
        replay.source_jobs(request,source,False)


def test_worker_rechecks_its_frozen_refit_inputs(retained_evaluation):
    request,source,fit=retained_evaluation
    files,metadata=replay.source_jobs(request,source,False)
    runtime=fit/'runtime';pf=runtime/replay.PF;pf.parent.mkdir(parents=True);pf.write_text('archived fixture')
    reference=source/'reference.npz';reference.write_bytes(b'probability fixture')
    job=dict(source_sha256={},replay_source_sha256={},runtime=str(runtime),archived_pf_sha256=replay.digest(pf),
             reference=str(reference),reference_sha256=replay.digest(reference),original_job=str(files[0]),
             original_job_sha256=replay.digest(files[0]),refit_replay_inputs=metadata['refit_replay_inputs'])
    replay.verify_sources(job)
    file=fit/'nominees.json';file.write_text(file.read_text()+' ')
    with pytest.raises(ValueError,match='binding changed'):replay.verify_sources(job)


@pytest.mark.parametrize('key,value', [('parameter_estimation_scope','full'),
    ('parameter_candidate_policy','declared_point_no_new_selection'),('structural_intervention','RESET'),
    ('retained_fit',''),('refit_nomination_sha256','')])
def test_oral_scope_requires_complete_refitted_intervention_provenance(retained_evaluation,key,value):
    request,source,_=retained_evaluation
    _,metadata=replay.source_jobs(request,source,False);metadata[key]=value
    with pytest.raises(ValueError,match='provenance'):estimation_context(metadata)

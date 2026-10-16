"""End-to-end score aggregation, data boundaries, and artifact integrity."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from src.Bayesian_state.evaluation.model_0923_precision_refit import summarize
from src.Bayesian_state.workflows.analysis.model_0923_h_validation import write, digest
from src.Bayesian_state.workflows.analysis.model_0923_precision_refit import RATE, INIT


def save_array(path, **arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:np.savez_compressed(stream,**arrays)
    return digest(path)


@pytest.fixture
def fixture_run(tmp_path):
    fit=tmp_path/'fit';fit.mkdir();evaluation=tmp_path/'evaluation';evaluation.mkdir()
    arms=['ADD1','ADD10','LOG25','LOG100'];nominees=[];groups=[];conditional=[];autonomous=[]
    categories=np.tile([1,2],32);choices=categories.copy();choices[::4]=3-choices[::4]
    # Different seed probabilities make log(mean p) differ from mean(log p).
    p=np.tile(np.array([.4,.6,.8])[:,None],(1,64));p[:,32:]*=.5
    generated=np.tile(categories,(4,1))
    for i in range(4):generated[i,i::4]=3-generated[i,i::4]
    for model in arms:
        finals=[]
        for index in range(2):
            point=dict(id=f'p{index}',hyperparams={RATE:.1,INIT:5.},sources=['test_fixture'])
            for stage in ('screen','refine','selection','confirmation'):
                f=fit/stage/'subject_101'/model/(point['id']+'.npz')
                seeds=[1,2,3] if stage=='confirmation' else [4,5,6]
                sha=save_array(f,observed_probability=p[:,:32]*.99**index,seeds=seeds)
                row=dict(subject=101,model=model,candidate=point,predictions_sha256=sha,
                    particles=8,seeds=seeds,numeric_pools=[dict(num_threads=1)],trials=32,cpu_seconds=.1)
                write(f.with_suffix('.json'),row)
                chunk=f.parent/'chunks'/f'{point["id"]}.json';chunk.parent.mkdir(exist_ok=True)
                write(chunk,row)
            finals.append(dict(point,mean_nll=float(-np.log(p[:,:32].mean(axis=0)*.99**index).mean()),
                score_file=str(f.with_suffix('.json'))))
        nominees.append(dict(subject=101,model=model,candidate=finals[0],finalists=finals,reference_id='p1'))
        groups.append(dict(subject=101,model=model,rows=finals))
        base=dict(subject=101,model=model,candidate=finals[0],condition=1,training_trials=32,particles=8)
        f=evaluation/'conditional/evaluate/subject_101'/model/'p0.npz'
        sha=save_array(f,observed_probability=p,correct_probability=p,seeds=[7,8,9],choices=choices,categories=categories)
        conditional.append(dict(base,predictions_sha256=sha))
        f=evaluation/'autonomous/evaluate/subject_101'/model/'p0.npz'
        sha=save_array(f,choices=generated,categories=categories,seeds=[10,11,12,13])
        autonomous.append(dict(base,predictions_sha256=sha))
    for stage in ('screen','refine','selection','confirmation'):
        write(fit/(stage+'_completion.json'),dict(status='complete',groups=groups))
    write(fit/'nominees.json',dict(status='frozen_before_test',smoke=False,nominees=nominees,
        settings=dict(subjects=[101],mechanism=dict(kind='E',capacity='evidence_mean'),
                      arms={m:dict(rate_scale=10. if m=='ADD10' else 1.,upper=100. if m=='LOG100' else 25.,rates=[0.,.1,1.]) for m in arms},
                      autonomous_paths=4,budgets=dict(evaluate=dict(seeds=3)))))
    write(evaluation/'protocol.json',dict(nomination_sha256=digest(fit/'nominees.json')))
    write(evaluation/'completion.json',dict(status='complete'))
    for role,rows in [('conditional',conditional),('autonomous',autonomous)]:
        write(evaluation/role/'evaluation.json',dict(status='complete',rows=rows))
    return fit,evaluation,tmp_path/'summary'


def test_pool_before_log_and_keep_time_boundary(fixture_run):
    summarize(*fixture_run)
    output=fixture_run[-1];scores=pd.read_csv(output/'scores.csv')
    assert len(scores)==12
    np.testing.assert_allclose(scores.loc[scores.phase=='training','choice_nll'],-np.log(.6))
    np.testing.assert_allclose(scores.loc[scores.phase=='test','choice_nll'],-np.log(.3))
    assert len(pd.read_csv(output/'paths.csv'))==16
    contrasts=pd.read_csv(output/'contrasts.csv')
    np.testing.assert_allclose(contrasts[['delta_choice_nll','delta_energy','delta_crps']],0.)
    np.testing.assert_allclose(contrasts[['delta_choice_nll_mc_se','delta_energy_mc_se','delta_crps_mc_se']],0.)
    budgets=pd.read_csv(output/'budgets.csv')
    assert len(budgets)==16 and (budgets.state_updates==1536).all()
    repeatability=pd.read_csv(output/'training_repeatability.csv')
    np.testing.assert_allclose(repeatability.delta_nll,0.)
    np.testing.assert_allclose(repeatability.independent_mc_se,
        np.hypot(repeatability.confirmation_mc_se,repeatability.repeated_mc_se))
    additive=pd.read_csv(output/'additive_search_comparison.csv')
    assert additive.add10_point_feasible_in_add1.all()
    cohort=pd.read_csv(output/'cohort_scores.csv')
    np.testing.assert_allclose(cohort.loc[cohort.phase=='training','participant_mean_choice_nll'],-np.log(.6))
    np.testing.assert_allclose(cohort.loc[cohort.phase=='test','trial_weighted_choice_nll'],-np.log(.3))


def test_changed_prediction_archive_is_rejected(fixture_run):
    fit,evaluation,output=fixture_run
    f=evaluation/'autonomous/evaluate/subject_101/LOG25/p0.npz'
    f.write_bytes(f.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='predictions changed'):summarize(fit,evaluation,output)
    assert not output.exists()


def test_declared_block_width_controls_curves(fixture_run):
    fit,evaluation,output=fixture_run
    file=fit/'nominees.json';frozen=json.loads(file.read_text())
    frozen['settings']['features']={'block_width':16};file.write_text(json.dumps(frozen))
    (evaluation/'protocol.json').write_text(json.dumps(dict(nomination_sha256=digest(file))))
    summarize(fit,evaluation,output)
    curves=pd.read_csv(output/'curves.csv')
    assert len(curves)==16
    assert set(curves.trial)=={16,32,48,64}


def make_rate_profile(fit):
    profile=fit.parent/'profile';profile.mkdir()
    frozen=json.loads((fit/'nominees.json').read_text())
    reference=next(r['candidate'] for r in frozen['nominees'] if r['model']=='ADD10')
    write(profile/'protocol.json',dict(audit=dict(subject=101,model='ADD10'),
        reference_id=reference['id'],reference_parameters=reference['hyperparams']))
    pd.DataFrame([dict(subject=101,model='ADD10',reference=True,effective_rate=1.,choice_nll=-np.log(.6))]).to_csv(profile/'rate_profile.csv',index=False)
    record=dict(status='complete',smoke=False,points=1,nomination_sha256=digest(fit/'nominees.json'),
        protocol_sha256=digest(profile/'protocol.json'),summary_sha256=digest(profile/'rate_profile.csv'),inputs_sha256={})
    write(profile/'completion.json',record)
    return profile,record


def test_rate_profile_is_preserved_with_provenance(fixture_run):
    fit,evaluation,output=fixture_run;profile,_=make_rate_profile(fit)
    summarize(fit,evaluation,output,profile)
    assert (profile/'rate_profile.csv').read_bytes()==(output/'rate_profile.csv').read_bytes()
    manifest=json.loads((output/'manifest.json').read_text())
    assert manifest['inputs_sha256'][str(profile/'completion.json')]==digest(profile/'completion.json')


def test_rate_profile_from_other_nomination_is_rejected(fixture_run):
    fit,evaluation,output=fixture_run;profile,record=make_rate_profile(fit)
    record['nomination_sha256']='changed'
    (profile/'completion.json').write_text(json.dumps(record))
    with pytest.raises(ValueError,match='Rate profile used different fits'):summarize(fit,evaluation,output,profile)
    assert not output.exists()

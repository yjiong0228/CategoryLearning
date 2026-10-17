"""Complete-path scoring, temporal boundaries and archive-integrity checks."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from src.Bayesian_state.evaluation.model_0923_episodic_admission import compare_score_runs,read_run,summarize
from src.Bayesian_state.workflows.analysis.model_0923_h_validation import write,digest


def fixture(tmp_path):
    data=tmp_path/'subject';data.mkdir();hashes={}
    for name in ('training.npz','sealed_test.npz'):
        np.savez(data/name,choices=np.ones(64,int),categories=np.ones(64,int))
        hashes[name]=digest(data/name)
    write(data/'prepared.json',dict(training_trials=64,trials=128,condition=1,array_sha256=hashes))
    settings=dict(subjects=[101],arms=['BASE','POOL'],paths=2,score=dict(seeds=[31,32]),features=dict(block_width=32))
    for stage in ('autonomous','score'):
        root=tmp_path/stage;root.mkdir();jobs=[]
        for arm in settings['arms']:
            target=root/arm;candidate=dict(id='point',hyperparams={'same':1})
            job=dict(name=arm,subject=101,model=arm,target=str(target),candidate=candidate,particles=16,
                seeds=[31,32],folder=str(data),mechanism={'kind':'E'},precision={'precision':'fast'})
            if stage=='autonomous':
                choices=np.ones((2,128),int)
                choices[0,:64]=2
                if arm=='BASE':choices[1,:]=2
                p=np.full((2,128,2),.5)
                zero=np.zeros((2,128));present=(choices==1).astype(int)
                np.savez(target.with_suffix('.npz'),choices=choices,categories=np.ones(128,int),oracle=0,
                    probabilities=p,correct_probability=p[:,:,0],feedback=present,seeds=[31,32],
                    present=present,contribution=present*.8,proposed=np.full((2,128),-1),rejected=zero,
                    target_exited=zero,target_capacity_dropped=zero,active_count=zero+2)
            else:
                p=np.full((2,64),.4 if arm=='BASE' else .6)
                np.savez(target.with_suffix('.npz'),observed_probability=p,correct_probability=p,seeds=[31,32])
            write(target.with_suffix('.json'),dict(candidate=candidate,particles=16,
                predictions_sha256=digest(target.with_suffix('.npz'))))
            jobs.append(job)
        write(root/'protocol.json',dict(settings=settings,stage=stage,smoke=False,jobs=jobs,
            source_sha256={},nomination_sha256='frozen'))
        write(root/'completion.json',dict(status='complete',jobs=2,protocol_sha256=digest(root/'protocol.json')))
    return tmp_path/'autonomous',tmp_path/'score'


def test_full_path_energy_and_training_scores_are_independently_reproduced(tmp_path):
    auto,score=fixture(tmp_path);output=tmp_path/'summary'
    summarize(auto,[score],output)
    frame=pd.read_csv(output/'scores.csv').set_index('model')
    # Block vectors are [0,0,1,1] and [1,1,1,1] for POOL, y=[1,1,1,1].
    expected=.5*np.sqrt(2)/2 - .5*(2*np.sqrt(2)/4)/2
    assert frame.loc['POOL','energy']==pytest.approx(expected)
    tr=pd.read_csv(output/'training.csv').set_index('model')
    assert tr.loc['POOL','choice_nll']==pytest.approx(-np.log(.6))
    assert tr.loc['POOL','delta_nll']==pytest.approx(np.log(.4/.6))
    assert tr.loc['POOL','mc_se']==0.
    outcomes=pd.read_csv(output/'training_outcomes.csv')
    combined=outcomes.groupby('model').contribution.sum()
    np.testing.assert_allclose(combined.reindex(tr.index),tr.delta_nll,atol=1e-14,rtol=0)
    assert (outcomes[outcomes.outcome=='incorrect']['count']==0).all()
    assert len(pd.read_csv(output/'path_blocks.csv'))==16
    assert len(pd.read_csv(output/'paths.csv'))==4
    with pytest.raises(FileExistsError):summarize(auto,[score],output)


def test_changed_archive_is_rejected(tmp_path):
    auto,_=fixture(tmp_path)
    with (auto/'BASE.npz').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='array changed'):read_run(auto)


def test_error_and_correct_contributions_use_the_total_training_denominator(tmp_path):
    auto,score=fixture(tmp_path);data=tmp_path/'subject'
    np.savez(data/'training.npz',choices=np.r_[np.full(32,2),np.ones(32,int)],categories=np.ones(64,int))
    meta=json.loads((data/'prepared.json').read_text())
    meta['array_sha256']['training.npz']=digest(data/'training.npz')
    (data/'prepared.json').write_text(json.dumps(meta))
    for arm,values in [('BASE',[.9,.6]),('POOL',[.4,.8])]:
        p=np.broadcast_to(np.repeat(values,32),(2,64))
        file=score/(arm+'.npz');np.savez(file,observed_probability=p,correct_probability=p,seeds=[31,32])
        row=json.loads(file.with_suffix('.json').read_text());row['predictions_sha256']=digest(file)
        file.with_suffix('.json').write_text(json.dumps(row))
    summarize(auto,[score],tmp_path/'summary')
    rows=pd.read_csv(tmp_path/'summary'/'training_outcomes.csv').query("model == 'POOL'").set_index('outcome')
    assert rows.loc['incorrect','contribution']==pytest.approx(.5*np.log(.9/.4))
    assert rows.loc['correct','contribution']==pytest.approx(.5*np.log(.6/.8))
    assert (rows['count']==32).all()


def test_changed_training_boundary_is_rejected(tmp_path):
    auto,score=fixture(tmp_path);meta=tmp_path/'subject'/'prepared.json'
    values=json.loads(meta.read_text());values['training_trials']=32;meta.write_text(json.dumps(values))
    with pytest.raises(ValueError,match='boundary'):summarize(auto,[score],tmp_path/'summary')


def test_numerical_comparison_uses_independent_run_errors():
    common=dict(subject=101,model='POOL',particles=16,seeds=2)
    rows=[dict(common,run='first',choice_nll=.6,mc_se=.03,delta_nll=.1,delta_nll_mc_se=.003),
          dict(common,run='second',choice_nll=.5,mc_se=.04,delta_nll=.09,delta_nll_mc_se=.004)]
    seeds={('first',101,'POOL'):[1,2],('second',101,'POOL'):[3,4]}
    row=compare_score_runs(rows,seeds)[0]
    assert row['choice_nll_difference']==pytest.approx(-.1)
    assert row['choice_nll_difference_mc_se']==pytest.approx(.05)
    assert row['delta_nll_difference']==pytest.approx(-.01)
    assert row['delta_nll_difference_mc_se']==pytest.approx(.005)
    seeds['second',101,'POOL']=[2,3]
    with pytest.raises(ValueError,match='independent'):compare_score_runs(rows,seeds)

"""Numerical finalist audits retain the training boundary and seed pooling."""
from pathlib import Path

import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis.model_0923_h_training_audit import combine
from src.Bayesian_state.workflows.analysis.model_0923_h_validation import digest, read, score_jobs, write


def audit_chunks(tmp_path: Path) -> list[dict]:
    chunks=[]
    for index,p in enumerate([.1,.9]):
        target=tmp_path/f'chunk{index}'
        np.savez(target.with_suffix('.npz'),observed_probability=np.full((1,3),p),
                 correct_probability=np.full((1,3),p),seeds=[index+10])
        write(target.with_suffix('.json'),dict(seconds=2.,cpu_seconds=1.,
            predictions_sha256=digest(target.with_suffix('.npz'))))
        chunks.append(dict(target=str(target)))
    return chunks


def test_audit_pools_seed_probabilities_before_training_score(tmp_path):
    rows=audit_chunks(tmp_path)
    result=combine(dict(target=str(tmp_path/'combined'),seeds=[10,11],
        candidate=dict(id='frozen_candidate',hyperparams={})),rows)
    assert result['mean_nll']==pytest.approx(np.log(2))
    assert read(tmp_path/'combined.json')['role']=='training_only_high_particle_audit'
    with np.load(tmp_path/'combined.npz') as z:
        assert z['observed_probability'].shape==(2,3)
        np.testing.assert_array_equal(z['seeds'],[10,11])


def test_audit_rejects_changed_prediction_chunk(tmp_path):
    chunks=audit_chunks(tmp_path)
    Path(chunks[0]['target']).with_suffix('.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Audit chunk changed'):
        combine(dict(target=str(tmp_path/'combined'),seeds=[10,11],candidate={}),chunks)


def test_audit_scoring_jobs_cannot_open_test_phase(tmp_path):
    data=tmp_path/'data';data.mkdir()
    write(data/'preparation_manifest.json',dict(settings=dict(source_fit='unused_source')))
    settings=dict(protocol='training_audit_test',seed=7,
        budgets=dict(audit=dict(particles=256,seeds=4)),models=dict(B=dict(kind='B')))
    jobs=score_jobs(data,tmp_path/'output',settings,{(101,'B'):[dict(id='one',hyperparams={})]},'audit',False)
    assert len(jobs)==1
    assert jobs[0]['phase']=='training'
    assert jobs[0]['particles']==256 and len(jobs[0]['seeds'])==4

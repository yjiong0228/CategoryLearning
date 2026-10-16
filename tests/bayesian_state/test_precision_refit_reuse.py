"""Only complete, identical scientific jobs may enter a recovered screen."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis import model_0923_precision_refit_recovery as recovery


@pytest.fixture
def complete_job(tmp_path):
    v=recovery.base.validation();parent=tmp_path/'parent';output=tmp_path/'output'
    for root in (parent,output):(root/'jobs').mkdir(parents=True)
    src=parent/'screen/chunk';src.parent.mkdir();dst=output/'screen/chunk'
    scientific=dict(phase='training',folder='prepared',model='LOG100',mechanism={'kind':'E'},
        candidate={'id':'fixed','hyperparams':{'rate':.1}},runtime='archive',seeds=[7,8],particles=8,smoke=True,
        arm_settings={'precision':'log','upper':100.})
    old=dict(scientific,name='screen_example',target=str(src),source_sha256={'base.py':'unchanged'})
    new=dict(scientific,name='screen_example',target=str(dst),
             source_sha256={'base.py':'unchanged',recovery.GUARD:'new_guard',recovery.ENTRY:'new_entry'})
    v.write(parent/'jobs/screen_example.json',old)
    (parent/'jobs/screen_example.log').write_text('')
    with src.with_suffix('.npz').open('xb') as f:np.savez_compressed(f,observed_probability=np.ones((2,8))*.5,seeds=[7,8])
    v.write(src.with_suffix('.json'),dict(predictions_sha256=v.digest(src.with_suffix('.npz'))))
    v.write(src.with_suffix('.experiment.json'),dict(corrections=[]))
    return new,parent,output


def test_valid_reuse_is_bitwise_and_records_provenance(complete_job):
    job,parent,output=complete_job;v=recovery.base.validation()
    assert recovery.reuse_job(job,parent,output)
    assert v.digest(parent/'screen/chunk.npz')==v.digest(output/'screen/chunk.npz')
    assert v.read(output/'screen/chunk.snapshot.json')['corrections']==[]
    assert (output/'reused_jobs/screen_example.json').exists()


def test_changed_seed_or_algorithm_forbids_reuse(complete_job):
    job,parent,output=complete_job
    wrong=deepcopy(job);wrong['seeds']=[9,10]
    with pytest.raises(ValueError,match='scientific protocol'):recovery.reuse_job(wrong,parent,output)
    wrong=deepcopy(job);wrong['source_sha256']['base.py']='different'
    with pytest.raises(ValueError,match='implementation changed'):recovery.reuse_job(wrong,parent,output)
    assert not (output/'screen').exists()


def test_failed_or_incomplete_job_requires_recomputation(complete_job):
    job,parent,output=complete_job
    (parent/'jobs/screen_example.log').write_text('Traceback: failure')
    assert not recovery.reuse_job(job,parent,output)
    assert not (output/'screen').exists()


def test_prediction_hash_corruption_forbids_reuse(complete_job):
    job,parent,output=complete_job
    f=parent/'screen/chunk.npz';f.write_bytes(f.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='predictions changed'):recovery.reuse_job(job,parent,output)
    assert not (output/'screen').exists()

"""Reuse only verified retained-precision banks with their exact seed ordering."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis.model_0923_online_evidence_development import checked_source


def fixture(tmp_path):
    target = tmp_path / 'bank'
    np.savez(target.with_suffix('.npz'), seeds=np.array([9, 3]))
    np.savez(target.with_suffix('.precision.npz'), counts=np.zeros((2, 4), int))
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    job = dict(target=str(target),smoke=False,phase='autonomous',model='SEARCH',
               retain_precision=True,candidate={'id':'fixed'},seeds=[9,3])
    record = dict(candidate=job['candidate'],paths=2,model='SEARCH',
                  predictions_sha256=digest(target.with_suffix('.npz')))
    precision = dict(status='complete',retained_precision=True,
                     arrays_sha256=digest(target.with_suffix('.precision.npz')))
    for suffix, row in (('.json',record),('.precision.json',precision)):
        target.with_suffix(suffix).write_text(json.dumps(row))
    return job, SimpleNamespace(read=lambda p:json.loads(Path(p).read_text()),digest=digest)


def test_original_array_seed_order_is_preserved(tmp_path):
    job,v=fixture(tmp_path)
    assert checked_source(job,v)['paths']==2


@pytest.mark.parametrize('field,value',[('retain_precision',False),('smoke',True),
    ('model','POOL'),('candidate',{'id':'changed'}),('seeds',[3,9])])
def test_rejects_changed_cognitive_or_seed_identity(tmp_path,field,value):
    job,v=fixture(tmp_path)
    job=deepcopy(job);job[field]=value
    with pytest.raises((ValueError,AssertionError)):
        checked_source(job,v)

"""Reject reused trajectories with changed candidates, seeds or mechanisms."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis import model_0923_reentry_finalist_generation as workflow


@pytest.fixture
def completed_chunk(tmp_path):
    target = tmp_path / 'chunk'
    np.savez(target.with_suffix('.npz'), seeds=[11, 22], choices=np.ones((2, 16)))
    v = SimpleNamespace(read=lambda p: json.loads(Path(p).read_text()),
                        digest=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest())
    candidate = dict(id='fixed', hyperparams={'capacity': 4})
    row = dict(candidate=candidate, model='POOL', paths=2, trials=16,
               role='autonomous_from_start', numeric_pools=[dict(num_threads=1)],
               predictions_sha256=v.digest(target.with_suffix('.npz')))
    precision = dict(structural_intervention=workflow.retention.INTERVENTION,
                     retained_precision=True, event_unit='autonomous path transitions', transitions=32)
    target.with_suffix('.json').write_text(json.dumps(row))
    target.with_suffix('.precision.json').write_text(json.dumps(precision))
    job = dict(target=str(target), candidate=deepcopy(candidate), model='POOL', seeds=[11, 22],
               retain_precision=True, structural_intervention=workflow.retention.INTERVENTION)
    return job, v


def test_complete_retention_chunk_is_accepted(completed_chunk):
    job, v = completed_chunk
    assert workflow.checked_chunk(job, v)['paths'] == 2


@pytest.mark.parametrize('change', ['candidate', 'seed_order', 'reset', 'partial_execution', 'changed_array'])
def test_incompatible_chunk_reuse_is_rejected(completed_chunk, change):
    job, v = completed_chunk
    if change == 'candidate':
        job['candidate']['hyperparams']['capacity'] = 6
    elif change == 'seed_order':
        job['seeds'].reverse()
    elif change == 'reset':
        job['retain_precision'] = False
    elif change == 'partial_execution':
        file = Path(job['target']).with_suffix('.precision.json')
        row = v.read(file)
        row['transitions'] -= 1
        file.write_text(json.dumps(row))
    else:
        with Path(job['target']).with_suffix('.npz').open('ab') as stream:
            stream.write(b'changed')
    with pytest.raises((ValueError, AssertionError)):
        workflow.checked_chunk(job, v)

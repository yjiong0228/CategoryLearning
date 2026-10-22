"""Protect chronology, recovery blindness and full-behavior scope labels."""
from copy import deepcopy

import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis import model_1022_compact_scope_control as control


def test_full_fixture_preserves_values_and_declares_later_data_use(tmp_path):
    source=tmp_path/'original';source.mkdir()
    control.write(source/'engine.json',{'unchanged':'engine'})
    chunks=[]
    for start,stop,name in [(0,4,'training.npz'),(4,7,'sealed_test.npz')]:
        arrays=dict(stimulus=np.arange(start*4,stop*4).reshape(-1,4),
                    categories=np.arange(start,stop)%4,choices=np.arange(start,stop)%3,
                    feedback=np.arange(start,stop)%2)
        chunks.append(arrays)
        control.save_arrays(source,name,arrays)
    control.write(source/'prepared.json',dict(subject=205,trials=7,training_trials=4,test_trials=3,
        split='chronological_early_later',
        engine_sha256=control.digest(source/'engine.json'),
        array_sha256={n:control.digest(source/n) for n in ('training.npz','sealed_test.npz')}))
    joined,_=control.joined_arrays(source)
    for key in joined: np.testing.assert_array_equal(joined[key],np.concatenate([r[key] for r in chunks]))
    destination=tmp_path/'new'/'subject_205'
    control.save_fixture(source,destination,joined,'archive/source','observed_full_behavior_reconstruction')
    meta=control.read(destination/'prepared.json')
    assert (meta['training_trials'],meta['test_trials'],meta['original_training_trials'])==(7,0,4)
    assert meta['data_role']=='observed_full_behavior_reconstruction'
    assert meta['split']=='observed_full_behavior_reconstruction'
    assert meta['original_split']=='chronological_early_later'
    with np.load(destination/'training.npz') as z:
        for key in joined: np.testing.assert_array_equal(z[key],joined[key])
    with np.load(destination/'sealed_test.npz') as z: assert all(len(z[k])==0 for k in z.files)
    synthetic=tmp_path/'synthetic'/'subject_205'
    control.save_fixture(source,synthetic,chunks[0],'archive/source','synthetic_training_only')
    synthetic_meta=control.read(synthetic/'prepared.json')
    assert synthetic_meta['split']=='synthetic_training_only'
    assert synthetic_meta['original_split']=='chronological_early_later'
    assert (synthetic_meta['training_trials'],synthetic_meta['test_trials'])==(4,0)
    # Any alteration to an original array must stop preparation.
    (source/'sealed_test.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):control.joined_arrays(source)


def test_writer_scope_changes_only_full_human_records_and_restores_on_failure(tmp_path):
    human,recovered=tmp_path/'human',tmp_path/'recovery';human.mkdir();recovered.mkdir()
    cases=[dict(kind='human',output=str(human)),dict(kind='recovery',output=str(recovered))]
    original=control.refit.write
    record=dict(status='frozen_before_test',settings=dict(example=1))
    saved=deepcopy(record)
    with pytest.raises(RuntimeError,match='intentional'):
        with control.full_behavior_metadata(cases):
            control.refit.write(human/'nominees.json',record)
            control.refit.write(human/'completion.json',dict(status='complete',test_choices_used=False))
            control.refit.write(recovered/'nominees.json',dict(status='frozen_before_truth'))
            raise RuntimeError('intentional')
    assert control.refit.write is original and record==saved
    full=control.read(human/'nominees.json')
    assert full['status']=='frozen_before_external_oral_check'
    assert full['original_later_choices_used'] is True and full['oral_used_for_fitting'] is False
    assert control.read(human/'completion.json')['test_choices_used'] is True
    assert control.read(recovered/'nominees.json')==dict(status='frozen_before_truth')

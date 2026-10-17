"""Check complete CPU reservations and frozen source identities for early replay."""
from copy import deepcopy
from itertools import product
import json
from pathlib import Path

import pytest

from src.Bayesian_state.workflows.analysis import model_0923_decision_basis_stream as stream


@pytest.fixture
def plan(tmp_path):
    folder = tmp_path / 'participant'; folder.mkdir()
    (folder / 'prepared.json').write_text(json.dumps({'subject': 207}))
    settings = dict(sample_role='development', subjects=[207], arms=['POOL', 'SEARCH'], seeds_per_arm=2, workers=4)
    jobs = [dict(phase='evaluate', name=f'{arm}_{seed}', target=f'output/{arm}_{seed}',
                 folder=str(folder), model=arm, seeds=[seed]) for arm in settings['arms'] for seed in [3, 7]]
    return settings, dict(jobs=jobs)


def test_reservation_covers_pending_sources_and_replays(plan):
    settings, protocol = plan
    jobs = stream.planned_jobs(settings, protocol, smoke=False, cpus=128)
    assert len(jobs) == 4
    # Each pair is awaiting its source, replaying, or completed. A source and
    # its replay cannot both be active; every unfinished pair remains reserved.
    for states in product(['source', 'replay', 'done'], repeat=len(jobs)):
        reserved = len(jobs) - states.count('done')
        recovery_slots = 128 - reserved
        active_sources = states.count('source'); active_replays = states.count('replay')
        assert active_sources + active_replays + recovery_slots == 128
    assert stream.planned_jobs(settings, protocol, smoke=True, cpus=1) == jobs[:1]


@pytest.mark.parametrize('change', ['omit_arm', 'omit_seed', 'duplicate_seed', 'duplicate_target',
                                  'duplicate_name', 'duplicate_subject', 'too_few_workers', 'multiple_seeds'])
def test_incomplete_reservations_rejected(plan, change):
    settings, protocol = deepcopy(plan); jobs = protocol['jobs']
    if change == 'omit_arm': settings['arms'] = ['POOL']
    elif change == 'omit_seed': jobs.pop()
    elif change == 'duplicate_seed': jobs[1]['seeds'] = jobs[0]['seeds']
    elif change == 'duplicate_target': jobs[1]['target'] = jobs[0]['target']
    elif change == 'duplicate_name': jobs[1]['name'] = jobs[0]['name']
    elif change == 'duplicate_subject': settings['subjects'].append(207)
    elif change == 'too_few_workers': settings['workers'] = 3
    elif change == 'multiple_seeds': jobs[0]['seeds'] = [3, 7]
    with pytest.raises(ValueError): stream.planned_jobs(settings, protocol, smoke=False, cpus=128)


def test_available_cpu_limit(plan):
    with pytest.raises(ValueError): stream.planned_jobs(*plan, smoke=False, cpus=3)


@pytest.fixture
def reference(tmp_path, monkeypatch):
    monkeypatch.setattr(stream.replay, 'ROOT', tmp_path)
    monkeypatch.setattr(stream.replay, 'PF', 'pf.py')
    runtime = tmp_path / 'runtime'; runtime.mkdir(); (runtime / 'pf.py').write_text('archived worker')
    original = dict(name='saved_seed', phase='evaluate', folder='folder', model='SEARCH',
        candidate={'id': 'frozen', 'hyperparams': {'gamma': .8}}, runtime=str(runtime), seeds=[13],
        particles=512, source_sha256={}, arm_settings={'precision': 'fast'}, mechanism={'kind': 'E'}, smoke=False)
    file = tmp_path / 'job.json'; file.write_text(json.dumps(original))
    array = tmp_path / 'original.npz'; array.write_bytes(b'saved probabilities')
    array.with_suffix('.experiment.json').write_text('{}')
    record = dict(candidate=deepcopy(original['candidate']), seeds=[13], particles=512, model='SEARCH',
                  predictions_sha256=stream.replay.digest(array))
    array.with_suffix('.json').write_text(json.dumps(record))
    pending = dict(original, original_job=str(file), reference=str(array), replay_source_sha256={},
                   archived_pf_sha256=stream.replay.digest(runtime / 'pf.py'))
    return pending, record


def test_binding_keeps_frozen_worker_inputs(reference):
    pending, record = reference
    result = stream.bind_reference(pending)
    assert result['reference_sha256'] == record['predictions_sha256']
    assert result['original_job_sha256'] == stream.replay.digest(Path(pending['original_job']))
    for key, value in pending.items(): assert result[key] == value
    assert 'reference_sha256' not in pending


def test_reference_waits_for_final_marker(reference):
    pending, _ = reference; array = Path(pending['reference']); marker = array.with_suffix('.experiment.json')
    marker.unlink()
    assert not stream.reference_ready(array)
    marker.write_text('{')
    assert not stream.reference_ready(array)
    with pytest.raises(ValueError): stream.bind_reference(pending)
    marker.write_text('{}')
    assert stream.reference_ready(array)


@pytest.mark.parametrize('change', ['job_parameter', 'record_parameter', 'record_seed', 'record_particles', 'array'])
def test_changed_reference_rejected(reference, change):
    pending, record = reference; array = Path(pending['reference'])
    if change == 'job_parameter':
        file = Path(pending['original_job']); original = json.loads(file.read_text())
        original['candidate']['hyperparams']['gamma'] = .3; file.write_text(json.dumps(original))
    elif change == 'array': array.write_bytes(b'different probabilities')
    else:
        if change == 'record_parameter': record['candidate']['hyperparams']['gamma'] = .3
        elif change == 'record_seed': record['seeds'] = [17]
        elif change == 'record_particles': record['particles'] = 256
        array.with_suffix('.json').write_text(json.dumps(record))
    with pytest.raises(ValueError): stream.bind_reference(pending)

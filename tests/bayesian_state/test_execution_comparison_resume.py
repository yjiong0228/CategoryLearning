"""Continuation provenance, immutable artifacts and additional CPU accounting."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis.model_1024_execution_comparison_resume import (
    digest, prepare_continuation, write_new,
)


def source_run(tmp_path, autonomous=False):
    source, checkout = tmp_path / 'original', tmp_path / 'checkout'
    source.mkdir(); checkout.mkdir()
    (checkout / 'scientific.py').write_text('frozen source\n')
    fixture = tmp_path / 'fixture'
    fixture.mkdir()
    (fixture / 'training.npz').write_bytes(b'fixed source data')
    stages = dict(calibration=1., full=2., prefix=2., recovery=2., report=1.)
    protocol = dict(smoke=True, policy=dict(seed=42, protocol='original-seeds',
        search=dict(particles=16), budget=dict(wall_hours=1., cpu_hours=8., workers=1,
                                             stage_cpu_hours=stages)),
        sources={'scientific.py': digest(checkout / 'scientific.py')},
        region_sources={'engine': 'registered-engine'}, input_bindings={},
        cases=[dict(id='case', template=dict(folder=str(fixture)),
                    input_hashes={'training.npz': digest(fixture / 'training.npz')})])
    write_new(source / 'protocol.json', protocol)
    write_new(source / 'protocol_digest.json', dict(sha256=digest(source / 'protocol.json')))
    write_new(source / 'completion.json', dict(status='bounded_run_finished'))
    write_new(source / 'recovery_cases.json', dict(status='complete', cases=[]))
    # Opaque content verifies that preparation never parses the held-back truth.
    (source / 'recovery_truths.json').write_bytes(b'opaque uninspected truth bytes')
    candidate = dict(id='point', hyperparams={'kappa': 0})
    job = dict(name='good', candidate=candidate, particles=0 if autonomous else 16, seeds=[3, 4],
               expected_trials=3, phase='autonomous' if autonomous else 'training', region_sources=protocol['region_sources'],
               target=str(source / 'numerical/good'))
    (source / 'numerical').mkdir()
    arrays = dict(choices=np.ones((2, 3)), feedback=np.zeros((2, 3))) if autonomous else dict(
        observed_probability=np.full((2, 3), .5), correct_probability=np.full((2, 3), .5))
    np.savez(source / 'numerical/good.npz', seeds=[3, 4], **arrays)
    meta = dict(candidate=candidate, trials=3, predictions_sha256=digest(source / 'numerical/good.npz'))
    meta.update(dict(paths=2) if autonomous else dict(particles=16, seeds=[3, 4]))
    write_new(source / 'numerical/good.json', meta)
    write_new(source / 'jobs/good.json', job)
    write_new(source / 'jobs/good.usage.json', dict(status='complete', source_sha256=protocol['region_sources']))
    (source / 'jobs/good.log').write_text('worker completed\n')
    write_new(source / 'ledger/good.json', dict(name='good', status='complete', returncode=0,
                                             stage='full', cpu_seconds=360.))
    write_new(source / 'ledger/interrupted.json', dict(name='interrupted', status='budget_terminated',
        returncode=-15, stage='prefix', cpu_seconds=100.))
    allowance = dict(wall_hours=2., stage_cpu_hours={key: .5 for key in stages})
    return source, checkout, allowance


@pytest.mark.parametrize('autonomous', [False, True])
def test_reuse_preserves_scientific_policy_and_separates_new_budget(tmp_path, autonomous):
    source, checkout, allowance = source_run(tmp_path, autonomous)
    before = {str(p.relative_to(source)): digest(p) for p in source.rglob('*') if p.is_file()}
    output = tmp_path / 'continued'
    result = prepare_continuation(source, output, checkout, allowance)
    old = json.loads((source / 'protocol.json').read_text())
    new = json.loads((output / 'protocol.json').read_text())
    assert result['reused_jobs'] == 1 and result['reused_cpu_hours'] == pytest.approx(.1)
    assert new['policy']['budget']['cpu_hours'] == pytest.approx(2.6)
    assert new['policy']['budget']['stage_cpu_hours']['full'] == pytest.approx(.6)
    restored = deepcopy(new)
    restored['policy']['budget'] = old['policy']['budget']
    assert restored == old
    assert (output / 'numerical/good.npz').is_symlink()
    assert (output / 'numerical/good.npz').samefile(source / 'numerical/good.npz')
    assert not (output / 'ledger/interrupted.json').exists()
    assert result['excluded_noncomplete_jobs']['interrupted']['cpu_seconds'] == 100.
    assert (output / 'recovery_truths.json').read_bytes() == b'opaque uninspected truth bytes'
    assert not (output / 'run_clock.json').exists()
    assert not (output / 'frozen').exists()
    assert before == {str(p.relative_to(source)): digest(p) for p in source.rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError):
        prepare_continuation(source, output, checkout, allowance)


@pytest.mark.parametrize('mutation,match', [
    ('running', 'terminal completion'),
    ('source', 'scientific source changed'),
    ('input', 'case input changed'),
    ('predictions', 'prediction digest mismatch'),
    ('formal', 'terminal supervisor receipt'),
])
def test_unverified_source_is_rejected_before_creating_output(tmp_path, mutation, match):
    source, checkout, allowance = source_run(tmp_path)
    if mutation == 'running':
        (source / 'completion.json').unlink()
    elif mutation == 'source':
        (checkout / 'scientific.py').write_text('changed')
    elif mutation == 'input':
        (tmp_path / 'fixture/training.npz').write_bytes(b'changed')
    elif mutation == 'predictions':
        (source / 'numerical/good.npz').write_bytes(b'changed')
    else:
        protocol = json.loads((source / 'protocol.json').read_text())
        protocol['smoke'] = False
        (source / 'protocol.json').write_text(json.dumps(protocol))
        (source / 'protocol_digest.json').write_text(json.dumps(dict(sha256=digest(source / 'protocol.json'))))
    with pytest.raises(ValueError, match=match):
        prepare_continuation(source, tmp_path / 'continued', checkout, allowance)
    assert not (tmp_path / 'continued').exists()


@pytest.mark.parametrize('value', [-1., float('nan'), float('inf')])
def test_invalid_additional_allowance_is_rejected(tmp_path, value):
    source, checkout, allowance = source_run(tmp_path)
    allowance['stage_cpu_hours']['prefix'] = value
    with pytest.raises(ValueError, match='allowances must be finite'):
        prepare_continuation(source, tmp_path / 'continued', checkout, allowance)

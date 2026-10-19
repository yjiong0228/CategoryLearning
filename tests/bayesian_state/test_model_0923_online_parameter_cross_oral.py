"""Linked state banks require the exact cognitive execution, not just a label."""
from copy import deepcopy

import numpy as np
import pytest
import yaml

from src.Bayesian_state.workflows.analysis.model_0923_online_parameter_cross_oral import CELLS, assemble, same_execution
from src.Bayesian_state.workflows.analysis.model_0923_decision_basis_replay import digest, read, write


def fixture():
    job = dict(phase='evaluate', folder='prepared', runtime='frozen', model='SEARCH', mechanism={'kind':'E'},
               seeds=[47], particles=512, arm_settings={'precision':'fast'}, retain_precision=True,
               structural_intervention='retained_rule_precision', candidate={'id':'fixed'})
    record = dict(candidate={'id':'fixed'}, seeds=[47], particles=512, exact_behavior_replay=True)
    return job, deepcopy(job), record


def test_accepts_original_nominee_with_unchanged_cognitive_inputs():
    expected, actual, record = fixture()
    actual.update(reported_arm='SEARCH', name='original_nominee')
    same_execution(expected, actual, record, 'RETAIN_RETAIN')


def test_accepts_online_nominee_only_when_actual_execution_is_online():
    expected, actual, record = fixture()
    expected['online_belief'] = actual['online_belief'] = record['online_belief'] = True
    same_execution(expected, actual, record, 'ONLINE_ONLINE')


@pytest.mark.parametrize('change', ['candidate', 'seed', 'runtime', 'mechanism', 'precision', 'online', 'replay'])
def test_rejects_incompatible_state_bank(change):
    expected, actual, record = fixture()
    if change == 'candidate': record['candidate'] = {'id':'other'}
    elif change == 'seed': actual['seeds'] = [48]
    elif change == 'runtime': actual['runtime'] = 'changed'
    elif change == 'mechanism': actual['mechanism'] = {'kind':'R1'}
    elif change == 'precision': actual['retain_precision'] = False
    elif change == 'online': record['online_belief'] = True
    elif change == 'replay': record['exact_behavior_replay'] = False
    with pytest.raises(ValueError): same_execution(expected, actual, record, 'RETAIN_RETAIN')


def build_banks(tmp_path):
    cross = tmp_path / 'cross'; cross.mkdir(); (cross / 'chunks').mkdir()
    sources = {key: tmp_path / key for key in ('retained_states', 'online_states', 'cross_states')}
    for key, source in sources.items():
        (source / 'states').mkdir(parents=True); (source / 'jobs').mkdir()
        count = 24 if key == 'cross_states' else 8
        write(source / 'completion.json', dict(status='complete', smoke=False, jobs=count, exact_behavior_replays=count))
        write(source / 'protocol.json', dict(source_sha256={}))
    banks = []; predictions = np.array([[.2, .8], [.7, .3]])
    for cell in CELLS:
        source = sources['retained_states' if cell == 'RETAIN_RETAIN' else
                         'online_states' if cell == 'ONLINE_ONLINE' else 'cross_states']
        arm = 'SEARCH' if cell in ('RETAIN_RETAIN', 'ONLINE_ONLINE') else cell
        plans = []; candidate = {'id': cell.split('_')[1]}
        for seed in range(8):
            job, _, record = fixture(); name = cell + str(seed)
            job.update(candidate=candidate, seeds=[seed], target=str(cross / 'chunks' / name),
                       online_belief=cell.startswith('ONLINE_'))
            record.update(candidate=candidate, seeds=[seed], online_belief=job['online_belief'],
                          arm=arm, subject=208, trials=2)
            state = source / 'states' / (name + '.npz')
            choice = cross / 'chunks' / (name + '.npz')
            np.savez(state, seeds=[seed], choices=[1, 2], prediction=predictions)
            np.savez(choice, seeds=[seed], choices=[1, 2], probabilities=predictions[None])
            record['array_sha256'] = digest(state)
            write(state.with_suffix('.json'), record)
            write(source / 'jobs' / (name + '.json'), dict(job, reported_arm=arm))
            write(choice.with_suffix('.json'), dict(predictions_sha256=digest(choice)))
            plans.append({'job': job})
        banks.append(dict(cell=cell, candidate=candidate, roles={'evaluate': plans}))
    write(cross / 'protocol.json', dict(banks=banks))
    write(cross / 'completion.json', dict(status='complete', smoke=False, nominees_changed=False,
                                        protocol_sha256=digest(cross / 'protocol.json')))
    config = tmp_path / 'assembly.yaml'
    config.write_text(yaml.safe_dump(dict(subject=208, cross=str(cross), **{k:str(p) for k,p in sources.items()})))
    return config, sources


def test_assembly_reuses_all_forty_arrays_with_explicit_cell_labels(tmp_path):
    config, sources = build_banks(tmp_path); output = tmp_path / 'index'
    original = {str(p): digest(p) for source in sources.values() for p in source.rglob('*') if p.is_file()}
    assemble(config, output)
    assert read(output / 'completion.json')['exact_behavior_replays'] == 40
    files = list((output / 'states').glob('*.npz'))
    assert len(files) == 40 and all(p.is_symlink() for p in files)
    assert {read(p.with_suffix('.json'))['arm'] for p in files} == set(CELLS)
    assert {str(p): digest(p) for source in sources.values() for p in source.rglob('*') if p.is_file()} == original


def test_assembly_rejects_different_predictions_even_with_consistent_state_hash(tmp_path):
    config, sources = build_banks(tmp_path); file = next((sources['cross_states'] / 'states').glob('*.npz'))
    with np.load(file) as z: arrays = {k:z[k].copy() for k in z.files}
    arrays['prediction'][0] = [.5, .5]; np.savez(file, **arrays)
    meta = file.with_suffix('.json'); record = read(meta); record['array_sha256'] = digest(file)
    import json
    meta.write_text(json.dumps(record))
    with pytest.raises(AssertionError): assemble(config, tmp_path / 'invalid')
    assert not (tmp_path / 'invalid').exists()

"""Protect parameter identities and data-use boundaries in retrospective replay."""
from copy import deepcopy
import json

import pytest

from src.Bayesian_state.workflows.analysis import model_0923_decision_basis_replay as replay
from src.Bayesian_state.workflows.analysis.model_0923_decision_basis_alignment import estimation_context


def save(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def scope(tmp_path):
    (tmp_path / 'jobs').mkdir()
    data = tmp_path / 'data'; folder = data / 'subject_308'; folder.mkdir(parents=True)
    save(folder / 'prepared.json', dict(subject=308))
    settings = dict(source_kind='scope_confirmation', estimation_objective='full',
                    subjects=[308], arms=['POOL', 'SEARCH'], seeds_per_arm=2)
    arm_settings = {arm: {'precision': 'fast'} for arm in settings['arms']}
    candidates = {'prefix': dict(id='early', hyperparams={'gamma': .3}),
                  'full': dict(id='late', hyperparams={'gamma': .99})}
    rows = []
    for arm in settings['arms']:
        for objective, point in candidates.items():
            rows.append(dict(subject=308, model=arm, objective=objective,
                             candidate=dict(point, particles=512, seeds=[5, 7])))
            for index, seed in enumerate([5, 7]):
                name = f'confirmation_S308_{arm}_{point["id"]}_C{index:03d}'
                save(tmp_path / 'jobs' / (name + '.json'), dict(phase='evaluate', name=name,
                     folder=str(folder), model=arm, seeds=[seed], particles=512, candidate=point,
                     source_sha256={}, arm_settings=arm_settings[arm], mechanism={'kind': 'E'}))
    save(tmp_path / 'protocol.json', dict(smoke=False, source_sha256={},
         settings=dict(arms=arm_settings, mechanism={'kind': 'E'})))
    save(tmp_path / 'nominees.json', dict(status='frozen_for_retrospective_diagnostics', smoke=False,
         source_sha256={}, data=str(data), nominees=rows))
    save(tmp_path / 'completion.json', dict(status='complete', smoke=False,
         nomination_sha256=replay.digest(tmp_path / 'nominees.json')))
    return settings, tmp_path


def test_scope_binds_only_selected_parameters_and_all_seeds(scope):
    settings, source = scope
    files, metadata = replay.source_jobs(settings, source, False)
    assert len(files) == 4 and all('_late_' in p.name for p in files)
    assert metadata['parameter_estimation_scope'] == 'full'
    assert metadata['evaluation_use'] == 'development_retrospective_comparison'
    files, metadata = replay.source_jobs(dict(settings, estimation_objective='prefix'), source, False)
    assert len(files) == 4 and all('_early_' in p.name for p in files)
    assert metadata['parameter_estimation_scope'] == 'prefix'
    assert len(replay.source_jobs(settings, source, True)[0]) == 1


@pytest.mark.parametrize('change', ['parameter', 'seed', 'particles', 'model', 'subject', 'source', 'mechanism',
                                   'arm_settings', 'nomination_checksum', 'nomination_status', 'missing_seed'])
def test_invalid_scope_binding_rejected(scope, change):
    settings, source = deepcopy(scope)
    file = source / 'jobs' / 'confirmation_S308_POOL_late_C001.json'; job = replay.read(file)
    if change == 'parameter': job['candidate']['hyperparams']['gamma'] = .8
    elif change == 'seed': job['seeds'] = [5]
    elif change == 'particles': job['particles'] = 256
    elif change == 'model': job['model'] = 'SEARCH'
    elif change == 'subject':
        save(source / 'data' / 'subject_308' / 'prepared.json', dict(subject=207))
    elif change == 'source': job['source_sha256'] = {'changed.py': 'changed'}
    elif change == 'mechanism': job['mechanism']['kind'] = 'R1'
    elif change == 'arm_settings': job['arm_settings']['precision'] = 'base'
    elif change.startswith('nomination'):
        frozen = replay.read(source / 'nominees.json'); frozen['status'] = 'frozen_before_test'
        save(source / 'nominees.json', frozen)
        if change == 'nomination_status':
            done = replay.read(source / 'completion.json')
            done['nomination_sha256'] = replay.digest(source / 'nominees.json'); save(source / 'completion.json', done)
    if change == 'missing_seed': file.unlink()
    else: save(file, job)
    with pytest.raises(ValueError): replay.source_jobs(settings, source, False)


def test_smoke_sources_require_explicit_smoke(scope):
    settings, source = scope
    for name in ('nominees.json', 'protocol.json', 'completion.json'):
        value = replay.read(source / name); value['smoke'] = True
        if name == 'completion.json': value['nomination_sha256'] = replay.digest(source / 'nominees.json')
        save(source / name, value)
    with pytest.raises(ValueError): replay.source_jobs(settings, source, False)
    assert len(replay.source_jobs(settings, source, True)[0]) == 1


def test_original_evaluation_selection_is_compatible(scope):
    settings, source = scope
    settings.pop('source_kind'); settings.pop('estimation_objective')
    for file in list((source / 'jobs').glob('*late*')):
        job = replay.read(file); job['name'] = job['name'].replace('confirmation_', 'evaluate_')
        save(source / 'jobs' / (job['name'] + '.json'), job)
    files, metadata = replay.source_jobs(settings, source, False)
    assert len(files) == 4 and all(p.name.startswith('evaluate_') for p in files)
    assert metadata['evaluation_use'] == 'original_evaluation_protocol'


@pytest.mark.parametrize('objective', ['prefix', 'full'])
def test_oral_report_keeps_retrospective_labels(objective):
    context, note = estimation_context(dict(parameter_estimation_scope=objective,
                                           evaluation_use='development_retrospective_comparison'))
    assert context['window_definition'] == 'original_chronological_boundary'
    assert '回顾性开发诊断' in note and '已使用数据' in note
    assert '完整行为序列' in note if objective == 'full' else '原前段行为序列' in note
    assert estimation_context({})[1] == ''

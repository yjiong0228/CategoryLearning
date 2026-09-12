"""Existing artifacts survive reruns, writer failures and concurrent publication."""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.Bayesian_state.simulation.config import dump_stream, save_json
from src.Bayesian_state.utils.paths import ROOT_DIR
from src.Bayesian_state.utils.streaming import StreamList


@pytest.mark.parametrize('kind', ['json', 'stream'])
def test_writers_refuse_existing_artifacts(tmp_path, kind):
    path = tmp_path / ('subject.json' if kind == 'json' else 'cache/subject_101_raw_runs.gz')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'existing research artifact')
    with pytest.raises(FileExistsError):
        if kind == 'json':
            save_json({'new': 1}, path)
        else:
            dump_stream([{'new': 1}], tmp_path, 101, 'raw_runs')
    assert path.read_bytes() == b'existing research artifact'


@pytest.mark.parametrize('kind', ['json', 'stream'])
def test_failed_serialization_leaves_no_partial_artifact(tmp_path, kind):
    if kind == 'json':
        path = tmp_path / 'subject.json'
        with pytest.raises(TypeError):
            save_json({'valid': 1, 'invalid': object()}, path)
    else:
        path = tmp_path / 'cache/subject_101_raw_runs.gz'
        with pytest.raises((AttributeError, TypeError)):
            dump_stream([{'valid': 1}, lambda: None], tmp_path, 101, 'raw_runs')
    assert not path.exists()
    assert not list(tmp_path.rglob('*.tmp'))


def test_json_and_stream_preserve_existing_formats(tmp_path):
    path = tmp_path / 'subjects/subject_101.json'
    save_json({'values': np.array([1, 2]), 'name': '口述'}, path)
    assert json.loads(path.read_text()) == {'values': [1, 2], 'name': '口述'}
    ref = dump_stream([{'trial': 1}, {'trial': 2}], tmp_path, 101, 'raw_runs')
    assert ref == {'format': 'stream-gzip-pickle', 'path': 'cache/subject_101_raw_runs.gz', 'count': 2}
    assert list(StreamList(str(tmp_path / ref['path']), ref['count'])) == [{'trial': 1}, {'trial': 2}]


def _simulation_config(tmp_path):
    source = ROOT_DIR / 'CategoryLearning_codes/Bayesian_model/configs/smoke_simulation.yaml'
    cfg = yaml.safe_load(source.read_text())
    cfg['engine_config_path'] = str(source.parent / 'model_0826.yaml')
    cfg['dataset']['processed_dir'] = str(ROOT_DIR / 'data/exp123/processed')
    cfg['output_dir'] = str(tmp_path / 'simulation')
    cfg['engine_config'] = {'inference': {'particle_count': 2}}
    cfg.update(max_trials=4, window_size=2, keep_logs=True)
    return cfg


@pytest.mark.parametrize('artifact', ['subjects/subject_102.json', 'cache/subject_102_raw_runs.gz'])
def test_all_subject_outputs_checked_before_any_model_work(tmp_path, monkeypatch, artifact):
    entry = importlib.import_module('src.Bayesian_state.run_simulation')
    cfg = _simulation_config(tmp_path)
    cfg['subjects'] = [101, 102]
    second_output = tmp_path / 'second_subject'
    cfg['subject_overrides'] = {'102': {'output_dir': str(second_output)}}
    path = second_output / artifact
    path.parent.mkdir(parents=True)
    path.write_bytes(b'keep')
    config_path = tmp_path / 'simulation.yaml'
    config_path.write_text(yaml.safe_dump(cfg))

    def forbid_compute(*args, **kwargs):
        pytest.fail('Existing outputs must be detected before constructing a model runner')

    monkeypatch.setattr(entry, 'StateModelSimulationRunner', forbid_compute)
    with pytest.raises(FileExistsError):
        entry.run_simulation(config_path)
    assert path.read_bytes() == b'keep'
    assert not (tmp_path / 'simulation/subjects/subject_101.json').exists()


def test_short_simulation_writes_new_outputs_and_rejects_rerun(tmp_path):
    entry = importlib.import_module('src.Bayesian_state.run_simulation')
    cfg = _simulation_config(tmp_path)
    output = Path(cfg['output_dir'])
    output.mkdir()
    metadata = output / 'hyper_best_source.json'
    metadata.write_text('{"source": "existing workflow metadata"}')
    config_path = tmp_path / 'simulation.yaml'
    config_path.write_text(yaml.safe_dump(cfg))
    paths = entry.run_simulation(config_path)
    assert paths == [output / 'subjects/subject_101.json']
    payload = json.loads(paths[0].read_text())
    assert payload['subject_id'] == 101
    ref = payload['raw_runs_ref']
    records = list(StreamList(str(paths[0].parent / ref['path']), ref['count']))
    assert len(records) == 1
    snapshot = {p: p.read_bytes() for p in output.rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError):
        entry.run_simulation(config_path)
    assert snapshot == {p: p.read_bytes() for p in output.rglob('*') if p.is_file()}
    assert not list(output.glob('*.lock'))


def test_json_refuses_destination_created_during_serialization(tmp_path):
    path = tmp_path / 'subject.json'

    class RacingPayload(dict):
        def items(self):
            path.write_bytes(b'concurrent completed result')
            return super().items()

    with pytest.raises(FileExistsError):
        save_json(RacingPayload(new=1), path)
    assert path.read_bytes() == b'concurrent completed result'
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('blocked_by', ['lock', 'duplicate_subject'])
def test_run_rejects_reserved_or_duplicate_subject_before_compute(tmp_path, monkeypatch, blocked_by):
    entry = importlib.import_module('src.Bayesian_state.run_simulation')
    cfg = _simulation_config(tmp_path)
    output = Path(cfg['output_dir'])
    output.mkdir()
    if blocked_by == 'lock':
        lock = output / '.subject_101.lock'
        lock.write_text('another process')
        expected_error = FileExistsError
    else:
        cfg['subjects'] = [101, 101]
        expected_error = ValueError
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(cfg))

    def forbid_compute(*args, **kwargs):
        pytest.fail('Reservation conflicts must be rejected before model work')

    monkeypatch.setattr(entry, 'StateModelSimulationRunner', forbid_compute)
    with pytest.raises(expected_error):
        entry.run_simulation(path)
    if blocked_by == 'lock':
        assert lock.read_text() == 'another process'


def test_model_failure_releases_only_its_own_reservation(tmp_path, monkeypatch):
    entry = importlib.import_module('src.Bayesian_state.run_simulation')
    cfg = _simulation_config(tmp_path)
    output = Path(cfg['output_dir'])
    output.mkdir()
    unrelated = output / '.subject_999.lock'
    unrelated.write_text('other process')
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(cfg))

    def fail_model(*args, **kwargs):
        assert (output / '.subject_101.lock').is_file()
        raise RuntimeError('model interrupted')

    monkeypatch.setattr(entry, 'StateModelSimulationRunner', fail_model)
    with pytest.raises(RuntimeError, match='model interrupted'):
        entry.run_simulation(path)
    assert not (output / '.subject_101.lock').exists()
    assert unrelated.read_text() == 'other process'

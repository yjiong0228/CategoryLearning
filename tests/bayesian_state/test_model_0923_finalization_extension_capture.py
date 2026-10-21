"""Preserve scientific worker boundaries when adding observational capture."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from src.Bayesian_state.workflows.analysis import model_0923_finalization_extension_capture as capture
from src.Bayesian_state.workflows.analysis import model_0923_finalization_extension_captured_states as states


def job():
    return dict(name='evaluate_filter', phase='evaluate', extension_candidate='SEARCH_ADD10',
        extension_scope='extension_validation', extension_recommendation='recommendation.json', smoke=False,
        model='SEARCH', arm_settings=dict(precision='fast', rate_scale=10., upper=25.),
        seeds=[73], online_belief=True, retain_precision=True,
        decision_basis_target='states/filter', decision_basis_sources=capture.sources())


@pytest.mark.parametrize('field,value', [('phase', 'training'), ('seeds', [73, 74]),
    ('decision_basis_sources', {}), ('decision_basis_target', ''), ('online_belief', False),
    ('retain_precision', False), ('extension_candidate', 'SEARCH_LOG25')])
def test_invalid_capture_is_rejected_before_execution(field, value):
    row = job()
    row[field] = value
    with pytest.raises(ValueError):
        capture.check_capture_job(row)


def test_valid_original_filter():
    capture.check_capture_job(job())


def test_autonomous_worker_delegates_without_observation(tmp_path, monkeypatch):
    row = job()
    row['phase'] = 'autonomous'
    for key in capture.CAPTURE_KEYS:
        row.pop(key)
    file = tmp_path / 'job.json'
    file.write_text(json.dumps(row))
    calls = []
    monkeypatch.setattr(capture.driver, 'worker', calls.append)
    capture.worker(file)
    assert calls == [file]


def test_configure_keeps_original_jobs_and_budget(tmp_path, monkeypatch):
    received = []
    module = SimpleNamespace(base=SimpleNamespace(execute=lambda *args: received.append(args)))
    monkeypatch.setattr(capture.driver, 'configure', lambda *args: module)
    # ENTRY is process-local launcher routing; restore it after the test.
    monkeypatch.setattr(capture.driver, 'ENTRY', capture.driver.ENTRY)
    capture.configure({}, {}, {})
    row = job()
    for key in capture.CAPTURE_KEYS:
        row.pop(key)
    automatic = dict(deepcopy(row), phase='autonomous', name='AUTO_filter', seeds=[11, 12, 13, 14])
    jobs = [row, automatic]
    before = deepcopy(jobs)
    module.base.execute(jobs, tmp_path, 37)
    tagged, output, budget = received[0]
    assert jobs == before and budget == 37 and output == tmp_path
    assert {k: v for k, v in tagged[0].items() if k not in capture.CAPTURE_KEYS} == row
    assert tagged[1] == automatic
    declaration = capture.read(tmp_path / 'decision_basis_capture_protocol.json')
    assert declaration['additional_filters'] == 0 and not declaration['scientific_settings_changed']


def test_partial_collection_cannot_enter_formal_oral_alignment(tmp_path):
    capture.write(tmp_path / 'protocol.json', dict(smoke=False, jobs=24))
    capture.write(tmp_path / 'completion.json', dict(status='complete', smoke=False, jobs=23,
        original_filter_captures=23, exact_behavior_replays=0, additional_filters=0))
    with pytest.raises(ValueError, match='all original'):
        states.align(tmp_path, tmp_path / 'unused_oral', tmp_path / 'forbidden')
    assert not (tmp_path / 'forbidden').exists()


def test_replay_record_cannot_be_mislabelled_as_capture(tmp_path):
    file = tmp_path / 'state.json'
    capture.write(file, dict(status='complete', exact_behavior_replay=True))
    with pytest.raises(ValueError, match='original-filter'):
        states.validate_record(file)

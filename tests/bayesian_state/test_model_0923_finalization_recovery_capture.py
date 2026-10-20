"""Check that observational capture preserves fitting and truth boundaries."""
import json

import pytest

from src.Bayesian_state.workflows.analysis import model_0923_finalization_recovery_capture as capture


def test_target_identity_does_not_open_generated_states_or_parameters(tmp_path, monkeypatch):
    protocol = tmp_path / 'protocol.json'
    protocol.write_text(json.dumps(dict(config=dict(scope='finalization_recovery'),
        cases=[dict(subject=205, replicate=0, target_rule=42,
                    states_truth=str(tmp_path / 'states_truth.npz'))])))
    original = capture.read
    opened = []

    def read(path):
        opened.append(path)
        return original(path)

    monkeypatch.setattr(capture, 'read', read)
    settings = dict(recovery_case=dict(subject=205, replicate=0))
    assert capture.target_from_protocol(settings, {str(protocol): 'bound'}) == 42
    assert opened == [protocol]


@pytest.mark.parametrize('cases', [[], [dict(subject=306, replicate=0, target_rule=42)],
    [dict(subject=205, replicate=0, target_rule=-1)],
    [dict(subject=205, replicate=0, target_rule=42)] * 2])
def test_missing_or_ambiguous_case_is_rejected(tmp_path, cases):
    protocol = tmp_path / 'protocol.json'
    protocol.write_text(json.dumps(dict(config=dict(scope='finalization_recovery'), cases=cases)))
    with pytest.raises(ValueError, match='exactly one'):
        capture.target_from_protocol(dict(recovery_case=dict(subject=205, replicate=0)), {str(protocol): 'bound'})


def test_screening_delegates_without_state_observation(tmp_path, monkeypatch):
    file = tmp_path / 'job.json'
    file.write_text(json.dumps(dict(name='screen_S205_SEARCH_point_C000')))
    calls = []
    monkeypatch.setattr(capture.driver, 'check_job', lambda job: None)
    monkeypatch.setattr(capture.driver, 'worker', calls.append)
    capture.worker(file)
    assert calls == [file]


def test_confirmation_capture_rejects_multiple_filters_before_execution(tmp_path, monkeypatch):
    file = tmp_path / 'job.json'
    file.write_text(json.dumps(dict(name='confirmation_S205_SEARCH_point_C000', seeds=[1, 2])))
    monkeypatch.setattr(capture.driver, 'check_job', lambda job: None)
    with pytest.raises(ValueError, match='capture declaration'):
        capture.worker(file)

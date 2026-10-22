"""Prevent overlapping phase queues or advancing after a failed phase."""
import json
import subprocess

import pytest

from src.Bayesian_state.workflows.analysis import model_1023_reliability_sequence as sequence


@pytest.mark.parametrize('active,failed', [(1, 0), (0, 1)])
def test_completed_diagnostic_requires_idle_successful_queue(tmp_path, active, failed):
    (tmp_path/'completion.json').write_text(json.dumps(dict(status='complete', smoke=False, phase='diagnostic')))
    (tmp_path/'resource_status.json').write_text(json.dumps(dict(active=active, failed=failed)))
    with pytest.raises(ValueError, match='no remaining numerical workers'):
        sequence.wait_for_diagnostic(tmp_path)


def test_failure_stops_all_downstream_phases(tmp_path, monkeypatch):
    config = tmp_path/'settings.yaml'
    config.write_text('test: true\n')
    oral = tmp_path/'configs/shared/model1022_compact_evaluation.yaml'
    oral.parent.mkdir(parents=True)
    oral.write_text('test: true\n')
    monkeypatch.setattr(sequence, 'ROOT', tmp_path)
    monkeypatch.setattr(sequence, 'wait_for_diagnostic', lambda folder: None)
    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(sequence.subprocess, 'run', fail)
    with pytest.raises(subprocess.CalledProcessError):
        sequence.run(config, tmp_path/'diagnostic', tmp_path/'continued')
    assert len(calls) == 1
    failure = json.loads((tmp_path/'continued/failure.json').read_text())
    assert failure['stage'] == 'diagnostic_summary'
    assert not (tmp_path/'continued/completion.json').exists()

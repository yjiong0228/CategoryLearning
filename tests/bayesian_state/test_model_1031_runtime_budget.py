"""The user wall limit and CPU budget survive a document-reserve amendment."""
from datetime import datetime, timezone
import json

import pytest

from src.Bayesian_state.workflows.analysis.model_1031_runtime_budget import compute_deadline
from src.Bayesian_state.workflows.analysis.model_1031_scheduled import LongestFirstScheduler
from src.Bayesian_state.workflows.analysis.model_1031_recovery import include_case_cpu
from src.Bayesian_state.workflows.runs.handoff_model_1031_supervisor import supervisor_command_matches


POLICY = dict(started_at_utc="2026-10-31T16:33:19+00:00", deadline_utc="2026-11-01T04:33:19+00:00",
              budget=dict(computation_stop_hours=9.5, cpu_hours=1152))


def test_amendment_preserves_wall_reserve_and_cpu_cap(tmp_path):
    assert datetime.fromtimestamp(compute_deadline(tmp_path, POLICY), timezone.utc).hour == 2
    path = tmp_path / "runtime_budget_amendment.json"
    path.write_text(json.dumps(dict(compute_deadline_utc="2026-11-01T03:03:19+00:00", cpu_hours=1152)))
    assert datetime.fromtimestamp(compute_deadline(tmp_path, POLICY), timezone.utc).hour == 3
    path.write_text(json.dumps(dict(compute_deadline_utc="2026-11-01T03:03:20+00:00", cpu_hours=1152)))
    with pytest.raises(ValueError, match="90 minutes"):
        compute_deadline(tmp_path, POLICY)
    path.write_text(json.dumps(dict(compute_deadline_utc="2026-11-01T03:03:19+00:00", cpu_hours=1153)))
    with pytest.raises(ValueError, match="CPU budget"):
        compute_deadline(tmp_path, POLICY)


def test_recovery_accounting_keeps_unrecorded_cpu_allowance(tmp_path):
    (tmp_path / "jobs").mkdir()
    (tmp_path / "recovery/case/jobs").mkdir(parents=True)
    (tmp_path / "jobs/one.json").write_text(json.dumps(dict(cpu_seconds=10.)))
    (tmp_path / "recovery/case/jobs/two.json").write_text(json.dumps(dict(cpu_seconds=30.)))
    (tmp_path / "supervisor_handoff_accounting.json").write_text(json.dumps(dict(unrecorded_cpu_allowance_seconds=20.)))
    scheduler = LongestFirstScheduler(tmp_path, POLICY)
    include_case_cpu(scheduler)
    assert scheduler.cpu_used() == 60.


def test_handoff_matches_both_module_and_study_path(tmp_path):
    command = ["python", "-m", "src.Bayesian_state.workflows.runs.supervise_model_1031_finalization", "--output", str(tmp_path)]
    assert supervisor_command_matches(command, tmp_path)
    assert not supervisor_command_matches(command, tmp_path / "another")
    command[2] = "another.module"
    assert not supervisor_command_matches(command, tmp_path)

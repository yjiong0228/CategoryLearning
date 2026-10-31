"""A supervisor handoff admits work only after the replacement is recorded."""
import time

import pytest

from src.Bayesian_state.workflows.analysis.model_1031_pipeline import await_supervisor_handoff


def test_handoff_gate_requires_completion_and_honors_stop(tmp_path):
    await_supervisor_handoff(tmp_path, time.time() + 10)
    (tmp_path / "supervisor_handoff_plan.json").write_text("{}")
    with pytest.raises(TimeoutError, match="handoff stopped"):
        await_supervisor_handoff(tmp_path, time.time() - 1)
    (tmp_path / "STOP_REQUESTED").write_text("")
    with pytest.raises(TimeoutError, match="handoff stopped"):
        await_supervisor_handoff(tmp_path, time.time() + 10)
    (tmp_path / "supervisor_launch_v3.json").write_text("{}")
    await_supervisor_handoff(tmp_path, time.time() + 10)

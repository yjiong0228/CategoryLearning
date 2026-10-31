"""Read an explicit compute/document-time amendment within the user deadline."""
from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
import time


def compute_deadline(output: Path, policy: dict) -> float:
    original = datetime.fromisoformat(policy["started_at_utc"]).timestamp() + policy["budget"]["computation_stop_hours"] * 3600
    path = output / "runtime_budget_amendment.json"
    if not path.exists():
        return original
    amendment = json.loads(path.read_text())
    deadline = datetime.fromisoformat(amendment["compute_deadline_utc"]).timestamp()
    delivery = datetime.fromisoformat(policy["deadline_utc"]).timestamp()
    if not original <= deadline <= delivery - 90 * 60:
        raise ValueError("Runtime amendment must preserve at least 90 minutes before the authorized delivery deadline")
    if amendment["cpu_hours"] != policy["budget"]["cpu_hours"]:
        raise ValueError("The runtime amendment cannot increase the registered CPU budget")
    return deadline


def unrecorded_cpu_allowance(output: Path) -> float:
    path = output / "supervisor_handoff_accounting.json"
    if not path.exists():
        return 0.
    value = float(json.loads(path.read_text())["unrecorded_cpu_allowance_seconds"])
    if not math.isfinite(value) or value < 0:
        raise ValueError("Unrecorded CPU allowance must be finite and nonnegative")
    return value


def read_job_record(path: Path) -> dict:
    """Allow the brief interval between immutable file creation and flush."""
    for attempt in range(4):
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            if attempt == 3:
                raise
            time.sleep(.05)
    raise AssertionError("Unreachable record-read state")

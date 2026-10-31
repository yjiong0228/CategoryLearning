"""Replace the study watchdog after development results have been written.

This handoff preserves completed jobs and carries the external-interruption
allowance into the new CPU counter. It signals only the recorded supervisor
for this study and waits for its owned compute group to stop before resuming.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from ..analysis.model_1031_finalization import ROOT, digest, write_new
from ..analysis.model_1031_runtime_budget import compute_deadline


def live(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


def supervisor_command_matches(command: list[str], output: Path) -> bool:
    module = "src.Bayesian_state.workflows.runs.supervise_model_1031_finalization"
    try:
        return (command[command.index("-m") + 1] == module
                and Path(command[command.index("--output") + 1]).resolve() == output.resolve())
    except (ValueError, IndexError):
        return False


def group_is_live(group: int) -> bool:
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().split(") ", 1)[1].split()
            if int(fields[2]) == group and fields[0] != "Z":
                return True
        except (OSError, ValueError, IndexError):
            continue
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--numeric-root", type=Path, required=True)
    parser.add_argument("--old-pid", type=int, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    policy = json.loads((output / "registration.json").read_text())["policy"]
    deadline = compute_deadline(output, policy)
    launch = json.loads((output / "supervisor_launch.json").read_text())
    if launch["pid"] != args.old_pid or not supervisor_command_matches(launch["command"], output):
        raise ValueError("The recorded supervisor does not belong to this study")
    write_new(output / "supervisor_handoff_plan.json", dict(pid=os.getpid(), old_pid=args.old_pid,
        trigger="development_parameters.json exists", deadline_unix=deadline,
        source_sha256=digest(Path(__file__))))
    print(json.dumps(dict(waiting_for="development_parameters.json", old_pid=args.old_pid)), flush=True)
    while not (output / "development_parameters.json").exists():
        if (output / "STOP_REQUESTED").exists() or not live(args.old_pid) or time.time() >= deadline:
            raise SystemExit("Handoff stopped before the development completion marker")
        time.sleep(5.)
    command = Path(f"/proc/{args.old_pid}/cmdline").read_bytes().decode().strip("\0").split("\0")
    if not supervisor_command_matches(command, output) or (output / "STOP_REQUESTED").exists():
        raise SystemExit("Supervisor identity or stop request prevents handoff")
    os.kill(args.old_pid, signal.SIGTERM)
    stopped_by = time.time() + 60.
    while live(args.old_pid) and time.time() < stopped_by:
        time.sleep(1.)
    if live(args.old_pid):
        raise RuntimeError("Previous supervisor remains active; no new process launched")
    heartbeats = list(output.glob("supervisor_heartbeat_*.jsonl"))
    last = json.loads(max(heartbeats, key=lambda p: p.stat().st_mtime).read_text().splitlines()[-1])
    if group_is_live(last["child_pid"]):
        raise RuntimeError("Previous compute group remains active; no new process launched")
    records = list((output / "jobs").glob("*.json")) + list((output / "recovery").glob("*/jobs/*.json"))
    recorded = sum(json.loads(path.read_text())["cpu_seconds"] for path in records)
    stages = [json.loads(path.read_text()) for path in output.glob("supervisor_stage_*.json")]
    observed = max([last["cpu_seconds"], *[stage["cpu_seconds_upper"] for stage in stages]])
    clarification = json.loads((output / "cpu_accounting_clarification.json").read_text())
    supplement = clarification["additional_conservative_cpu_seconds"]
    # Charge the complete prior uncertainty, a polling interval, and one CPU
    # hour for all targeted validation/reporting outside the worker tree.
    prior = max(observed, recorded) + supplement + policy["budget"]["workers"] * 15 + 3600.
    write_new(output / "supervisor_handoff_accounting.json", dict(
        recorded_job_cpu_seconds=recorded, previous_watchdog_upper_seconds=observed,
        external_interruption_supplement_seconds=supplement, polling_allowance_seconds=policy["budget"]["workers"] * 15,
        validation_reporting_allowance_seconds=3600., prior_cpu_seconds=prior,
        unrecorded_cpu_allowance_seconds=prior - recorded,
        explanation="The prior wall-time CPU bound is superseded by an explicit conservative CPU counter after the deadline amendment"))
    if (output / "STOP_REQUESTED").exists() or time.time() >= deadline or prior >= policy["budget"]["cpu_hours"] * 3600:
        raise SystemExit("Stop request or budget prevents new supervisor admission")
    new_command = [sys.executable, "-u", "-m", "src.Bayesian_state.workflows.runs.supervise_model_1031_finalization",
        "--output", str(output), "--numeric-root", str(args.numeric_root.resolve()), "--prior-cpu-seconds", str(prior)]
    with (output / "supervisor_stdout_v3.log").open("x") as stream:
        process = subprocess.Popen(new_command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    write_new(output / "supervisor_launch_v3.json", dict(pid=process.pid, command=new_command,
        launched_unix=time.time(), compute_deadline_unix=deadline, old_pid=args.old_pid,
        old_compute_group_stopped=True))
    print(json.dumps(dict(resumed_pid=process.pid, prior_cpu_hours=prior / 3600)), flush=True)


if __name__ == "__main__":
    main()

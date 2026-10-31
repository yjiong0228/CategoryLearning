"""Precompute reserved-subject training fits while development precision runs.

Each possible finalist receives exactly the registered two-model training
search. Future choices are scored only after structure selection. The separate
watchdog adds this process tree to the existing study CPU counter; numerical
worker quotas sum to the original limit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

from . import model_1031_finalization as core
from .model_1031_runtime_budget import compute_deadline
from ..runs.handoff_model_1031_supervisor import group_is_live, live, supervisor_command_matches
from ..runs.supervise_model_1031_finalization import stop_group, tree_cpu


def prefit_path(output: Path, variant: str) -> Path:
    return output / "evaluation_prefits" / f"{variant}_parameters.json"


def run_training_pairs(output: Path) -> None:
    """Redirect only the completion table; reuse the unchanged fitting code."""
    from .model_1031_scheduled import LongestFirstScheduler

    plan = json.loads((output / "evaluation_prefit_plan.json").read_text())
    previous_environment = {key: os.environ.get(key) for key in ("MODEL1031_PHASE_WORKERS", "MODEL1031_COMPANION_MARKER")}
    os.environ["MODEL1031_PHASE_WORKERS"] = str(plan["workers"])
    os.environ.pop("MODEL1031_COMPANION_MARKER", None)
    original_write, original_scheduler = core.write_new, core.Scheduler
    try:
        core.Scheduler = LongestFirstScheduler
        for pair in plan["variant_pairs"]:
            destination = prefit_path(output, pair[0])
            if destination.exists():
                continue

            def redirected_write(path, value):
                original_write(destination if path == output / "evaluation_parameters.json" else path, value)

            core.write_new = redirected_write
            print(json.dumps(dict(training_pair=pair)), flush=True)
            core.fit_role(output, "evaluation", pair)
    finally:
        core.write_new, core.Scheduler = original_write, original_scheduler
        for key, value in previous_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def await_prefits(output: Path, deadline: float) -> dict | None:
    if not (output / "evaluation_prefit_plan.json").exists():
        return None
    marker = output / "evaluation_prefit_finished.json"
    while not marker.exists():
        if (output / "STOP_REQUESTED").exists() or time.time() >= deadline:
            raise TimeoutError("Prefit handoff did not finish before the compute deadline")
        launch = output / "evaluation_prefit_launch.json"
        if launch.exists() and not live(json.loads(launch.read_text())["pid"]):
            raise RuntimeError("Prefit watchdog stopped without closing its numerical worker group")
        time.sleep(.5)
    finished = json.loads(marker.read_text())
    if not finished.get("workers_stopped"):
        raise ValueError("Prefit completion must confirm that its numerical process group stopped")
    return finished


def reuse_selected_prefit(output: Path) -> bool:
    """Publish the exact completed pair selected using development data only."""
    selection = json.loads((output / "structure_selection.json").read_text())
    path = prefit_path(output, selection["selected_variant"])
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    variants = list(dict.fromkeys(group["variant"] for group in record["groups"]))
    if variants != selection["evaluation_variants"]:
        raise ValueError("Precomputed training search used a different model pair")
    core.write_new(output / "evaluation_parameters.json", record)
    core.write_new(output / "evaluation_prefit_reused.json", dict(
        source=str(path.relative_to(output)), source_sha256=core.digest(path),
        selected_variant=selection["selected_variant"], reused_at_utc=datetime.now(timezone.utc).isoformat(),
        structure_frozen_at_utc=selection["frozen_at_utc"], future_choices_scored=False))
    return True


def main_counter(output: Path) -> float:
    paths = list(output.glob("supervisor_heartbeat_*.jsonl"))
    latest = max(paths, key=lambda path: path.stat().st_mtime)
    # The writer can be between writing the final bytes and the newline.
    lines = latest.read_text().splitlines()
    for line in reversed(lines):
        try:
            return float(json.loads(line)["cpu_seconds"])
        except json.JSONDecodeError:
            continue
    raise ValueError("No complete study CPU heartbeat")


def close_worker_group(process: subprocess.Popen) -> None:
    stop_group(process)
    deadline = time.time() + 10.
    while group_is_live(process.pid) and time.time() < deadline:
        time.sleep(.1)
    if group_is_live(process.pid):
        raise RuntimeError("Prefit numerical process group remains active")


def supervise(output: Path, main_pid: int) -> None:
    register = core.validate_source(output)
    policy = register["policy"]
    plan = json.loads((output / "evaluation_prefit_plan.json").read_text())
    if plan["workers"] + plan["concurrent_precision_workers"] > policy["budget"]["workers"]:
        raise ValueError("Concurrent numerical workers exceed the registered allocation")
    if not 1 <= plan["workers"] or (output / "structure_selection.json").exists():
        raise ValueError("Prefit admission requires a positive quota before structure selection")
    if (output / "evaluation_parameters.json").exists() or (output / "evaluation_evaluation.json").exists():
        raise ValueError("Evaluation fitting or future scoring already started")
    order = json.loads((output / plan["precision_submission_order"]).read_text())
    expected_precision = min(order["workers"], len(set(order["submitted_order"])))
    if plan["concurrent_precision_workers"] != expected_precision:
        raise ValueError("Precision worker bound differs from the recorded job queue")
    command = Path(f"/proc/{main_pid}/cmdline").read_bytes().decode().strip("\0").split("\0")
    if not supervisor_command_matches(command, output):
        raise ValueError("Existing supervisor identity does not match this study")
    deadline = compute_deadline(output, policy)
    requested = False

    def on_signal(signum, frame):
        nonlocal requested
        requested = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    worker_command = [sys.executable, "-u", "-m", __spec__.name, "--phase", "worker", "--output", str(output)]
    completed_record = False
    with (output / "evaluation_prefit_worker.log").open("x") as stream:
        process = subprocess.Popen(worker_command, cwd=core.ROOT, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        maximum, previous_main = 0., 0.
        reason = None
        with (output / "evaluation_prefit_heartbeat.jsonl").open("x") as heartbeat:
            try:
                while True:
                    own = resource.getrusage(resource.RUSAGE_SELF)
                    child = resource.getrusage(resource.RUSAGE_CHILDREN)
                    external = own.ru_utime + own.ru_stime + child.ru_utime + child.ru_stime + tree_cpu(process.pid)
                    previous_main = max(previous_main, main_counter(output))
                    # Cover the main watchdog's polling interval before it
                    # reports newly consumed CPU from its own process tree.
                    maximum = max(maximum, previous_main + external + policy["budget"]["workers"] * 15)
                    heartbeat.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),
                        cpu_seconds_upper=maximum, main_counter=previous_main, prefit_tree_cpu=external)) + "\n")
                    heartbeat.flush()
                    status = process.poll()
                    if status is not None and not completed_record:
                        close_worker_group(process)
                        core.write_new(output / "evaluation_prefit_finished.json", dict(exit_code=status,
                            finished_at_utc=datetime.now(timezone.utc).isoformat(), workers_stopped=True,
                            completed_pairs=[pair for pair in plan["variant_pairs"] if prefit_path(output, pair[0]).exists()],
                            future_choices_scored=False, worker_source_sha256=core.digest(Path(__file__))))
                        completed_record = True
                    reason = ("stop_requested" if requested or (output / "STOP_REQUESTED").exists() else
                              "absolute_compute_deadline" if time.time() >= deadline else
                              "global_cpu_limit" if maximum >= policy["budget"]["cpu_hours"] * 3600 else
                              "main_supervisor_stopped" if not live(main_pid) else None)
                    if reason:
                        if reason == "global_cpu_limit" and live(main_pid):
                            os.kill(main_pid, signal.SIGTERM)
                        break
                    if completed_record and (output / "supervisor_completed.json").exists():
                        break
                    time.sleep(10.)
            finally:
                close_worker_group(process)
    if not completed_record:
        core.write_new(output / "evaluation_prefit_finished.json", dict(exit_code=process.returncode,
            finished_at_utc=datetime.now(timezone.utc).isoformat(), workers_stopped=True,
            completed_pairs=[pair for pair in plan["variant_pairs"] if prefit_path(output, pair[0]).exists()],
            stop_reason=reason, future_choices_scored=False))
    core.write_new(output / "evaluation_prefit_supervisor_completed.json", dict(
        completed_at_utc=datetime.now(timezone.utc).isoformat(), stop_reason=reason,
        cpu_seconds_upper=maximum, additional_process_tree_cpu_seconds=external))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("supervise", "worker"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--main-pid", type=int)
    args = parser.parse_args()
    if args.phase == "worker":
        run_training_pairs(args.output.resolve())
    else:
        if args.main_pid is None:
            parser.error("--main-pid is required for supervision")
        supervise(args.output.resolve(), args.main_pid)


if __name__ == "__main__":
    main()

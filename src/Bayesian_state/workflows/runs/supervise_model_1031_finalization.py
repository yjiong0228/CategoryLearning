"""Persist the authorized Model 1031 study under one absolute resource watchdog.

The supervisor launches one compute phase at a time and terminates its owned
process group at the registered compute deadline, CPU limit, or a STOP_REQUESTED
file. It never deletes outputs. A detached launch can survive a tool session.
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

from ..analysis.model_1031_finalization import ROOT, digest, write_new
from ..analysis.model_1031_runtime_budget import compute_deadline


def tree_cpu(pid: int, proc_root: Path = Path("/proc")) -> float:
    """Include live descendants and descendants already reaped by live parents."""
    try:
        fields = (proc_root / str(pid) / "stat").read_text().split(") ", 1)[1].split()
        own = sum(float(fields[i]) for i in (11, 12, 13, 14)) / os.sysconf("SC_CLK_TCK")
        # A subprocess belongs to the thread that launched it. Concurrent
        # pipeline branches launch coordinators from non-main threads.
        children = set()
        for task in (proc_root / str(pid) / "task").iterdir():
            try:
                children.update(int(child) for child in (task / "children").read_text().split())
            except (OSError, ValueError):
                continue
        return own + sum(tree_cpu(child, proc_root) for child in children)
    except (OSError, ValueError, IndexError):
        return 0.


def stop_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    if process.poll() is None:
        try:
            process.wait(timeout=10.)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--numeric-root", required=True, type=Path)
    parser.add_argument("--prior-cpu-seconds", required=True, type=float)
    args = parser.parse_args()
    output = args.output.resolve()
    register = json.loads((output / "registration.json").read_text())
    policy = register["policy"]
    deadline = compute_deadline(output, policy)
    cpu_limit = policy["budget"]["cpu_hours"] * 3600
    if args.prior_cpu_seconds < 0:
        raise ValueError("Prior CPU accounting must be nonnegative")
    requested = False
    def on_signal(signum, frame):
        nonlocal requested
        requested = True
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    stages = [
        ("development_fit", "development_parameters.json", args.numeric_root.resolve(),
         ["src.Bayesian_state.workflows.analysis.model_1031_finalization", "--phase", "fit-development"]),
        ("remaining_pipeline", "pipeline_completed.json", ROOT,
         ["src.Bayesian_state.workflows.analysis.model_1031_pipeline", "--numeric-root", str(args.numeric_root.resolve())]),
    ]
    heartbeat = output / f"supervisor_heartbeat_{time.time_ns()}.jsonl"
    write_new(output / f"supervisor_plan_{time.time_ns()}.json", dict(pid=os.getpid(), deadline_unix=deadline,
        cpu_limit_seconds=cpu_limit, prior_cpu_seconds=args.prior_cpu_seconds,
        source_sha256=digest(Path(__file__)), heartbeat=str(heartbeat)))
    with heartbeat.open("x") as stream:
        for phase, marker, cwd, arguments in stages:
            if (output / marker).exists():
                continue
            if time.time() >= deadline or args.prior_cpu_seconds >= cpu_limit:
                raise SystemExit("Registered compute budget exhausted before phase admission")
            log = output / f"{phase}_resumed_{time.time_ns()}.log"
            command = [sys.executable, "-u", "-m", *arguments, "--output", str(output)]
            started = time.time()
            with log.open("x") as phase_stream:
                process = subprocess.Popen(command, cwd=cwd, env=env, stdout=phase_stream,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                reason = None
                last_cpu = args.prior_cpu_seconds
                try:
                    while True:
                        child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
                        self_usage = resource.getrusage(resource.RUSAGE_SELF)
                        cpu = (args.prior_cpu_seconds + child_usage.ru_utime + child_usage.ru_stime
                               + self_usage.ru_utime + self_usage.ru_stime + tree_cpu(process.pid))
                        last_cpu = max(last_cpu, cpu)
                        stream.write(json.dumps(dict(phase=phase, child_pid=process.pid,
                            utc=datetime.now(timezone.utc).isoformat(), cpu_seconds=last_cpu)) + "\n")
                        stream.flush()
                        if process.poll() is not None:
                            break
                        reason = ("stop_requested" if requested or (output / "STOP_REQUESTED").exists() else
                                  "absolute_compute_deadline" if time.time() >= deadline else
                                  "cpu_limit" if cpu >= cpu_limit else None)
                        if reason:
                            stop_group(process)
                            break
                        time.sleep(15.)
                finally:
                    stop_group(process)
            write_new(output / f"supervisor_stage_{time.time_ns()}.json", dict(phase=phase, command=command,
                seconds=time.time() - started, exit_code=process.returncode, stop_reason=reason,
                cpu_seconds_upper=last_cpu + (policy["budget"]["workers"] * 15 if process.returncode else 0.),
                log=str(log), log_sha256=digest(log)))
            if process.returncode or reason or not (output / marker).exists():
                raise SystemExit(f"Stopped {phase}: {reason or process.returncode}; results retained in {log}")
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    write_new(output / "supervisor_completed.json", dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        cpu_seconds_including_prior=args.prior_cpu_seconds + usage.ru_utime + usage.ru_stime))


if __name__ == "__main__":
    main()

"""Continue the registered study after the development fit.

Each phase has an immutable completion record. The numerical implementation
checkout remains separate from the decision and recovery coordinators.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .model_1031_finalization import ROOT, digest, validate_source, write_new
from .model_1031_runtime_budget import compute_deadline


def await_supervisor_handoff(output: Path, deadline: float) -> None:
    """Keep the old pipeline idle during an explicitly registered handoff."""
    while ((output / "supervisor_handoff_plan.json").exists()
           and not (output / "supervisor_launch_v3.json").exists()):
        if (output / "STOP_REQUESTED").exists() or time.time() >= deadline:
            raise TimeoutError("Supervisor handoff stopped before pipeline admission")
        time.sleep(.5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--numeric-root", type=Path, required=True)
    args = parser.parse_args()
    output, numeric_root = args.output.resolve(), args.numeric_root.resolve()
    registration = validate_source(output)
    deadline = compute_deadline(output, registration["policy"])
    await_supervisor_handoff(output, deadline)
    plan = output / "pipeline_plan.json"
    if not plan.exists():
        write_new(plan, dict(coordinator_sha256=digest(Path(__file__)), frozen_numerical_reference_root=str(numeric_root),
            coordinating_root=str(ROOT), deadline_unix=deadline,
            submission_policy="Long jobs first; registered numerical files reverified in the coordinating checkout before execution",
            order=["development evaluation", "two-structure precision check", "structure freeze",
                   "parallel branches: evaluation subject fits and prediction; simulation fitting and state recovery"]))
    while not (output / "development_parameters.json").exists():
        if time.time() >= deadline:
            raise TimeoutError("Development fitting did not finish before the compute deadline")
        time.sleep(15.)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    def run(module, phase, marker, root, extras=(), *, workers=None, companion_marker=None):
        if (output / marker).exists():
            print(json.dumps(dict(phase=phase, reused_marker=marker)), flush=True)
            return
        if time.time() >= deadline:
            raise TimeoutError("Compute deadline reached between phases")
        scheduled = phase in ("evaluate-development", "fit-evaluation", "evaluate-evaluation", "refine", "fit", "run", "prepare", "state")
        if scheduled:
            command = [sys.executable, "-u", "-m", "src.Bayesian_state.workflows.analysis.model_1031_scheduled",
                       "--module", module, "--phase", phase, "--output", str(output), *extras]
            root = ROOT
        else:
            command = [sys.executable, "-u", "-m", f"src.Bayesian_state.workflows.analysis.{module}",
                       "--phase", phase, "--output", str(output), *extras]
        stamp = time.time_ns()
        log = output / "pipeline_logs" / f"{stamp}_{phase}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        print(json.dumps(dict(phase=phase, log=str(log))), flush=True)
        started = time.time()
        phase_env = dict(env)
        if workers is not None:
            phase_env["MODEL1031_PHASE_WORKERS"] = str(workers)
            phase_env["MODEL1031_COMPANION_MARKER"] = str(companion_marker)
        with log.open("x") as stream:
            completed = subprocess.run(command, cwd=root, env=phase_env, stdout=stream, stderr=subprocess.STDOUT)
        write_new(output / "pipeline_steps" / f"{stamp}_{phase}.json", dict(
            command=command, cwd=str(root), started_unix=started, seconds=time.time() - started,
            exit_code=completed.returncode, log_sha256=digest(log), expected_marker=marker,
            initial_worker_quota=workers, companion_marker=str(companion_marker) if companion_marker else None))
        if completed.returncode != 0 or not (output / marker).exists():
            raise RuntimeError(f"Phase {phase} failed; inspect {log}")
    run("model_1031_finalization", "evaluate-development", "development_evaluation.json", numeric_root)
    run("model_1031_decision", "pool-development", "development_evaluation_grouped.json", ROOT)
    run("model_1031_decision", "shortlist", "structural_shortlist.json", ROOT)
    run("model_1031_decision", "refine", "structural_precision.json", ROOT)
    run("model_1031_decision", "freeze", "structure_selection.json", ROOT)
    selection = json.loads((output / "structure_selection.json").read_text())
    total = min(registration["policy"]["budget"]["workers"], len(os.sched_getaffinity(0)))
    evaluation_done = output / "evaluation_branch_completed.json"
    recovery_done = output / "recovery_branch_completed.json"
    def evaluation_branch(workers):
        kwargs = dict(workers=workers, companion_marker=recovery_done)
        run("model_1031_finalization", "fit-evaluation", "evaluation_parameters.json", numeric_root,
            ["--variants", *selection["evaluation_variants"]], **kwargs)
        run("model_1031_finalization", "evaluate-evaluation", "evaluation_evaluation.json", numeric_root, **kwargs)
        run("model_1031_decision", "pool-evaluation", "evaluation_evaluation_grouped.json", ROOT, **kwargs)
        if not evaluation_done.exists():
            write_new(evaluation_done, dict(completed_unix=time.time()))
    def recovery_branch(workers):
        kwargs = dict(workers=workers, companion_marker=evaluation_done)
        run("model_1031_recovery", "prepare", "recovery_prepared.json", ROOT, **kwargs)
        run("model_1031_recovery", "fit", "recovery_fits.json", ROOT, **kwargs)
        run("model_1031_recovery", "state", "recovery_states.json", ROOT, **kwargs)
        if not recovery_done.exists():
            write_new(recovery_done, dict(completed_unix=time.time()))
    if total < 2:
        evaluation_branch(total)
        recovery_branch(total)
    else:
        # Threads only wait on coordinators. Numerical workers remain a single
        # process layer with disjoint quotas whose sum is the registered cap.
        with ThreadPoolExecutor(max_workers=2) as coordinators:
            futures = [coordinators.submit(evaluation_branch, total // 2),
                       coordinators.submit(recovery_branch, total - total // 2)]
            for future in futures:
                future.result()
    write_new(output / "pipeline_completed.json", dict(completed_unix=time.time(),
        selected_variant=selection["selected_variant"], scope="registered numerical phases; scientific interpretation follows"))
    if (output / "evaluation_precision_plan.json").exists():
        run("model_1031_evaluation_precision", "run", "evaluation_precision.json", ROOT)


if __name__ == "__main__":
    main()

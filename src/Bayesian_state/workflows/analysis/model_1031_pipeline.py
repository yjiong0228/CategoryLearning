"""Continue the registered study sequentially after the development fit.

Each phase has an immutable completion record. The numerical implementation
checkout remains separate from the decision and recovery coordinators.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .model_1031_finalization import ROOT, digest, validate_source, write_new


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--numeric-root", type=Path, required=True)
    args = parser.parse_args()
    output, numeric_root = args.output.resolve(), args.numeric_root.resolve()
    registration = validate_source(output)
    deadline = datetime.fromisoformat(registration["policy"]["started_at_utc"]).timestamp() + registration["policy"]["budget"]["computation_stop_hours"] * 3600
    plan = output / "pipeline_plan.json"
    if not plan.exists():
        write_new(plan, dict(coordinator_sha256=digest(Path(__file__)), numerical_root=str(numeric_root),
            coordinating_root=str(ROOT), deadline_unix=deadline,
            order=["development evaluation", "two-structure precision check", "structure freeze",
                   "evaluation subject fits and prediction", "simulation model/parameter recovery", "known-parameter state recovery"]))
    while not (output / "development_parameters.json").exists():
        if time.time() >= deadline:
            raise TimeoutError("Development fitting did not finish before the compute deadline")
        time.sleep(15.)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    def run(module, phase, marker, root, extras=()):
        if (output / marker).exists():
            print(json.dumps(dict(phase=phase, reused_marker=marker)), flush=True)
            return
        if time.time() >= deadline:
            raise TimeoutError("Compute deadline reached between phases")
        command = [sys.executable, "-u", "-m", f"src.Bayesian_state.workflows.analysis.{module}",
                   "--phase", phase, "--output", str(output), *extras]
        stamp = time.time_ns()
        log = output / "pipeline_logs" / f"{stamp}_{phase}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        print(json.dumps(dict(phase=phase, log=str(log))), flush=True)
        started = time.time()
        with log.open("x") as stream:
            completed = subprocess.run(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
        write_new(output / "pipeline_steps" / f"{stamp}_{phase}.json", dict(
            command=command, cwd=str(root), started_unix=started, seconds=time.time() - started,
            exit_code=completed.returncode, log_sha256=digest(log), expected_marker=marker))
        if completed.returncode != 0 or not (output / marker).exists():
            raise RuntimeError(f"Phase {phase} failed; inspect {log}")
    run("model_1031_finalization", "evaluate-development", "development_evaluation.json", numeric_root)
    run("model_1031_decision", "shortlist", "structural_shortlist.json", ROOT)
    run("model_1031_decision", "refine", "structural_precision.json", ROOT)
    run("model_1031_decision", "freeze", "structure_selection.json", ROOT)
    selection = json.loads((output / "structure_selection.json").read_text())
    run("model_1031_finalization", "fit-evaluation", "evaluation_parameters.json", numeric_root,
        ["--variants", *selection["evaluation_variants"]])
    run("model_1031_finalization", "evaluate-evaluation", "evaluation_evaluation.json", numeric_root)
    run("model_1031_recovery", "prepare", "recovery_prepared.json", ROOT)
    run("model_1031_recovery", "fit", "recovery_fits.json", ROOT)
    run("model_1031_recovery", "state", "recovery_states.json", ROOT)
    write_new(output / "pipeline_completed.json", dict(completed_unix=time.time(),
        selected_variant=selection["selected_variant"], scope="registered numerical phases; scientific interpretation follows"))


if __name__ == "__main__":
    main()

"""Run an existing joint-recovery case with an explicit worker allocation.

The prepared scientific settings, candidate counts, particles and seeds remain
unchanged. Only the budget passed to the existing process dispatcher changes;
each stage records that override separately. The original shared worker limiter
and one-numeric-thread worker environment still apply. Smoke always uses one
worker. Use a new output directory, including for numerical regression checks.

Example: python -m src.Bayesian_state.workflows.analysis.model_0923_joint_recovery_allocation
--phase search --prepared PREPARED --replicate 1 --workers 96 --output NEW_DIRECTORY
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import os
from pathlib import Path

from . import model_0923_joint_parameter_recovery as recovery
from .model_0923_h_validation import write
from ...evaluation.model_0923_h_validation import digest

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
SCIENTIFIC_JOB_FIELDS = (
    "phase", "folder", "model", "mechanism", "candidate", "runtime",
    "seeds", "particles", "smoke", "arm_settings",
)


@contextmanager
def worker_allocation(workers: int, *, smoke: bool = False):
    """Override dispatch resources while preserving prepared scientific policy."""
    if type(workers) is not int or not 1 <= workers <= 128:
        raise ValueError("Declare a worker allocation between 1 and 128")
    allocated = 1 if smoke else min(workers, len(os.sched_getaffinity(0)))
    original_configure = recovery.configure

    def configure(settings, inputs):
        inputs = dict(inputs)
        inputs[str(Path(__file__).resolve())] = digest(Path(__file__))
        module = original_configure(settings, inputs)
        original_execute = module.base.execute
        original_hashes = module.base.source_hashes

        def hashes(validation):
            return {**original_hashes(validation), ENTRY: digest(ROOT / ENTRY)}

        def execute(jobs, output, prepared_budget):
            stages = {job["name"].split("_", 1)[0] for job in jobs}
            if len(stages) != 1:
                raise ValueError("Resource records require one nonempty fitting stage")
            stage = stages.pop()
            path = output / f"{stage}_worker_allocation.json"
            if path.exists():
                raise FileExistsError("A stage allocation has already been recorded")
            identities = [
                {key: deepcopy(job[key]) for key in SCIENTIFIC_JOB_FIELDS if key in job}
                for job in jobs
            ]
            policy = deepcopy(settings)
            used = min(allocated, len(jobs))
            write(path, dict(
                stage=stage, prepared_worker_budget=prepared_budget,
                allocated_worker_ceiling=allocated, ready_tasks=len(jobs),
                dispatched_worker_ceiling=used, numeric_threads=1, smoke=smoke,
                dispatcher="existing shared-budget process dispatcher",
                scientific_policy="unchanged prepared settings, jobs, particles and seeds",
                source_sha256=digest(Path(__file__)),
            ))
            original_execute(jobs, output, used)
            assert settings == policy, "Dispatch changed the scientific fitting settings"
            for job, expected in zip(jobs, identities):
                assert {key: job[key] for key in expected} == expected

        module.base.source_hashes = hashes
        module.base.execute = execute
        return module

    recovery.configure = configure
    try:
        yield
    finally:
        recovery.configure = original_configure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("search", "score-truth"), required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--fit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    if args.phase == "score-truth" and args.fit is None:
        parser.error("--fit is required for score-truth")
    with worker_allocation(args.workers, smoke=args.smoke):
        if args.phase == "search":
            recovery.search(args.prepared, args.replicate, args.output, args.smoke)
        else:
            recovery.score_truth(args.prepared, args.replicate, args.fit, args.output, args.smoke)


if __name__ == "__main__":
    main()

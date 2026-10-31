"""Submit long registered jobs first while retaining their numerical identity.

This launcher changes task order only. Shared worker functions, candidate
selection, seeds, particle counts and the absolute budget remain unchanged.
The checked-out numerical files are verified against the study registration.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import importlib
import json
from pathlib import Path
import sys
import time

from .model_1031_finalization import Scheduler, digest, identity, validate_source, write_new


def ordered_jobs(jobs: list[dict], records: list[dict], full_lengths: dict[int, int]) -> tuple[list[dict], dict]:
    rates = defaultdict(list)
    for record in records:
        job = record["job"]
        if job["kind"] == "fit":
            rates[job["subject"], job["config"]].append(record["cpu_seconds"] / (job["particles"] * record["trials"]))
    estimates = {}
    for job in jobs:
        n = job["train_trials"] if job.get("scope") == "train" else full_lengths[job["subject"]]
        if job.get("max_trials") is not None:
            n = min(n, job["max_trials"])
        rate = max(rates[job["subject"], job["config"]], default=.003 if job["condition"] == 1 else .009)
        count = job["particles"] if job["kind"] == "fit" else len(job["seeds"])
        estimates[job["id"]] = float(rate * count * n)
    result = sorted(jobs, key=lambda job: -estimates[job["id"]])
    if sorted(job["id"] for job in result) != sorted(job["id"] for job in jobs):
        raise RuntimeError("Submission ordering changed the registered job multiset")
    return result, estimates


class LongestFirstScheduler(Scheduler):
    """Reuse the registered pool and worker without adding another parallel layer."""

    def run(self, jobs: list[dict], label: str) -> dict[str, dict]:
        register = validate_source(self.output)
        lengths = {subject["subject"]: subject["trials"] for subject in register["subjects"]}
        paths = list((self.output / "jobs").glob("*.json")) + list((self.output / "recovery").glob("*/jobs/*.json"))
        records = [json.loads(path.read_text()) for path in paths]
        ordered, estimates = ordered_jobs(jobs, records, lengths)
        write_new(self.output / "submission_orders" / f"{time.time_ns()}_{label}.json", dict(
            phase=label, policy="descending observed CPU cost, with condition fallback", estimates=estimates,
            original_order=[job["id"] for job in jobs], submitted_order=[job["id"] for job in ordered],
            original_jobs_sha256=identity(dict(jobs=jobs)), numerical_jobs_changed=False,
            scheduler_source_sha256=digest(Path(__file__))))
        return super().run(ordered, label)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--module", required=True, choices=("model_1031_finalization", "model_1031_decision",
        "model_1031_recovery", "model_1031_evaluation_precision"))
    args, remaining = parser.parse_known_args()
    module = importlib.import_module(f"{__package__}.{args.module}")
    module.Scheduler = LongestFirstScheduler
    sys.argv = [sys.argv[0], *remaining]
    module.main()


if __name__ == "__main__":
    main()

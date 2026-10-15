"""Generate every frozen finalist with its explicitly declared H sampler.

The existing finalist workflow owns simulation budgets, trajectory assembly and
scoring. The candidate adapter installs fixed-cardinality competition only for
named models and checks fixed parameter values in each worker. No nominees are
changed. Use --fit FIT --config FINALIST_CONFIG --output NEW_OUTPUT from the
repository root; small software fixtures should declare one worker explicitly.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import model_0923_h_candidate_validation as adapter
from . import model_0923_h_finalist_generation as finalists


def run(fit: Path, config: Path, output: Path) -> None:
    validation = adapter.validation()
    frozen = validation.read(fit / 'nominees.json')
    adapter.configure(validation, frozen['settings'])
    original_hashes = validation.source_hashes

    def source_hashes() -> dict:
        hashes = original_hashes()
        hashes[str(Path(__file__).resolve().relative_to(adapter.REPO))] = validation.digest(Path(__file__))
        return hashes

    finalists.execute = validation.execute
    finalists.score_jobs = validation.score_jobs
    finalists.source_hashes = source_hashes
    finalists.run(fit, config, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['fit', 'config', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.fit, args.config, args.output)

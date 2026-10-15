"""Audit task reconstruction after objective fits are frozen.

Task-order, block-completeness and allocation mismatches exclude a participant
from this environment diagnostic only. File changes and missing inputs stop the
audit. The original fitting cohort remains intact.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from . import model_0923_h_validation as v
from .model_0923_h_adaptive_task import prepare_schedules


def run(fit: Path, config: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new task-audit destination')
    frozen = v.read(fit/'nominees.json')
    if frozen['smoke'] or frozen['status'] != 'frozen_before_test':
        raise ValueError('Complete scientific parameter freezing before auditing future task inputs')
    settings = yaml.safe_load(config.read_text())
    output.mkdir(parents=True)
    checks, exclusions = [], {}
    for subject in sorted({r['subject'] for r in frozen['nominees']}):
        destination = output/f'subject_{subject}'
        destination.mkdir()
        try:
            rows = prepare_schedules(v, frozen, destination, [subject])
        except ValueError as exc:
            if str(exc) not in ('Original task order is ambiguous', 'Need complete original blocks'):
                raise
            exclusions[subject] = str(exc)
        except AssertionError as exc:
            # Array assertions in prepare_schedules audit labels, prepared trial
            # alignment and adaptive allocations after verifying input hashes.
            exclusions[subject] = 'Task reconstruction array mismatch: '+str(exc).strip()
        else:
            checks.extend(rows)
    settings['excluded_subjects'] = dict(settings.get('excluded_subjects', {})) | exclusions
    with (output/'task_config.yaml').open('x') as handle:
        yaml.safe_dump(settings, handle, sort_keys=False)
    v.write(output/'completion.json', dict(status='complete',
        nomination_sha256=v.digest(fit/'nominees.json'), config_sha256=v.digest(config),
        source_sha256=v.digest(Path(__file__)), checks=checks, exclusions=exclusions,
        scope='task_environment_only; primary_fitting_and_prediction_cohort_unchanged'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fit', 'config', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.fit, args.config, args.output)


if __name__ == '__main__':
    main()

"""Continue a running simplification fit through behavior and oral evaluation.

Use --fit FIT --capture-validation VALIDATION --output NEW. Wait at most three
hours for both completed inputs, then execute the registered frozen evaluation,
behavior summary and two existing oral codings. Each stage writes a new log and
completion marker. A failed stage stops the chain and preserves all artifacts.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .model_0923_h_validation import digest, read, write
from .model_1021_simplification_capture import sources

ROOT = Path(__file__).resolve().parents[4]


def completed(path: Path) -> dict | None:
    try:
        value = read(path / 'completion.json')
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if value.get('status') != 'complete':
        raise ValueError('Unexpected prerequisite status: ' + str(path))
    return value


def run(fit: Path, validation: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new followthrough directory')
    fit, validation, output = fit.resolve(), validation.resolve(), output.resolve()
    output.mkdir(parents=True)
    evaluation = output / 'evaluation'
    steps = [
        ('evaluation', 'src.Bayesian_state.workflows.analysis.model_1021_simplification_capture',
         ['--fit', str(fit), '--output', str(evaluation)]),
        ('behavior_summary', 'src.Bayesian_state.evaluation.model_1021_simplification',
         ['--fit', str(fit), '--evaluation', str(evaluation), '--output', str(output / 'behavior_summary')]),
    ]
    for name, audit in [('oral_midpoint', 'oral_midpoint_audit_20261018_v1'),
                        ('oral_relative', 'oral_relative_sensitivity_20261018_v1')]:
        steps.append((name, 'src.Bayesian_state.workflows.analysis.model_1021_simplification_oral',
            ['--evaluation', str(evaluation), '--oral-audit', str(ROOT / 'results/model_0923' / audit),
             '--output', str(output / name)]))
    write(output / 'plan.json', dict(fit=str(fit), capture_validation=str(validation),
        source_sha256=digest(Path(__file__)), capture_sources=sources(),
        steps=[dict(name=n, command=[sys.executable, '-u', '-m', m, *args]) for n, m, args in steps],
        role='Development mechanism comparison; individual parameters selected by original training behavior'))
    started = time.monotonic()
    while True:
        fitted, checked = completed(fit), completed(validation)
        if fitted and checked:
            break
        if time.monotonic() - started > 10800:
            raise TimeoutError('Prerequisite fit or capture check did not finish within three hours')
        time.sleep(10)
    if (fitted.get('smoke') is not False or fitted.get('test_choices_used') is not False
            or checked.get('capture_source_sha256') != sources() or len(checked.get('cases', [])) != 6
            or any(row.get('exact') is not True for row in checked['cases'])):
        raise ValueError('Require the scientific prefix fit and six exact capture checks')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', NUMBA_NUM_THREADS='1')
    for name, module, args in steps:
        print('Starting ' + name, flush=True)
        command = [sys.executable, '-u', '-m', module, *args]
        write(output / (name + '_started.json'), dict(command=command, started_unix=time.time()))
        with (output / (name + '.log')).open('x') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            write(output / (name + '_failed.json'), dict(returncode=result.returncode, command=command))
            raise RuntimeError('Followthrough failed; see ' + str(output / (name + '.log')))
        write(output / (name + '_completed.json'), dict(status='complete', finished_unix=time.time()))
        print('Completed ' + name, flush=True)
    write(output / 'completion.json', dict(status='complete', steps=[n for n, _, _ in steps],
        elapsed_seconds=time.monotonic()-started, parameter_recovery='next evidence-dependent stage'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--capture-validation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.fit, args.capture_validation, args.output)

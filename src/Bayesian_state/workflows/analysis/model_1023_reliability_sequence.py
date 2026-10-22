"""Run the registered estimation phases sequentially after a live diagnostic.

Use --config YAML --diagnostic DIR --output NEW.
The diagnostic's worker snapshots provide a heartbeat across process sandboxes.
After successful completion and numerical-budget audit, launch the three fresh
recovery cases, six human prefix fits, paired frozen evaluation and summaries.
Only one numerical queue runs at a time. A failed command stops downstream work
and retains logs and results. Scientific interpretation and figures follow the
completed summaries; this controller never declares the research goal achieved.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .model_1023_estimation_reliability import ROOT, read, write, digest


def wait_for_diagnostic(folder: Path) -> None:
    while True:
        try:
            complete = read(folder/'completion.json')
        except (FileNotFoundError, json.JSONDecodeError):
            complete = None
        if complete is not None:
            break
        snapshots = list((folder/'resource_snapshots').glob('*.json'))
        final = folder/'resource_status.json'
        if final.exists():
            snapshots.append(final)
        if not snapshots:
            raise RuntimeError('Diagnostic has no worker heartbeat')
        latest = max(snapshots, key=lambda p: p.stat().st_mtime)
        try:
            failed = read(latest)['failed']
        except json.JSONDecodeError:
            failed = False  # The newest exclusive-create snapshot may still be writing.
        if failed or time.time()-latest.stat().st_mtime > 600:
            raise RuntimeError('Diagnostic failed or its worker heartbeat stopped for ten minutes')
        time.sleep(10)
    resources = read(folder/'resource_status.json')
    if (complete['status'] != 'complete' or complete['smoke'] or complete['phase'] != 'diagnostic'
            or resources['active'] or resources['failed']):
        raise ValueError('Require a successful scientific diagnostic with no remaining numerical workers')


def run(config: Path, diagnostic: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new continuation directory')
    output.mkdir(parents=True)
    oral_config = ROOT/'configs/shared/model1022_compact_evaluation.yaml'
    inputs = {str(p.resolve()): digest(p) for p in (config, oral_config, Path(__file__))}
    write(output/'protocol.json', dict(status='waiting_for_diagnostic', diagnostic=str(diagnostic.resolve()),
        inputs_sha256=inputs, numerical_queues='One at a time, each capped at 128',
        output_policy='All new directories; previous diagnostic is read-only'))
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR='/tmp/model1023_mpl')
    estimator = 'src.Bayesian_state.workflows.analysis.model_1023_estimation_reliability'
    auditor = 'src.Bayesian_state.evaluation.model_1023_estimation_reliability'
    evaluation = output/'human_evaluation'
    steps = [("diagnostic_summary", auditor, ['--study', diagnostic, '--output', output/'diagnostic_summary'])]
    for phase in ('validation', 'human'):
        study = output/phase
        steps.extend([(phase, estimator, ['--config', config, '--phase', phase, '--output', study]),
                      (phase+'_summary', auditor, ['--study', study, '--output', output/(phase+'_summary')])])
    steps.extend([
        ('human_evaluation', 'src.Bayesian_state.workflows.analysis.model_1023_frozen_evaluate',
         ['--study', output/'human', '--output', evaluation]),
        ('human_external_summary', 'src.Bayesian_state.evaluation.model_1023_frozen_evaluate',
         ['--evaluation', evaluation, '--config', oral_config, '--output', output/'human_external_summary']),
    ])
    write(output/'planned_steps.json', dict(steps=[dict(name=name, module=module, args=list(map(str, args)))
                                                  for name, module, args in steps]))
    current = 'diagnostic_wait'
    try:
        wait_for_diagnostic(diagnostic)
        for number, (current, module, args) in enumerate(steps):
            for file, sha in inputs.items():
                if digest(Path(file)) != sha:
                    raise ValueError('Registered sequence input changed: '+file)
            command = [sys.executable, '-u', '-m', module, *map(str, args)]
            started = time.time()
            write(output/f'{number:02d}_{current}_started.json', dict(command=command, unix_time=started))
            print(current+': started', flush=True)
            with (output/(current+'.log')).open('x') as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            write(output/f'{number:02d}_{current}_completed.json',
                  dict(returncode=result.returncode, seconds=time.time()-started, unix_time=time.time()))
            print(current+': completed', flush=True)
    except BaseException as error:
        write(output/'failure.json', dict(stage=current, error=repr(error), unix_time=time.time()))
        raise
    write(output/'completion.json', dict(status='complete', steps=len(steps),
        summaries_complete=True, scientific_interpretation_and_figures='pending'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'diagnostic', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.diagnostic, args.output)

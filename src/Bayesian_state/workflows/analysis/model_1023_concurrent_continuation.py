"""Continue the frozen study using a shared cap across independent phases.

Validation computation may use idle diagnostic capacity. Human fitting starts
after the diagnostic completes, while independent validation calculations can
continue. Each phase retains its original inputs, seeds and candidate budget;
reports and substantive decisions use the completed diagnostic/validation
results. Controllers dispatch work only; all numerical subprocesses share one
set of at most 128 single-thread running slots. Existing serial continuation
must be stopped before starting this replacement controller.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import os
import shutil
import subprocess
import sys
from threading import Event
import time

from .model_1023_compute_budget import create_budget, summarize_budget
from .model_1023_estimation_reliability import ROOT, read, write, digest
from .model_1023_reliability_sequence import wait_for_diagnostic


def run(config: Path, diagnostic: Path, output: Path, figures: Path) -> None:
    if output.exists() or figures.exists():
        raise FileExistsError('Use new continuation and figure destinations')
    output.mkdir(parents=True)
    figures.mkdir(parents=True)
    budget = output/'global_budget'
    create_budget(budget, min(128, len(os.sched_getaffinity(0))), diagnostic)
    oral_config = ROOT/'configs/shared/model1022_compact_evaluation.yaml'
    config_hashes = {config.resolve(): digest(config), oral_config.resolve(): digest(oral_config)}
    write(output/'protocol.json', dict(config=str(config.resolve()), config_sha256=digest(config),
        diagnostic=str(diagnostic.resolve()), figures=str(figures.resolve()),
        source_sha256={str(Path(__file__).resolve()): digest(Path(__file__))},
        scheduling='Independent phase controllers share global numerical leases; human fitting follows diagnostic completion',
        scientific_plan='Original registered data, seeds, support and per-case candidate/particle counts',
        validation_use='Validation and human numerical computations are independent; substantive interpretation waits for both'))
    shutil.copyfile(Path(__file__), output/'source_snapshot.py')
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR='/tmp/model1023_mpl')
    gate_module = 'src.Bayesian_state.workflows.analysis.model_1023_compute_budget'
    audit_module = 'src.Bayesian_state.evaluation.model_1023_estimation_reliability'
    figure_module = 'CategoryLearning_codes.figures.fig3.model1023_estimation_reliability'
    diagnostic_finished = Event()

    def step(name, module, args):
        if any(digest(path) != sha for path, sha in config_hashes.items()):
            raise ValueError('Registered search or oral configuration changed')
        command = [sys.executable, '-u', '-m', module, *map(str, args)]
        start = time.time()
        write(output/(name+'_started.json'), dict(command=command, unix_time=start))
        print(name+': started', flush=True)
        with (output/(name+'.log')).open('x') as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        write(output/(name+'_finished.json'), dict(returncode=result.returncode, seconds=time.time()-start))
        if result.returncode:
            raise RuntimeError(name+' failed; see '+str(output/(name+'.log')))
        print(name+': complete', flush=True)

    def diagnostic_branch():
        try:
            wait_for_diagnostic(diagnostic)
            step('diagnostic_summary', audit_module, ['--study', diagnostic, '--output', output/'diagnostic_summary'])
        finally:
            diagnostic_finished.set()
        step('diagnostic_figures', figure_module, ['--summary', output/'diagnostic_summary', '--output', figures/'diagnostic'])

    def validation_branch():
        step('validation', gate_module, ['--kind', 'fit', '--phase', 'validation', '--config', config,
             '--budget', budget, '--output', output/'validation'])
        step('validation_summary', audit_module, ['--study', output/'validation', '--output', output/'validation_summary'])
        step('validation_figures', figure_module, ['--summary', output/'validation_summary', '--output', figures/'validation'])

    def human_branch():
        diagnostic_finished.wait()
        if not (output/'diagnostic_summary/completion.json').exists():
            raise RuntimeError('Human phase requires the completed diagnostic budget audit')
        step('human', gate_module, ['--kind', 'fit', '--phase', 'human', '--config', config,
             '--budget', budget, '--output', output/'human'])
        step('human_summary', audit_module, ['--study', output/'human', '--output', output/'human_summary'])
        step('human_evaluation', gate_module, ['--kind', 'evaluate', '--study', output/'human',
             '--budget', budget, '--output', output/'human_evaluation'])
        step('human_external_summary', 'src.Bayesian_state.evaluation.model_1023_frozen_evaluate',
             ['--evaluation', output/'human_evaluation', '--config', oral_config, '--output', output/'human_external_summary'])
        step('human_state_sensitivity', 'src.Bayesian_state.evaluation.model_1023_state_sensitivity',
             ['--evaluation', output/'human_evaluation', '--audit-summary', output/'human_summary', '--output', output/'human_state_sensitivity'])
        step('human_parameter_figures', figure_module, ['--summary', output/'human_summary', '--output', figures/'human_parameters'])
        step('human_figures', 'CategoryLearning_codes.figures.fig3.model1023_frozen_evaluate',
             ['--source', output/'human_external_summary', '--states', output/'human_state_sensitivity', '--output', figures/'human'])

    failures = []
    with ThreadPoolExecutor(max_workers=3) as controls:
        futures = {controls.submit(function): name for name, function in
                   [('diagnostic', diagnostic_branch), ('validation', validation_branch), ('human', human_branch)]}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                failure = dict(branch=futures[future], error=repr(error))
                failures.append(failure)
                write(output/(futures[future]+'_failure.json'), failure)
    resources = summarize_budget(budget)
    write(output/'global_resource_audit.json', resources)
    if failures:
        write(output/'failure.json', dict(failures=failures, resources=resources))
        raise RuntimeError('Continuation has failed stages; independent completed results were retained')
    write(output/'completion.json', dict(status='complete', figures=str(figures.resolve()),
        numerical_and_summary_steps='complete', scientific_interpretation_and_visual_qa='pending', resources=resources))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'diagnostic', 'output', 'figures'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.diagnostic, args.output, args.figures)

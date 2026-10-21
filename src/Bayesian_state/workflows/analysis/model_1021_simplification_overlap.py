"""Overlap frozen evaluation and COMPACT recovery in one numerical worker pool.

Use --fit FIT --capture-validation VALIDATION --output NEW. The original
evaluation workers retain their seeds and settings. Once every autonomous
worker exits successfully, its saved paths supply the registered recovery.
Evaluation and recovery share at most 128 single-threaded numerical processes.
Behavior and oral summaries follow the completed evaluation. A generation
snapshot is checked against the eventual canonical evaluation arrays.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess
import sys
from threading import Event
import time

import numpy as np

from . import model_1021_simplification_capture as capture
from . import model_1021_simplification_recovery as recovery
from .model_1021_simplification_followthrough import completed
from ...evaluation.model_0923_h_validation import checked_arrays
from ...utils.seeding import stable_seed

ROOT = Path(__file__).resolve().parents[4]
read, write, digest = recovery.read, recovery.write, recovery.digest


class EvaluationDispatch:
    """Signal generation readiness only after every autonomous process exits."""

    def __init__(self, pool: ThreadPoolExecutor, launch):
        self.pool = pool
        self.launch = launch
        self.generation_ready = Event()
        self.error: BaseException | None = None

    def __call__(self, jobs: list[dict], folder: Path, budget: int) -> None:
        if self.pool._max_workers > budget or len({j['name'] for j in jobs}) != len(jobs):
            raise ValueError('Invalid shared budget or duplicate evaluation job')
        remaining = {j['name'] for j in jobs if j['phase'] == 'autonomous'}
        if not remaining:
            raise ValueError('Require autonomous generation jobs')
        # Each delegated launcher runs one original worker. Waiting threads
        # perform orchestration; all numerical processes occupy this one pool.
        futures = {self.pool.submit(self.launch, [job], folder, 1): job for job in jobs}
        try:
            for future in as_completed(futures):
                future.result()
                remaining.discard(futures[future]['name'])
                if not remaining:
                    self.generation_ready.set()
        except BaseException as error:
            self.error = error
            self.generation_ready.set()
            raise


def snapshot_generation(evaluation: Path, output: Path) -> None:
    """Aggregate completed generation with the original combination routine."""
    if output.exists():
        raise FileExistsError('Use a new generation snapshot directory')
    declared = read(evaluation/'protocol.json')
    fit = Path(declared['fit'])
    frozen = read(fit/'nominees.json')
    if frozen['smoke'] or declared['nomination_sha256'] != digest(fit/'nominees.json'):
        raise ValueError('Require generation at the complete frozen scientific fit')
    settings = frozen['settings']
    nominees = {(r['subject'], r['model']): r['candidate'] for r in frozen['nominees']}
    groups = {key: [] for key in nominees}
    bindings = {str((evaluation/'protocol.json').resolve()): digest(evaluation/'protocol.json')}
    jobs = [j for j in declared['jobs'] if j['phase'] == 'autonomous']
    output.mkdir(parents=True)
    (output/'jobs').mkdir()
    for expected in jobs:
        file = evaluation/'jobs'/(expected['name']+'.json')
        job = read(file)
        subject = read(Path(job['folder'])/'prepared.json')['subject']
        key = subject, job['model']
        if (key not in groups or job['candidate'] != nominees[key] or job['seeds'] != expected['seeds']
                or job['source_sha256'] != frozen['source_sha256'] or job['phase'] != 'autonomous'
                or job['target'] != expected['target'] or job['folder'] != expected['folder']):
            raise ValueError('Generation job differs from its frozen declaration')
        target = Path(job['target'])
        record = read(target.with_suffix('.json'))
        read(target.with_suffix('.mechanism.json'))
        if (record['subject'], record['model'], record['candidate']) != (subject, job['model'], job['candidate']):
            raise ValueError('Generation record differs from its worker job')
        if digest(target.with_suffix('.npz')) != record['predictions_sha256']:
            raise ValueError('Generation chunk changed')
        bindings[str(file.resolve())] = digest(file)
        bindings[str(target.with_suffix('.npz').resolve())] = record['predictions_sha256']
        shutil.copyfile(file, output/'jobs'/file.name)
        groups[key].append(job)
    rows = []
    for (subject, arm), chunks in groups.items():
        seeds = [stable_seed(dict(protocol=settings['protocol'], base=settings['seed'], role='autonomous',
                                 subject=subject, path=i)) for i in range(settings['autonomous_paths'])]
        order = {seed: i for i, seed in enumerate(seeds)}
        chunks.sort(key=lambda c: order[c['seeds'][0]])
        if [seed for c in chunks for seed in c['seeds']] != seeds:
            raise ValueError('Missing, reordered or duplicated autonomous paths')
        aggregate = deepcopy(chunks[0])
        target = output/'autonomous/evaluate'/f'subject_{subject}'/arm/nominees[subject, arm]['id']
        target.parent.mkdir(parents=True)
        aggregate.update(target=str(target.resolve()), seeds=seeds)
        rows.append(capture.driver.io.combine_evaluation_chunks(aggregate, chunks, True))
    write(output/'autonomous/evaluation.json', dict(status='complete', rows=rows))
    write(output/'protocol.json', dict(fit=str(fit.resolve()), nomination_sha256=digest(fit/'nominees.json'),
        kind='autonomous_generation_snapshot', source_evaluation=str(evaluation.resolve()),
        source_sha256=frozen['source_sha256'], scope='Completed generation only; conditional evaluation may be running'))
    write(output/'completion.json', dict(status='complete', kind='autonomous_generation_snapshot',
        jobs=len(jobs), inputs_sha256=bindings, nomination_sha256=digest(fit/'nominees.json')))


def check_snapshot(snapshot: Path, evaluation: Path) -> dict:
    """Require exact arrays after the original evaluator finishes its own merge."""
    if completed(evaluation) is None:
        raise ValueError('Canonical evaluation is incomplete')
    rows = read(evaluation/'autonomous/evaluation.json')['rows']
    canonical = {(r['subject'], r['model']): r for r in rows}
    early = read(snapshot/'autonomous/evaluation.json')['rows']
    if len(early) != len(rows):
        raise ValueError('Generation snapshot omitted cells')
    for row in early:
        a = checked_arrays(snapshot/'autonomous', row)
        b = checked_arrays(evaluation/'autonomous', canonical[row['subject'], row['model']])
        if set(a) != set(b):
            raise ValueError('Generation fields changed')
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
    return dict(status='complete', cells=len(rows), all_generation_arrays_exact=True,
        snapshot_completion_sha256=digest(snapshot/'completion.json'),
        evaluation_completion_sha256=digest(evaluation/'completion.json'))


def evaluate_with_recovery(fit: Path, output: Path, *, software_check: bool = False) -> None:
    if output.exists():
        raise FileExistsError('Use a new overlap destination')
    if capture.driver.base is recovery.driver.base:
        raise RuntimeError('Evaluation and recovery require independent controller namespaces')
    output.mkdir(parents=True)
    workers = 1 if software_check else min(128, len(os.sched_getaffinity(0)))
    write(output/'overlap_protocol.json', dict(fit=str(fit.resolve()), software_check=software_check,
        workers=workers, numeric_threads=1, source_sha256=digest(Path(__file__)),
        recovery_sources=recovery.recovery_hashes(recovery.driver.io),
        boundary='All parameters frozen; recovery starts after every autonomous worker exits'))
    evaluation, snapshot = output/'evaluation', output/'generation_ready'
    prepared, recovered, scored = [output/name for name in ('recovery_prepared', 'recovery_fit', 'recovery_scores')]
    original = capture.driver.execute
    with ThreadPoolExecutor(max_workers=workers) as pool:
        dispatch = EvaluationDispatch(pool, original)
        capture.driver.execute = dispatch

        def evaluate():
            try:
                capture.evaluate(fit, evaluation)
            except BaseException as error:
                dispatch.error = error
                dispatch.generation_ready.set()
                raise

        def recover():
            dispatch.generation_ready.wait()
            if dispatch.error is not None:
                raise RuntimeError('Evaluation failed before recovery readiness') from dispatch.error
            snapshot_generation(evaluation, snapshot)
            print('Generation complete; starting joint recovery in the shared pool', flush=True)
            recovery.prepare(snapshot, prepared)
            recovery.search(prepared, recovered, software_check, numerical_pool=pool)
            recovery.score_truth(prepared, recovered, scored, numerical_pool=pool)
            from ...evaluation.model_1021_simplification_recovery_states import summarize
            summarize(prepared, recovered, scored, output/'recovery_states')

        try:
            with ThreadPoolExecutor(max_workers=2) as controllers:
                futures = [controllers.submit(evaluate), controllers.submit(recover)]
                for future in as_completed(futures):
                    future.result()
        finally:
            capture.driver.execute = original
    agreement = check_snapshot(snapshot, evaluation)
    write(output/'generation_agreement.json', agreement)
    write(output/'overlap_completed.json', dict(status='complete', software_check=software_check,
        evaluation_complete=True, recovery_complete=True, shared_worker_limit=workers))


def run(fit: Path, validation: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new followthrough directory')
    started = time.monotonic()
    while completed(fit) is None or completed(validation) is None:
        if time.monotonic()-started > 10800:
            raise TimeoutError('Prerequisites did not complete within three hours')
        time.sleep(10)
    fitted, checked = completed(fit), completed(validation)
    if (fitted.get('smoke') is not False or fitted.get('test_choices_used') is not False
            or checked.get('capture_source_sha256') != capture.sources() or len(checked.get('cases', [])) != 6
            or any(row.get('exact') is not True for row in checked['cases'])):
        raise ValueError('Require scientific fitting and all six exact capture checks')
    evaluate_with_recovery(fit.resolve(), output.resolve())
    steps = [('behavior_summary', 'src.Bayesian_state.evaluation.model_1021_simplification',
        ['--fit', str(fit.resolve()), '--evaluation', str((output/'evaluation').resolve())])]
    for name, audit in [('oral_midpoint', 'oral_midpoint_audit_20261018_v1'),
                        ('oral_relative', 'oral_relative_sensitivity_20261018_v1')]:
        steps.append((name, 'src.Bayesian_state.workflows.analysis.model_1021_simplification_oral',
            ['--evaluation', str((output/'evaluation').resolve()), '--oral-audit', str(ROOT/'results/model_0923'/audit)]))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', NUMBA_NUM_THREADS='1')
    for name, module, arguments in steps:
        command = [sys.executable, '-u', '-m', module, *arguments, '--output', str((output/name).resolve())]
        write(output/(name+'_started.json'), dict(command=command, started_unix=time.time()))
        with (output/(name+'.log')).open('x') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            write(output/(name+'_failed.json'), dict(returncode=result.returncode))
            raise RuntimeError('Summary failed; see '+str(output/(name+'.log')))
        print('Completed '+name, flush=True)
    write(output/'completion.json', dict(status='complete', steps=['evaluation', 'joint_recovery', *[s[0] for s in steps]],
        elapsed_seconds=time.monotonic()-started, shared_worker_limit=min(128, len(os.sched_getaffinity(0)))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fit', 'capture-validation', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.fit, args.capture_validation, args.output)

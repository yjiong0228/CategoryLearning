"""Run frozen episodic-admission fits in the shared exp123 task environment.

Use --config CONFIG --output NEW_DIRECTORY. --subjects may declare a subset.
--smoke runs one subject, all declared arms, one path per task policy and one
worker; adaptive paths stop at two blocks while fixed replay remains complete.
The original path seeds and fixed-schedule references are retained. Cognitive
operations and task policies are delegated to the existing shared task runner.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from threading import Condition
import time

import numpy as np
import pandas as pd
import yaml

from . import model_0923_h_adaptive_task as adaptive

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
WORKER = 'src/Bayesian_state/workflows/analysis/model_0923_h_adaptive_task.py'


class SharedWorkerBudget:
    """Use released slots while the preceding exact-replay queue finishes.

    The predecessor owns up to its declared worker count. Counting all its
    unfinished tasks is conservative, including tasks not yet launched.
    This gate coordinates worker launches only; no cognitive RNG is consumed.
    """
    def __init__(self, budget: int, predecessor: Path | None):
        self.budget, self.predecessor = budget, predecessor
        self.active, self.condition = 0, Condition()
        self.protocol = (json.loads((predecessor / 'protocol.json').read_text())
                         if predecessor is not None else None)

    def reserved(self) -> int:
        if self.predecessor is None or (self.predecessor / 'completion.json').exists():
            return 0
        completed = len(list((self.predecessor / 'states').glob('*.json')))
        return min(self.protocol['workers'], max(0, self.protocol['jobs'] - completed))

    @contextmanager
    def slot(self):
        with self.condition:
            while self.active >= self.budget - self.reserved():
                self.condition.wait(timeout=2.)
            self.active += 1
        try:
            yield
        finally:
            with self.condition:
                self.active -= 1
                self.condition.notify_all()


def prepare(config: Path, output: Path, smoke: bool, subjects: list[int] | None) -> tuple[dict, list[dict]]:
    """Verify fit, every seed group and task allocations before launching jobs."""
    v = adaptive.validation()
    settings = yaml.safe_load(config.read_text())
    source, fit = ROOT / settings['reference'], ROOT / settings['fit']
    frozen = v.read(fit / 'nominees.json')
    if output.exists():
        raise FileExistsError('Use a new adaptive admission study directory')
    if frozen['smoke'] or frozen['status'] != 'frozen_before_test':
        raise ValueError('Need frozen scientific fits')
    completion = v.read(source / 'completion.json')
    if completion['status'] != 'complete' or completion['nomination_sha256'] != v.digest(fit / 'nominees.json'):
        raise ValueError('Reference evaluation differs from the frozen fit')
    for name, sha in frozen['source_sha256'].items():
        if v.digest(ROOT / name) != sha:
            raise ValueError(f'Frozen implementation changed: {name}')
    selected = settings['subjects'] if subjects is None else subjects
    if len(set(selected)) != len(selected) or not set(selected) <= set(settings['subjects']):
        raise ValueError('Select a unique subset of declared development subjects')
    if settings['policies'] != ['fixed', 'balanced', 'adaptive'] or settings['arms'] != ['BASE', 'NEW', 'POOL', 'DRAW']:
        raise ValueError('Require the declared four arms and three task policies')
    if smoke:
        selected = selected[:1]
        settings.update(paths=1, max_blocks=2, parallel_budget=1)
    if not selected or settings['paths'] < 1 or settings['parallel_budget'] < 1:
        raise ValueError('Need subjects, paths and workers')
    originals = []
    nominees = {(r['subject'], r['model']): r['candidate'] for r in frozen['nominees']}
    for subject in selected:
        seed_banks = []
        for arm in settings['arms']:
            files = sorted((source / 'jobs').glob(f'AUTO_evaluate_S{subject}_{arm}_*.json'))
            bank, seeds = [], []
            for file in files:
                job = v.read(file)
                nominee = nominees[subject, arm]
                if job['candidate']['id'] != nominee['id'] or job['candidate']['hyperparams'] != nominee['hyperparams']:
                    raise ValueError('Saved autonomous path uses another parameter candidate')
                reference = Path(job['target']).with_suffix('.npz')
                record = v.read(reference.with_suffix('.json'))
                if v.digest(reference) != record['predictions_sha256']:
                    raise ValueError('Original autonomous array changed')
                with np.load(reference) as arrays:
                    np.testing.assert_array_equal(arrays['seeds'], job['seeds'])
                seeds.extend(job['seeds'])
                bank.append(dict(job, reference=str(reference), reference_sha256=record['predictions_sha256'],
                                 original_job=str(file.resolve()), original_job_sha256=v.digest(file)))
            if len(set(seeds)) != len(seeds) or len(seeds) < settings['paths']:
                raise ValueError(f'Insufficient or duplicated path seeds: {subject} {arm}')
            seed_banks.append(seeds[:settings['paths']])
            left = settings['paths']
            for job in bank:
                if left <= 0:
                    break
                job['seeds'] = job['seeds'][:left]
                left -= len(job['seeds'])
                originals.append(job)
        for seeds in seed_banks[1:]:
            np.testing.assert_array_equal(seeds, seed_banks[0])
    output.mkdir(parents=True)
    (output / 'jobs').mkdir(); (output / 'schedules').mkdir()
    checks = adaptive.prepare_schedules(v, frozen, output / 'schedules', selected)
    v.write(output / 'task_policy_audit.json', dict(all_matched=True, checks=checks))
    extra = {name: v.digest(ROOT / name) for name in [ENTRY, WORKER, *adaptive.TASK_FILES]}
    jobs = []
    for original in originals:
        subject = v.read(Path(original['folder']) / 'prepared.json')['subject']
        schedule = output / 'schedules' / f'schedule_{subject}.npz'
        for policy in settings['policies']:
            name = f'{original["name"]}_{policy}'
            directory = output / 'paths' / f'S{subject}' / original['model'] / policy
            directory.mkdir(parents=True, exist_ok=True)
            jobs.append(dict(original, name=name, policy=policy, settings=settings,
                             schedule=str(schedule.resolve()), schedule_sha256=v.digest(schedule),
                             directory=str(directory.resolve()),
                             target=str((output / 'jobs' / (name + '_result.json')).resolve()),
                             source_sha256={**original['source_sha256'], **extra},
                             episodic_arm_settings=original['arm_settings']))
    hashes = {name: sha for job in jobs for name, sha in job['source_sha256'].items()}
    for name in hashes:
        destination = output / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    shutil.copy2(config, output / 'protocol.yaml')
    cpu_budget = min(settings['parallel_budget'], len(os.sched_getaffinity(0)))
    workers = min(cpu_budget, len(jobs))
    protocol = dict(settings=settings, subjects=selected, smoke=smoke, workers=workers, cpu_budget=cpu_budget, jobs=len(jobs),
                    fit=str(fit.resolve()), reference=str(source.resolve()), source_sha256=hashes,
                    nomination_sha256=v.digest(fit / 'nominees.json'), config_sha256=v.digest(config),
                    original_completion_sha256=v.digest(source / 'completion.json'),
                    python=sys.version, versions={name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'pandas', 'PyYAML')},
                    git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    role='development_fixed_parameter_environment_and_stopping_diagnostic',
                    assumptions=['first observed block repeated at session starts', 'cognition persists across sessions',
                                 'NumPy distributional sampling; MATLAB timing draws not replayed',
                                 'administrative horizon retained with all censored paths',
                                 'no oral input; categories supplied by task feedback only'],
                    task_program_sha256={name: v.digest(ROOT / name) for name in
                                         ('exp_program/exp123/Task2_learning.m', 'exp_program/exp123/sample_2.m')})
    if settings.get('resource_predecessor') and not smoke:
        predecessor = ROOT / settings['resource_predecessor']
        previous = v.read(predecessor / 'protocol.json')
        if previous['smoke'] or not (predecessor / 'states').is_dir():
            raise ValueError('Resource predecessor must be the declared scientific state replay')
        protocol['resource_predecessor'] = str(predecessor.resolve())
        protocol['predecessor_protocol_sha256'] = v.digest(predecessor / 'protocol.json')
    v.write(output / 'protocol.json', protocol)
    return protocol, jobs


def run(config: Path, output: Path, smoke: bool, subjects: list[int] | None = None) -> None:
    protocol, jobs = prepare(config, output, smoke, subjects)
    v = adaptive.validation()
    budget = SharedWorkerBudget(protocol['cpu_budget'],
                               Path(protocol['resource_predecessor']) if 'resource_predecessor' in protocol else None)
    def launch(job):
        file = output / 'jobs' / (job['name'] + '.json')
        v.write(file, job)
        launcher = ('import runpy,sys;root,script=sys.argv[1:3];sys.path.insert(0,root);'
                    "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        with budget.slot(), tempfile.TemporaryDirectory(prefix='admission_task_') as cache, file.with_suffix('.log').open('x') as log:
            env.update(MPLCONFIGDIR=cache, NUMBA_CACHE_DIR=cache)
            result = subprocess.run([sys.executable, '-c', launcher, job['runtime'], str(ROOT / WORKER),
                                     '--worker', str(file.resolve())], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'Task worker failed: {file.with_suffix(".log")}')
        return v.read(Path(job['target']))
    print(f'{len(jobs)} task jobs; {protocol["workers"]} workers; one numeric thread', flush=True)
    start, results = time.monotonic(), []
    with ThreadPoolExecutor(max_workers=protocol['workers']) as pool:
        for future in as_completed([pool.submit(launch, job) for job in jobs]):
            results.append(future.result())
            if len(results) % 32 == 0 or len(results) == len(jobs):
                print(f'{len(results)}/{len(jobs)} complete', flush=True)
    frame = pd.DataFrame([row for result in results for row in result['records']])
    if frame.duplicated(['subject', 'model', 'policy', 'seed']).any():
        raise ValueError('Duplicated generated path')
    frame.to_csv(output / 'paths.csv', index=False)
    summary = frame.groupby(['subject', 'model', 'policy']).agg(paths=('seed', 'size'),
        pass_fraction=('passed', 'mean'), pass_by_observed_fraction=('passed_by_observed_horizon', 'mean'),
        mean_blocks_to_stop_or_censor=('blocks', 'mean'), final_correct_probability=('final_correct_probability', 'mean'),
        mean_search_probability=('mean_search_probability', 'mean')).reset_index()
    summary.to_csv(output / 'summary.csv', index=False)
    if any(p['num_threads'] != 1 for result in results for p in result['numeric_pools']):
        raise ValueError('Numeric worker exceeded one thread')
    v.write(output / 'completion.json', dict(status='complete', smoke=smoke, jobs=len(jobs), paths=len(frame),
        workers=protocol['workers'], seconds=time.monotonic()-start, cpu_seconds=sum(r['cpu_seconds'] for r in results),
        exact_fixed_replay_paths=sum(r['exact_fixed_replay_paths'] for r in results),
        artifacts={name: v.digest(output / name) for name in ('paths.csv', 'summary.csv')}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--subjects', type=int, nargs='+')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.config, args.output, args.smoke, args.subjects)


if __name__ == '__main__':
    main()

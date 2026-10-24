"""Run geometry audits and passive execution-state capture at frozen estimates.

Use --study FIT --output NEW [--smoke]. The shared numerical queue and launcher
are also used by synthetic recovery. Every process has one numeric thread.
Human capture replays saved evaluation seeds and requires identical behavior
probabilities. The smoke path uses a saved eight-trial confirmation filter.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
STUDY = 'src/Bayesian_state/workflows/analysis/model_1023_execution_study.py'
CAPTURE = 'src/Bayesian_state/workflows/analysis/model_1023_execution_capture.py'
spec = importlib.util.spec_from_file_location('execution_validation_study', ROOT / STUDY)
study = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = study
spec.loader.exec_module(study)
read, write, digest = study.read, study.write, study.digest


def sources() -> dict:
    capture = study.base.load('execution_validation_capture', CAPTURE)
    return dict(capture.sources(), **{ENTRY: digest(ROOT / ENTRY)})


def worker(file: Path) -> None:
    job = read(file)
    if job['validation_sources'] != sources():
        raise ValueError('Validation source changed')
    if job.get('capture_sources'):
        capture = study.base.load('execution_validation_worker_capture', CAPTURE)
        capture.worker(file)
    else:
        study.worker(file)


class NumericalQueue:
    """One shared process budget across independent case-controller threads."""

    def __init__(self, pool: ThreadPoolExecutor):
        self.pool = pool

    @staticmethod
    def launch(original: dict, folder: Path) -> None:
        job = deepcopy(original)
        if job['phase'] not in ('prepare', 'audit') and job['model'] != 'COMPACT':
            job.update(execution_mode=job['model'], comparison_arm='ADAPT', model='COMPACT')
            job['mechanism']['capacity'] = 'evidence'
        job['validation_sources'] = sources()
        directory = folder / 'jobs'; directory.mkdir(exist_ok=True)
        file = directory / (job['name'] + '.json')
        write(file, job)
        launcher = ('import runpy,sys;runtime,script=sys.argv[1:3];sys.path.insert(0,runtime);'
                    "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")
        with tempfile.TemporaryDirectory(prefix='execution_validation_') as cache:
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', NUMBA_NUM_THREADS='1', NUMBA_CACHE_DIR=cache,
                MPLCONFIGDIR=cache)
            with file.with_suffix('.log').open('x') as log:
                result = subprocess.run([sys.executable, '-c', launcher, job['runtime'], str(ROOT / ENTRY),
                    '--worker', str(file.resolve())], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError('Execution validation failed: ' + str(file.with_suffix('.log')))

    def execute(self, jobs: list[dict], folder: Path, budget: int) -> None:
        if self.pool._max_workers > budget:
            raise ValueError('Numerical queue exceeds the declared budget')
        for count, future in enumerate(as_completed([self.pool.submit(self.launch, job, folder) for job in jobs]), 1):
            future.result()
            if count % 16 == 0 or count == len(jobs):
                print(f'{folder.name}: {count}/{len(jobs)} jobs complete', flush=True)


def geometry_jobs(fit: Path, output: Path) -> list[dict]:
    """Select every training-perfect rule without consulting human choices."""
    protocol = read(fit / 'protocol.json')
    settings = protocol['settings']
    data = Path(settings['data'])
    preparation = read(data / 'preparation_manifest.json')
    runtime = str((ROOT / preparation['settings']['source_fit'] / 'source').resolve())
    rows = read(fit / 'nominees.json')['nominees']
    return [dict(phase='audit', name=f'audit_S{r["subject"]}', subject=r['subject'],
        candidate=r['candidate'], folder=str((data / f'subject_{r["subject"]}').resolve()),
        runtime=runtime, mechanism=settings['mechanism'], source_sha256=study.sources(study.driver.io),
        geometry_beta=25., target=str((output / 'geometry' / f'subject_{r["subject"]}').resolve()))
        for r in rows if r['arm'] == 'MIX']


def run(fit: Path, output: Path, smoke: bool) -> None:
    if output.exists():
        raise FileExistsError('Use a new state-validation destination')
    completed = read(fit / 'completion.json')
    protocol = read(fit / 'protocol.json')
    if completed['status'] != 'complete' or completed['smoke'] != smoke:
        raise ValueError('Require a completed matching numerical study')
    if protocol['source_sha256'] != study.sources(study.driver.io):
        raise ValueError('Fitted numerical implementation changed')
    output.mkdir(parents=True)
    audits = geometry_jobs(fit, output)
    budget = 1 if smoke else min(128, len(os.sched_getaffinity(0)))
    capture = study.base.load('execution_validation_capture_plan', CAPTURE)
    with ThreadPoolExecutor(max_workers=budget) as pool:
        queue = NumericalQueue(pool)
        queue.execute(audits, output, budget)
        frozen = read(fit / 'nominees.json')['nominees']
        jobs = []
        for nominee in frozen:
            sid, mode = nominee['subject'], nominee['arm']
            geometry = read(output / 'geometry' / f'subject_{sid}' / 'audit.json')
            if not geometry['training_perfect_rules']:
                raise ValueError('No training-perfect rule for target-state interpretation')
            stage = 'confirmation' if smoke else 'evaluate'
            original_jobs = sorted((fit / 'jobs').glob(f'{stage}_S{sid}_{mode}_{nominee["candidate"]["id"]}_*.json'))
            if not original_jobs:
                raise ValueError('Original scoring jobs missing')
            # Formal capture uses the first two independent saved evaluation
            # seeds. Exact replay checks prevent observer-induced changes.
            for original_file in original_jobs[:1 if smoke else 2]:
                original = read(original_file)
                reference = Path(original['target']).with_suffix('.npz')
                for seed in original['seeds'][:1]:
                    job = deepcopy(original)
                    name = 'capture_' + original['name'] + f'_F{seed}'
                    job.update(name=name, seeds=[seed], capture_sources=capture.sources(),
                        target_rules=geometry['training_perfect_rules'], capture_oral=True,
                        state_capture_target=str((output / 'states' / name).resolve()),
                        target=str((output / 'scores' / name).resolve()),
                        original_score=str(reference.resolve()), original_score_sha256=digest(reference))
                    jobs.append(job)
        write(output / 'protocol.json', dict(study=str(fit.resolve()), smoke=smoke, jobs=jobs,
            fit_nominees_sha256=digest(fit / 'nominees.json'), source_sha256=sources(),
            inference='current and past choices; pre-feedback; no oral observations',
            target_definition='all training-geometry-perfect rules',
            scope='developmental state reconstruction; no empirical latent-state ground truth'))
        queue.execute(jobs, output, budget)
    write(output / 'completion.json', dict(status='complete', smoke=smoke, filters=len(jobs),
        exact_behavior_replay=True, source_sha256=sources(),
        protocol_sha256=digest(output / 'protocol.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('study', 'output', 'worker'):
        parser.add_argument('--' + key, type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.study and args.output:
        run(args.study, args.output, args.smoke)
    else:
        parser.error('Use --study and --output, or internal --worker')

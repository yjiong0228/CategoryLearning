"""Replay frozen belief evaluation filters to capture their oral decision bases.

Run --study COMPLETED_COMPARISON --output NEW [--arms SUPPORT SLOT]. Every
replayed prediction must exactly equal the saved original prediction. The
shared passive collector reads the decision before current feedback, with no
oral input. Replay filters are explicitly counted as additional computation.
Use --smoke with a completed short belief comparison to check all fitted arms
on a sixteen-trial software fixture and two-particle single-seed filters.
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

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
PILOT = 'src/Bayesian_state/workflows/analysis/model_1022_belief_comparison.py'
CAPTURE = 'src/Bayesian_state/workflows/analysis/model_1021_simplification_capture.py'
spec = importlib.util.spec_from_file_location('belief_capture_pilot', ROOT / PILOT)
pilot = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pilot
spec.loader.exec_module(pilot)
read, write, digest = pilot.read, pilot.write, pilot.digest
capture = pilot.driver.base.load('belief_capture_observer', CAPTURE)


def sources() -> dict:
    return dict(pilot.sources(pilot.driver.io), **capture.sources(), **{ENTRY: digest(ROOT / ENTRY)})


def assert_replay(reference: Path, replay: Path, expected_sha: str) -> None:
    """Require exact arrays, including identities, probabilities and seeds."""
    if digest(reference) != expected_sha:
        raise ValueError('Original evaluation predictions changed')
    with np.load(reference) as original, np.load(replay) as current:
        if set(original.files) != set(current.files):
            raise ValueError('Replay output fields differ from the original filter')
        for key in original.files:
            np.testing.assert_array_equal(original[key], current[key], err_msg=f'Replay differs: {key}')


def worker(file: Path) -> None:
    job = read(file)
    if job['belief_capture_sources'] != sources():
        raise ValueError('Belief capture source changed')
    if job.get('software_plain_filter'):
        pilot.worker(file)
        return
    original = capture.driver.worker
    capture.driver.worker = pilot.worker
    try:
        capture.worker(file)
    finally:
        capture.driver.worker = original
    reference, replay = Path(job['replay_reference']), Path(job['target']).with_suffix('.npz')
    assert_replay(reference, replay, job['replay_reference_sha256'])
    write(Path(job['target']).with_suffix('.replay.json'), dict(status='complete',
        arm=job['comparison_arm'], source_sha256=sources(), reference=str(reference),
        reference_sha256=job['replay_reference_sha256'], replay_sha256=digest(replay),
        exact_array_replay=True, oral_inputs_used=False, additional_filters=1,
        decision_basis_target=job['decision_basis_target']))


def execute(jobs: list[dict], output: Path, budget: int) -> None:
    directory = output / 'jobs'
    directory.mkdir(exist_ok=True)

    def launch(job):
        file = directory / (job['name'] + '.json')
        write(file, job)
        launcher = ('import runpy,sys;runtime,script=sys.argv[1:3];sys.path.insert(0,runtime);'
                    "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")
        with tempfile.TemporaryDirectory(prefix='belief_capture_') as cache, file.with_suffix('.log').open('x') as log:
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', NUMBA_NUM_THREADS='1', NUMBA_CACHE_DIR=cache)
            subprocess.run([sys.executable, '-c', launcher, job['runtime'], str(ROOT / ENTRY),
                            '--worker', str(file.resolve())], cwd=ROOT, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    workers = min(budget, len(os.sched_getaffinity(0)), len(jobs))
    if workers < 1:
        raise ValueError('Capture requires at least one declared job')
    print(f'{len(jobs)} frozen filter jobs; {workers} single-thread workers', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, future in enumerate(as_completed([pool.submit(launch, j) for j in jobs]), 1):
            future.result()
            if i % 12 == 0 or i == len(jobs):
                print(f'{i}/{len(jobs)} complete', flush=True)


def plan_jobs(study: Path, output: Path, arms: list[str] | None, smoke: bool) -> tuple[list[dict], list[dict], dict]:
    """Bind every case to its frozen parameter point and saved filter inputs."""
    protocol, nomination, done = [read(study / name) for name in ('protocol.json', 'nominees.json', 'completion.json')]
    if (done['status'] != 'complete' or nomination['smoke'] is not smoke
            or nomination['status'] != 'frozen_before_test' or nomination['later_choices_used']):
        raise ValueError('Require a completed matching training-only belief comparison')
    available = {r['arm'] for r in nomination['nominees']}
    chosen = available if arms is None else set(arms)
    if not chosen or not chosen <= available:
        raise ValueError('Requested capture arm is absent from the completed comparison')
    inputs = {str((study / name).resolve()): digest(study / name)
              for name in ('protocol.json', 'nominees.json', 'completion.json')}
    original_jobs = [(p, read(p)) for p in sorted((study / 'jobs').glob('*.json'))]
    jobs, cases = [], []
    fixture_data = None
    if smoke:
        from src.Bayesian_state.workflows.analysis.model_1022_compact_evaluate import smoke_fixture
        sid = nomination['nominees'][0]['subject']
        fixture_data = smoke_fixture(ROOT / protocol['settings']['data'], output / 'software_fixture', sid)
    for row in nomination['nominees']:
        if row['arm'] not in chosen:
            continue
        matching = []
        for file, job in original_jobs:
            meta = read(Path(job['folder']) / 'prepared.json')
            if (meta['subject'] == row['subject'] and job['comparison_arm'] == row['arm']
                    and job['candidate']['id'] == row['candidate']['id']
                    and job['phase'] == ('training' if smoke else 'evaluate')):
                if job['candidate']['hyperparams'] != row['candidate']['hyperparams']:
                    raise ValueError('Evaluation parameters differ from the frozen nominee')
                matching.append((file, job))
        if smoke:
            matching = matching[:1]
        expected = 1 if smoke else protocol['settings']['budgets']['evaluate']['seeds']
        if len(matching) != expected:
            raise ValueError('Capture must preserve every independent evaluation filter')
        paths, seeds = [], []
        for i, (file, original) in enumerate(matching):
            job = deepcopy(original)
            inputs[str(file.resolve())] = digest(file)
            name = f'S{row["subject"]}_{row["arm"]}_C{i:03d}'
            job.update(name=name, target=str((output / 'conditional' / name).resolve()),
                       decision_basis_target=str((output / 'state_capture' / name).resolve()),
                       decision_basis_sources=capture.sources(), belief_capture_sources=sources())
            if smoke:
                job.update(phase='evaluate', folder=str((fixture_data / f'subject_{row["subject"]}').resolve()),
                           particles=2, seeds=job['seeds'][:1], source_sha256=pilot.sources(pilot.driver.io))
                reference = output / 'software_plain' / (name + '.npz')
            else:
                if len(job['seeds']) != 1:
                    raise ValueError('Require the original single-seed evaluation chunks')
                for path, sha in job['source_sha256'].items():
                    if digest(ROOT / path) != sha:
                        raise ValueError('Fitted scientific source changed: ' + path)
                reference = Path(original['target']).with_suffix('.npz')
                meta_file = reference.with_suffix('.json')
                score = read(meta_file)
                if digest(reference) != score['predictions_sha256']:
                    raise ValueError('Saved evaluation predictions changed')
                inputs[str(reference)] = score['predictions_sha256']
                inputs[str(meta_file)] = digest(meta_file)
            job['replay_reference'] = str(reference.resolve())
            paths.append(job['decision_basis_target'] + '.json')
            seeds.extend(job['seeds'])
            jobs.append(job)
        if len(set(seeds)) != expected:
            raise ValueError('Captured filters require distinct original seeds')
        cases.append(dict(subject=row['subject'], arm=row['arm'], candidate=row['candidate'], captures=paths,
            seeds=seeds, cohort=protocol['policy'].get('cohort', 'development')))
    return jobs, cases, inputs


def run(study: Path, output: Path, arms: list[str] | None, smoke: bool) -> None:
    if output.exists():
        raise FileExistsError('Use a new belief capture directory')
    output.mkdir(parents=True)
    jobs, cases, inputs = plan_jobs(study, output, arms, smoke)
    pilot.driver.base.archive(output, study / 'config.yaml', sources())
    write(output / 'protocol.json', dict(study=str(study.resolve()), smoke=smoke, source_sha256=sources(),
        inputs_sha256=inputs, additional_filters=len(jobs), oral_inputs_used=False,
        timing='after_choice_before_current_feedback', replay_policy='exact equality of all saved prediction arrays'))
    if smoke:
        plain = []
        for job in jobs:
            row = deepcopy(job)
            row.update(name='plain_' + job['name'], target=str(Path(job['replay_reference']).with_suffix('')),
                       software_plain_filter=True)
            plain.append(row)
        execute(plain, output, 1)
    for job in jobs:
        job['replay_reference_sha256'] = digest(Path(job['replay_reference']))
    write(output / 'cases.json', dict(cases=cases, inputs_sha256=inputs))
    execute(jobs, output, 1 if smoke else 128)
    for path, sha in inputs.items():
        if digest(Path(path)) != sha:
            raise ValueError('Original evaluation input changed during capture')
    write(output / 'completion.json', dict(status='complete', smoke=smoke, cases=len(cases),
        filters=len(jobs), additional_filters=len(jobs), exact_array_replay=True, oral_inputs_used=False,
        cases_sha256=digest(output / 'cases.json'), source_sha256=sources()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'output', 'worker'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--arms', nargs='+')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.study and args.output:
        run(args.study, args.output, args.arms, args.smoke)
    else:
        parser.error('Supply --study and --output, or --worker')

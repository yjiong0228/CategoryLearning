"""Replay frozen finite-H evaluations with per-particle region decision states.

Use --fit COMPLETED_FIT --output NEW_DIRECTORY [--smoke]. The full run replays
every independent evaluation seed at its original parameters and particle count.
--smoke requires a completed smoke fit and retains its small original budget.
Workers import the exact fit snapshot, then install the separately archived
read-only state collector. Every original behavior array must reproduce exactly.
Oral reports are absent. This entry saves A, q and within-particle rule weights;
geometric component projection and oral scoring are subsequent analysis steps.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from src.Bayesian_state.evaluation.model_0923_h_validation import digest, load
from src.Bayesian_state.workflows.analysis.model_0923_h_validation import write

ROOT = Path(__file__).resolve().parents[4]
COLLECTOR = 'src/Bayesian_state/evaluation/oral/finite_region_states.py'
ENTRY = 'src/Bayesian_state/workflows/analysis/model_0923_finite_region_states.py'
EXTRA_FIELDS = ('capture_source_job', 'capture_source_arrays', 'capture_source_record',
                'capture_runtime', 'capture_extension', 'capture_bindings')


def verify(bindings: dict) -> None:
    for filename, expected in bindings.items():
        if digest(Path(filename)) != expected:
            raise ValueError('State replay input changed: ' + filename)


def worker(file: Path) -> None:
    job = load(file)
    verify(job['capture_bindings'])
    source = load(Path(job['capture_source_job']))
    restored = {k: v for k, v in job.items() if k not in EXTRA_FIELDS}
    restored['target'], restored['bindings'] = source['target'], source['bindings']
    if restored != source or job['target'] == source['target'] or job['stage'] != 'evaluation':
        raise ValueError('Capture must preserve the frozen evaluation job')
    from src.Bayesian_state.workflows.analysis import model_0923_finite_region_fit as original
    if not Path(original.__file__).resolve().is_relative_to(Path(job['capture_runtime'])):
        raise ValueError('Worker did not import the frozen fit runtime')
    name = 'src.Bayesian_state.evaluation.oral._finite_region_state_extension'
    spec = importlib.util.spec_from_file_location(name, job['capture_extension'])
    extension = importlib.util.module_from_spec(spec)
    sys.modules[name] = extension
    spec.loader.exec_module(extension)
    base_filter = original.run_state_model_particle_filter
    records = []

    def capture_filter(**kwargs):
        if kwargs.get('state_observer') is not None or kwargs.get('observe_decision_basis'):
            raise ValueError('The original filter already has a state observer')
        with extension.capture_finite_region_states(choices=kwargs['choices'],
                particle_count=kwargs['particle_count']) as collector:
            result = base_filter(**kwargs, state_observer=collector, observe_decision_basis=True)
            records.append(collector.arrays())
        return result

    original.run_state_model_particle_filter = capture_filter
    try:
        original.worker(file)
    finally:
        original.run_state_model_particle_filter = base_filter
    if len(records) != 1:
        raise ValueError('Expected one complete filter per state job')
    captured = records[0]
    target = Path(job['target'])
    original_elements = 0
    with np.load(job['capture_source_arrays']) as old, np.load(target.with_suffix('.npz')) as new:
        if set(old.files) != set(new.files):
            raise ValueError('State capture changed the behavioral output schema')
        for key in old.files:
            np.testing.assert_array_equal(new[key], old[key], err_msg=key)
            original_elements += old[key].size
        np.testing.assert_allclose(captured['prediction'], old['probabilities'], atol=1e-12, rtol=0)
        np.testing.assert_allclose(captured['choice_probability'], old['observed_probability'], atol=1e-12, rtol=0)
        np.testing.assert_array_equal(captured['choices'], old['choices'])
    if captured['rule_ids'].shape[:2] != (len(captured['choices']), job['particles']):
        raise ValueError('State capture lost trial or particle identity')
    verify(job['capture_bindings'])
    state_file = target.with_suffix('.regions.npz')
    with state_file.open('xb') as handle:
        np.savez_compressed(handle, **captured)
    write(target.with_suffix('.regions.json'), dict(status='complete', schema_version=extension.VERSION,
        source_job_sha256=digest(Path(job['capture_source_job'])), job_sha256=digest(file),
        arrays_sha256=digest(state_file), behavior_arrays_sha256=digest(target.with_suffix('.npz')),
        original_elements_exact=original_elements, trials=len(captured['choices']), particles=job['particles'],
        active_slots=captured['rule_ids'].shape[-1], subject=job['subject'], arm=job['arm'],
        point=job['point'], seed=job['seed'], choice_floor_events=int(captured['choice_floor_active'].sum()),
        cognitive_runtime=job['capture_runtime'], collector_file=str(Path(extension.__file__).resolve()),
        timing='After H transition and before current feedback; both predictive and actual filtered weights',
        oral_used=False))


def run(fit: Path, output: Path, smoke: bool = False, workers: int = 16) -> None:
    fit, output = fit.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError('Use a new finite-region state directory')
    if not 1 <= workers <= 64:
        raise ValueError('Use at most 64 single-thread state workers')
    done, protocol, frozen = (load(fit/name) for name in ('completion.json', 'protocol.json', 'nominees.json'))
    if (done['status'] != 'complete' or bool(done['smoke']) != smoke
            or bool(protocol['smoke']) != smoke or bool(frozen['smoke']) != smoke
            or done['protocol_sha256'] != digest(fit/'protocol.json')
            or done['nomination_sha256'] != digest(fit/'nominees.json')
            or frozen['status'] != 'frozen_before_later_choices'):
        raise ValueError('Require a complete matching smoke/full frozen behavior fit')
    bindings = {str(fit/name): digest(fit/name) for name in ('completion.json', 'protocol.json', 'nominees.json')}
    runtime = Path(protocol['source_runtime'])
    for name, expected in done['artifacts_sha256'].items():
        path = fit/name
        if digest(path) != expected:
            raise ValueError('Completed fit artifact changed: ' + name)
    source_jobs = []
    for file in sorted((fit/'jobs').glob('evaluation_*.json')):
        job = load(file)
        record_file = Path(job['target']).with_suffix('.json')
        record = load(record_file)
        arrays = Path(job['target']).with_suffix('.npz')
        if (record['job_sha256'] != digest(file) or record['arrays_sha256'] != digest(arrays)
                or job['stage'] != 'evaluation' or not any(n['subject'] == job['subject']
                and n['arm'] == job['arm'] and n['point'] == job['point'] for n in frozen['nominees'])):
            raise ValueError('Evaluation does not match its frozen point or output')
        for path in (file, record_file, arrays):
            bindings[str(path)] = digest(path)
        source_jobs.append((file, job, record_file, arrays))
    expected_jobs = len(frozen['nominees']) * (1 if smoke else protocol['settings']['evaluation']['repeats'])
    if len(source_jobs) != expected_jobs:
        raise ValueError('Incomplete original evaluation seed bank')
    output.mkdir(parents=True)
    (output/'jobs').mkdir(); (output/'chunks').mkdir()
    for name in (ENTRY, COLLECTOR):
        target = output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
        bindings[str(target)] = digest(target)
    all_jobs = []
    for source_file, source, source_record, source_arrays in source_jobs:
        job = deepcopy(source)
        job.update(target=str(output/'chunks'/Path(source['target']).name),
            capture_source_job=str(source_file), capture_source_record=str(source_record),
            capture_source_arrays=str(source_arrays), capture_runtime=str(runtime),
            capture_extension=str(output/'source'/COLLECTOR), capture_bindings=bindings)
        job['bindings'].update(bindings)
        all_jobs.append(job)
    budget = min(1 if smoke else workers, len(all_jobs), len(os.sched_getaffinity(0)))
    write(output/'protocol.json', dict(status='frozen_before_execution', fit=str(fit), smoke=smoke,
        nomination_sha256=digest(fit/'nominees.json'), bindings=bindings, jobs=len(all_jobs), workers=budget,
        numeric_threads=1, timing='Pre-feedback particle-level A,q and decision contribution',
        selection='All completed evaluation filters; original parameters, particles and seeds', oral_used=False))

    def launch(job):
        file = output/'jobs'/(job['name']+'.json'); write(file, job)
        launcher = ('import runpy,sys;sys.path.insert(0,sys.argv[1]);'
                    'sys.argv=sys.argv[2:];runpy.run_path(sys.argv[0],run_name="__main__")')
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        with file.with_suffix('.log').open('x') as log:
            subprocess.run([sys.executable, '-c', launcher, str(runtime), str(output/'source'/ENTRY),
                            '--worker', str(file)], cwd=ROOT, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        record = load(Path(job['target']).with_suffix('.regions.json'))
        if (record['status'] != 'complete' or record['job_sha256'] != digest(file)
                or record['arrays_sha256'] != digest(Path(job['target']).with_suffix('.regions.npz'))):
            raise ValueError('Incomplete region state output')
        return record

    print(f'{len(all_jobs)} exact state replays; {budget} workers', flush=True)
    records = []
    with ThreadPoolExecutor(max_workers=budget) as pool:
        for i, future in enumerate(as_completed([pool.submit(launch, j) for j in all_jobs]), 1):
            records.append(future.result()); print(f'{i}/{len(all_jobs)} complete', flush=True)
    verify(bindings)
    artifacts = {str(path.relative_to(output)): digest(path) for path in output.rglob('*')
                 if path.is_file() and 'source' not in path.relative_to(output).parts and path.suffix != '.log'}
    write(output/'completion.json', dict(status='complete', smoke=smoke, jobs=len(all_jobs),
        protocol_sha256=digest(output/'protocol.json'), nomination_sha256=digest(fit/'nominees.json'),
        original_elements_exact=sum(r['original_elements_exact'] for r in records), artifacts_sha256=artifacts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.fit and args.output:
        run(args.fit, args.output, args.smoke, args.workers)
    else:
        parser.error('Supply --worker or both --fit and --output')

"""Classify existing pilot paths without model simulation or parameter fitting.

Run from the repository root with --study, --output NEW, and --config. Every
screening and confirmation path is included. Records are processed in a single
process-parallel layer with one numeric thread per worker. Classification
validation remains deferred; source hashes and per-path identities are saved.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import warnings

import numpy as np
import scipy
import yaml

from ...metrics.trajectory_classification import classify_path
from .model_1024_individual_regions import completed_arrays, load_observed


def _read(path: Path):
    return json.loads(path.read_text())


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _batch(task: dict, policy: dict) -> dict:
    start = time.process_time()
    records = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for index, correct in enumerate(task['correct']):
            row = classify_path(correct, policy)
            row.update(path_index=task['start']+index,
                       seed=int(task['seeds'][index]) if task['seeds'] is not None else None)
            records.append(row)
    return dict(key=task['key'], start=task['start'], records=records,
                cpu_seconds=time.process_time()-start,
                warnings={message:sum(str(w.message)==message for w in caught)
                          for message in set(str(w.message) for w in caught)})


def run(study: Path, output: Path, config: Path, workers: int) -> None:
    if output.exists():
        raise FileExistsError('Use a new classification directory')
    for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        if os.environ.get(name)!='1':
            raise ValueError(f'Set {name}=1 before starting this process')
    started = time.time()
    parent_start = time.process_time()
    policy = yaml.safe_load(config.read_text())
    protocol = _read(study/'protocol.json')
    output.mkdir(parents=True)
    (output/'chunks').mkdir()
    sources = [Path(__file__), Path('src/Bayesian_state/metrics/trajectory_classification.py'), config]
    _write(output/'registration.json', dict(version=policy['version'], policy=policy,
        study=str(study), sources={str(p):_sha(p) for p in sources},
        method_validation='deferred_by_user', model_jobs=0, parameter_scores_changed=False,
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        created_unix=started))
    (output/'classification_source.py').write_bytes(sources[1].read_bytes())
    (output/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    (output/'config.yaml').write_bytes(config.read_bytes())
    tasks, groups, inputs = [], {}, {str(study/'protocol.json'):_sha(study/'protocol.json')}
    for case in protocol['cases']:
        sid = case['subject']
        _, actual = load_observed(Path(case['template']['folder']))
        key = f'observed_S{sid}'
        real = actual['choices']==actual['categories']
        groups[key] = dict(subject=sid, phase='observed', candidate=None, paths=1,
                           outcome_sha256=hashlib.sha256(real.astype('uint8').tobytes()).hexdigest())
        tasks.append(dict(key=key, start=0, correct=real[None, :], seeds=None))
        for phase, prefix in [('screen', 'search'), ('confirmation', 'confirmed')]:
            source = study/f'{prefix}_S{sid}.json'
            inputs[str(source)] = _sha(source)
            for row in _read(source):
                choices = completed_arrays(row['generation_jobs'], 'choices')
                seeds = completed_arrays(row['generation_jobs'], 'seeds')
                if choices is None or seeds is None:
                    raise ValueError('Every requested path must have a complete source')
                correct = choices==actual['categories'][None, :]
                key = f"{phase}_S{sid}_{row['candidate']['id']}"
                groups[key] = dict(subject=sid, phase=phase, candidate=row['candidate']['id'], paths=len(choices),
                    choices_sha256=hashlib.sha256(np.asarray(choices,dtype='<i8').tobytes()).hexdigest(),
                    outcome_sha256=hashlib.sha256(correct.astype('uint8').tobytes()).hexdigest(),
                    seeds_sha256=hashlib.sha256(np.asarray(seeds,dtype='<i8').tobytes()).hexdigest())
                for job in row['generation_jobs']:
                    path = Path(job['target']).with_suffix('.npz')
                    inputs[str(path)] = _sha(path)
                for first in range(0, len(choices), 32):
                    tasks.append(dict(key=key, start=first, correct=correct[first:first+32], seeds=seeds[first:first+32]))
    workers = min(workers, len(os.sched_getaffinity(0)), len(tasks))
    _write(output/'inputs.json', inputs)
    records = {key:[] for key in groups}
    worker_cpu, warning_counts = 0., {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_batch, task, policy) for task in tasks]
        for number, future in enumerate(as_completed(futures), 1):
            result = future.result()
            _write(output/'chunks'/f"{result['key']}_{result['start']:04d}.json", result)
            records[result['key']].extend(result['records'])
            worker_cpu += result['cpu_seconds']
            for message, count in result['warnings'].items():
                warning_counts[message] = warning_counts.get(message, 0)+count
            if number%16==0 or number==len(tasks):
                print(f'{number}/{len(tasks)} batches; {worker_cpu:.1f} worker CPU seconds', flush=True)
    for key, group in groups.items():
        ordered = sorted(records[key], key=lambda r:r['path_index'])
        if [r['path_index'] for r in ordered]!=list(range(group['paths'])):
            raise ValueError('Missing or duplicate path index')
        _write(output/f'{key}.json', dict(**group, records=ordered))
    for source in sources:
        if _sha(source)!=_read(output/'registration.json')['sources'][str(source)]:
            raise ValueError('Classification source changed during execution')
    _write(output/'index.json', dict(version=policy['version'], groups=groups))
    _write(output/'completion.json', dict(status='complete', method_validation='deferred_by_user',
        groups=len(groups), observed_paths=sum(g['paths'] for g in groups.values() if g['phase']=='observed'),
        generated_paths=sum(g['paths'] for g in groups.values() if g['phase']!='observed'),
        workers=workers, worker_cpu_seconds=worker_cpu, coordinator_cpu_seconds=time.process_time()-parent_start,
        wall_seconds=time.time()-started, warnings=warning_counts, model_jobs=0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/shared/model1024_trajectory_classification.yaml'))
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    if args.workers<1:
        parser.error('--workers must be positive')
    run(args.study, args.output, args.config, args.workers)


if __name__=='__main__':
    main()

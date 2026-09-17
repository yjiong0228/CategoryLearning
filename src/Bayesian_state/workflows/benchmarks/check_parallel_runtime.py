"""Check a requested worker budget using bounded PF prefixes and thread probes."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
from time import monotonic, sleep

from joblib import Parallel, cpu_count, delayed
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info
import yaml

from ...inference.backends.particle_filter import run_state_model_particle_filter
from ...utils.parallel import parallel_job_count, single_threaded_processes


def check_worker(config, data, reference, barrier, count):
    # One initial task per worker waits until all workers have entered. This
    # verifies actual process capacity, not just the requested Parallel value.
    directory = Path(barrier)
    (directory / str(os.getpid())).touch(exist_ok=False)
    started = monotonic()
    while len(list(directory.iterdir())) < count:
        if monotonic() - started > 120:
            raise RuntimeError('Not all requested workers entered within 120 seconds')
        sleep(.05)
    result = run_state_model_particle_filter(
        engine_config=config, subject_id=129, condition=1,
        stimulus=data[[f'feature{i}' for i in range(1, 5)]].to_numpy(),
        choices=data.choice.to_numpy(), feedback=data.feedback.to_numpy(),
        particle_count=16, choice_readout_power=1., filter_seed=8326,
    )
    values = {
        'result/observation_probabilities/prior_t': result.marginal_probabilities,
        'result/state_probabilities/hypothesis_prior': result.marginal_hypothesis_prior,
        'result/state_probabilities/active_probability': result.marginal_active_probability,
        'result/diagnostics/pre_choice_ess': result.pre_choice_ess,
        'result/diagnostics/post_choice_ess': result.post_choice_ess,
        'result/diagnostics/resampled': result.resampled,
    }
    for key, value in values.items():
        np.testing.assert_array_equal(value, reference[key][:len(data)], err_msg=key)
    libraries = threadpool_info()
    assert all(pool['num_threads'] == 1 for pool in libraries), libraries
    with single_threaded_processes():
        inner_jobs = parallel_job_count(128, 128)
    assert inner_jobs == 1
    return {'pid': os.getpid(), 'exact_arrays': len(values), 'inner_jobs': inner_jobs,
            'threadpools': libraries,
            'thread_environment': {name: os.environ.get(name) for name in (
                'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    if not 2 <= args.workers <= cpu_count():
        parser.error('workers must be between 2 and the available CPU count')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config_path = Path('configs/exp123/model_struct/pmh_model_cond1_0826.yaml')
    data_path = Path('data/exp123/processed/Task2_processed.csv')
    config = yaml.safe_load(config_path.read_text())
    config['modules']['hypo_transitions_mod']['kwargs']['capacity'] = 3
    config['modules']['hypo_transitions_mod']['kwargs']['persistent_execution']['enabled'] = False
    frame = pd.read_csv(data_path)
    data = frame.loc[frame.iSub.eq(129)].iloc[:8]
    with np.load(args.reference, allow_pickle=False) as arrays:
        reference = {key: arrays[key] for key in arrays.files
                     if key.startswith(('result/observation_probabilities/',
                                        'result/state_probabilities/', 'result/diagnostics/'))}
    started = monotonic()
    with TemporaryDirectory(prefix='model0826_workers_') as barrier, single_threaded_processes():
        jobs = parallel_job_count(args.workers, args.workers)
        rows = Parallel(n_jobs=jobs, batch_size=1)(
            delayed(check_worker)(config, data, reference, barrier, jobs) for _ in range(jobs)
        )
    assert len({row['pid'] for row in rows}) == args.workers
    report = {
        'requested_workers': args.workers, 'unique_worker_count': len(rows),
        'available_cpus': cpu_count(), 'affinity_cpus': len(os.sched_getaffinity(0)),
        'wall_seconds': monotonic() - started, 'trials_per_task': len(data),
        'particles': 16, 'filter_seed': 8326, 'engine_config': config,
        'python': platform.python_version(),
        'versions': {name: importlib.metadata.version(name)
                     for name in ('numpy', 'numba', 'joblib', 'threadpoolctl')},
        'workers': rows,
        'sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in (config_path, data_path, args.reference,
                                *Path('src/Bayesian_state').rglob('*.py'))},
    }
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f"{len(rows)} unique workers; all numeric threads=1; {len(rows)*6} arrays exact; "
          f"wall={report['wall_seconds']:.2f}s")


if __name__ == '__main__':
    main()

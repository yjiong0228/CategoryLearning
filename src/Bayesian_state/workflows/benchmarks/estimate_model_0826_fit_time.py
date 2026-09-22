"""Bounded timing inputs for an estimate, not a parameter search or full fit."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from time import perf_counter

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import yaml

from ...hypothesis_space.geometry import warmup_dykstra_numba
from ...simulation.data import SubjectTrialDataLoader
from ...simulation.execution import evaluate_state_model_run
from ...utils.parallel import parallel_job_count, single_threaded_processes


def time_case(case: dict) -> dict:
    config_path = Path(f"configs/exp123/model_struct/pmh_model_cond{case['condition']}_0826.yaml")
    config = yaml.safe_load(config_path.read_text())
    config['inference']['particle_count'] = case['particles']
    transition = config['modules']['hypo_transitions_mod']['kwargs']
    transition['capacity'] = case['capacity']
    transition['persistent_execution']['enabled'] = bool(case['chi'])
    loader = SubjectTrialDataLoader(config)
    frame = loader._get_subject_frame(case['subject'], 1.)
    arrays = loader._extract_arrays(frame, case.get('max_trials'))
    with single_threaded_processes():
        warmup_dykstra_numba()
        started = perf_counter()
        result = evaluate_state_model_run(
            subject_id=case['subject'], condition=case['condition'], arrays=arrays,
            params={}, engine_config_template=config,
            processed_data_dir=Path('data/exp123/processed').resolve(), window_size=16,
            keep_logs=False, prediction_mode='prior_t', selection_prediction_mode='prior_t',
            loss_metric='choice_nll', trajectory_seed=8326,
        )
        elapsed = perf_counter() - started
    assert np.isfinite(result.mean_error)
    return {**case, 'trials': len(arrays.feedback), 'seconds': elapsed,
            'choice_nll': float(result.mean_error), 'engine_config': deepcopy(config)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    # Start with a single small call before the twelve full-sequence timing jobs.
    smoke = time_case(dict(condition=1, subject=129, particles=4,
                           capacity=3, chi=0, max_trials=32))
    cases = [dict(condition=c, subject=s, particles=r, capacity=m, chi=chi)
             for c, s in ((1, 129), (2, 229), (3, 301))
             for r in (16, 64) for m, chi in ((3, 0), (5, 1))]
    with single_threaded_processes():
        rows = Parallel(n_jobs=parallel_job_count(128, len(cases)), batch_size=1)(
            delayed(time_case)(case) for case in cases)
    data_path = Path('data/exp123/processed/Task2_processed.csv')
    data = pd.read_csv(data_path)
    counts = data.groupby(['condition', 'iSub']).size()
    report = {
        'purpose': 'bounded single-seed timing; not a full fit; no fitted parameters',
        'seed': 8326, 'smoke': smoke, 'timings': rows,
        'python': platform.python_version(), 'cpu_affinity': len(os.sched_getaffinity(0)),
        'load_average_at_end': os.getloadavg(),
        'versions': {name: importlib.metadata.version(name)
                     for name in ('numpy', 'scipy', 'numba', 'joblib', 'pandas')},
        'subject_trials': [{'condition': int(c), 'subject': int(s), 'trials': int(n)}
                           for (c, s), n in counts.items()],
        'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                   (data_path, *Path('src/Bayesian_state').rglob('*.py'),
                    *Path('configs/exp123/model_struct').glob('*0826.yaml'))},
    }
    (args.output_dir/'timings.json').write_text(json.dumps(report, indent=2) + '\n')
    for row in rows:
        print({k: v for k, v in row.items() if k != 'engine_config'})


if __name__ == '__main__':
    main()

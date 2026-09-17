"""Bounded, reproducible PF/generation checks against separately saved outputs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
from time import perf_counter

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import yaml

from ...inference.backends.particle_filter import run_state_model_particle_filter
from ...optimization.model_0826 import build_model_0826_cell_engine
from ...simulation.autonomous import run_autonomous_category_learning


def numerical_outputs(result) -> dict[str, np.ndarray]:
    """Collect numeric public arrays and nested diagnostics, including genealogy."""
    arrays = {}
    def visit(name, value):
        if isinstance(value, np.ndarray):
            if value.dtype.kind in 'biufc':
                arrays[name] = value
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(f'{name}/{key}', item)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(f'{name}/{index}', item)
        elif isinstance(value, (int, float, bool, np.number)):
            arrays[name] = np.asarray(value)
    visit('result', vars(result))
    if hasattr(result, 'trajectory'):
        visit('trajectory', vars(result.trajectory))
    return arrays


def cases() -> list[dict]:
    rows = []
    for condition, subject, settings in (
        (1, 129, [(1,0),(3,0),(3,1),(5,0),(5,1)]),
        (2, 229, [(3,0),(3,1)]),
        (3, 301, [(1,0),(3,0),(3,1)]),
    ):
        for capacity, execution in settings:
            rows.append(dict(name=f'c{condition}_M{capacity}_chi{execution}', condition=condition,
                subject=subject, capacity=capacity, execution=execution, cell='PMH', trials=32,
                particles=16 if (condition,capacity,execution)==(1,3,0) else 4, seed=8326))
    for cell in ('P','PM','PH'):
        rows.append(dict(name=f'c1_{cell}', condition=1, subject=129, capacity=3,
            execution=0, cell=cell, trials=32, particles=4, seed=8326))
    rows.append(dict(name='c1_long', condition=1, subject=129, capacity=3,
        execution=1, cell='PMH', trials=None, particles=4, seed=8326))
    for condition, subject in ((1,129),(3,301)):
        rows.append(dict(name=f'c{condition}_generation', condition=condition, subject=subject,
            capacity=3, execution=1, cell='PMH', trials=32, particles=0, seed=8326))
    return rows


def run_case(case: dict, mode: str, repeats: int) -> tuple[dict, dict[str, np.ndarray]]:
    config_path = Path(f"configs/exp123/model_struct/pmh_model_cond{case['condition']}_0826.yaml")
    engine = yaml.safe_load(config_path.read_text())
    transition = engine['modules']['hypo_transitions_mod']['kwargs']
    transition['capacity'] = case['capacity']
    transition['persistent_execution']['enabled'] = bool(case['execution'])
    engine = build_model_0826_cell_engine(engine, case['cell'])
    if mode != 'default':
        options = engine['partition']['kwargs']
        options['zero_beta_fast_path'] = mode != 'disabled'
        options['boundary_distance_cache_max_entries'] = 0
    data = pd.read_csv('data/exp123/processed/Task2_processed.csv')
    data = data.loc[data.iSub.eq(case['subject'])].iloc[:case['trials']]
    kwargs = dict(engine_config=engine, subject_id=case['subject'], condition=case['condition'],
        stimulus=data[[f'feature{i}' for i in range(1,5)]].to_numpy())
    if case['particles']:
        kwargs.update(choices=data.choice.to_numpy(), feedback=data.feedback.to_numpy(),
            particle_count=case['particles'], choice_readout_power=1., filter_seed=case['seed'])
        execute = run_state_model_particle_filter
    else:
        kwargs.update(categories=data.category.to_numpy(), trajectory_seed=case['seed'])
        execute = run_autonomous_category_learning
    seconds = []
    previous = None
    for _ in range(repeats + 1):
        start = perf_counter()
        result = execute(**kwargs)
        seconds.append(perf_counter() - start)
        arrays = numerical_outputs(result)
        if previous is not None:
            compare_arrays(previous, arrays)
        previous = arrays
    return {**case, 'trials':len(data), 'mode':mode, 'cold_seconds':seconds[0],
        'warm_seconds':seconds[1:], 'warm_median_seconds':float(np.median(seconds[1:])),
        'numeric_array_count':len(arrays), 'process_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'engine_config':engine}, arrays


def compare_arrays(reference, actual) -> None:
    if set(reference) != set(actual):
        raise AssertionError('Numeric output keys differ')
    for name in actual:
        np.testing.assert_array_equal(actual[name], reference[name], err_msg=name)


def compare_modes(case: dict, repeats: int):
    """Rotate mode order; each measurement starts a fresh PF/learner."""
    modes = ('disabled', 'zero-only', 'default')
    rows, saved = {}, {}
    for repeat in range(repeats):
        for offset in range(len(modes)):
            mode = modes[(repeat + offset) % len(modes)]
            row, arrays = run_case(case, mode, 1)
            if saved:
                compare_arrays(next(iter(saved.values())), arrays)
            saved[mode] = arrays
            if mode not in rows:
                rows[mode] = row
            else:
                rows[mode]['warm_seconds'].extend(row['warm_seconds'])
            rows[mode]['warm_median_seconds'] = float(np.median(rows[mode]['warm_seconds']))
    return [(rows[mode], saved[mode]) for mode in modes]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reference', type=Path, help='Independently saved benchmark directory')
    parser.add_argument('--cases', nargs='+', help='Subset of case names; default is the bounded validation matrix')
    parser.add_argument('--mode', choices=('default','zero-only','disabled'), default='default')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--compare-modes', action='store_true', help='Rotate disabled/zero-only/default timing within each case')
    args = parser.parse_args()
    if args.repeats < 1 or not 1 <= args.jobs <= 4:
        parser.error('repeats >= 1 and jobs in [1,4] are required')
    if args.compare_modes and args.jobs != 1:
        parser.error('compare-modes uses one process to measure serial latency')
    selected = cases()
    if args.cases:
        unknown = set(args.cases) - {case['name'] for case in selected}
        if unknown:
            parser.error(f'Unknown cases: {sorted(unknown)}')
        selected = [case for case in selected if case['name'] in args.cases]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    if args.compare_modes:
        outputs = [output for case in selected for output in compare_modes(case, args.repeats)]
    else:
        outputs = Parallel(n_jobs=args.jobs)(delayed(run_case)(case,args.mode,args.repeats) for case in selected)
    elapsed = perf_counter() - started
    rows = []
    for row, arrays in outputs:
        suffix = '_'+row['mode'] if args.compare_modes else ''
        destination = args.output_dir / (row['name']+suffix+'.npz')
        np.savez_compressed(destination, **arrays)
        if args.reference:
            with np.load(args.reference / (row['name']+'.npz'), allow_pickle=False) as reference:
                compare_arrays(reference, arrays)
            row['exact_reference_match'] = True
        rows.append(row)
        print(f"{row['name']}: {row['warm_median_seconds']:.3f}s; {len(arrays)} arrays", flush=True)
    paths = [Path('data/exp123/processed/Task2_processed.csv'), *Path('src/Bayesian_state').rglob('*.py'),
        *Path('configs/exp123/model_struct').glob('*0826.yaml')]
    report = {'cases':rows, 'jobs':args.jobs, 'wall_seconds':elapsed, 'python':platform.python_version(),
        'versions':{name:importlib.metadata.version(name) for name in ('numpy','scipy','pandas','numba','joblib')},
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)},
        'reference':str(args.reference) if args.reference else None}
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()

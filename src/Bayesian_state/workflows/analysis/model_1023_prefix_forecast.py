"""Continue frozen Model 1022 states after the original observed prefix.

Run --config YAML --output NEW [--smoke]. Three archived candidates per subject
retain their parameters, task boundaries, original filter seeds and cognitive
runtime. Each filter samples sixteen equally weighted future branches. Future
choices and feedback come exclusively from the model and task semantics.
--smoke uses two four-particle filters and two branches on 8+8 training trials,
one selected candidate for each represented condition, and one worker. It tests
exact independent prefix replay, passive capture and reversed branch ordering.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
HELPER = 'src/Bayesian_state/simulation/filtered_continuation.py'
OBSERVER = 'src/Bayesian_state/simulation/continuation_diagnostics.py'
PROBE = 'src/Bayesian_state/workflows/analysis/model_1021_entry_precision_probe.py'


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify(bindings: dict) -> None:
    for path, sha in bindings.items():
        if digest(Path(path)) != sha:
            raise ValueError('Frozen input or source changed: ' + str(path))


def forecast_inputs(training, future, smoke: bool) -> tuple[dict, dict]:
    """Select only prefix observations and future task inputs."""
    if smoke:
        return ({k: training[k][:8].copy() for k in ('stimulus', 'choices', 'feedback')},
                {k: training[k][8:16].copy() for k in ('stimulus', 'categories')})
    return ({k: training[k].copy() for k in ('stimulus', 'choices', 'feedback')},
            {k: future[k].copy() for k in ('stimulus', 'categories')})


def worker(file: Path) -> None:
    import numpy as np
    from threadpoolctl import threadpool_info
    job = read(file)
    verify(job['bindings'])
    probe = load('compact_prefix_precision_driver', PROBE)
    helper = load('src.Bayesian_state.simulation.filtered_continuation', HELPER)
    observer = load('src.Bayesian_state.simulation.continuation_diagnostics', OBSERVER)
    from src.Bayesian_state.inference.backends.particle_filter import run_state_model_particle_filter
    from src.Bayesian_state.simulation.parameters import apply_fixed_hyperparams_to_engine_config
    from src.Bayesian_state.utils.seeding import stable_seed
    folder = Path(job['folder'])
    meta = read(folder / 'prepared.json')
    if (job['compact_variant'], job['entry_control']) != ('TRANSFER', 'PRECISION'):
        raise ValueError('Need frozen TRANSFER mechanism')
    verify({str(Path(job['runtime']) / k): v for k, v in meta['frozen_cognitive_source_sha256'].items()})
    verify({str(folder / k): v for k, v in meta['array_sha256'].items()})
    verify({str(folder / 'engine.json'): meta['engine_sha256']})
    engine = read(folder / 'engine.json')
    engine['modules']['hypo_transitions_mod']['kwargs']['intervention'] = job['mechanism']
    engine = apply_fixed_hyperparams_to_engine_config(engine, job['candidate']['hyperparams'])
    with np.load(folder / 'training.npz') as training, np.load(folder / 'sealed_test.npz') as future:
        prefix, future = forecast_inputs(training, future, job['smoke'])
    kwargs = dict(engine_config=engine, subject_id=meta['subject'], condition=meta['condition'], **prefix,
        particle_count=job['particles'], particle_blocks=1, filter_seed=job['seeds'][0], choice_readout_power=1.,
        processed_data_dir=meta['paths']['processed_dir'], dataset_paths=meta['paths'])
    start, cpu = time.monotonic(), time.process_time()
    called = []

    def forecast(installed_job, workflow):
        # Existing worker contexts install exactly the fitted equations. Only
        # the terminal scoring operation is replaced by prefix continuation.
        called.append(True)
        models, result = helper.capture_prefix(**kwargs)
        prediction = result.observation_probabilities['prior_t']
        if job['smoke']:
            independent = run_state_model_particle_filter(**kwargs)
            expected = independent.observation_probabilities['prior_t']
            np.testing.assert_array_equal(result.final_weights, independent.final_weights)
        else:
            with np.load(job['reference']) as saved:
                np.testing.assert_array_equal(saved['seeds'], job['seeds'])
                np.testing.assert_array_equal(saved['choices'][:len(prefix['choices'])], prefix['choices'])
                expected = saved['probabilities'][0, :len(prefix['choices'])]
        np.testing.assert_array_equal(prediction, expected, err_msg='Prefix changed from original filter')
        endpoints = [observer.endpoint_state(m, job['target_rule']) for m in models]
        seeds = [stable_seed(dict(protocol=job['settings']['protocol'], base=job['settings']['seed'],
                    subject=meta['subject'], filter_seed=job['seeds'][0], branch=i))
                 for i in range(job['settings']['branches_per_filter'])]
        generate = dict(models=models, weights=result.final_weights, observed_choices=prefix['choices'],
            observed_feedback=prefix['feedback'], **future, condition=meta['condition'])
        capture = observer.ContinuationObserver(job['target_rule'])
        arrays = helper.generate_continuations(**generate, branch_seeds=seeds, observer=capture)
        if job['smoke']:
            plain = helper.generate_continuations(**generate, branch_seeds=seeds)
            for k in plain:
                np.testing.assert_array_equal(plain[k], arrays[k], err_msg='Observer changed ' + k)
            reverse = helper.generate_continuations(**generate, branch_seeds=seeds[::-1], observer=capture)
            for k, value in arrays.items():
                other = reverse[k] if k in ('terminal_weights', 'categories') else reverse[k][::-1]
                np.testing.assert_array_equal(value, other, err_msg='Branch order changed ' + k)
        arrays.update({'endpoint_' + k: np.asarray([e[k] for e in endpoints]) for k in endpoints[0]})
        arrays.update(prefix_probabilities=prediction, filter_seed=np.asarray(job['seeds'][0]))
        target = Path(job['target']); target.parent.mkdir(parents=True, exist_ok=True)
        with target.with_suffix('.npz').open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        write(target.with_suffix('.json'), dict(status='complete', subject=meta['subject'],
            condition=meta['condition'], candidate=job['candidate'], case_id=job['case_id'], roles=job['roles'],
            cohort=job['cohort'], filter_seed=job['seeds'][0], particles=job['particles'], paths=len(seeds),
            training_trials=len(prefix['choices']), forecast_trials=len(future['categories']),
            target_rule=job['target_rule'], prefix_exact=True, smoke=job['smoke'],
            branch_order_check=job['smoke'], observer_passivity_check=job['smoke'],
            seconds=time.monotonic()-start, cpu_seconds=time.process_time()-cpu, numeric_pools=threadpool_info(),
            predictions_sha256=digest(target.with_suffix('.npz'))))

    original = probe.driver.io.score
    probe.driver.io.score = forecast
    try:
        probe.worker(file)
    finally:
        probe.driver.io.score = original
    if len(called) != 1:
        raise ValueError('Need exactly one prefix filter execution')
    verify(job['bindings'])


def prepare(config: Path, output: Path, smoke: bool) -> tuple[dict, list[dict]]:
    import yaml
    settings = yaml.safe_load(config.read_text())
    source = ROOT / settings['evaluation']
    plan, done = read(source/'cases.json'), read(source/'completion.json')
    if done['status'] != 'complete' or done['smoke'] or digest(source/'cases.json') != done['cases_sha256']:
        raise ValueError('Need completed frozen candidate evaluations')
    if (settings['subjects'] != [205, 306, 309] or settings['filters'] != 8
            or settings['particles'] != 1024 or settings['branches_per_filter'] != 16):
        raise ValueError('Preserve the registered nine-point 8 x 16 forecast budget')
    verify(plan['inputs_sha256'])
    cases = [deepcopy(c) for c in plan['cases'] if c['subject'] in settings['subjects']]
    if any(sum(c['subject'] == sid for c in cases) != 3 for sid in settings['subjects']):
        raise ValueError('Require three archived candidates per subject')
    if smoke:
        chosen = {}
        for c in cases:
            if 'selected' in c['roles']:
                chosen.setdefault(c['condition'], c)
        cases = list(chosen.values())
    effective = dict(settings)
    if smoke:
        effective.update(filters=2, particles=4, branches_per_filter=2, workers=1)
    hashes = {str((ROOT/k).resolve()): digest(ROOT/k) for k in (ENTRY, HELPER, OBSERVER)}
    bindings = {str((source/k).resolve()): digest(source/k) for k in ('cases.json', 'completion.json', 'protocol.json')}
    bindings[str(config.resolve())] = digest(config)
    jobs, records = [], []
    for c in cases:
        own = Path(c['evaluation'])
        record, = read(own/'autonomous/evaluation.json')['rows']
        original_files = sorted((own/'jobs').glob('evaluate_*.json'))
        group = [f for f in original_files if read(f)['candidate']['id'] == c['candidate']]
        if len(group) != settings['filters']:
            raise ValueError('Missing original conditional filter jobs')
        seeds = [read(f)['seeds'][0] for f in group]
        if seeds != c['reference_seeds']['conditional']:
            raise ValueError('Filter seeds differ from frozen candidate evaluation')
        original_protocol = read(own/'protocol.json')
        fit = Path(original_protocol['fit'])
        if (original_protocol['compact_variant'] != 'TRANSFER'
                or digest(fit/'nominees.json') != original_protocol['nomination_sha256']):
            raise ValueError('Fitted mechanism or parameters changed')
        item = {k: c[k] for k in ('case_id','subject','condition','cohort','candidate','roles','evaluation','candidate_hyperparams')}
        item['target_rule'] = record['target_rule']
        records.append(item)
        for f in group[:effective['filters']]:
            job = read(f)
            if job['candidate']['hyperparams'] != c['candidate_hyperparams'] or job['particles'] != settings['particles']:
                raise ValueError('Frozen parameters or numerical budget changed')
            reference = Path(job['target']).with_suffix('.npz')
            expected = read(reference.with_suffix('.json'))['predictions_sha256']
            if digest(reference) != expected:
                raise ValueError('Saved prefix reference changed')
            own_bindings = dict(hashes, **{str(ROOT/k): v for k,v in job['source_sha256'].items()})
            own_bindings.update({str(f.resolve()): digest(f), str(reference): expected,
                str((fit/'nominees.json').resolve()): digest(fit/'nominees.json')})
            verify(own_bindings)
            name = c['case_id'] + '_' + f.stem.rsplit('_', 1)[-1]
            for k in ('decision_basis_target','decision_basis_sources','evaluation_route_sources'):
                job.pop(k, None)
            job.update(name=name, settings=effective, roles=c['roles'], cohort=c['cohort'], case_id=c['case_id'],
                smoke=smoke, particles=effective['particles'], reference=str(reference), bindings=own_bindings,
                target_rule=record['target_rule'], target=str((output/'chunks'/name).resolve()))
            jobs.append(job)
    output.mkdir(parents=True)
    (output/'jobs').mkdir()
    sources = dict(hashes)
    for job in jobs:
        sources.update({str(ROOT/k): v for k,v in job['source_sha256'].items()})
    for name in sources:
        relative = Path(name).relative_to(ROOT)
        dest = output/'source'/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(name, dest)
    shutil.copy2(config, output/'protocol.yaml')
    protocol = dict(settings=effective, smoke=smoke, cases=records, jobs=len(jobs),
        workers=min(effective['workers'], len(os.sched_getaffinity(0)), len(jobs)),
        source_sha256=sources, inputs_sha256=bindings, python=sys.version,
        versions={k: importlib.metadata.version(k) for k in ('numpy','scipy','pandas','PyYAML')},
        conditioning='Original observed prefix, then own choices and task-generated feedback',
        uncertainty_unit='Independent prefix filter, including all descendant branches',
        data_use='Development diagnosis; expansion participants retain uncertain prior influence',
        endpoint_timing='After last prefix feedback; trial diagnostics before each future feedback')
    write(output/'protocol.json', protocol)
    return protocol, jobs


def run(config: Path, output: Path, smoke: bool) -> None:
    if output.exists():
        raise FileExistsError('Use a new continuation directory')
    protocol, jobs = prepare(config, output, smoke)
    start = time.monotonic()
    print(f"{len(jobs)} prefix jobs, {protocol['workers']} single-thread workers", flush=True)

    def launch(job):
        file = output/'jobs'/(job['name']+'.json')
        write(file, job)
        launcher = ('import runpy,sys;runtime,script=sys.argv[1:3];sys.path.insert(0,runtime);'
                    "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
            MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', NUMBA_NUM_THREADS='1')
        with tempfile.TemporaryDirectory(prefix='compact_prefix_') as cache, file.with_suffix('.log').open('x') as log:
            env.update(NUMBA_CACHE_DIR=cache, MPLCONFIGDIR=cache)
            subprocess.run([sys.executable, '-c', launcher, job['runtime'], str(ROOT/ENTRY), '--worker', str(file.resolve())],
                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        return read(Path(job['target']).with_suffix('.json'))

    rows = []
    with ThreadPoolExecutor(max_workers=protocol['workers']) as pool:
        for future in as_completed([pool.submit(launch, job) for job in jobs]):
            rows.append(future.result())
            print(f'{len(rows)}/{len(jobs)} prefix jobs complete', flush=True)
    verify(protocol['inputs_sha256']); verify(protocol['source_sha256'])
    write(output/'completion.json', dict(status='complete', smoke=smoke, rows=rows,
        jobs=len(jobs), paths=sum(r['paths'] for r in rows), seconds=time.monotonic()-start,
        protocol_sha256=digest(output/'protocol.json'), prefix_exact=all(r['prefix_exact'] for r in rows)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('config','output','worker'):
        parser.add_argument('--'+key, type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.config and args.output:
        run(args.config, args.output, args.smoke)
    else:
        parser.error('Supply --config and --output, or --worker')

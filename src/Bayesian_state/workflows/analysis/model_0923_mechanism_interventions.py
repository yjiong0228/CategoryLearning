"""Fixed-parameter, equal-seed interventions on archived Model 0923 fits.

Run with --output pointing to a new directory. --smoke uses one path, eight
trials and one worker. The sole cognitive override is the shared transition
module; every other cognitive module runs from the validated fit snapshot.
Baseline sparse traces must reproduce the independent saved cohort traces.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
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

REPO = Path(__file__).resolve().parents[4]
TRANSITION = 'src/Bayesian_state/model/modules/hypothesis_transition/mechanistic_rule_search.py'


def load_source(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def cohort():
    return load_source('intervention_cohort', 'src/Bayesian_state/workflows/analysis/model_0923_cohort_path_diagnosis.py')


def install_transition() -> None:
    """Replace one implementation in process; archived files remain read-only."""
    import src.Bayesian_state.model.assembly as assembly
    import src.Bayesian_state.model.modules.hypothesis_transition as package
    module = load_source('src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search', TRANSITION)
    cls = module.MechanisticRuleSearchHypothesisTransitionModule
    assembly.MechanisticRuleSearchHypothesisTransitionModule = cls
    package.MechanisticRuleSearchHypothesisTransitionModule = cls


def prepare(job: dict, c, d) -> None:
    import numpy as np
    import pandas as pd
    record, settings = job['record'], job['settings']
    directory = Path(job['directory'])
    old = REPO / settings['source_diagnosis'] / record['id']
    previous = c.load_json(old / 'prepared.json')
    directory.mkdir()
    cache = directory / 'resources_cache'
    cache.mkdir()
    validation = c.validate_record(record, d)
    for file in (old / 'resources_cache').glob('*.npz'):
        shutil.copy2(file, cache / file.name)
    inputs = c.resolve_record(record, cache, settings['smoke_trials'] if job['smoke'] else None)
    observed = pd.read_csv(old / 'observed_trials.csv').iloc[:len(inputs['arrays'].choices)]
    for column in ('choice', 'category', 'feedback'):
        np.testing.assert_array_equal(observed[column], getattr(inputs['arrays'], {'choice':'choices','category':'categories','feedback':'feedback'}[column]))
    if inputs['point'] != c.load_json(old / 'parameters.json'):
        raise ValueError('Fitted parameters changed')
    for name, expected in previous['cognitive_source_sha256'].items():
        if c.digest(Path(d.ROOT_DIR) / name) != expected:
            raise ValueError(f'Archived cognitive source changed: {name}')
    original_manifest = c.load_json(old.parent / 'manifest.json')
    baseline = {}
    for file in [old/'rule_catalog.csv', old/'observed_trials.csv', old/'parameters.json', old/'prepared.json', *sorted((old/'traces').glob('C_*.npz'))]:
        actual = c.digest(file)
        if actual != original_manifest['artifact_sha256'][str(file.relative_to(old.parent))]:
            raise ValueError(f'Independent baseline artifact changed: {file}')
        baseline[str(file.resolve())] = actual
    observed.to_csv(directory / 'observed_trials.csv', index=False)
    for name in ('rule_catalog.csv', 'parameters.json'):
        shutil.copy2(old / name, directory / name)
    seeds = previous['seeds'][:1 if job['smoke'] else settings['paths']]
    if len(seeds) != (1 if job['smoke'] else settings['paths']):
        raise ValueError('Requested path budget differs from saved seeds')
    c.write_json(directory / 'source_validation.json', validation)
    c.write_json(directory / 'prepared.json', dict(previous, trials=len(observed), seeds=seeds,
        baseline_sha256=baseline, smoke=job['smoke'], intervention_source_sha256=c.digest(REPO / TRANSITION)))
    for variant in settings['variants']:
        (directory / variant / 'traces').mkdir(parents=True)


def trace(job: dict, c, d) -> None:
    import numpy as np
    from threadpoolctl import threadpool_info
    directory, settings = Path(job['directory']), job['settings']
    prepared = c.load_json(directory / 'prepared.json')
    if c.digest(REPO / TRANSITION) != prepared['intervention_source_sha256']:
        raise ValueError('Intervention implementation changed during run')
    install_transition()
    inputs = c.resolve_record(job['record'], directory / 'resources_cache', settings['smoke_trials'] if job['smoke'] else None)
    engine = deepcopy(inputs['engine'])
    engine['modules']['hypo_transitions_mod']['kwargs']['intervention'] = settings['variants'][job['variant']]
    arrays = inputs['arrays']
    indices = job['indices']
    seeds = [prepared['seeds'][i] for i in indices]
    categories = engine['partition']['kwargs']['n_cats']
    capacity = engine['modules']['hypo_transitions_mod']['kwargs']['max_capacity']

    class Recorder(d.PathRecorder):
        def __init__(self):
            super().__init__(arrays.categories, len(seeds), capacity, categories, inputs['condition'] == 3)
            shape = self.arrays['choice'].shape
            for key in ('admitted', 'rejected'):
                self.arrays[key] = np.zeros(shape, dtype=bool)
            for key in ('search_probability', 'failure_pressure', 'accumulated_failure'):
                self.arrays[key] = np.full(shape, np.nan)
            self.arrays['evidence_score'] = np.full(self.arrays['rule'].shape, np.nan)
            self.histories = [{} for _ in seeds]

        def before(self, model, choice, feedback):
            token = super().before(model, choice, feedback)
            row, ids, _, _ = token
            t, path = row[:2]
            transition = model.engine.get_module(d.ModuleRole.HYPOTHESIS_TRANSITION)
            event = transition.transition_log[-1]
            self.arrays['admitted'][t, path] = event['admission_event']
            self.arrays['rejected'][t, path] = event.get('newcomer_rejected', False)
            for key, field in [('search_probability', 'current_event_probability'), ('failure_pressure','failure_pressure'), ('accumulated_failure','accumulated_failure')]:
                self.arrays[key][t, path] = getattr(transition, field, np.nan)
            history = self.histories[path]
            self.histories[path] = {int(h): history.get(int(h), []) for h in ids}
            if job['variant'] == 'R1':
                # Independent observer history uses recorded pre-beta evidence.
                np.testing.assert_allclose(transition.recent_log_evidence[ids].sum(axis=1),
                    [sum(self.histories[path][int(h)]) for h in ids], atol=1e-12, rtol=0)
            return token

        def after(self, model, token, result):
            super().after(model, token, result)
            row, ids, _, _ = token
            t, path = row[:2]
            feedback = self.arrays['feedback'][t, path]
            neutral = (.5 if feedback == 0 else .25) if inputs['condition'] == 3 else (1/categories if feedback == 1 else 1-1/categories)
            evidence = np.log(np.maximum(self.arrays['evidence'][row], np.finfo(float).tiny))-np.log(neutral)
            for slot, (h, value) in enumerate(zip(ids, evidence)):
                history = (self.histories[path][int(h)]+[float(value)])[-settings['evidence_window']:]
                self.histories[path][int(h)] = history
                self.arrays['evidence_score'][t, path, slot] = sum(history)

    recorder = Recorder()
    kwargs = dict(engine_config=engine, subject_id=inputs['subject'], condition=inputs['condition'],
        stimulus=arrays.stimulus, categories=arrays.categories,
        processed_data_dir=inputs['paths']['processed_dir'], dataset_paths=inputs['paths'])
    cpu_start, wall_start = time.process_time(), time.monotonic()
    with d.record_lifecycle(recorder):
        for seed in seeds:
            d.autonomous.run_autonomous_category_learning(**kwargs, trajectory_seed=seed)
    recorder.validate()
    baseline_checks = 0
    if job['variant'] == 'B':
        for name, expected in prepared['baseline_sha256'].items():
            if not name.endswith('.npz'):
                continue
            file = Path(name)
            if c.digest(file) != expected:
                raise ValueError('Saved baseline trace changed')
            with np.load(file) as old:
                for path, seed in enumerate(seeds):
                    matches = np.flatnonzero(old['seeds'] == seed)
                    if not len(matches):
                        continue
                    for key in old.files:
                        if key not in ('seeds', 'categories'):
                            np.testing.assert_array_equal(recorder.arrays[key][:, path], old[key][:len(arrays.choices), int(matches[0])], err_msg=f'{key}, seed={seed}')
                    baseline_checks += 1
        if baseline_checks != len(seeds):
            raise ValueError('Missing independently saved baseline paths')
    file = directory / job['variant'] / 'traces' / f'C_{indices[0]:03d}.npz'
    with file.open('xb') as stream:
        np.savez_compressed(stream, **recorder.arrays, seeds=seeds, categories=arrays.categories)
    c.write_json(file.with_suffix('.json'), dict(variant=job['variant'], paths=len(seeds), trials=len(arrays.choices),
        baseline_exact_paths=baseline_checks, sha256=c.digest(file), cpu_seconds=time.process_time()-cpu_start,
        wall_seconds=time.monotonic()-wall_start, numeric_pools=threadpool_info(), pid=os.getpid(),
        intervention=settings['variants'][job['variant']], intervention_source_sha256=prepared['intervention_source_sha256']))


def worker(file: Path) -> None:
    c = cohort()
    job = c.load_json(file)
    d = c.load_diagnosis()
    with d.single_threaded_processes():
        (prepare if job['phase']=='prepare' else trace)(job, c, d)


def launch(job: dict, output: Path, c) -> None:
    file = output / 'jobs' / f"{job['record']['id']}_{job['phase']}_{job.get('variant','')}_{job.get('indices',[0])[0]:03d}.json"
    c.write_json(file, job)
    launcher = ('import runpy,sys; source,script=sys.argv[1:3]; sys.path.insert(0,source); '
                "sys.argv=[script]+sys.argv[3:]; runpy.run_path(script,run_name='__main__')")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    with tempfile.TemporaryDirectory(prefix='model0923_intervention_numba_') as cache:
        env['NUMBA_CACHE_DIR'] = cache
        with file.with_suffix('.log').open('x') as stream:
            result = subprocess.run([sys.executable, '-c', launcher, str(Path(job['record']['runtime']).resolve()),
                str(Path(__file__).resolve()), '--worker', str(file)], env=env, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Worker failed: {file.with_suffix(".log")}')


def run(config: Path, output: Path, smoke: bool, only: list[int] | None) -> None:
    import yaml
    from src.Bayesian_state.utils.parallel import parallel_job_count
    c = cohort()
    settings = yaml.safe_load(config.read_text())
    if output.exists():
        raise FileExistsError('Choose a new output directory')
    if set(settings['variants']) != {'B','S_control','S1','R_control','R1','C1'}:
        raise ValueError('Expected the six predeclared variants')
    wanted = only or settings['subjects']
    records = [r for r in c.records_from_config(yaml.safe_load((REPO/settings['cohort_config']).read_text())) if r['primary'] and r['subject'] in wanted]
    if set(r['subject'] for r in records) != set(wanted):
        raise ValueError('A requested participant has no primary fit')
    output = output.resolve()
    (output/'jobs').mkdir(parents=True)
    (output/'effective_config.yaml').write_text(yaml.safe_dump(settings, sort_keys=False))
    start = time.monotonic()
    jobs = [dict(record=r, settings=settings, directory=str(output/r['id']), smoke=smoke, phase='prepare') for r in records]

    def execute(pending):
        workers = 1 if smoke else parallel_job_count(settings['parallel_budget'], len(pending))
        print(f'Starting {len(pending)} jobs with {workers} workers; numeric threads=1', flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(launch, j, output, c):j for j in pending}
            for i, future in enumerate(as_completed(futures), 1):
                future.result()
                j = futures[future]
                print(f"{i}/{len(pending)} {j['record']['id']} {j.get('variant','prepare')} {j.get('indices',[])}", flush=True)
        return workers

    execute(jobs)
    pending = []
    for job in jobs:
        n = len(c.load_json(Path(job['directory'])/'prepared.json')['seeds'])
        for variant in settings['variants']:
            for first in range(0, n, settings['chunk_size']):
                pending.append(dict(job, phase='trace', variant=variant, indices=list(range(first,min(first+settings['chunk_size'],n)))))
    workers = execute(pending)
    c.write_json(output/'simulation_manifest.json', dict(status='complete', protocol=settings['protocol'],
        records=records, settings=settings, smoke=smoke, actual_workers=workers, numeric_threads=1,
        paths=sum(len(j['indices']) for j in pending), refits=0, elapsed_seconds=time.monotonic()-start,
        scientific_status='fixed_parameter_intervention_not_refitted',
        source_sha256={str(p.relative_to(REPO)):c.digest(p) for p in (config.resolve(), Path(__file__).resolve(), REPO/TRANSITION)},
        versions={name:importlib.metadata.version(name) for name in ('numpy','scipy','pandas','joblib','PyYAML','numba')}, python=sys.version.split()[0]))
    print(f'Completed simulations: {output}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/shared/model0923_mechanism_interventions.yaml'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--subjects', type=int, nargs='+')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.output:
        run(args.config, args.output, args.smoke, args.subjects)
    else:
        parser.error('--output is required')


if __name__ == '__main__':
    main()

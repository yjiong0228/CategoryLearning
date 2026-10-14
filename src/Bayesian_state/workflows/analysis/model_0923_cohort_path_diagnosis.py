"""Diagnose all configured saved 0923 fits using each fit's frozen cognitive core.

One process layer executes independent subject/chunk jobs. Analysis modules are
loaded from this checkout; cognitive modules, configurations and data resolve
against the archived source. Existing paths must replay; older MEG fits without
paths generate explicitly labelled, fixed-parameter paths. See the cohort note.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[4]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_diagnosis():
    # Extend only the analysis namespace; archived cognitive imports stay intact.
    import src.Bayesian_state.evaluation as evaluation
    evaluation.__path__.insert(0, str(REPO / 'src/Bayesian_state/evaluation'))
    name = 'cohort_diagnosis_observer'
    path = REPO / 'src/Bayesian_state/workflows/analysis/model_0923_path_diagnosis.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def resolve_record(record: dict, cache: Path, max_trials: int | None):
    from src.Bayesian_state.evaluation.model_0923_fit_source import resolve_fit_subject
    if record['layout'] != 'meg_cd':
        return resolve_fit_subject(Path(record['fit']).resolve(), record['subject'], cache, max_trials)
    from src.Bayesian_state.simulation.config import load_yaml, resolve_engine_config
    from src.Bayesian_state.simulation.data import SubjectTrialDataLoader
    from src.Bayesian_state.simulation.parameters import apply_fixed_hyperparams_to_engine_config
    from src.Bayesian_state.utils.datasets import resolve_dataset_paths
    fit = Path(record['fit'])
    path = fit / 'run_configs/simulation.yaml'
    config = load_yaml(path)
    if config.get('max_trials') is not None:
        raise ValueError('MEG source is a truncated fit')
    best = load_json(fit / f"fit/subject_{record['subject']}/best_hyperparams.json")
    point = best['final_rescore_best']['hyperparams']
    bank = [json.loads(s) for s in (fit / f"fit/subject_{record['subject']}/final_rescore.jsonl").read_text().splitlines()]
    selected = [r for r in bank if r['hyperparams'] == point]
    if not selected or selected[0]['aggregated_error'] != best['final_rescore_best']['aggregated_error']:
        raise ValueError('MEG selected point and final rescore differ')
    engine = apply_fixed_hyperparams_to_engine_config(
        resolve_engine_config(config, path.parent, subject_id=record['subject']), point)
    engine['oral_readout'] = None
    engine['partition']['kwargs']['similarity_cache_dir'] = str(cache)
    engine['inference']['particle_blocks'] = 1
    paths = resolve_dataset_paths(config, path.parent)
    loader = SubjectTrialDataLoader(engine, dataset_paths=paths)
    loader.prepare_data(paths['learning_data'])
    frame = loader._get_subject_frame(record['subject'], 1.)
    arrays = loader._extract_arrays(frame, max_trials)
    if 'miss' in frame:
        frame = frame.loc[frame['miss'] != 1]
    frame = frame.iloc[:len(arrays.choices)].copy()
    return dict(subject=record['subject'], condition=loader._get_condition_value(frame),
                engine=engine, arrays=arrays, paths=paths, frame=frame, point=point,
                best=best, fit_status='initial_search_only')


def validate_record(record: dict, d) -> dict:
    fit = Path(record['fit'])
    if record['layout'] != 'meg_cd':
        return d.validate_fit_source(fit)
    status = load_json(fit / 'fit_status.json')
    if status['status'] != 'complete' or status['subjects'][str(record['subject'])]['status'] != 'complete':
        raise ValueError('MEG source fit incomplete')
    snapshot = load_json(fit / 'snapshot_manifest.json')
    source_hashes = snapshot['files']
    for name, expected in source_hashes.items():
        if digest(fit / 'source' / name) != expected:
            raise ValueError(f'MEG frozen input changed: {name}')
    from src.Bayesian_state.evaluation.model_0923_fit_source import _check_versions
    manifest = load_json(fit / 'run_manifest.json')
    _check_versions(manifest['versions'], manifest['versions']['python'])
    return dict(process_status='completed', source_fit_commit=manifest['git_commit'],
                subject_acceptance='initial_search_only', snapshot_checked_files=len(source_hashes),
                snapshot_manifest_sha256=digest(fit / 'snapshot_manifest.json'))


def prepare(record: dict, settings: dict, directory: Path, smoke: bool, d) -> None:
    import numpy as np
    import pandas as pd
    from src.Bayesian_state.evaluation.model_0923_postfit import rule_catalog
    directory.mkdir()
    (directory / 'traces').mkdir()
    cache = directory / 'resources_cache'
    cache.mkdir()
    validation = validate_record(record, d)
    inputs = resolve_record(record, cache, settings['smoke_trials'] if smoke else None)
    catalog = d.geometric_catalog(inputs, rule_catalog(inputs), settings['geometry_beta'])
    catalog.to_csv(directory / 'rule_catalog.csv', index=False)
    observed = pd.DataFrame(dict(valid_trial=np.arange(1, len(inputs['arrays'].choices)+1),
        choice=inputs['arrays'].choices, category=inputs['arrays'].categories, feedback=inputs['arrays'].feedback))
    sources = {}
    fit = Path(record['fit'])
    if record['layout'] == 'meg_cd':
        for name in ('run_configs/simulation.yaml', 'run_configs/hyper_cd.yaml', 'fit_status.json',
                     f"fit/subject_{record['subject']}/best_hyperparams.json",
                     f"fit/subject_{record['subject']}/final_rescore.jsonl"):
            sources[str(fit / name)] = digest(fit / name)
    if record.get('evaluation'):
        evaluation = Path(record['evaluation'])
        manifest_path = evaluation / 'manifest.json'
        manifest = load_json(manifest_path)
        trial_file = evaluation / f"subject_{record['subject']}/trials.csv"
        trials = pd.read_csv(trial_file).iloc[:len(observed)]
        for column in ('choice', 'category', 'feedback'):
            np.testing.assert_array_equal(trials[column], observed[column])
        observed['conditional_correct'] = trials.predicted_correct_probability.to_numpy()
        for name in ('trials.csv', 'autonomous_rollouts.npz'):
            path = trial_file.with_name(name)
            sources[str(path)] = digest(path)
            if 'artifacts_sha256' in manifest:
                key = 'report/' + path.relative_to(evaluation).as_posix()
                if sources[str(path)] != manifest['artifacts_sha256'][key]:
                    raise ValueError(f'Saved evaluation artifact changed: {path}')
        sources[str(manifest_path)] = digest(manifest_path)
        with np.load(trial_file.with_name('autonomous_rollouts.npz')) as saved:
            seeds = saved['seeds'].astype(int).tolist()
    else:
        from src.Bayesian_state.utils.seeding import stable_seed
        seeds = [stable_seed(dict(protocol=settings['protocol'], base=settings['new_meg_seed'],
                    subject=record['subject'], path=i)) for i in range(settings['new_meg_paths'])]
    if smoke:
        seeds = seeds[:1]
    if len(set(seeds)) != len(seeds):
        raise ValueError('Duplicated autonomous seeds')
    observed.to_csv(directory / 'observed_trials.csv', index=False)
    write_json(directory / 'parameters.json', inputs['point'])
    write_json(directory / 'source_validation.json', validation)
    write_json(directory / 'prepared.json', dict(record=record, seeds=seeds,
        trials=len(observed), condition=int(inputs['condition']), fit_status=inputs['fit_status'],
        source_sha256=sources, runtime_root=str(d.ROOT_DIR), smoke=smoke,
        cognitive_source_sha256={str(p.relative_to(d.ROOT_DIR)): digest(p)
            for subtree in ('model', 'hypothesis_space', 'simulation')
            for p in (d.ROOT_DIR / 'src/Bayesian_state' / subtree).rglob('*.py')}))


def trace_chunk(record: dict, settings: dict, directory: Path, indices: list[int], smoke: bool, d) -> None:
    import numpy as np
    prepared = load_json(directory / 'prepared.json')
    inputs = resolve_record(record, directory / 'resources_cache', settings['smoke_trials'] if smoke else None)
    output = directory / 'traces' / f'C_{indices[0]:03d}.npz'
    if record.get('evaluation'):
        source = Path(record['evaluation']) / f"subject_{record['subject']}/autonomous_rollouts.npz"
        if digest(source) != prepared['source_sha256'][str(source)]:
            raise ValueError('Source paths changed after preparation')
        check = d.replay_task(inputs, 'C', source, indices, output)
        check['mode'] = 'saved_path_replay'
    else:
        seeds = [prepared['seeds'][i] for i in indices]
        arrays = inputs['arrays']
        capacity = inputs['engine']['modules']['hypo_transitions_mod']['kwargs']['max_capacity']
        recorder = d.PathRecorder(arrays.categories, len(seeds), capacity)
        kwargs = dict(engine_config=deepcopy(inputs['engine']), subject_id=inputs['subject'],
            condition=inputs['condition'], stimulus=arrays.stimulus, categories=arrays.categories,
            processed_data_dir=inputs['paths']['processed_dir'], dataset_paths=inputs['paths'])
        # A separately executed unwrapped path verifies that recording consumes
        # no random draws and does not change choices, feedback or probabilities.
        reference = d.autonomous.run_autonomous_category_learning(**kwargs, trajectory_seed=seeds[0])
        with d.record_lifecycle(recorder):
            for seed in seeds:
                d.autonomous.run_autonomous_category_learning(**kwargs, trajectory_seed=seed)
        recorder.validate()
        np.testing.assert_allclose(recorder.arrays['prediction'][:, 0], reference.trajectory.observed_probabilities, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(recorder.arrays['choice'][:, 0], reference.trajectory.choices)
        np.testing.assert_array_equal(recorder.arrays['feedback'][:, 0], reference.trajectory.feedback)
        with output.open('xb') as stream:
            np.savez_compressed(stream, **recorder.arrays, seeds=seeds, categories=arrays.categories)
        check = dict(mode='new_fixed_parameter_paths', paths=len(seeds), trials=len(arrays.choices),
                     replay_max_difference=0., independent_wrapper_check_paths=1)
    check['sha256'] = digest(output)
    write_json(output.with_suffix('.json'), check)


def worker(job_path: Path) -> None:
    job = load_json(job_path)
    d = load_diagnosis()
    args = (job['record'], job['settings'], Path(job['directory']))
    with d.single_threaded_processes():
        if job['phase'] == 'prepare':
            prepare(*args, job['smoke'], d)
        else:
            trace_chunk(*args, job['indices'], job['smoke'], d)


def records_from_config(settings: dict) -> list[dict]:
    records = []
    for sid in settings['initial_subjects']:
        fit = settings['initial_fit']
        records.append(dict(id=f'exp123_{sid}_initial', subject=sid, dataset='exp123', layout='initial_cd',
            primary=sid not in settings['plus_subjects'], fit=fit, runtime=fit+'/source',
            evaluation=settings['initial_evaluation']))
    for sid in settings['plus_subjects']:
        root = settings['plus_root']
        records.append(dict(id=f'{"meg" if sid == 335 else "exp123"}_{sid}_plus', subject=sid,
            dataset='meg' if sid == 335 else 'exp123', layout='adaptive', primary=True,
            fit=f'{root}/subject_{sid}/continuation', runtime=root+'/source',
            evaluation=f'{root}/subject_{sid}/evaluation',
            reuse=f"{settings['previous_diagnosis']}/subject_{sid}"))
    for sid in settings['meg_subjects']:
        fit = settings['meg_fit']
        records.append(dict(id=f'meg_{sid}_initial', subject=sid, dataset='meg', layout='meg_cd',
            primary=sid not in settings['plus_subjects'], fit=fit, runtime=fit+'/source'))
    identities = [(r['dataset'], r['subject']) for r in records if r['primary']]
    if len(set(identities)) != len(identities):
        raise ValueError('A participant has multiple primary fits')
    return records


def launch(job: dict, output: Path) -> None:
    job_file = output / 'jobs' / (job['record']['id']+'_'+job['phase']+'_'+str(job.get('indices', [0])[0])+'.json')
    write_json(job_file, job)
    launcher = ('import runpy,sys; source,script=sys.argv[1:3]; '
                'sys.path.insert(0,source); sys.argv=[script]+sys.argv[3:]; '
                "runpy.run_path(script,run_name='__main__')")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    command = [sys.executable, '-c', launcher, str(Path(job['record']['runtime']).resolve()),
               str(Path(__file__).resolve()), '--worker', str(job_file)]
    with tempfile.TemporaryDirectory(prefix='model0923_cohort_numba_') as cache:
        env['NUMBA_CACHE_DIR'] = cache
        with job_file.with_suffix('.log').open('x') as stream:
            result = subprocess.run(command, env=env, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Job failed: {job_file.with_suffix('.log')}")


def run(config: Path, output: Path, smoke: bool = False, only: list[int] | None = None) -> None:
    import numpy as np
    import pandas as pd
    import yaml
    from src.Bayesian_state.utils.parallel import parallel_job_count
    settings = yaml.safe_load(config.read_text())
    for key in ('new_meg_paths', 'chunk_size', 'parallel_budget', 'late_window', 'retention_trials', 'smoke_trials'):
        if settings[key] < 1:
            raise ValueError(f'{key} must be positive')
    if output.exists():
        raise FileExistsError('Choose a new output directory')
    output = output.resolve()
    output.mkdir(parents=True)
    (output / 'jobs').mkdir()
    (output / 'effective_config.yaml').write_text(yaml.safe_dump(settings, sort_keys=False))
    records = records_from_config(settings)
    if only:
        records = [r for r in records if r['subject'] in only]
    start = time.monotonic()
    jobs = [dict(record=r, settings=settings, directory=str(output / r['id']), smoke=smoke, phase='prepare') for r in records]
    def execute(pending):
        workers = 1 if smoke else parallel_job_count(settings['parallel_budget'], len(pending))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(launch, job, output): job for job in pending}
            for i, future in enumerate(as_completed(futures), 1):
                future.result()
                print(f"{i}/{len(pending)} {futures[future]['record']['id']} {futures[future]['phase']}", flush=True)
        return workers
    execute(jobs)
    pending = []
    for job in jobs:
        r, directory = job['record'], Path(job['directory'])
        prepared = load_json(directory / 'prepared.json')
        if r.get('reuse') and not smoke:
            old = Path(r['reuse'])
            manifest = load_json(old.parent / 'manifest.json')
            observed = pd.read_csv(directory / 'observed_trials.csv')
            seen = []
            for path in sorted((old / 'traces').glob('C_*.npz')):
                expected = manifest['artifact_sha256'][str(path.relative_to(old.parent))]
                if digest(path) != expected:
                    raise ValueError(f'Existing trace changed: {path}')
                with np.load(path) as a:
                    seen.extend(a['seeds'].astype(int).tolist())
                    np.testing.assert_array_equal(a['categories'], observed.category)
                # Immutable traces are referenced rather than copied or overwritten.
                (directory / 'traces' / path.name).symlink_to(path.resolve())
                write_json(directory / 'traces' / path.with_suffix('.json').name,
                           dict(mode='reuse_validated_trace', sha256=expected))
            if seen != prepared['seeds']:
                raise ValueError('Previously diagnosed seeds differ')
        else:
            size = settings['chunk_size']
            for first in range(0, len(prepared['seeds']), size):
                pending.append(dict(job, phase='trace', indices=list(range(first, min(first+size, len(prepared['seeds']))))))
    workers = execute(pending)
    from src.Bayesian_state.evaluation.model_0923_cohort_paths import summarize_cohort
    summarize_cohort(output, records, settings)
    inventory = [dict(id=r['id'], subject=r['subject'], fit=r['fit'], included=True,
        role='primary' if r['primary'] else 'version_sensitivity',
        reason='complete_fixed_parameter_fit', trajectories='new_fixed_parameter' if not r.get('evaluation') else
        'reuse_validated_traces' if r.get('reuse') and not smoke else 'saved_path_replay') for r in records]
    # Audit the fit manifests outside immutable source snapshots and numeric caches.
    selected_fits = {Path(r['fit']).resolve() for r in records}
    for folder, dirs, files in os.walk(REPO / 'results/model_0923'):
        dirs[:] = [s for s in dirs if s not in ('source', 'cache', 'resources_cache', 'batches', 'hyper_cd', 'seeds', 'fit', 'jobs')]
        path = Path(folder)
        candidates = [name for name in ('run_manifest.json', 'manifest.json') if name in files]
        for name in candidates:
            manifest = load_json(path / name)
            if not isinstance(manifest, dict):
                continue
            is_fit = name == 'run_manifest.json' or manifest.get('config', {}).get('backend') == 'model0923_adaptive'
            if not is_fit or path.resolve() in selected_fits:
                continue
            reason = 'smoke_or_validation' if any(s in str(path) for s in ('smoke', 'preparation', 'validation')) else 'superseded_or_incomplete_fit'
            inventory.append(dict(id=str(path.relative_to(REPO)), fit=str(path.relative_to(REPO)), included=False,
                                  role='excluded', reason=reason))
    pd.DataFrame(inventory).to_csv(output / 'inventory.csv', index=False)
    artifacts = {str(p.relative_to(output)): digest(p) for p in output.rglob('*') if p.is_file() and p.suffix not in ('.log',)}
    sources = {str(config): digest(config), str(Path(__file__).relative_to(REPO)): digest(Path(__file__))}
    for name in ('workflows/analysis/model_0923_path_diagnosis.py', 'evaluation/model_0923_cohort_paths.py', 'evaluation/model_0923_fit_source.py'):
        p = REPO / 'src/Bayesian_state' / name
        sources[str(p.relative_to(REPO))] = digest(p)
    write_json(output / 'manifest.json', dict(status='complete', protocol=settings['protocol'], smoke=smoke,
        records=records, primary_participants=sum(r['primary'] for r in records), actual_workers=workers,
        refits=0, source_sha256=sources, artifact_sha256=artifacts, elapsed_seconds=time.monotonic()-start,
        new_meg_seed=settings['new_meg_seed'],
        versions={name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'pandas', 'joblib', 'PyYAML', 'numba')},
        python=sys.version.split()[0]))
    print(f'Completed: {output}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/shared/model0923_cohort_path_diagnosis.yaml'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--subjects', type=int, nargs='+')
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

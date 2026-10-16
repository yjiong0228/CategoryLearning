"""Generate fixed-schedule continuations from filtered training states.

--fit supplies frozen parameters, --reference their conditional predictions,
--config the filter/branch budget, and --output a new destination. Scientific
runs must reproduce every saved training-prefix prediction before generating
future responses. Workers decode future stimuli and categories only. Several
branches share one filter; their filter-seed grouping is retained in outputs.

--smoke uses one participant/model, two four-particle filters and two branches
per filter, with eight consecutive prefix trials and eight continuation trials.
It checks prefix predictions against the unmodified original filter execution.
Optional particle_overrides maps subject IDs to counts when the conditional
reference used different numerical budgets. Each count must match its reference.
Participant data roles follow the frozen fit and are carried into the forecast
protocol; the conditional reference supplies the fallback group description.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[4]
BASE = 'src/Bayesian_state/workflows/analysis/model_0923_h_validation.py'
HELPER = 'src/Bayesian_state/simulation/filtered_continuation.py'


def load_source(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, REPO / file)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validation():
    return load_source('prefix_forecast_validation', BASE)


def configured_particles(settings: dict, subject: int, smoke: bool = False) -> int:
    value = 4 if smoke else settings.get('particle_overrides', {}).get(subject, settings['particles'])
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValueError('Prefix particle counts must be integers >= 2')
    return value


def data_use_roles(frozen: dict, subjects: list[int], reference: dict) -> tuple[str,dict]:
    if frozen['status']=='frozen_for_development_evaluation':
        return 'development',{subject:'development' for subject in subjects}
    settings=frozen['settings']
    overall=frozen.get('data_use_status',settings.get('data_use_status',reference.get('data_use_status','development')))
    roles={}
    for subject in subjects:
        if subject in settings.get('new_validation_subjects',[]):roles[subject]='new_validation'
        elif subject in settings.get('development_subjects',[]):roles[subject]='development'
        else:roles[subject]=overall
    return overall,roles


def worker(file: Path) -> None:
    import numpy as np
    from threadpoolctl import threadpool_info
    v = validation(); job = v.read(file)
    for name, sha in job['source_sha256'].items():
        if v.digest(REPO / name) != sha:
            raise ValueError('Forecast implementation changed')
    w = v.intervention_workflow(); w.install_transition()
    if 'candidate_sampler' in job['mechanism']:
        adapter = load_source('prefix_candidate_adapter', 'src/Bayesian_state/workflows/analysis/model_0923_h_candidate_validation.py')
        adapter.fixed_models(dict(models={job['model']: job['mechanism']}, fixed_size_models=[job['model']]))
        sampler = w.load_source('src.Bayesian_state.model.modules.hypothesis_transition.single_proposal_competition', adapter.HELPER)
        sampler.install_worker_intervention(sys.modules['src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search'])
        job = adapter.core_job(dict(job, candidate_sampler=job['mechanism']['candidate_sampler']))
    helper = load_source('src.Bayesian_state.simulation.filtered_continuation', HELPER)
    from src.Bayesian_state.simulation.parameters import apply_fixed_hyperparams_to_engine_config
    from src.Bayesian_state.inference.backends.particle_filter import run_state_model_particle_filter
    from src.Bayesian_state.utils.seeding import stable_seed
    folder = Path(job['folder']); meta = v.read(folder / 'prepared.json')
    for name, sha in meta['frozen_cognitive_source_sha256'].items():
        if v.digest(Path(job['runtime']) / name) != sha:
            raise ValueError('Archived cognitive implementation changed')
    if v.digest(folder / 'engine.json') != meta['engine_sha256']:
        raise ValueError('Prepared engine changed')
    if v.digest(folder / 'training.npz') != meta['array_sha256']['training.npz']:
        raise ValueError('Training observations changed')
    engine = v.read(folder / 'engine.json')
    engine['modules']['hypo_transitions_mod']['kwargs']['intervention'] = job['mechanism']
    engine = apply_fixed_hyperparams_to_engine_config(engine, job['candidate']['hyperparams'])
    with np.load(folder / 'training.npz') as saved:
        training = {key: saved[key] for key in saved.files}
    if job['smoke']:
        future = {key: training[key][8:16] for key in ('stimulus', 'categories')}
        training = {key: value[:8] for key, value in training.items()}
    else:
        if v.digest(folder / 'sealed_test.npz') != meta['array_sha256']['sealed_test.npz']:
            raise ValueError('Future task schedule changed')
        with np.load(folder / 'sealed_test.npz') as saved:
            future = {key: saved[key] for key in ('stimulus', 'categories')}
    kwargs = dict(engine_config=engine, subject_id=meta['subject'], condition=meta['condition'],
        stimulus=training['stimulus'], choices=training['choices'], feedback=training['feedback'],
        particle_count=job['particles'], particle_blocks=1, filter_seed=job['filter_seed'],
        choice_readout_power=1., processed_data_dir=meta['paths']['processed_dir'], dataset_paths=meta['paths'])
    start, cpu = time.monotonic(), time.process_time()
    diagnosis = w.cohort().load_diagnosis()
    with diagnosis.single_threaded_processes():
        models, prefix = helper.capture_prefix(**kwargs)
        if job['smoke']:
            reference = run_state_model_particle_filter(**kwargs)
            expected = reference.observation_probabilities['prior_t']
            np.testing.assert_array_equal(prefix.final_weights, reference.final_weights)
        else:
            reference = Path(job['reference'])
            if v.digest(reference) != job['reference_sha256']:
                raise ValueError('Saved conditional predictions changed')
            with np.load(reference) as saved:
                index = int(np.flatnonzero(saved['seeds'] == job['filter_seed'])[0])
                expected = saved['probabilities'][index, :len(training['choices'])]
        np.testing.assert_array_equal(prefix.observation_probabilities['prior_t'], expected)
        seeds = [stable_seed(dict(protocol=job['settings']['protocol'], base=job['settings']['seed'],
            subject=meta['subject'], filter_seed=job['filter_seed'], branch=i, role='prefix_forecast_branch'))
            for i in range(job['settings']['branches_per_filter'])]
        arrays = helper.generate_continuations(models=models, weights=prefix.final_weights,
            observed_choices=training['choices'], observed_feedback=training['feedback'],
            stimulus=future['stimulus'], categories=future['categories'], branch_seeds=seeds, condition=meta['condition'])
    arrays['prefix_observed_probability'] = expected[np.arange(len(training['choices'])), training['choices'] - 1]
    arrays['filter_seed'] = np.asarray(job['filter_seed'])
    target = Path(job['target']); target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix('.npz').open('xb') as handle:
        np.savez_compressed(handle, **arrays)
    v.write(target.with_suffix('.json'), dict(subject=meta['subject'], condition=meta['condition'], model=job['model'],
        candidate=job['candidate'], filter_seed=job['filter_seed'], particles=job['particles'], paths=len(seeds),
        training_trials=len(training['choices']), forecast_trials=len(future['categories']), prefix_exact=True,
        reference_type='separate_filter_execution' if job['smoke'] else 'saved_conditional_prefix',
        seconds=time.monotonic()-start, cpu_seconds=time.process_time()-cpu, numeric_pools=threadpool_info(),
        predictions_sha256=v.digest(target.with_suffix('.npz'))))


def combine(v, jobs: list[dict], target: Path) -> dict:
    """Retain branch-to-filter membership when concatenating forecast paths."""
    import numpy as np
    arrays, records = [], []
    for job in jobs:
        file = Path(job['target']); record = v.read(file.with_suffix('.json'))
        if v.digest(file.with_suffix('.npz')) != record['predictions_sha256']:
            raise ValueError('Forecast chunk changed')
        with np.load(file.with_suffix('.npz')) as saved:
            arrays.append({key: saved[key] for key in saved.files})
        records.append(record)
    for other in arrays[1:]:
        np.testing.assert_array_equal(other['categories'], arrays[0]['categories'])
    result = {}
    for key in arrays[0]:
        if key == 'categories':
            result[key] = arrays[0][key]
        elif key in ('terminal_weights', 'prefix_observed_probability', 'filter_seed'):
            result[key] = np.stack([a[key] for a in arrays])
        else:
            result[key] = np.concatenate([a[key] for a in arrays], axis=0)
    result['branch_filter_seed'] = np.concatenate([np.repeat(a['filter_seed'], len(a['branch_seeds'])) for a in arrays])
    np.testing.assert_array_equal(result['filter_seed'], [job['filter_seed'] for job in jobs])
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix('.npz').open('xb') as handle:
        np.savez_compressed(handle, **result)
    row = dict(records[0], filters=len(jobs), paths=len(result['branch_seeds']),
        filter_seeds=[job['filter_seed'] for job in jobs],
        cpu_seconds=sum(r['cpu_seconds'] for r in records), seconds=sum(r['seconds'] for r in records),
        predictions_sha256=v.digest(target.with_suffix('.npz')), target=str(target.resolve()))
    v.write(target.with_suffix('.json'), row)
    return row


def run(fit: Path, reference: Path, config: Path, output: Path, smoke: bool) -> None:
    import numpy as np
    import yaml
    from src.Bayesian_state.utils.parallel import parallel_job_count
    v = validation(); frozen = v.read(fit / 'nominees.json'); settings = yaml.safe_load(config.read_text())
    if output.exists():raise FileExistsError('Use a new prefix-forecast destination')
    if frozen['smoke'] or frozen['status'] not in ('frozen_before_test', 'frozen_for_development_evaluation'):
        raise ValueError('Need frozen scientific parameters')
    for name, sha in frozen['source_sha256'].items():
        if v.digest(REPO / name) != sha:raise ValueError('Fitted scientific source changed')
    reference_protocol=v.read(reference / 'evaluation_protocol.json')
    if reference_protocol['nomination_sha256'] != v.digest(fit / 'nominees.json'):
        raise ValueError('Conditional reference uses another frozen fit')
    if v.read(reference / 'evaluation.json')['status'] != 'complete':
        raise ValueError('Need complete conditional references')
    if smoke:settings.update(particles=4, filters=2, branches_per_filter=2, parallel_budget=1)
    selected = frozen['nominees'][:1] if smoke else frozen['nominees']
    indexed = {(row['subject'], row['model']): row for row in v.read(reference / 'evaluation.json')['rows']}
    banks = {(row['subject'], row['model']): [row['candidate']] for row in selected}
    bases = v.score_jobs(Path(frozen['data']), output, frozen['settings'], banks, 'evaluate', False)
    hashes = dict(frozen['source_sha256'])
    for name in (HELPER, str(Path(__file__).resolve().relative_to(REPO))):hashes[name] = v.digest(REPO / name)
    output.mkdir(parents=True); (output / 'jobs').mkdir()
    jobs, groups = [], {}
    for base in bases:
        subject = v.read(Path(base['folder']) / 'prepared.json')['subject']; key = subject, base['model']
        old = indexed[key]
        if old['candidate']['hyperparams'] != base['candidate']['hyperparams']:
            raise ValueError('Reference parameters differ')
        particles = configured_particles(settings, subject, smoke)
        if not smoke and old['particles'] != particles:
            raise ValueError('Exact prefix replay requires the reference particle count')
        file = reference / 'evaluate' / f'subject_{subject}' / base['model'] / (base['candidate']['id'] + '.npz')
        if v.digest(file) != old['predictions_sha256']:raise ValueError('Conditional reference changed')
        with np.load(file) as saved:seeds = saved['seeds'][:settings['filters']].tolist()
        if len(seeds) != settings['filters']:raise ValueError('Not enough reference filter seeds')
        groups[key] = []
        for index, seed in enumerate(seeds):
            name = f'S{subject}_{base["model"]}_F{index:03d}'
            job = dict(base, name=name, smoke=smoke, settings=settings, particles=particles,
                filter_seed=seed, source_sha256=hashes, reference=str(file.resolve()), reference_sha256=old['predictions_sha256'],
                target=str((output / 'chunks' / name).resolve()))
            groups[key].append(job); jobs.append(job)
    data_role,participant_roles=data_use_roles(frozen,sorted({key[0] for key in groups}),reference_protocol)
    v.write(output / 'forecast_protocol.json', dict(fit=str(fit.resolve()), reference=str(reference.resolve()),
        nomination_sha256=v.digest(fit / 'nominees.json'), settings=settings, smoke=smoke, source_sha256=hashes,
        data_use_status=data_role, participant_roles=participant_roles,
        conditioning='observed training prefix; subsequent recorded stimulus schedule',
        future_behavior='model choices and task-produced feedback only', uncertainty_unit='complete filter seed with all its branches'))
    lengths = {j['folder']: v.read(Path(j['folder']) / 'prepared.json')['training_trials'] for j in jobs}
    jobs.sort(key=lambda j: -lengths[j['folder']] * j['particles'])
    workers = parallel_job_count(settings['parallel_budget'], len(jobs))

    def launch(job):
        file = output / 'jobs' / (job['name'] + '.json'); v.write(file, job)
        launcher = 'import runpy,sys;source,script=sys.argv[1:3];sys.path.insert(0,source);sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name="__main__")'
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        with tempfile.TemporaryDirectory(prefix='h_prefix_forecast_') as cache, file.with_suffix('.log').open('x') as log:
            env['NUMBA_CACHE_DIR'] = cache
            subprocess.run([sys.executable, '-c', launcher, job['runtime'], str(Path(__file__).resolve()),
                '--worker', str(file.resolve())], env=env, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)

    print(f'Running {len(jobs)} filters with {workers} workers; branches remain serial within each filter', flush=True)
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, future in enumerate(as_completed([pool.submit(launch, job) for job in jobs]), 1):
            future.result()
            if index % 16 == 0 or index == len(jobs):print(f'{index}/{len(jobs)} filters complete', flush=True)
    rows = [combine(v, group, output / 'forecast' / f'subject_{key[0]}' / key[1] / group[0]['candidate']['id'])
            for key, group in groups.items()]
    v.write(output / 'forecast.json', dict(status='complete', smoke=smoke, rows=rows,
        seconds=time.monotonic()-start, filters=len(jobs), paths=sum(row['paths'] for row in rows)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fit', 'reference', 'config', 'output', 'worker'):parser.add_argument('--' + name, type=Path)
    parser.add_argument('--smoke', action='store_true'); args = parser.parse_args()
    if args.worker:worker(args.worker); return
    if not all([args.fit, args.reference, args.config, args.output]):parser.error('fit, reference, config and output are required')
    run(args.fit, args.reference, args.config, args.output, args.smoke)


if __name__ == '__main__':main()

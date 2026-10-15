"""Compare predictive likelihood and autonomous-distribution parameter fitting.

Both arms use the same ECM_S mechanism, support, initial points and number of
particle/path state updates at every fitting stage. Their adaptive searches
follow their own training objective. A common union of finalists is scored by
both objectives before independent confirmation freezes the estimates.

Generation fitting opens training.npz only; responses and feedback passed to
the model are generated internally. External stimuli are replayed. Full task
adaptation is evaluated separately. --smoke is an eight-trial software check.
All outputs are new and every worker uses the archived cognitive runtime.
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
import time

REPO = Path(__file__).resolve().parents[4]
BASE = 'src/Bayesian_state/workflows/analysis/model_0923_h_validation.py'
METRICS = 'src/Bayesian_state/metrics/autonomous_fit.py'
PROPOSALS = 'src/Bayesian_state/optimization/search/adaptive_proposals.py'


def source_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validation():
    return source_module('objective_validation', BASE)


def hashes(v) -> dict:
    result = v.source_hashes()
    for relative in (METRICS, PROPOSALS, str(Path(__file__).resolve().relative_to(REPO))):
        result[relative] = v.digest(REPO / relative)
    return result


def validate_settings(settings: dict) -> None:
    if set(settings['models']) != {'NLL', 'GEN'} or settings['models']['NLL'] != settings['models']['GEN']:
        raise ValueError('Two objective arms must use exactly the same mechanism')
    if settings['objectives'] != dict(NLL='predictive_nll', GEN='autonomous_energy_u'):
        raise ValueError('Unknown objective assignment')
    if settings['models']['NLL'] != dict(kind='E', decay=.95, capacity='evidence_mean', search='accumulate'):
        raise ValueError('This comparison freezes the ECM_S structure')
    for stage in ('screen', 'refine', 'selection', 'confirmation'):
        b = settings['budgets'][stage]
        if b['particles']*b['seeds'] != b['paths'] or b['seeds'] < 2 or b['paths'] < 3:
            raise ValueError('Particle/path state-update budgets must match')
    f = settings['features']
    if f['block_width'] < 1 or not 0 < f['temporal_weight'] < 1:
        raise ValueError('Invalid frozen feature specification')


def generation_score(job: dict, v) -> None:
    """Generate and score training paths without opening future data."""
    import numpy as np
    from threadpoolctl import threadpool_info
    from src.Bayesian_state.simulation.autonomous import run_autonomous_category_learning
    from src.Bayesian_state.simulation.parameters import apply_fixed_hyperparams_to_engine_config
    metrics = source_module('objective_metrics', METRICS)
    folder = Path(job['folder']); meta = v.read(folder/'prepared.json')
    if v.digest(folder/'engine.json') != meta['engine_sha256']:
        raise ValueError('Engine changed')
    for name, sha in meta.get('frozen_cognitive_source_sha256', {}).items():
        if v.digest(Path(job['runtime'])/name) != sha:
            raise ValueError('Archived cognitive runtime changed')
    if v.digest(folder/'training.npz') != meta['array_sha256']['training.npz']:
        raise ValueError('Training arrays changed')
    with np.load(folder/'training.npz') as saved:
        training = {key: saved[key] for key in ('stimulus', 'categories', 'choices')}
    if job['smoke']:
        training = {key: value[:8] for key, value in training.items()}
    engine = v.read(folder/'engine.json')
    engine['modules']['hypo_transitions_mod']['kwargs']['intervention'] = job['mechanism']
    engine = apply_fixed_hyperparams_to_engine_config(engine, job['candidate']['hyperparams'])
    category_count = 2 if meta['condition'] == 1 else 4
    feature_kwargs = dict(n_categories=category_count, **job['features'])
    observed = metrics.behavior_features(training['choices'][None, :], training['categories'], **feature_kwargs)[0]
    generated = []; start, cpu = time.monotonic(), time.process_time()
    for seed in job['seeds']:
        result = run_autonomous_category_learning(engine_config=engine, subject_id=meta['subject'],
            condition=meta['condition'], stimulus=training['stimulus'], categories=training['categories'],
            trajectory_seed=seed, processed_data_dir=meta['paths']['processed_dir'], dataset_paths=meta['paths'])
        generated.append(np.asarray(result.trajectory.choices))
    choices = np.asarray(generated)
    features = metrics.behavior_features(choices, training['categories'], **feature_kwargs)
    score = metrics.energy_u(features, observed)
    target = Path(job['target']); target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix('.npz').open('xb') as handle:
        np.savez_compressed(handle, features=features, observed_features=observed,
                            seeds=np.asarray(job['seeds']), choices=choices, categories=training['categories'])
    v.write(target.with_suffix('.json'), dict(subject=meta['subject'], case_id=job['case_id'],
        condition=meta['condition'], model=job['model'], candidate=job['candidate'],
        objective='autonomous_energy_u', objective_score=score['value'], objective_mc_se=score['mc_se'],
        paths=len(choices), trials=len(training['categories']), state_updates=int(choices.size),
        seconds=time.monotonic()-start, cpu_seconds=time.process_time()-cpu,
        numeric_pools=threadpool_info(), predictions_sha256=v.digest(target.with_suffix('.npz')),
        data_boundary='training_only', task_environment='replayed_external_stimuli_with_own_choices_and_feedback'))


def worker(file: Path) -> None:
    v = validation(); job = v.read(file)
    for name, sha in job['source_sha256'].items():
        if v.digest(REPO/name) != sha:
            raise ValueError('Frozen comparison source changed')
    w = v.intervention_workflow(); w.install_transition()
    diagnosis = w.cohort().load_diagnosis()
    with diagnosis.single_threaded_processes():
        if job['objective'] == 'autonomous_energy_u':
            generation_score(job, v)
        elif job['objective'] == 'predictive_nll':
            v.score(job, w)
        else:
            raise ValueError('Unknown fitting objective')


def execute(jobs: list[dict], output: Path, budget: int) -> None:
    v = validation()
    from src.Bayesian_state.utils.parallel import parallel_job_count
    workers = parallel_job_count(budget, len(jobs))
    directory = output/'jobs'; directory.mkdir(exist_ok=True)

    def launch(job):
        file = directory/(job['name']+'.json'); v.write(file, job)
        launcher = ('import runpy,sys;source,script=sys.argv[1:3];sys.path.insert(0,source);'
                    'sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name="__main__")')
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        with tempfile.TemporaryDirectory(prefix='h_objective_') as cache, file.with_suffix('.log').open('x') as log:
            env['NUMBA_CACHE_DIR'] = cache
            result = subprocess.run([sys.executable, '-c', launcher, job['runtime'], str(Path(__file__).resolve()),
                '--worker', str(file.resolve())], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'Objective worker failed: {file.with_suffix(".log")}')

    print(f'Running {len(jobs)} objective jobs with {workers} workers; numeric threads=1', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(launch, job) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            if i % 16 == 0 or i == len(jobs):
                print(f'{i}/{len(jobs)} complete', flush=True)


def training_jobs(v, data: Path, output: Path, settings: dict, banks: dict,
                  stage: str, smoke: bool) -> list[dict]:
    from src.Bayesian_state.utils.seeding import stable_seed
    jobs = v.score_jobs(data, output, settings, banks, stage, smoke)
    for job in jobs:
        case = next(s for s, m in banks if str((data/f'subject_{s}').resolve()) == job['folder'] and m == job['model'])
        job.update(case_id=case, objective=settings['objectives'][job['model']],
                   features=settings['features'], source_sha256=hashes(v))
        if job['model'] == 'GEN':
            job['seeds'] = [stable_seed(dict(protocol=settings['protocol'], base=settings['seed'],
                role='objective_training_generation', stage=stage, subject=case, path=i))
                for i in range(4 if smoke else settings['budgets'][stage]['paths'])]
    return jobs


def ranked_rows(v, output: Path, stage: str, subject: int, arm: str) -> list[dict]:
    rows = []
    for file in sorted((output/stage/f'subject_{subject}'/arm).glob('*.json')):
        record = v.read(file)
        score = record['mean_nll'] if arm == 'NLL' else record['objective_score']
        rows.append(dict(record['candidate'], objective_score=score, score_file=str(file.resolve())))
    return sorted(rows, key=lambda x: (x['objective_score'], x['id']))


def diverse_elites(rows: list[dict], count: int) -> list[dict]:
    from src.Bayesian_state.optimization.search.adaptive_proposals import select_elites
    v = validation()
    # Reuse the established diversity selection; this key is internal only.
    ranked = [dict(row, mean_nll=row['objective_score']) for row in rows]
    result = select_elites(ranked, count, exclude_key=v.H_PATH+'max_capacity')
    return [{k: value for k, value in row.items() if k != 'mean_nll'} for row in result]


def shared_union(rows: dict, subjects: list[int], per_arm: int) -> dict:
    banks = {}
    for subject in subjects:
        union = {row['id']: row for arm in ('NLL', 'GEN') for row in rows[(subject, arm)][:per_arm]}
        points = [{k: row[k] for k in ('id', 'hyperparams', 'sources', 'origin')} for row in union.values()]
        for arm in ('NLL', 'GEN'):
            banks[(subject, arm)] = deepcopy(points)
    return banks


def search(config: Path, data: Path, output: Path, smoke: bool) -> None:
    import yaml
    from src.Bayesian_state.optimization.search.adaptive_proposals import initial_points, candidate, propose_round
    from src.Bayesian_state.utils.seeding import stable_seed
    v = validation(); settings = yaml.safe_load(config.read_text()); validate_settings(settings)
    if output.exists():
        raise FileExistsError('Use a new objective-comparison output')
    prep = v.read(data/'preparation_manifest.json')
    subjects = [s for s in prep['subjects'] if (data/f'subject_{s}'/'prepared.json').exists()]
    if smoke:
        subjects = subjects[:1]
    if not subjects:
        raise ValueError('No eligible prepared participants')
    output.mkdir(parents=True)
    v.write(output/'search_protocol.json', dict(settings=settings, data=str(data.resolve()), smoke=smoke,
        source_sha256=hashes(v), data_manifest_sha256=v.digest(data/'preparation_manifest.json'),
        subjects=subjects, budget_definition='equal particle/path state updates; actual CPU time reported separately'))
    banks, spaces, seen = {}, {}, {}
    for subject in subjects:
        space, anchor = v.model_space(settings, 'NLL')
        points = initial_points(space, anchor, 5 if smoke else settings['initial_candidates'],
            stable_seed(dict(base=settings['seed'], subject=subject, role='initial')), cover_key=v.H_PATH+'max_capacity')
        for arm in ('NLL', 'GEN'):
            key = subject, arm
            banks[key] = [candidate(point, 'objective_initial') for point in points]
            spaces[key] = space; seen[key] = {row['id'] for row in banks[key]}
    execute(training_jobs(v, data, output, settings, banks, 'screen', smoke), output,
            1 if smoke else settings['parallel_budget'])
    rows = {key: ranked_rows(v, output, 'screen', *key) for key in banks}
    for index in range(1 if smoke else settings['refinement_rounds']):
        stage = f'refine{index+1}'; settings['budgets'][stage] = settings['budgets']['refine']
        for key in banks:
            elites = diverse_elites(rows[key], 1 if smoke else settings['elites'])
            count = 1 if smoke else settings['proposals_per_elite']
            proposed = propose_round(elites, spaces[key], seen[key], count,
                stable_seed(dict(base=settings['seed'], subject=key[0], role=stage)), .5, .25)
            if len(proposed) != len(elites)*count:
                raise ValueError('Unequal proposal budget')
            banks[key] = proposed+elites; seen[key].update(row['id'] for row in proposed)
        execute(training_jobs(v, data, output, settings, banks, stage, smoke), output,
                1 if smoke else settings['parallel_budget'])
        rows = {key: ranked_rows(v, output, stage, *key) for key in banks}
    for stage, count in [('selection', settings['union_per_arm']), ('confirmation', settings['confirmation_per_arm'])]:
        banks = shared_union(rows, subjects, 1 if smoke else count)
        execute(training_jobs(v, data, output, settings, banks, stage, smoke), output,
                1 if smoke else settings['parallel_budget'])
        rows = {key: ranked_rows(v, output, stage, *key) for key in banks}
    nominees = [dict(subject=subject, model=arm, candidate=rows[(subject, arm)][0],
        finalists=rows[(subject, arm)], scientific_status='bounded_training_objective_comparison',
        unique_search_candidates=len(seen[(subject, arm)]), objective=settings['objectives'][arm])
        for subject in subjects for arm in ('NLL', 'GEN')]
    v.write(output/'nominees.json', dict(status='frozen_before_test', settings=settings, smoke=smoke,
        data=str(data.resolve()), source_sha256=hashes(v), nominees=nominees))
    v.write(output/'completion.json', dict(status='complete', participants=len(subjects), arms=2,
        test_choices_used=False, fitted_coordinates=len(spaces[(subjects[0], 'NLL')]), smoke=smoke))
    print('Both objectives frozen; training jobs never opened test arrays', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'data', 'output', 'worker'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif all((args.config, args.data, args.output)):
        search(args.config, args.data, args.output, args.smoke)
    else:
        parser.error('Supply --config, --data and --output')


if __name__ == '__main__':
    main()

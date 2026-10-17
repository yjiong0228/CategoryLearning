"""Compare frozen H fits under fixed, balanced, and response-adaptive tasks.

Every run uses a new destination and one outer process layer. The archived
cognitive runtime remains frozen; only the current H and task adapters are
loaded explicitly. Fixed-schedule paths must exactly replay saved predictions
up to their first passing block. All paths, including censored ones, are saved.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, contextmanager
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[4]
TASK_FILES = ['src/Bayesian_state/simulation/exp123_adaptive_task.py',
              'src/Bayesian_state/simulation/adaptive_autonomous.py']


def load_source(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validation():
    return load_source('adaptive_task_validation', REPO / 'src/Bayesian_state/workflows/analysis/model_0923_h_validation.py')


@contextmanager
def cognitive_adapters(job: dict):
    """Restore the frozen episodic-refit interventions when explicitly supplied.

    The ordinary H-study jobs retain their original implementation. Episodic
    jobs carry the precision settings and admission identity from their saved
    autonomous reference; task generation changes neither of these settings.
    """
    settings = job.get('episodic_arm_settings')
    corrections = {}
    with ExitStack() as stack:
        if settings is not None:
            if job['model'] not in ('BASE', 'NEW', 'POOL', 'DRAW'):
                raise ValueError('Unsupported episodic task arm')
            prefix = 'src/Bayesian_state/model/modules/'
            feedback = load_source('src.Bayesian_state.model.modules.hypothesis_transition.feedback_guided',
                                   REPO / (prefix + 'hypothesis_transition/feedback_guided.py'))
            precision = load_source('src.Bayesian_state.model.modules.precision_interventions',
                                    REPO / (prefix + 'precision_interventions.py'))
            guard = load_source('src.Bayesian_state.model.modules.pairing_snapshot_roundoff',
                                REPO / (prefix + 'pairing_snapshot_roundoff.py'))
            history = load_source('src.Bayesian_state.model.modules.hypothesis_transition.episodic_admission',
                                  REPO / (prefix + 'hypothesis_transition/episodic_admission.py'))
            from src.Bayesian_state.model.modules.beta import BetaModule
            from src.Bayesian_state.model.modules.pairing_memory import HierarchicalPairingMemoryModule
            cls = sys.modules['src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search'].MechanisticRuleSearchHypothesisTransitionModule
            corrections['feedback'] = stack.enter_context(feedback.retention_roundoff_guard(cls))
            corrections['snapshot'] = stack.enter_context(guard.pairing_snapshot_roundoff_guard(HierarchicalPairingMemoryModule))
            stack.enter_context(precision.precision_intervention(BetaModule, mode=settings['precision'], upper=settings['upper']))
            if job['model'] != 'BASE':
                corrections['recall'] = stack.enter_context(history.episodic_admission(cls, mode=job['model']))
        yield corrections


def worker(file: Path) -> None:
    import numpy as np
    from threadpoolctl import threadpool_info
    v = validation(); job = v.read(file)
    workflow = v.intervention_workflow(); diagnosis = workflow.cohort().load_diagnosis()
    workflow.install_transition()
    for name, sha in job['source_sha256'].items():
        if v.digest(REPO / name) != sha:
            raise ValueError('Executing source changed')
    if 'candidate_sampler' in job['mechanism']:
        adapter = load_source('adaptive_candidate_validation', REPO / 'src/Bayesian_state/workflows/analysis/model_0923_h_candidate_validation.py')
        adapter.fixed_models(dict(models={job['model']:job['mechanism']}, fixed_size_models=[job['model']]))
        sampler = workflow.load_source('src.Bayesian_state.model.modules.hypothesis_transition.single_proposal_competition', adapter.HELPER)
        sampler.install_worker_intervention(sys.modules['src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search'])
        job = adapter.core_job(dict(job, candidate_sampler=job['mechanism']['candidate_sampler']))
    task_module = load_source('src.Bayesian_state.simulation.exp123_adaptive_task', REPO / TASK_FILES[0])
    runner = load_source('src.Bayesian_state.simulation.adaptive_autonomous', REPO / TASK_FILES[1])
    from src.Bayesian_state.simulation.parameters import apply_fixed_hyperparams_to_engine_config
    from src.Bayesian_state.utils.seeding import stable_seed
    folder = Path(job['folder']); meta = v.read(folder / 'prepared.json')
    for name, sha in meta['frozen_cognitive_source_sha256'].items():
        if v.digest(Path(job['runtime']) / name) != sha:
            raise ValueError('Archived cognitive source changed')
    if v.digest(folder / 'engine.json') != meta['engine_sha256']:
        raise ValueError('Prepared engine changed')
    engine = v.read(folder / 'engine.json')
    engine['modules']['hypo_transitions_mod']['kwargs']['intervention'] = job['mechanism']
    engine = apply_fixed_hyperparams_to_engine_config(engine, job['candidate']['hyperparams'])
    schedule = Path(job['schedule'])
    if v.digest(schedule) != job['schedule_sha256'] or v.digest(Path(job['reference'])) != job['reference_sha256']:
        raise ValueError('Task schedule or replay reference changed')
    with np.load(schedule) as data:
        blocks = [task_module.TaskBlock(data['stimulus'][i:i+64], data['categories'][i:i+64], data['ambiguous'][i:i+64])
                  for i in range(0, len(data['categories']), 64)]
    with np.load(job['reference']) as data:
        reference = {key: data[key] for key in ('seeds', 'choices', 'feedback', 'correct_probability')}
    records = []
    start, cpu = time.monotonic(), time.process_time()
    with diagnosis.single_threaded_processes(), cognitive_adapters(job) as corrections:
        for seed in job['seeds']:
            task_seed = stable_seed(dict(protocol=job['settings']['protocol'], base=job['settings']['seed'],
                                         subject=meta['subject'], trajectory_seed=seed, role='task'))
            if job['policy'] == 'fixed':
                task = task_module.FixedScheduleTask(blocks, condition=meta['condition'], seed=task_seed)
            else:
                task = task_module.AdaptiveTask(blocks[0], condition=meta['condition'], seed=task_seed,
                    max_blocks=job['settings']['max_blocks'], adaptive=job['policy'] != 'balanced',
                    repeat_session_rng=job['policy'] == 'adaptive_repeat',
                    blocks_per_session=job['settings']['blocks_per_session'], threshold=job['settings']['threshold'],
                    beta=job['settings']['beta'], ambiguity_width=job['settings']['ambiguity_width'])
            result = runner.run_adaptive_category_learning(engine_config=engine, subject_id=meta['subject'],
                task=task, trajectory_seed=seed, processed_data_dir=meta['paths']['processed_dir'], dataset_paths=meta['paths'])
            arrays = result['arrays']; n = len(arrays['choices'])
            correct = arrays['probabilities'][np.arange(n), arrays['categories'] - 1]
            if job['policy'] == 'fixed':
                row = int(np.flatnonzero(reference['seeds'] == seed)[0])
                for name in ('choices', 'feedback'):
                    np.testing.assert_array_equal(arrays[name], reference[name][row, :n])
                np.testing.assert_array_equal(correct, reference['correct_probability'][row, :n])
            target = Path(job['directory']) / f'path_{seed}.npz'
            with target.open('xb') as handle:
                np.savez_compressed(handle, **arrays, trajectory_seed=seed, task_seed=task_seed,
                    block_accuracy=np.array([b['accuracy'] for b in result['blocks']]),
                    block_dprime=np.array([b['dprime'] for b in result['blocks']]),
                    block_passed=np.array([b['passed'] for b in result['blocks']]))
            records.append(dict(subject=meta['subject'], model=job['model'], policy=job['policy'], seed=seed,
                task_seed=task_seed, blocks=n // 64, passed=result['passed'], observed_blocks=len(blocks),
                passed_by_observed_horizon=bool(result['passed'] and n <= len(blocks) * 64),
                final_correct_probability=float(correct[-64:].mean()), final_sample_accuracy=float((arrays['feedback'][-64:] == 1).mean()),
                mean_search_probability=float(arrays['search_probability'].mean()),
                near_one_search_fraction=float((arrays['search_probability'] >= .95).mean()),
                path=str(target.resolve()), path_sha256=v.digest(target)))
    for name, sha in job['source_sha256'].items():
        if v.digest(REPO / name) != sha:
            raise ValueError('Executing source changed during task generation')
    v.write(Path(job['target']), dict(records=records, seconds=time.monotonic()-start, cpu_seconds=time.process_time()-cpu,
        numeric_pools=threadpool_info(), exact_fixed_replay_paths=len(records) if job['policy'] == 'fixed' else 0,
        episodic_arm_settings=job.get('episodic_arm_settings'), corrections=corrections))


def prepare_schedules(v, frozen: dict, output: Path, subjects: list[int]) -> list[dict]:
    """Audit original adaptive allocations before running model-generated choices."""
    import numpy as np
    import pandas as pd
    from src.Bayesian_state.simulation.exp123_adaptive_task import block_performance, allocate_species, stimulus_labels
    checks = []
    for subject in subjects:
        folder = Path(frozen['data']) / f'subject_{subject}'
        meta = v.read(folder / 'prepared.json'); file = Path(meta['paths']['learning_data'])
        if v.digest(file) != meta['learning_data_sha256']:
            raise ValueError('Original task metadata changed')
        frame = pd.read_csv(file); frame = frame[frame.iSub == subject].copy()
        if 'miss' in frame:
            frame = frame[frame['miss'] == 0]
        index = pd.MultiIndex.from_frame(frame[['iSession', 'iBlock', 'iTrial']])
        if not index.is_unique or not index.is_monotonic_increasing:
            raise ValueError('Original task order is ambiguous')
        stimulus = frame[[f'feature{i}' for i in range(1, 5)]].to_numpy()
        categories, ambiguous = stimulus_labels(stimulus, meta['condition'])
        np.testing.assert_array_equal(categories, frame.category)
        np.testing.assert_array_equal(ambiguous, frame.ambiguous)
        prepared_parts = []
        for name in ('training.npz', 'sealed_test.npz'):
            if v.digest(folder / name) != meta['array_sha256'][name]:
                raise ValueError('Prepared sequence changed')
            with np.load(folder / name) as saved:
                prepared_parts.append(saved['stimulus'])
        np.testing.assert_array_equal(stimulus, np.concatenate(prepared_parts))
        previous, last = None, None
        for (session, block), group in frame.groupby(['iSession', 'iBlock'], sort=False):
            if len(group) != 64:
                raise ValueError('Need complete original blocks')
            if block > 1:
                prediction = allocate_species(previous, meta['condition'])
                if meta['condition'] == 1:
                    prediction = prediction.reshape(2, 2).sum(axis=1)
                actual = np.bincount(group.category, minlength=3 if meta['condition'] == 1 else 5)[1:]
                np.testing.assert_array_equal(prediction, actual)
                checks.append(dict(subject=subject, session=int(session), block=int(block), counts=actual.tolist(), match=True))
            accuracy, dprime = block_performance(group.category.to_numpy(), group.choice.to_numpy(),
                group.ambiguous.to_numpy(), 2 if meta['condition'] == 1 else 4)
            previous = (last + dprime) / 2 if session > 1 and block == 2 else dprime
            if block == 5:
                last = dprime
        with (output / f'schedule_{subject}.npz').open('xb') as handle:
            np.savez_compressed(handle, stimulus=stimulus, categories=categories, ambiguous=ambiguous)
    return checks


def select_task_nominees(frozen: dict, settings: dict, smoke: bool = False) -> tuple[list[dict], list[dict]]:
    """Apply explicit participant exclusions to this task diagnostic only.

    excluded_subjects maps integer participant IDs to documented task-provenance
    reasons. The original fitted cohort and chronological predictions are kept.
    Every retained participant keeps all fitted models.
    """
    excluded = settings.get('excluded_subjects', {})
    available = {row['subject'] for row in frozen['nominees']}
    if (not isinstance(excluded, dict)
            or any(isinstance(k, bool) or not isinstance(k, int) or k not in available
                   or not isinstance(v, str) or not v.strip() for k, v in excluded.items())):
        raise ValueError('excluded_subjects must map fitted integer IDs to task-provenance reasons')
    selected = [row for row in frozen['nominees'] if row['subject'] not in excluded]
    if not selected:
        raise ValueError('Task diagnostic must retain at least one participant')
    records = [dict(subject=subject, scope='adaptive_task_only', reason=reason)
               for subject, reason in sorted(excluded.items())]
    return (selected[:1] if smoke else selected), records


def run(fit: Path, reference: Path, config: Path, output: Path, smoke: bool) -> None:
    import numpy as np
    import pandas as pd
    import yaml
    from src.Bayesian_state.utils.parallel import parallel_job_count
    v = validation(); frozen = v.read(fit / 'nominees.json'); settings = yaml.safe_load(config.read_text())
    if output.exists():
        raise FileExistsError('Use a new adaptive-task study destination')
    if frozen['smoke'] or frozen['status'] not in ('frozen_before_test', 'frozen_for_development_evaluation'):
        raise ValueError('Need frozen scientific fits')
    for name, sha in frozen['source_sha256'].items():
        if v.digest(REPO / name) != sha:
            raise ValueError('Fitted scientific implementation changed')
    if v.read(reference / 'evaluation_protocol.json')['nomination_sha256'] != v.digest(fit / 'nominees.json'):
        raise ValueError('Replay reference uses another fit')
    selected, exclusions = select_task_nominees(frozen, settings, smoke)
    if smoke:
        settings.update(paths=1, max_blocks=2, parallel_budget=1, chunk_size=1)
    output.mkdir(parents=True); (output / 'jobs').mkdir(); (output / 'schedules').mkdir()
    checks = prepare_schedules(v, frozen, output / 'schedules', sorted({r['subject'] for r in selected}))
    v.write(output / 'task_policy_audit.json', dict(checks=checks, all_matched=True))
    rows = {(r['subject'], r['model']): r for r in v.read(reference / 'evaluation.json')['rows']}
    banks = {(r['subject'], r['model']): [r['candidate']] for r in selected}
    bases = v.score_jobs(Path(frozen['data']), output, frozen['settings'], banks, 'evaluate', False)
    hashes = dict(frozen['source_sha256'])
    for name in TASK_FILES + [str(Path(__file__).resolve().relative_to(REPO))]:
        hashes[name] = v.digest(REPO / name)
    jobs = []
    for base in bases:
        subject = v.read(Path(base['folder']) / 'prepared.json')['subject']
        old = rows[(subject, base['model'])]
        if old['candidate'] != base['candidate']:
            raise ValueError('Reference parameters differ')
        file = reference / 'evaluate' / f'subject_{subject}' / base['model'] / (base['candidate']['id'] + '.npz')
        if v.digest(file) != old['predictions_sha256']:
            raise ValueError('Reference predictions changed')
        with np.load(file) as saved:
            seeds = saved['seeds'][:settings['paths']].tolist()
        if len(seeds) != settings['paths']:
            raise ValueError('Not enough saved independent path seeds')
        schedule = output / 'schedules' / f'schedule_{subject}.npz'
        for policy in settings['policies']:
            directory = output / 'paths' / f'S{subject}' / base['model'] / policy
            directory.mkdir(parents=True)
            for first in range(0, len(seeds), settings['chunk_size']):
                name = f'S{subject}_{base["model"]}_{policy}_{first:03d}'
                jobs.append(dict(base, name=name, policy=policy, settings=settings, seeds=seeds[first:first+settings['chunk_size']],
                    schedule=str(schedule.resolve()), schedule_sha256=v.digest(schedule), reference=str(file.resolve()),
                    reference_sha256=old['predictions_sha256'], directory=str(directory.resolve()),
                    target=str((output / 'jobs' / (name + '_result.json')).resolve()), source_sha256=hashes))
    program_files = ['exp_program/exp123/Task2_learning.m', 'exp_program/exp123/sample_2.m']
    v.write(output / 'protocol.json', dict(fit=str(fit.resolve()), reference=str(reference.resolve()),
        nomination_sha256=v.digest(fit / 'nominees.json'), settings=settings, source_sha256=hashes, smoke=smoke,
        included_subjects=sorted({r['subject'] for r in selected}), excluded_subjects=exclusions,
        original_task_source_sha256={name:v.digest(REPO / name) for name in program_files},
        data_use_status=frozen['settings'].get('data_use_status', 'development'),
        role='task_environment_and_stopping_diagnostic',
        assumptions=['first observed block repeated in observed order at session starts', 'cognition persists across sessions',
            'NumPy distributional sampling; MATLAB random timing draws not replayed', 'explicit administrative horizon',
            'all failed and censored paths retained; no successful-path selection']))
    workers = parallel_job_count(settings['parallel_budget'], len(jobs))
    def launch(job):
        file = output / 'jobs' / (job['name'] + '.json'); v.write(file, job)
        launcher = 'import runpy,sys;source,script=sys.argv[1:3];sys.path.insert(0,source);sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name="__main__")'
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        with tempfile.TemporaryDirectory(prefix='h_adaptive_') as cache, file.with_suffix('.log').open('x') as log:
            env['NUMBA_CACHE_DIR'] = cache
            subprocess.run([sys.executable, '-c', launcher, job['runtime'], str(Path(__file__).resolve()),
                '--worker', str(file.resolve())], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f'Running {len(jobs)} chunks with {workers} workers', flush=True)
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, future in enumerate(as_completed([pool.submit(launch, job) for job in jobs]), 1):
            future.result()
            if index % 16 == 0 or index == len(jobs):
                print(f'{index}/{len(jobs)} complete', flush=True)
    results = [v.read(Path(job['target'])) for job in jobs]
    frame = pd.DataFrame([row for result in results for row in result['records']])
    frame.to_csv(output / 'paths.csv', index=False)
    summary = frame.groupby(['subject', 'model', 'policy'], sort=False).agg(paths=('seed','size'),
        pass_fraction=('passed','mean'), pass_by_observed_fraction=('passed_by_observed_horizon','mean'),
        mean_blocks_to_stop_or_censor=('blocks','mean'), final_correct_probability=('final_correct_probability','mean'),
        mean_search_probability=('mean_search_probability','mean'), near_one_search_fraction=('near_one_search_fraction','mean')).reset_index()
    summary.to_csv(output / 'summary.csv', index=False)
    if any(pool['num_threads'] != 1 for result in results for pool in result['numeric_pools']):
        raise ValueError('Numeric worker exceeded one thread')
    v.write(output / 'completion.json', dict(status='complete', smoke=smoke, paths=len(frame), jobs=len(jobs), workers=workers,
        seconds=time.monotonic()-start, cpu_seconds=sum(r['cpu_seconds'] for r in results),
        exact_fixed_replay_paths=sum(r['exact_fixed_replay_paths'] for r in results)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fit', 'reference', 'config', 'output', 'worker'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    else:
        if any(getattr(args, name) is None for name in ('fit', 'reference', 'config', 'output')):
            parser.error('fit, reference, config and output are required')
        run(args.fit, args.reference, args.config, args.output, args.smoke)


if __name__ == '__main__':
    main()

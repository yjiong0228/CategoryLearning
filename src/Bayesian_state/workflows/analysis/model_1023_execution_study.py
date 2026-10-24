"""Compare MIX, MAP and persistent STICKY execution on feedback-based beliefs.

Run --phase prepare --config YAML --output NEW, then --phase fit --config
YAML --data PREPARED --output NEW [--smoke]. All arms reuse the shared Model
1023 belief modules, numerical filters, simulation and adaptive optimizer.
Only the execution adapter changes. The participant's execution_inertia
coordinate is kappa for STICKY and zero for MIX/MAP. Fits use training choices
only, and independently confirmed nominees freeze before suffix evaluation.
Autonomous runs retain sparse rule beliefs, execution and forced switches.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from copy import deepcopy
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
ADAPTER = 'src/Bayesian_state/model/modules/hypothesis_transition/belief_execution.py'
PREPARATION_HELPER = 'src/Bayesian_state/evaluation/model_0923_fit_source.py'
PILOT = 'src/Bayesian_state/workflows/analysis/model_1022_belief_comparison.py'
spec = importlib.util.spec_from_file_location('execution_study_belief_pilot', ROOT / PILOT)
pilot = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pilot
spec.loader.exec_module(pilot)
driver, base = pilot.driver, pilot.driver.base
read, write, digest = pilot.read, pilot.write, pilot.digest
KAPPA = driver.H + 'execution_inertia'


def sources(v) -> dict:
    """Bind all shared scientific sources plus this opt-in extension."""
    return dict(pilot.sources(v), **{p: digest(ROOT / p) for p in (ENTRY, ADAPTER, PREPARATION_HELPER)})


def autonomous(job: dict, workflow) -> None:
    """Generate paths with the shared simulator and passive sparse states."""
    from threadpoolctl import threadpool_info
    v = base.validation()
    d = workflow.cohort().load_diagnosis()
    inputs = base.load('execution_study_inputs', driver.refit.INPUTS)
    engine, meta, arrays = inputs.inputs(job, v)
    n = min(16, len(arrays['categories'])) if job['smoke'] else len(arrays['categories'])
    if job.get('generation_trials') is not None:
        n = min(n, int(job['generation_trials']))
    categories = arrays['categories'][:n]
    capacity = engine['modules']['hypo_transitions_mod']['kwargs']['max_capacity']
    bank = []

    class Recorder:
        def __init__(self):
            self.rows, self.next_path = [], 0

        def before(self, model, choice, feedback):
            h = model.engine.get_module(d.ModuleRole.HYPOTHESIS_TRANSITION)
            event = h.transition_log[-1]
            active = np.asarray(h.active)
            rule = np.full(capacity, -1, dtype=np.int16)
            rule[:len(active)] = active
            row = dict(rule=rule)
            for key, values in [('belief', model._pending_trial.prior),
                                ('precision', model._pending_trial.beta),
                                ('contribution', h.execution_probabilities)]:
                padded = np.zeros(capacity, dtype=np.float64)
                padded[:len(active)] = values[active]
                row[key] = padded
            row.update(executed=-1 if job['execution_mode'] == 'MIX' else int(h.executed_hypothesis),
                switch=bool(event.get('execution_switch_event', False)) if job['execution_mode'] != 'MIX' else False,
                forced=bool(event.get('execution_forced_switch', False)), search=bool(event['search_event']),
                search_probability=float(event['search_probability']),
                proposal=-1 if event['proposed_hypothesis'] is None else int(event['proposed_hypothesis']))
            self.rows.append(row)

        def after(self, *args):
            pass

    started, cpu = time.monotonic(), time.process_time()
    for seed in job['seeds']:
        recorder = Recorder()
        with d.record_lifecycle(recorder):
            result = d.autonomous.run_autonomous_category_learning(engine_config=engine,
                subject_id=meta['subject'], condition=meta['condition'], stimulus=arrays['stimulus'][:n],
                categories=categories, trajectory_seed=seed, processed_data_dir=meta['paths']['processed_dir'],
                dataset_paths=meta['paths'])
        p = np.asarray(result.trajectory.observed_probabilities)
        if not np.isfinite(p).all() or np.any(p < 0):
            raise ValueError('Invalid autonomous probabilities')
        np.testing.assert_allclose(p.sum(axis=1), 1., atol=1e-12, rtol=0)
        if len(recorder.rows) != n:
            raise ValueError('Execution recorder missed a trial')
        row = {key: np.array([r[key] for r in recorder.rows]) for key in recorder.rows[0]}
        row.update(choices=np.asarray(result.trajectory.choices), probabilities=p,
            feedback=np.asarray(result.trajectory.feedback), correct_probability=p[np.arange(n), categories - 1])
        bank.append(row)
    target = Path(job['target']); target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix('.npz').open('xb') as stream:
        np.savez_compressed(stream, **{k: np.stack([r[k] for r in bank]) for k in bank[0]},
                            seeds=job['seeds'], categories=categories)
    write(target.with_suffix('.json'), dict(subject=meta['subject'], model=job['execution_mode'],
        condition=meta['condition'], candidate=job['candidate'], paths=len(bank), trials=n,
        training_trials=meta['training_trials'], seconds=time.monotonic() - started,
        cpu_seconds=time.process_time() - cpu, numeric_pools=threadpool_info(),
        role='autonomous_from_start', predictions_sha256=digest(target.with_suffix('.npz'))))


def worker(file: Path) -> None:
    """Install execution after the existing ADAPT modules in one worker."""
    job = read(file)
    if job['phase'] in ('prepare', 'audit'):
        v = base.validation()
        workflow = v.intervention_workflow(); workflow.install_transition()
        if job['phase'] == 'prepare':
            base.load('src.Bayesian_state.evaluation.model_0923_fit_source', PREPARATION_HELPER)
            v.prepare_subject(job, workflow)
        else:
            inputs = base.load('execution_study_geometry', driver.refit.INPUTS)
            inputs.audit(job, v)
        return
    if (job['source_sha256'] != sources(driver.io) or job['execution_mode'] not in ('MIX', 'MAP', 'STICKY')
            or job['comparison_arm'] != 'ADAPT'):
        raise ValueError('Execution job lost its registered mechanism or source identity')
    kappa = job['candidate']['hyperparams'][KAPPA]
    if not np.isfinite(kappa) or kappa < 0 or job['execution_mode'] != 'STICKY' and kappa != 0:
        raise ValueError('Only STICKY has a nonzero execution parameter')
    original_load, original_fixed = base.load, driver.fixed_coordinates
    original_autonomous = driver.refit.autonomous

    def load(name, path):
        module = original_load(name, path)
        if path == pilot.ERROR_ENTRY:
            execution = original_load('src.Bayesian_state.model.modules.hypothesis_transition.belief_execution', ADAPTER)
            original_entry = module.belief_error_entry

            @contextmanager
            def install(cls):
                with original_entry(cls) as entries, execution.belief_execution(cls, mode=job['execution_mode']) as audit:
                    entries['execution'] = audit
                    yield entries

            module.belief_error_entry = install
        return module

    def fixed(arm, condition):
        # Keep the candidate and its identity intact throughout scoring.
        # The new adapter validates kappa and shields belief initialization.
        return {k: v for k, v in original_fixed(arm, condition).items() if k != KAPPA}

    base.load, driver.fixed_coordinates, driver.refit.autonomous = load, fixed, autonomous
    try:
        pilot.worker(file)
    finally:
        base.load, driver.fixed_coordinates, driver.refit.autonomous = original_load, original_fixed, original_autonomous


def execute(jobs: list[dict], output: Path, budget: int) -> None:
    """Dispatch a single layer of fresh, single-threaded numerical workers."""
    directory = output / 'jobs'; directory.mkdir(exist_ok=True)

    def launch(original):
        job = deepcopy(original)
        if job['phase'] not in ('prepare', 'audit'):
            job.update(execution_mode=job['model'], comparison_arm='ADAPT', model='COMPACT')
            job['mechanism']['capacity'] = 'evidence'
        file = directory / (job['name'] + '.json')
        write(file, job)
        launcher = ('import runpy,sys;runtime,script=sys.argv[1:3];sys.path.insert(0,runtime);'
                    "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")
        with tempfile.TemporaryDirectory(prefix='belief_execution_') as cache:
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', NUMBA_NUM_THREADS='1', NUMBA_CACHE_DIR=cache,
                MPLCONFIGDIR=cache)
            with file.with_suffix('.log').open('x') as log:
                subprocess.run([sys.executable, '-c', launcher, job['runtime'], str(ROOT / ENTRY),
                    '--worker', str(file.resolve())], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    workers = min(budget, len(os.sched_getaffinity(0)), len(jobs))
    if workers < 1:
        raise ValueError('No execution study jobs')
    print(f'{len(jobs)} jobs; {workers} workers; one numerical thread each', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for count, future in enumerate(as_completed([pool.submit(launch, j) for j in jobs]), 1):
            future.result()
            if count % 16 == 0 or count == len(jobs):
                print(f'{count}/{len(jobs)} completed', flush=True)


def prepare(config: Path, output: Path, smoke: bool) -> None:
    """Use the existing chronological preparation and original task mappings."""
    if output.exists():
        raise FileExistsError('Use a new preparation directory')
    policy = yaml.safe_load(config.read_text())
    output.mkdir(parents=True)
    settings = dict(policy['preparation'], protocol=policy['protocol'] + '_data',
                    subjects=policy['subjects'], parallel_budget=policy['parallel_budget'])
    subjects = settings['subjects'][:1] if smoke else settings['subjects']
    fit = (ROOT / settings['source_fit']).resolve()
    jobs = [dict(phase='prepare', name=f'prepare_S{s}', subject=s, condition=c, source_fit=str(fit),
        runtime=str(fit / 'source'), folder=str((output / f'subject_{s}').resolve()),
        cache_sources=settings['cache_sources'], training_fraction=settings['training_fraction']) for s, c in subjects]
    execute(jobs, output, 1 if smoke else policy['parallel_budget'])
    write(output / 'preparation_manifest.json', dict(status='complete', settings=settings, smoke=smoke,
        subjects=[s for s, c in subjects], source_sha256=sources(driver.io),
        artifacts_sha256={str(p.relative_to(output)): digest(p) for p in output.glob('subject_*/*') if p.is_file()}))


def parameter_space(settings: dict, mode: str, condition: int, kappas: list) -> dict:
    shared = deepcopy(settings)
    shared['arms']['COMPACT'] = deepcopy(next(iter(settings['arms'].values())))
    space = driver.model_space(shared, 'COMPACT', condition)
    space[KAPPA] = list(kappas) if mode == 'STICKY' else [0.]
    return space


def confirmation_bank(screen: dict, refined: dict, anchors: dict, subjects: list,
                      modes: list[str], quota: int, leaders: int) -> dict:
    """Give every arm the same confirmation budget and its own best starts.

    Cross-arm projections follow round-robin rank order. Each arm retains
    its leading screened/refined candidate and the common training anchor;
    STICKY's own leaders retain their estimated kappa.
    """
    from src.Bayesian_state.optimization.search.adaptive_proposals import candidate
    if quota < 3 or leaders < 1:
        raise ValueError('Confirmation requires at least three candidates')
    result = {}
    for sid in subjects:
        shared = []
        for rank in range(leaders):
            for rows_by_arm in (screen, refined):
                for mode in modes:
                    rows = rows_by_arm[sid, mode]
                    if rank < len(rows):
                        shared.append(candidate(dict(rows[rank]['hyperparams'], **{KAPPA: 0.}),
                                                'cross_arm_projection'))
        for mode in modes:
            own = {}
            rows = [screen[sid, mode][0], refined[sid, mode][0], anchors[sid, mode]]
            rows += shared + refined[sid, mode] + screen[sid, mode]
            for row in rows:
                own.setdefault(row['id'], candidate(row['hyperparams'], 'balanced_confirmation'))
                if len(own) == quota:
                    break
            if len(own) != quota:
                raise ValueError('Insufficient distinct candidates for equal confirmation budgets')
            result[sid, mode] = list(own.values())
    return result


def fit(config: Path, data: Path, output: Path, smoke: bool) -> None:
    """Fit each mechanism, confirm common projections and freeze nominees."""
    from src.Bayesian_state.optimization.search.adaptive_proposals import (
        candidate, initial_points, propose_round, select_elites)
    from src.Bayesian_state.utils.seeding import stable_seed
    if output.exists():
        raise FileExistsError('Use a new fit directory')
    policy = yaml.safe_load(config.read_text())
    if policy['arms'] != ['MIX', 'MAP', 'STICKY'] or not 1 <= policy['parallel_budget'] <= 128:
        raise ValueError('Require the declared three execution mechanisms')
    settings = yaml.safe_load((ROOT / policy['base_config']).read_text())
    precision = deepcopy(settings['arms']['COMPACT'])
    for key in ('protocol', 'seed', 'parallel_budget', 'budgets', 'autonomous_paths', 'autonomous_chunk'):
        settings[key] = deepcopy(policy[key])
    subjects = policy['subjects'][:1] if smoke else policy['subjects']
    settings.update(data=str(data.resolve()), arms={a: deepcopy(precision) for a in policy['arms']},
                    subjects=[s for s, c in subjects])
    if smoke:
        settings['parallel_budget'] = 1
    output.mkdir(parents=True)
    base.source_hashes, base.execute = sources, execute
    base.archive(output, config, sources(driver.io))
    warm, inputs = {}, {}
    for name in policy.get('warm_nominations', []):
        file = ROOT / name
        if not file.exists():
            raise FileNotFoundError(f'Declared warm nomination missing: {name}')
        old = read(file)
        if old['smoke'] or old['later_choices_used'] or old['status'] != 'frozen_before_test':
            raise ValueError('Warm anchors require training-only frozen nominations')
        inputs[name] = digest(file)
        for row in old['nominees']:
            if row['arm'] == 'ADAPT':
                warm[row['subject']] = row['candidate']['hyperparams']
    banks, spaces, anchors = {}, {}, {}
    count = 8 if smoke else policy['initial_candidates']
    for sid in settings['subjects']:
        meta = read(data / f'subject_{sid}' / 'prepared.json')
        common = parameter_space(settings, 'MIX', meta['condition'], policy['kappa_values'])
        anchor = deepcopy(warm.get(sid, {k: v[len(v) // 2] for k, v in common.items()}))
        anchor[KAPPA] = 0.
        if set(anchor) != set(common) or any(anchor[k] not in v for k, v in common.items()):
            raise ValueError('Anchor outside shared parameter support')
        points = initial_points(common, anchor, count,
            stable_seed(dict(base=policy['seed'], subject=sid, stage='shared_initial')),
            cover_key=driver.H + 'max_capacity')
        for mode in policy['arms']:
            key = sid, mode
            spaces[key] = parameter_space(settings, mode, meta['condition'], policy['kappa_values'])
            own = deepcopy(points)
            if mode == 'STICKY':
                for index, p in enumerate(own):
                    p[KAPPA] = policy['kappa_values'][index % len(policy['kappa_values'])]
            banks[key] = [candidate(p, 'shared_initial_coordinates') for p in own]
            anchors[key] = candidate(anchor, 'common_training_anchor')
    write(output / 'protocol.json', dict(settings=settings, policy=policy, smoke=smoke,
        source_sha256=sources(driver.io), inputs_sha256=inputs, python=sys.version,
        versions={k: importlib.metadata.version(k) for k in ('numpy', 'scipy', 'pandas', 'PyYAML')},
        data_boundary='chronological training only; frozen later evaluation; developmental cohort',
        parameter_spaces=[dict(subject=k[0], mode=k[1], space=v) for k, v in spaces.items()],
        initial_banks=[dict(subject=k[0], mode=k[1], candidates=v) for k, v in banks.items()]))
    stage = lambda bank, name: base.stage(settings, data, output, bank, name, smoke)
    screen = stage(banks, 'screen')
    seen = {k: {r['id'] for r in v} for k, v in screen.items()}
    refined = screen
    for index in range(1 if smoke else policy['refinement_rounds']):
        name = f'refine_{index + 1}'
        next_bank = {}
        for key, rows in refined.items():
            elites = select_elites(rows, 1 if smoke else policy['elites'], exclude_key=driver.H + 'max_capacity')
            quota = 2 if smoke else policy['refinement_candidates']
            new = propose_round(elites, spaces[key], seen[key], quota // len(elites),
                stable_seed(dict(base=policy['seed'], subject=key[0], stage=name)),
                policy['proposal_fractions']['local'], policy['proposal_fractions']['jump'])
            if len(new) != quota:
                raise ValueError('Adaptive search quota not met')
            seen[key].update(r['id'] for r in new)
            next_bank[key] = new + elites
        refined = stage(next_bank, name)
    confirmation = confirmation_bank(screen, refined, anchors, settings['subjects'], policy['arms'],
        4 if smoke else policy['confirmation_candidates'], 1 if smoke else policy['confirmation_per_arm'])
    confirmed = stage(confirmation, 'confirmation')
    nominees = [dict(subject=k[0], arm=k[1], candidate=v[0], finalists=v) for k, v in confirmed.items()]
    write(output / 'nominees.json', dict(status='frozen_before_test', nominees=nominees, smoke=smoke,
        later_choices_used=False, oral_used=False, selection='independent_training_NLL'))
    if not smoke:
        stage({(r['subject'], r['arm']): [r['candidate']] for r in nominees}, 'evaluate')
    preparation = read(data / 'preparation_manifest.json')
    runtime = str((ROOT / preparation['settings']['source_fit'] / 'source').resolve())
    jobs = []
    for row in nominees:
        seeds = [stable_seed(dict(base=policy['seed'], subject=row['subject'], stage='autonomous', repeat=i))
                 for i in range(2 if smoke else policy['autonomous_paths'])]
        for index in range(0, len(seeds), policy['autonomous_chunk']):
            name = f'autonomous_S{row["subject"]}_{row["arm"]}_{index:03d}'
            jobs.append(dict(phase='autonomous', name=name, subject=row['subject'], model=row['arm'],
                candidate=row['candidate'], mechanism=settings['mechanism'],
                folder=str((data / f'subject_{row["subject"]}').resolve()), runtime=runtime,
                arm_settings=precision, seeds=seeds[index:index + policy['autonomous_chunk']],
                smoke=smoke, source_sha256=sources(driver.io), target=str((output / 'autonomous' / name).resolve())))
    write(output / 'autonomous_jobs.json', dict(jobs=jobs))
    execute(jobs, output, settings['parallel_budget'])
    write(output / 'completion.json', dict(status='complete', smoke=smoke, cases=len(nominees),
        stage='numerical_fits_and_generation', scientific_summary_pending=True,
        source_sha256=sources(driver.io), nominees_sha256=digest(output / 'nominees.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'fit'))
    for name in ('config', 'data', 'output', 'worker'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.phase == 'prepare' and args.config and args.output:
        prepare(args.config, args.output, args.smoke)
    elif args.phase == 'fit' and args.config and args.data and args.output:
        fit(args.config, args.data, args.output, args.smoke)
    else:
        parser.error('Choose prepare or fit with --config, a new --output, and --data for fit')

"""Equal-budget joint refits of BASE, NEW, POOL and DRAW admission.

All arms share fourteen parameter supports, initial candidates and scoring
seeds. Each gets one adaptive refinement, then evaluates the same union of
leading candidates. Training pre-choice NLL selects parameters. Independent
confirmation precedes frozen later-trial and autonomous evaluation. All nine
participants are development data. New output directories are required.

The precision-refit scheduler, probability scoring and archived cognitive core
are reused; opt-in admission adapters implement the experimental differences.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from copy import deepcopy
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import yaml

# File loading also works when the subprocess imports its frozen model package.
import importlib.util
REPO = Path(__file__).resolve().parents[4]
ENTRY = 'src/Bayesian_state/workflows/analysis/model_0923_episodic_refit.py'
spec = importlib.util.spec_from_file_location('episodic_refit_base',
    REPO/'src/Bayesian_state/workflows/analysis/model_0923_precision_refit.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
HELPER = 'src/Bayesian_state/model/modules/hypothesis_transition/episodic_admission.py'
GUARD = 'src/Bayesian_state/model/modules/pairing_snapshot_roundoff.py'
INPUTS = 'src/Bayesian_state/workflows/analysis/model_0923_h_rule_diagnosis.py'


def starting_banks(settings: dict, frozen: dict, smoke: bool) -> tuple[dict, dict]:
    """Reuse both additive optima, then cover capacities and joint coordinates."""
    from src.Bayesian_state.optimization.search.adaptive_proposals import candidate, initial_points
    from src.Bayesian_state.utils.seeding import stable_seed
    banks, required = {}, {}
    space = base.model_space(settings, 'BASE')
    if any(base.model_space(settings, arm) != space for arm in settings['arms']):
        raise ValueError('Admission arms require identical parameter supports')
    for subject in settings['subjects'][:1] if smoke else settings['subjects']:
        prior = {r['model']: r for r in frozen['nominees'] if r['subject'] == subject}
        anchor = deepcopy(prior['ADD10']['candidate']['hyperparams'])
        mapped = deepcopy(prior['ADD1']['candidate']['hyperparams'])
        mapped[base.RATE] /= settings['arms']['BASE']['rate_scale']
        count = 5 if smoke else settings['initial_candidates']
        points = [(anchor, 'prior_ADD10'), (mapped, 'prior_ADD1_exact_mapping')]
        starts = initial_points(space, anchor, count + 16,
            stable_seed(dict(base=settings['seed'], subject=subject, role='shared_initial')),
            cover_key=base.validation().H_PATH+'max_capacity')
        points.extend((point, 'common_capacity_and_joint_start') for point in starts)
        rows = {}
        for point, source in points:
            if set(point) != set(space) or any(point[k] not in values for k, values in space.items()):
                raise ValueError('Warm start outside common support')
            row = candidate(point, source)
            rows.setdefault(row['id'], row)
            if len(rows) == count: break
        if len(rows) != count: raise ValueError('Insufficient distinct initial candidates')
        for arm in settings['arms']:
            banks[subject, arm] = deepcopy(list(rows.values()))
            required[subject, arm] = deepcopy(next(iter(rows.values())))
    return banks, required


def exchange_banks(rows: dict, required: dict, count: int) -> dict:
    """Every arm scores the same per-subject union, including the old reference."""
    from src.Bayesian_state.optimization.search.adaptive_proposals import candidate
    shared = {}
    for (subject, arm), values in rows.items():
        bank = shared.setdefault(subject, {})
        for row in [required[subject, arm]] + values[:count]:
            item = bank.setdefault(row['id'], candidate(row['hyperparams'], 'cross_arm_exchange'))
            item.setdefault('proposed_by', [])
            if arm not in item['proposed_by']: item['proposed_by'].append(arm)
    return {key: deepcopy(list(shared[key[0]].values())) for key in rows}


def autonomous(job: dict, workflow) -> None:
    """Generate complete paths and record target-rule availability and use."""
    from threadpoolctl import threadpool_info
    v = base.validation(); d = workflow.cohort().load_diagnosis()
    inputs = base.load('episodic_refit_inputs', INPUTS)
    subject = v.read(Path(job['folder'])/'prepared.json')['subject']
    engine, meta, arrays = inputs.inputs(dict(job, subject=subject), v)
    audit = Path(job['audit'])
    if v.digest(audit) != job['audit_sha256']: raise ValueError('Geometry audit changed')
    target_rule = v.read(audit)['target']
    n = 16 if job['smoke'] else len(arrays['categories'])
    categories = arrays['categories'][:n]; bank = []

    class Recorder:
        def __init__(self): self.rows = []; self.next_path = 0
        def before(self, model, choice, outcome):
            h = model.engine.get_module(d.ModuleRole.HYPOTHESIS_TRANSITION)
            event = h.transition_log[-1]
            self.rows.append(dict(present=int(target_rule in h.active),
                contribution=float(h.execution_probabilities[target_rule]),
                beta=float(model._pending_trial.beta[target_rule]) if target_rule in h.active else 0.,
                proposed=-1 if event['proposed_hypothesis'] is None else int(event['proposed_hypothesis']),
                search=int(event['search_event']), rejected=int(event.get('newcomer_rejected', False)),
                target_exited=int(target_rule in event['autonomous_exits']),
                target_capacity_dropped=int(event['capacity_dropped'] == target_rule),
                active_count=len(h.active)))
        def after(self, *args): pass

    start, cpu = time.monotonic(), time.process_time()
    for seed in job['seeds']:
        recorder = Recorder()
        with d.record_lifecycle(recorder):
            result = d.autonomous.run_autonomous_category_learning(engine_config=engine,
                subject_id=subject, condition=meta['condition'], stimulus=arrays['stimulus'][:n],
                categories=categories, trajectory_seed=seed, processed_data_dir=meta['paths']['processed_dir'],
                dataset_paths=meta['paths'])
        p = np.asarray(result.trajectory.observed_probabilities)
        if not np.isfinite(p).all() or np.any(p < 0): raise ValueError('Invalid autonomous probabilities')
        np.testing.assert_allclose(p.sum(axis=1), 1., atol=1e-12, rtol=0)
        row = {k: np.array([r[k] for r in recorder.rows]) for k in recorder.rows[0]}
        row.update(choices=np.asarray(result.trajectory.choices), probabilities=p,
            feedback=np.asarray(result.trajectory.feedback), correct_probability=p[np.arange(n), categories-1])
        bank.append(row)
    target = Path(job['target']); target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix('.npz').open('xb') as stream:
        np.savez_compressed(stream, **{k: np.stack([r[k] for r in bank]) for k in bank[0]},
                            seeds=job['seeds'], categories=categories)
    v.write(target.with_suffix('.json'), dict(subject=subject, model=job['model'], condition=meta['condition'],
        candidate=job['candidate'], paths=len(bank), trials=n, training_trials=meta['training_trials'],
        seconds=time.monotonic()-start, cpu_seconds=time.process_time()-cpu, numeric_pools=threadpool_info(),
        target_rule=target_rule, role='autonomous_from_start', predictions_sha256=v.digest(target.with_suffix('.npz'))))


def worker(file: Path) -> None:
    v = base.validation(); job = v.read(file)
    workflow = v.intervention_workflow(); workflow.install_transition()
    diagnosis = workflow.cohort().load_diagnosis()
    feedback = base.load('src.Bayesian_state.model.modules.hypothesis_transition.feedback_guided', base.FEEDBACK)
    precision = base.load('src.Bayesian_state.model.modules.precision_interventions', base.PRECISION)
    guard = base.load('src.Bayesian_state.model.modules.pairing_snapshot_roundoff', GUARD)
    history = base.load('src.Bayesian_state.model.modules.hypothesis_transition.episodic_admission', HELPER)
    from src.Bayesian_state.model.modules.beta import BetaModule
    from src.Bayesian_state.model.modules.pairing_memory import HierarchicalPairingMemoryModule
    cls = sys.modules['src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search'].MechanisticRuleSearchHypothesisTransitionModule
    with ExitStack() as stack:
        stack.enter_context(diagnosis.single_threaded_processes())
        feedback_fixes = stack.enter_context(feedback.retention_roundoff_guard(cls))
        snapshot_fixes = stack.enter_context(guard.pairing_snapshot_roundoff_guard(HierarchicalPairingMemoryModule))
        arm = job['arm_settings']
        stack.enter_context(precision.precision_intervention(BetaModule, mode=arm['precision'], upper=arm['upper']))
        recall_fixes = []
        if job['model'] != 'BASE':
            recall_fixes = stack.enter_context(history.episodic_admission(cls, mode=job['model']))
        if job['phase'] == 'autonomous': autonomous(job, workflow)
        else: v.score(job, workflow)
        v.write(Path(job['target']).with_suffix('.experiment.json'), dict(arm_settings=arm,
            feedback_corrections=feedback_fixes, snapshot_corrections=snapshot_fixes, recall_corrections=recall_fixes))


def execute(jobs: list[dict], output: Path, budget: int) -> None:
    from src.Bayesian_state.utils.parallel import parallel_job_count
    v = base.validation(); directory = output/'jobs'; directory.mkdir(exist_ok=True)
    def launch(job):
        if job['phase'] == 'autonomous':
            settings = v.read(Path(v.read(output/'protocol.json')['fit'])/'nominees.json')['settings']
            subject = v.read(Path(job['folder'])/'prepared.json')['subject']
            audit = REPO/settings['audit']/f'audit/subject_{subject}/audit.json'
            job.update(audit=str(audit), audit_sha256=v.digest(audit))
        file = directory/(job['name']+'.json'); v.write(file, job)
        launcher = ('import runpy,sys;source,script=sys.argv[1:3];sys.path.insert(0,source);'
                    "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        with tempfile.TemporaryDirectory(prefix='episodic_refit_') as cache:
            env['NUMBA_CACHE_DIR'] = cache
            with file.with_suffix('.log').open('x') as stream:
                result = subprocess.run([sys.executable, '-c', launcher, job['runtime'], str(REPO/ENTRY),
                    '--worker', str(file)], cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError(f'Worker failed: {file.with_suffix(".log")}')
    workers = parallel_job_count(budget, len(jobs))
    print(f'{len(jobs)} jobs; {workers} workers; numeric threads=1', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, future in enumerate(as_completed([pool.submit(launch, j) for j in jobs]), 1):
            future.result()
            if i % 32 == 0 or i == len(jobs): print(f'{i}/{len(jobs)} complete', flush=True)


def configure() -> None:
    original = base.source_hashes
    def hashes(v):
        result = original(v)
        for name in (ENTRY, HELPER, GUARD, INPUTS): result[name] = v.digest(REPO/name)
        return result
    base.source_hashes = hashes
    base.execute = execute


def search(config: Path, output: Path, smoke: bool) -> None:
    from src.Bayesian_state.optimization.search.adaptive_proposals import propose_round, select_elites
    from src.Bayesian_state.utils.seeding import stable_seed
    v = base.validation(); settings = yaml.safe_load(config.read_text())
    if list(settings['arms']) != ['BASE', 'NEW', 'POOL', 'DRAW']: raise ValueError('Require four declared arms')
    data = REPO/settings['data']; frozen_file = REPO/settings['warm_start_fit']/'nominees.json'
    frozen = v.read(frozen_file)
    if frozen['smoke'] or frozen['status'] != 'frozen_before_test': raise ValueError('Need scientific warm starts')
    for name, sha in frozen['source_sha256'].items():
        if v.digest(REPO/name) != sha: raise ValueError(f'Warm-start source changed: {name}')
    if output.exists(): raise FileExistsError('Use a new episodic refit directory')
    banks, required = starting_banks(settings, frozen, smoke)
    seen = {key: {r['id'] for r in rows} for key, rows in banks.items()}
    output.mkdir(parents=True); hashes = base.source_hashes(v); base.archive(output, config, hashes)
    shutil.copy2(REPO/settings['parameter_support'], output/'parameter_support.yaml')
    v.write(output/'parameter_spaces.json', {arm: base.model_space(settings, arm) for arm in settings['arms']})
    v.write(output/'protocol.json', dict(settings=settings, smoke=smoke, source_sha256=hashes,
        warm_start_nomination_sha256=v.digest(frozen_file), data_manifest_sha256=v.digest(data/'preparation_manifest.json'),
        data_boundary='training only; all subjects are development participants',
        initial_banks=[dict(subject=k[0], model=k[1], rows=rows) for k, rows in banks.items()],
        python=sys.version, versions={k: importlib.metadata.version(k) for k in ('numpy','scipy','pandas','PyYAML')}))
    screen = base.stage(settings, data, output, banks, 'screen', smoke)
    for key in banks:
        elites = select_elites(screen[key], 1 if smoke else settings['elites'], exclude_key=v.H_PATH+'max_capacity')
        proposals = propose_round(elites, base.model_space(settings, key[1]), seen[key],
            1 if smoke else settings['proposals_per_elite'],
            stable_seed(dict(base=settings['seed'], subject=key[0], role='joint_refine')), .5, .25)
        expected = 1 if smoke else settings['elites']*settings['proposals_per_elite']
        if len(proposals) != expected: raise ValueError('Unequal adaptive proposal budget')
        banks[key] = proposals + elites; seen[key].update(r['id'] for r in proposals)
    refined = base.stage(settings, data, output, banks, 'refine', smoke)
    leading = {}
    for key in screen:
        merged = {r['id']: r for r in screen[key] + refined[key]}
        leading[key] = sorted(merged.values(), key=lambda r: (r['mean_nll'], r['id']))
    banks = exchange_banks(leading, required, 2 if smoke else settings['exchange_per_arm'])
    for key in banks: seen[key].update(r['id'] for r in banks[key])
    rows = base.stage(settings, data, output, banks, 'exchange', smoke)
    for name, count in [('selection', settings['selection_candidates']), ('confirmation', settings['confirmation_candidates'])]:
        banks = {key: base.shortlist(values, required[key], 2 if smoke else min(count, len(values)))
                 for key, values in rows.items()}
        rows = base.stage(settings, data, output, banks, name, smoke)
    nominees = [dict(subject=k[0], model=k[1], candidate=r[0], finalists=r,
        unique_candidates=len(seen[k]), reference_id=required[k]['id']) for k, r in rows.items()]
    v.write(output/'nominees.json', dict(status='frozen_before_test', settings=settings, smoke=smoke,
        source_sha256=hashes, data=str(data.resolve()), nominees=nominees))
    v.write(output/'completion.json', dict(status='complete', test_choices_used=False, smoke=smoke,
        nomination_sha256=v.digest(output/'nominees.json')))
    print('All admission nominees frozen before evaluation', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path); parser.add_argument('--phase', choices=('search','evaluate'))
    for name in ('config','fit','output'): parser.add_argument('--'+name, type=Path)
    parser.add_argument('--smoke', action='store_true'); args = parser.parse_args()
    if args.worker: worker(args.worker); return
    configure()
    if args.phase == 'search' and args.config and args.output: search(args.config, args.output, args.smoke)
    elif args.phase == 'evaluate' and args.fit and args.output and not args.smoke: base.evaluate(args.fit, args.output)
    else: parser.error('Use search with --config, or evaluate with --fit, and a new --output')


if __name__ == '__main__': main()

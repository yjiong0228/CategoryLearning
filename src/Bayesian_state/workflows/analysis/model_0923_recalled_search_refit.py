"""Equal-budget behavior-only refits of POOL and history-guided SEARCH.

Reuse the established fitting stages, particle objective and autonomous engine.
Initial banks include prior fits and explicit memory, recall, capacity and
execution-inertia rays, followed by common joint proposals. Independent filter
seeds confirm the nominees before subsequent-trial evaluation. All participants
are development data; oral reports never enter parameter selection.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from copy import deepcopy
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
PROPOSAL = 'src/Bayesian_state/model/modules/hypothesis_transition/episodic_proposal.py'
BUDGET = 'src/Bayesian_state/workflows/analysis/model_0923_episodic_adaptive_task.py'
# Workers import their frozen cognitive package; load the current orchestration
# by file so that new workflow modules need not be added to that package.
spec = importlib.util.spec_from_file_location('recalled_search_base',
    ROOT / 'src/Bayesian_state/workflows/analysis/model_0923_episodic_refit.py')
refit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = refit
spec.loader.exec_module(refit)
base = refit.base


def starting_banks(settings: dict, frozen: dict, smoke: bool) -> tuple[dict, dict]:
    """Give both mechanisms identical anchored rays and joint starting points."""
    from src.Bayesian_state.optimization.search.adaptive_proposals import candidate, initial_points
    from src.Bayesian_state.utils.seeding import stable_seed
    space = base.model_space(settings, 'POOL')
    if base.model_space(settings, 'SEARCH') != space:
        raise ValueError('Require identical active parameter supports')
    banks, required = {}, {}
    for subject in settings['subjects'][:1] if smoke else settings['subjects']:
        previous = {r['model']: r['candidate']['hyperparams'] for r in frozen['nominees'] if r['subject'] == subject}
        anchor = previous['POOL']; points = [(anchor, 'previous_POOL')]
        points += [(previous[arm], 'previous_' + arm) for arm in ('BASE', 'NEW', 'DRAW')]
        for coordinate in settings['initial_ray_coordinates']:
            for value in space[coordinate]:
                points.append((dict(anchor, **{coordinate: value}), 'ray:' + coordinate))
        count = 6 if smoke else settings['initial_candidates']
        points += [(point, 'common_joint_start') for point in initial_points(space, anchor, count + 32,
            stable_seed(dict(base=settings['seed'], subject=subject, role='shared_initial')),
            cover_key=base.validation().H_PATH + 'max_capacity')]
        rows = {}
        for point, origin in points:
            if set(point) != set(space) or any(point[k] not in levels for k, levels in space.items()):
                raise ValueError('Starting point outside declared common support')
            row = candidate(point, origin)
            rows.setdefault(row['id'], row)
            if len(rows) == count:
                break
        if len(rows) != count:
            raise ValueError('Insufficient distinct common starting points')
        for arm in settings['arms']:
            banks[subject, arm] = deepcopy(list(rows.values()))
            required[subject, arm] = deepcopy(next(iter(rows.values())))
    return banks, required


def worker(file: Path) -> None:
    v = base.validation(); job = v.read(file)
    for name, sha in job['source_sha256'].items():
        if v.digest(ROOT / name) != sha:
            raise ValueError(f'Executing source changed: {name}')
    workflow = v.intervention_workflow(); workflow.install_transition()
    diagnosis = workflow.cohort().load_diagnosis()
    feedback = base.load('src.Bayesian_state.model.modules.hypothesis_transition.feedback_guided', base.FEEDBACK)
    precision = base.load('src.Bayesian_state.model.modules.precision_interventions', base.PRECISION)
    guard = base.load('src.Bayesian_state.model.modules.pairing_snapshot_roundoff', refit.GUARD)
    history = base.load('src.Bayesian_state.model.modules.hypothesis_transition.episodic_admission', refit.HELPER)
    proposal = base.load('src.Bayesian_state.model.modules.hypothesis_transition.episodic_proposal', PROPOSAL)
    from src.Bayesian_state.model.modules.beta import BetaModule
    from src.Bayesian_state.model.modules.pairing_memory import HierarchicalPairingMemoryModule
    cls = sys.modules['src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search'].MechanisticRuleSearchHypothesisTransitionModule
    if job['model'] not in ('POOL', 'SEARCH'):
        raise ValueError('Require the declared POOL versus SEARCH comparison')
    with ExitStack() as stack:
        stack.enter_context(diagnosis.single_threaded_processes())
        feedback_fixes = stack.enter_context(feedback.retention_roundoff_guard(cls))
        snapshot_fixes = stack.enter_context(guard.pairing_snapshot_roundoff_guard(HierarchicalPairingMemoryModule))
        arm = job['arm_settings']
        stack.enter_context(precision.precision_intervention(BetaModule, mode=arm['precision'], upper=arm['upper']))
        recall_fixes = stack.enter_context(history.episodic_admission(cls, mode='POOL'))
        stack.enter_context(proposal.episodic_proposal(cls, proposal=job['model'] == 'SEARCH'))
        (refit.autonomous if job['phase'] == 'autonomous' else v.score)(job, workflow)
    for name, sha in job['source_sha256'].items():
        if v.digest(ROOT / name) != sha:
            raise ValueError(f'Source changed during execution: {name}')
    v.write(Path(job['target']).with_suffix('.experiment.json'), dict(
        proposal=job['model'] == 'SEARCH', prior=False, feedback_corrections=feedback_fixes,
        snapshot_corrections=snapshot_fixes, recall_corrections=recall_fixes))


def execute(jobs: list[dict], output: Path, budget: int) -> None:
    from src.Bayesian_state.workflows.analysis.model_0923_episodic_adaptive_task import SharedWorkerBudget
    from src.Bayesian_state.utils.parallel import parallel_job_count
    v = base.validation(); protocol = v.read(output / 'protocol.json')
    settings = protocol.get('settings')
    if settings is None:
        settings = v.read(Path(protocol['fit']) / 'nominees.json')['settings']
    predecessor = ROOT / settings['resource_predecessor'] if budget > 1 else None
    slots = SharedWorkerBudget(min(budget, len(os.sched_getaffinity(0))), predecessor)
    directory = output / 'jobs'; directory.mkdir(exist_ok=True)

    def launch(job):
        if job['phase'] == 'autonomous':
            subject = v.read(Path(job['folder']) / 'prepared.json')['subject']
            audit = ROOT / settings['audit'] / f'audit/subject_{subject}/audit.json'
            job.update(audit=str(audit), audit_sha256=v.digest(audit))
        file = directory / (job['name'] + '.json'); v.write(file, job)
        launcher = ('import runpy,sys;source,script=sys.argv[1:3];sys.path.insert(0,source);'
                    "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        with slots.slot(), tempfile.TemporaryDirectory(prefix='recalled_search_fit_') as cache:
            env.update(NUMBA_CACHE_DIR=cache, MPLCONFIGDIR=cache)
            with file.with_suffix('.log').open('x') as log:
                result = subprocess.run([sys.executable, '-c', launcher, job['runtime'], str(ROOT / ENTRY),
                    '--worker', str(file)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'Worker failed: {file.with_suffix(".log")}')

    workers = parallel_job_count(budget, len(jobs))
    print(f'{len(jobs)} jobs; {workers} workers; shared budget; numeric threads=1', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, future in enumerate(as_completed([pool.submit(launch, job) for job in jobs]), 1):
            future.result()
            if i % 16 == 0 or i == len(jobs):
                print(f'{i}/{len(jobs)} complete', flush=True)


def configure() -> None:
    refit.configure()
    original = base.source_hashes
    def hashes(v):
        result = original(v)
        result.update({name: v.digest(ROOT / name) for name in (ENTRY, PROPOSAL, BUDGET)})
        return result
    base.source_hashes = hashes
    base.execute = execute


def search(config: Path, output: Path, smoke: bool) -> None:
    from src.Bayesian_state.optimization.search.adaptive_proposals import propose_round, select_elites
    from src.Bayesian_state.utils.seeding import stable_seed
    v = base.validation(); settings = yaml.safe_load(config.read_text())
    if list(settings['arms']) != ['POOL', 'SEARCH']:
        raise ValueError('Require POOL then SEARCH arms')
    data = ROOT / settings['data']; frozen_file = ROOT / settings['warm_start_fit'] / 'nominees.json'
    frozen = v.read(frozen_file)
    if frozen['smoke'] or frozen['status'] != 'frozen_before_test':
        raise ValueError('Need scientific behavior-only warm starts')
    for name, sha in frozen['source_sha256'].items():
        if v.digest(ROOT / name) != sha:
            raise ValueError(f'Warm-start source changed: {name}')
    if output.exists():
        raise FileExistsError('Use a new recalled-search fit directory')
    banks, required = starting_banks(settings, frozen, smoke)
    seen = {key: {r['id'] for r in rows} for key, rows in banks.items()}
    output.mkdir(parents=True); hashes = base.source_hashes(v); base.archive(output, config, hashes)
    v.write(output / 'parameter_spaces.json', {arm: base.model_space(settings, arm) for arm in settings['arms']})
    v.write(output / 'protocol.json', dict(settings=settings, smoke=smoke, source_sha256=hashes,
        warm_start_nomination_sha256=v.digest(frozen_file), data_manifest_sha256=v.digest(data / 'preparation_manifest.json'),
        data_boundary='training choices only; all subjects development; oral excluded',
        initial_banks=[dict(subject=k[0], model=k[1], rows=rows) for k, rows in banks.items()],
        python=sys.version, versions={name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'pandas', 'PyYAML')}))
    screened = base.stage(settings, data, output, banks, 'screen', smoke)
    for key, rows in screened.items():
        elites = select_elites(rows, 1 if smoke else settings['elites'], exclude_key=v.H_PATH + 'max_capacity')
        proposals = propose_round(elites, base.model_space(settings, key[1]), seen[key],
            1 if smoke else settings['proposals_per_elite'],
            stable_seed(dict(base=settings['seed'], subject=key[0], role='joint_refine')), .5, .25)
        expected = 1 if smoke else settings['elites'] * settings['proposals_per_elite']
        if len(proposals) != expected:
            raise ValueError('Unequal joint refinement budget')
        banks[key] = proposals + elites; seen[key].update(r['id'] for r in proposals)
    refined = base.stage(settings, data, output, banks, 'refine', smoke)
    leading = {}
    for key in screened:
        unique = {r['id']: r for r in screened[key] + refined[key]}
        leading[key] = sorted(unique.values(), key=lambda r: (r['mean_nll'], r['id']))
    banks = refit.exchange_banks(leading, required, 2 if smoke else settings['exchange_per_arm'])
    for key, rows in banks.items():
        seen[key].update(r['id'] for r in rows)
    rows = base.stage(settings, data, output, banks, 'exchange', smoke)
    for name, count in [('selection', settings['selection_candidates']), ('confirmation', settings['confirmation_candidates'])]:
        banks = {key: base.shortlist(values, required[key], 2 if smoke else min(count, len(values)))
                 for key, values in rows.items()}
        rows = base.stage(settings, data, output, banks, name, smoke)
    for name, sha in hashes.items():
        if v.digest(ROOT / name) != sha:
            raise ValueError(f'Fitting source changed: {name}')
    nominees = [dict(subject=k[0], model=k[1], candidate=r[0], finalists=r,
        unique_candidates=len(seen[k]), reference_id=required[k]['id']) for k, r in rows.items()]
    v.write(output / 'nominees.json', dict(status='frozen_before_test', settings=settings, smoke=smoke,
        source_sha256=hashes, data=str(data.resolve()), nominees=nominees))
    v.write(output / 'completion.json', dict(status='complete', test_choices_used=False, smoke=smoke,
        nomination_sha256=v.digest(output / 'nominees.json')))
    print('History-guided-search nominees frozen before evaluation', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--phase', choices=('search', 'evaluate'))
    for name in ('config', 'fit', 'output'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--smoke', action='store_true'); args = parser.parse_args()
    if args.worker:
        worker(args.worker); return
    configure()
    if args.phase == 'search' and args.config and args.output:
        search(args.config, args.output, args.smoke)
    elif args.phase == 'evaluate' and args.fit and args.output and not args.smoke:
        base.evaluate(args.fit, args.output)
    else:
        parser.error('Use search with --config or evaluate with --fit, and a new --output')


if __name__ == '__main__':
    main()

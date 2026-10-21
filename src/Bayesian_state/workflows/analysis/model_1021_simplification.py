"""Refit REFERENCE and COMPACT with the existing training-only numerical policy.

Run --phase search --config YAML --output NEW [--smoke], then --phase evaluate
--fit FIT --output NEW. Each arm receives the same bounded candidate budget.
The shared archived cognitive runtime, particle likelihood, adaptive proposals,
stage aggregation and autonomous simulator supply the numerical implementation.
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
H = 'engine.modules.hypo_transitions_mod.kwargs.'
MODULES = 'src/Bayesian_state/model/modules/'
COMPACT = MODULES + 'hypothesis_transition/compact_evidence.py'
PROPOSAL = MODULES + 'hypothesis_transition/episodic_proposal.py'
ONLINE = MODULES + 'hypothesis_transition/online_evidence_belief.py'
RETAIN = MODULES + 'hypothesis_transition/reentry_precision.py'


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


refit = load('compact_shared_episodic_refit', 'src/Bayesian_state/workflows/analysis/model_0923_episodic_refit.py')
base = refit.base
io = base.validation()


def fixed_coordinates(arm: str, condition: int) -> dict:
    result = {H + 'transfer_scale': .01}
    if arm == 'COMPACT':
        result.update({H + 'exit_rate': 0., H + 'execution_inertia': 0., H + 'local_structure_weight': 1.,
                       H + 'search_controller.global_search_failure_gain': 0.})
        if condition != 3:
            result[base.MEMORY] = 1.
    elif arm != 'REFERENCE':
        raise ValueError('Unknown simplification arm')
    return result


def model_space(settings: dict, arm: str, condition: int) -> dict:
    space = base.model_space(settings, arm)
    for key, value in fixed_coordinates(arm, condition).items():
        if value not in space[key]:
            raise ValueError('Closed coordinate is outside the registered support')
        space[key] = [value]
    return space


def source_hashes(v) -> dict:
    result = ORIGINAL_HASHES(v)
    for path in (ENTRY, COMPACT, PROPOSAL, ONLINE, RETAIN, refit.ENTRY,
                 refit.GUARD, refit.HELPER, refit.INPUTS):
        result[path] = v.digest(ROOT / path)
    return result


ORIGINAL_HASHES = base.source_hashes


def worker(file: Path) -> None:
    job = io.read(file)
    for path, sha in job['source_sha256'].items():
        if io.digest(ROOT / path) != sha:
            raise ValueError('Executing simplification source changed: ' + path)
    arm_name = job['model']
    meta = io.read(Path(job['folder']) / 'prepared.json')
    for key, value in fixed_coordinates(arm_name, meta['condition']).items():
        if job['candidate']['hyperparams'][key] != value:
            raise ValueError('Closed coordinate differs from the declared mechanism')
    expected = 'evidence' if arm_name == 'COMPACT' else 'evidence_mean'
    if job['mechanism'] != dict(kind='E', decay=.95, capacity=expected, search='accumulate'):
        raise ValueError('Unexpected mechanism configuration')
    workflow = io.intervention_workflow()
    workflow.install_transition()
    diagnosis = workflow.cohort().load_diagnosis()
    feedback = base.load('src.Bayesian_state.model.modules.hypothesis_transition.feedback_guided', base.FEEDBACK)
    precision = base.load('src.Bayesian_state.model.modules.precision_interventions', base.PRECISION)
    guard = base.load('src.Bayesian_state.model.modules.pairing_snapshot_roundoff', refit.GUARD)
    from src.Bayesian_state.model.modules.beta import BetaModule
    from src.Bayesian_state.model.modules.pairing_memory import HierarchicalPairingMemoryModule
    cls = sys.modules['src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search'].MechanisticRuleSearchHypothesisTransitionModule
    with ExitStack() as stack:
        stack.enter_context(diagnosis.single_threaded_processes())
        fixes = stack.enter_context(feedback.retention_roundoff_guard(cls))
        snapshot = stack.enter_context(guard.pairing_snapshot_roundoff_guard(HierarchicalPairingMemoryModule))
        arm = job['arm_settings']
        if (arm['precision'], arm['upper'], arm['rate_scale']) != ('fast', 25., 10.):
            raise ValueError('Both arms require ADD10 precision')
        stack.enter_context(precision.precision_intervention(BetaModule, mode='fast', upper=25.))
        if arm_name == 'REFERENCE':
            history = base.load('src.Bayesian_state.model.modules.hypothesis_transition.episodic_admission', refit.HELPER)
            proposal = base.load('src.Bayesian_state.model.modules.hypothesis_transition.episodic_proposal', PROPOSAL)
            retain = base.load('src.Bayesian_state.model.modules.hypothesis_transition.reentry_precision', RETAIN)
            online = base.load('src.Bayesian_state.model.modules.hypothesis_transition.online_evidence_belief', ONLINE)
            stack.enter_context(history.episodic_admission(cls, mode='POOL'))
            stack.enter_context(proposal.episodic_proposal(cls, proposal=True))
            stack.enter_context(retain.reentry_precision(cls))
            events = stack.enter_context(online.online_evidence_belief(cls, enabled=True))
        else:
            compact = base.load('src.Bayesian_state.model.modules.hypothesis_transition.compact_evidence', COMPACT)
            events = stack.enter_context(compact.compact_evidence(cls))
        (refit.autonomous if job['phase'] == 'autonomous' else io.score)(job, workflow)
        audit = dict(events) if arm_name == 'COMPACT' else dict(reconstructions=len(events))
    io.write(Path(job['target']).with_suffix('.mechanism.json'), dict(
        model=arm_name, audit=audit, feedback_roundoff=fixes, pairing_roundoff=snapshot,
        fixed_coordinates=fixed_coordinates(arm_name, meta['condition']), oral_used_for_fitting=False))


def execute(jobs: list[dict], output: Path, budget: int) -> None:
    """One outer layer of fresh processes, largest ready jobs dispatched first."""
    settings = io.read(output / 'protocol.json')
    if 'settings' not in settings:
        settings = io.read(Path(settings['fit']) / 'nominees.json')
    settings = settings['settings']
    directory = output / 'jobs'
    directory.mkdir(exist_ok=True)

    def launch(original):
        job = deepcopy(original)
        job['mechanism']['capacity'] = 'evidence' if job['model'] == 'COMPACT' else 'evidence_mean'
        if job['phase'] == 'autonomous':
            subject = io.read(Path(job['folder']) / 'prepared.json')['subject']
            audit = ROOT / settings['audit'] / f'audit/subject_{subject}/audit.json'
            job.update(audit=str(audit), audit_sha256=io.digest(audit))
        file = directory / (job['name'] + '.json')
        io.write(file, job)
        launcher = ('import runpy,sys;runtime,script=sys.argv[1:3];sys.path.insert(0,runtime);'
                    "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")
        with tempfile.TemporaryDirectory(prefix='compact_h_worker_') as cache:
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                       NUMBA_NUM_THREADS='1', NUMBA_CACHE_DIR=cache, MPLCONFIGDIR=cache)
            with file.with_suffix('.log').open('x') as stream:
                result = subprocess.run([sys.executable, '-c', launcher, job['runtime'], str(ROOT / ENTRY),
                    '--worker', str(file)], env=env, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('Worker failed: ' + str(file.with_suffix('.log')))

    workers = min(budget, len(os.sched_getaffinity(0)), len(jobs))
    print(f'{len(jobs)} jobs; {workers} workers; numeric threads=1', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, future in enumerate(as_completed([pool.submit(launch, job) for job in jobs]), 1):
            future.result()
            if i % 16 == 0 or i == len(jobs):
                print(f'{i}/{len(jobs)} complete', flush=True)


def configure():
    base.source_hashes = source_hashes
    base.execute = execute


def starting_banks(settings: dict, smoke: bool) -> tuple[dict, dict, dict, dict]:
    from src.Bayesian_state.optimization.search.adaptive_proposals import candidate, initial_points
    from src.Bayesian_state.utils.seeding import stable_seed
    banks, required, spaces, inputs = {}, {}, {}, {}
    for subject in settings['subjects'][:1] if smoke else settings['subjects']:
        folder = ROOT / settings['data'] / f'subject_{subject}'
        meta = io.read(folder / 'prepared.json')
        source = ROOT / settings['warm_start_fits'][subject]
        nomination = source / 'nominees.json'
        previous = io.read(nomination)
        done = io.read(source / 'completion.json')
        if (previous['smoke'] or previous['status'] != 'frozen_before_test' or done['status'] != 'complete'
                or done['test_choices_used'] is not False or done['nomination_sha256'] != io.digest(nomination)):
            raise ValueError('Warm start must be a completed behavior-training nomination')
        inputs[str(nomination.resolve())] = io.digest(nomination)
        inputs[str((source / 'completion.json').resolve())] = io.digest(source / 'completion.json')
        training = folder / 'training.npz'
        if io.digest(training) != meta['array_sha256']['training.npz']:
            raise ValueError('Training boundary or arrays changed')
        inputs[str(training.resolve())] = io.digest(training)
        anchor, = [row['candidate']['hyperparams'] for row in previous['nominees']
                   if row['subject'] == subject and row['model'] == 'SEARCH']
        full_space = model_space(settings, 'REFERENCE', meta['condition'])
        count = 4 if smoke else settings['initial_candidates']
        starts = initial_points(full_space, anchor, count * 4,
            stable_seed(dict(base=settings['seed'], subject=subject, role='shared_initial')),
            cover_key=H + 'max_capacity')
        for arm in settings['arms']:
            key = subject, arm
            space = spaces[key] = model_space(settings, arm, meta['condition'])
            rows = {}
            for i, point in enumerate([anchor] + starts):
                point = dict(point, **fixed_coordinates(arm, meta['condition']))
                if set(point) != set(space) or any(point[k] not in values for k, values in space.items()):
                    raise ValueError('Starting point outside registered support')
                row = candidate(point, 'previous_training_projection' if i == 0 else 'shared_support_start')
                rows.setdefault(row['id'], row)
                if len(rows) == count:
                    break
            if len(rows) != count:
                raise ValueError('Insufficient distinct starts')
            banks[key] = list(rows.values())
            required[key] = banks[key][0]
    return banks, required, spaces, inputs


def search(config: Path, output: Path, smoke: bool) -> None:
    from src.Bayesian_state.optimization.search.adaptive_proposals import propose_round, select_elites
    from src.Bayesian_state.utils.seeding import stable_seed
    if output.exists():
        raise FileExistsError('Use a new simplification output directory')
    settings = yaml.safe_load(config.read_text())
    if (list(settings['arms']) != ['REFERENCE', 'COMPACT'] or not 1 <= settings['parallel_budget'] <= 128
            or settings['initial_candidates'] != 40 or settings['scope'] != 'simplification_development'):
        raise ValueError('Require the two registered arms and bounded development policy')
    configure()
    banks, required, spaces, inputs = starting_banks(settings, smoke)
    seen = {key: {r['id'] for r in rows} for key, rows in banks.items()}
    output.mkdir(parents=True)
    hashes = source_hashes(io)
    base.archive(output, config, hashes)
    io.write(output / 'protocol.json', dict(settings=settings, smoke=smoke, source_sha256=hashes,
        inputs_sha256=inputs, data_boundary='Original training choices only; development later trials reserved for evaluation',
        initial_banks=[dict(subject=k[0], model=k[1], rows=v) for k, v in banks.items()],
        parameter_spaces=[dict(subject=k[0], model=k[1], space=v,
            variable_coordinates=sum(len(values) > 1 for values in v.values())) for k, v in spaces.items()],
        python=sys.version, versions={n: importlib.metadata.version(n) for n in ('numpy', 'scipy', 'pandas', 'PyYAML')}))
    data = ROOT / settings['data']
    rows = base.stage(settings, data, output, banks, 'screen', smoke)
    for key, values in rows.items():
        elites = select_elites(values, 1 if smoke else settings['elites'], exclude_key=H + 'max_capacity')
        proposed = propose_round(elites, spaces[key], seen[key], 1 if smoke else settings['proposals_per_elite'],
            stable_seed(dict(base=settings['seed'], subject=key[0], role='joint_refine')), .5, .25)
        if len(proposed) != (1 if smoke else settings['elites'] * settings['proposals_per_elite']):
            raise ValueError('Unequal refinement budget')
        banks[key] = proposed + elites
        seen[key].update(r['id'] for r in proposed)
    rows = base.stage(settings, data, output, banks, 'refine', smoke)
    for stage, count in [('selection', settings['selection_candidates']),
                         ('confirmation', settings['confirmation_candidates'])]:
        banks = {k: base.shortlist(v, required[k], 2 if smoke else count) for k, v in rows.items()}
        rows = base.stage(settings, data, output, banks, stage, smoke)
    nominees = [dict(subject=k[0], model=k[1], candidate=v[0], finalists=v,
                     unique_candidates=len(seen[k]), reference_id=required[k]['id']) for k, v in rows.items()]
    io.write(output / 'nominees.json', dict(status='frozen_before_test', settings=settings, smoke=smoke,
        source_sha256=hashes, data=str(data.resolve()), nominees=nominees))
    io.write(output / 'completion.json', dict(status='complete', test_choices_used=False, smoke=smoke,
        nomination_sha256=io.digest(output / 'nominees.json')))
    print('Simplification nominees frozen before evaluation', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('search', 'evaluate'))
    for name in ('config', 'fit', 'output', 'worker'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.phase == 'search' and args.config and args.output:
        search(args.config, args.output, args.smoke)
    elif args.phase == 'evaluate' and args.fit and args.output and not args.smoke:
        configure()
        base.evaluate(args.fit, args.output)
    else:
        parser.error('Use search with --config, evaluate with --fit, and a new --output')


if __name__ == '__main__':
    main()

"""Jointly refit COMPACT and precision transfer, and audit recovery search.

Use --config YAML --output NEW [--smoke]. The two human mechanisms share starts,
parameter supports, scoring seeds and numerical budgets. Two bounded rounds
cover coordinate rays and joint moves. One shared pool launches independent
single-thread filter processes across all subjects and recovery cases.

The internal numerical arm remains COMPACT; compact_variant records whether
current precision transfers to newcomers. Frozen human parameters are not
evaluated on later choices here. Recovery truth is opened after all fits freeze.
Existing model equations, proposal routines and score aggregation are reused.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[4]
if __package__:
    from . import model_1021_simplification_recovery as recovery
    from ...optimization.search.adaptive_proposals import candidate, ray_proposals, propose_round, select_elites
    from ...utils.seeding import stable_seed
    driver = recovery.driver
else:
    # Numerical workers run against the existing archived cognitive package.
    # Load the current routing script by file without replacing that package.
    spec = importlib.util.spec_from_file_location('model1022_compact_driver',
        ROOT/'src/Bayesian_state/workflows/analysis/model_1021_simplification.py')
    driver = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = driver
    spec.loader.exec_module(driver)
read, write, digest = driver.io.read, driver.io.write, driver.io.digest
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
ENTRY_PROBE = 'src/Bayesian_state/workflows/analysis/model_1021_entry_precision_probe.py'
ENTRY_PRECISION = 'src/Bayesian_state/model/modules/hypothesis_transition/entry_precision.py'
CAPTURE = 'src/Bayesian_state/workflows/analysis/model_1021_simplification_recovery_capture.py'


def sources(v) -> dict:
    hashes = recovery.recovery_hashes(v)
    for name in (ENTRY, ENTRY_PROBE, ENTRY_PRECISION,
                 'src/Bayesian_state/model/modules/hypothesis_transition/centered_evidence.py',
                 'src/Bayesian_state/optimization/search/adaptive_proposals.py'):
        hashes[name] = digest(ROOT/name)
    return hashes


def worker(file: Path) -> None:
    job = read(file)
    variant = job['compact_variant']
    if variant not in ('COMPACT', 'TRANSFER') or job['model'] != 'COMPACT':
        raise ValueError('Unknown compact cognitive variant')
    model_worker = driver.worker
    if variant == 'TRANSFER':
        if job.get('entry_control') != 'PRECISION':
            raise ValueError('Transfer requires the declared current-precision adapter')
        probe = driver.load('model1022_precision_worker', ENTRY_PROBE)
        model_worker = probe.worker
    if 'state_capture_target' in job:
        capture = driver.load('model1022_recovery_capture', CAPTURE)
        capture.worker(file, model_worker=model_worker)
    else:
        model_worker(file)
    write(Path(job['target']).with_suffix('.variant.json'), dict(
        compact_variant=variant, case_id=job['case_id'], oral_used_for_fitting=False))


def configure_executor(pool: ThreadPoolExecutor) -> None:
    driver.base.source_hashes = sources

    def launch(original: dict, folder: Path):
        settings = read(folder/'protocol.json')['settings']
        job = deepcopy(original)
        job['mechanism']['capacity'] = 'evidence'
        job.update(compact_variant=settings['compact_variant'], case_id=settings['case_id'])
        if job['compact_variant'] == 'TRANSFER':
            job['entry_control'] = 'PRECISION'
        if job['phase'] != 'training':
            raise ValueError('This entry fits training behavior only')
        if job['name'].startswith('confirmation_') and 'recovery_target_rule' in settings:
            job.update(target_rule=settings['recovery_target_rule'],
                state_capture_target=str((folder/'state_capture'/job['name']).resolve()),
                target_capture_sources={name: digest(ROOT/name)
                    for name in (recovery.CAPTURE, recovery.PF, recovery.COLLECTOR)})
        directory = folder/'jobs'
        directory.mkdir(exist_ok=True)
        file = directory/(job['name']+'.json')
        write(file, job)
        launcher = ('import runpy,sys;runtime,script=sys.argv[1:3];sys.path.insert(0,runtime);'
                    "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")
        with tempfile.TemporaryDirectory(prefix='compact_refit_') as cache:
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', NUMBA_NUM_THREADS='1',
                       NUMBA_CACHE_DIR=cache, MPLCONFIGDIR=cache)
            with file.with_suffix('.log').open('x') as log:
                result = subprocess.run([sys.executable, '-c', launcher, job['runtime'], str(ROOT/ENTRY),
                    '--worker', str(file)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('Refit worker failed: '+str(file.with_suffix('.log')))

    def execute(jobs, folder, budget):
        from .model_0923_h_training_audit import combine
        expanded, groups = recovery.split_filter_jobs(jobs)
        for future in as_completed([pool.submit(launch, job, folder) for job in expanded]):
            future.result()
        for original, children in groups:
            combine(original, children)

    driver.base.execute = execute


def starting_bank(space: dict, anchors: list[dict], count: int, seed: int) -> list[dict]:
    """Preserve behavioral warm starts, then add support-only neutral starts."""
    bank = {}
    for point in anchors:
        if set(point) != set(space) or any(point[k] not in v for k, v in space.items()):
            raise ValueError('Warm start lies outside declared parameter support')
        row = candidate(point, 'previous_training_fit')
        bank.setdefault(row['id'], row)
    for row in recovery.neutral_starts(space, max(count*2, 40), seed):
        if len(bank) >= count:
            break
        bank.setdefault(row['id'], row)
    if len(bank) != count:
        raise ValueError('Could not create the declared number of starts')
    return list(bank.values())


def refine_bank(rows: list[dict], space: dict, seen: set[str], settings: dict, round_index: int,
                smoke: bool) -> list[dict]:
    elites = select_elites(rows, 1 if smoke else settings['elites'], exclude_key=driver.H+'max_capacity')
    seed = stable_seed(dict(base=settings['seed'], role='coordinate_and_joint', round=round_index))
    rays = ray_proposals(elites, space, seen, 1 if smoke else settings['rays_per_elite'], seed, 'coordinate_rays')
    joints = propose_round(elites, space, seen|{r['id'] for r in rays},
                           1 if smoke else settings['joint_per_elite'], seed+1, 0., 0.)
    expected = 2 if smoke else settings['elites']*(settings['rays_per_elite']+settings['joint_per_elite'])
    result = rays+joints
    if len(result) != expected or len({r['id'] for r in result}) != expected:
        raise ValueError('Declared bounded proposal quota was not met')
    return result


def configure_settings(original: dict, policy: dict, *, seed: int, variant: str, case_id: str) -> dict:
    settings = deepcopy(original)
    for key in ('protocol', 'parallel_budget', 'initial_candidates', 'rounds', 'elites', 'rays_per_elite',
                'joint_per_elite', 'carry_candidates', 'selection_candidates', 'confirmation_candidates',
                'budgets', 'autonomous_paths', 'autonomous_chunk', 'scope'):
        settings[key] = deepcopy(policy[key])
    settings.update(seed=seed, compact_variant=variant, case_id=case_id, particle_overrides={})
    settings['arms'] = {'COMPACT': deepcopy(original['arms']['COMPACT'])}
    return settings


def prepare(policy: dict, output: Path, smoke: bool) -> tuple[list[dict], Path]:
    previous = read(ROOT/policy['previous_fit']/'nominees.json')
    cases, bindings = [], {}
    for cohort, specification in policy['cohorts'].items():
        warm_file = ROOT/specification['warm_fit']/'nominees.json'
        warm = read(warm_file)
        done_file = warm_file.with_name('completion.json')
        done = read(done_file)
        if (warm['smoke'] or warm['status'] != 'frozen_before_test' or done['status'] != 'complete'
                or done['test_choices_used'] is not False or done['nomination_sha256'] != digest(warm_file)):
            raise ValueError('Human warm starts require frozen scientific training fits')
        bindings[str(warm_file.resolve())] = digest(warm_file)
        bindings[str(done_file.resolve())] = digest(done_file)
        subjects = specification['subjects'][:1] if smoke else specification['subjects']
        if smoke and cohort != 'development':
            continue
        for subject in subjects:
            folder = ROOT/specification['data']/f'subject_{subject}'
            meta = read(folder/'prepared.json')
            if digest(folder/'training.npz') != meta['array_sha256']['training.npz']:
                raise ValueError('Training data boundary changed')
            bindings[str((folder/'training.npz').resolve())] = digest(folder/'training.npz')
            anchor, = [r['candidate']['hyperparams'] for r in warm['nominees']
                       if (r['subject'], r['model']) == (subject, specification['warm_arm'])]
            anchor = dict(anchor, **driver.fixed_coordinates('COMPACT', meta['condition']))
            space = driver.model_space(previous['settings'], 'COMPACT', meta['condition'])
            seed = stable_seed(dict(base=policy['seed'], role='human_fit', subject=subject))
            starts = starting_bank(space, [anchor], 4 if smoke else policy['initial_candidates'], seed)
            for variant in policy['variants']:
                case_id = f'S{subject}_{variant}'
                settings = configure_settings(previous['settings'], policy, seed=seed, variant=variant, case_id=case_id)
                settings.update(data=specification['data'], audit=specification['audit'], subjects=[subject])
                cases.append(dict(case_id=case_id, subject=subject, condition=meta['condition'], cohort=cohort,
                    variant=variant, settings=settings, space=space, starts=deepcopy(starts), required=starts[0],
                    kind='human', output=str((output/'human_fit'/case_id).resolve())))
    original = ROOT/policy['original_recovery']/'recovery_prepared'
    recovery_cases = recovery.checked_cases(original)
    prepared = output/'recovery_prepared'
    prepared.mkdir()
    declared = []
    for old in recovery_cases[:1] if smoke else recovery_cases:
        old_fit = ROOT/policy['previous_recovery']/'fit'/old['case_id']
        old_nominee = read(old_fit/'nominees.json')['nominee']['candidate']
        scored = read(old_fit/'screen_completion.json')['groups'][0]['rows']
        elites = select_elites(scored, 7, exclude_key=driver.H+'max_capacity')
        anchors = [old_nominee['hyperparams']]+[r['hyperparams'] for r in elites]
        if smoke:
            anchors = anchors[:1]
        seed = stable_seed(dict(base=policy['seed'], role='optimizer_recovery', case_id=old['case_id']))
        settings = configure_settings(old['settings'], policy, seed=seed, variant='COMPACT', case_id=old['case_id'])
        starts = starting_bank(old['space'], anchors, 4 if smoke else policy['initial_candidates'], seed)
        case = dict(deepcopy(old), settings=settings, starts=starts, required=starts[0], kind='recovery',
                    variant='COMPACT', output=str((output/'recovery_fit'/old['case_id']).resolve()))
        file = prepared/old['case_id']/'case.json'
        file.parent.mkdir(); write(file, case)
        declared.append(dict(case_id=old['case_id'], file=str(file.resolve()), sha256=digest(file)))
        cases.append(case)
        for name in ('nominees.json', 'screen_completion.json'):
            bindings[str((old_fit/name).resolve())] = digest(old_fit/name)
    write(prepared/'protocol.json', dict(cases=declared, original=str(original.resolve()),
        truth_policy='Copied opaquely, opened after all fits freeze; starts depend only on past behavior scores'))
    shutil.copyfile(original/'truths.json', prepared/'truths.json')
    write(prepared/'completion.json', dict(status='complete', protocol_sha256=digest(prepared/'protocol.json'),
        truths_sha256=digest(prepared/'truths.json'), cases=len(declared)))
    write(output/'cases.json', dict(cases=cases, inputs_sha256=bindings, oral_used=False,
        human_later_choices_used=False, recovery_truth_used_for_proposals=False))
    return cases, prepared


def fit_case(case: dict, smoke: bool) -> dict:
    settings, output = case['settings'], Path(case['output'])
    output.mkdir(parents=True)
    hashes = sources(driver.io)
    write(output/'protocol.json', dict(settings=settings, case_id=case['case_id'], source_sha256=hashes,
        space=case['space'], smoke=smoke, cohort=case.get('cohort', 'synthetic_recovery')))
    key = case['subject'], 'COMPACT'
    seen = {r['id'] for r in case['starts']}
    rows = driver.base.stage(settings, ROOT/settings['data'], output, {key: case['starts']}, 'screen', smoke)[key]
    print(case['case_id']+': screen complete', flush=True)
    for index in range(1, (1 if smoke else settings['rounds'])+1):
        new = refine_bank(rows, case['space'], seen, settings, index, smoke)
        seen.update(r['id'] for r in new)
        carry = driver.base.shortlist(rows, case['required'], 2 if smoke else settings['carry_candidates'])
        rows = driver.base.stage(settings, ROOT/settings['data'], output, {key: new+carry}, f'refine_{index}', smoke)[key]
        print(case['case_id']+f': coordinate/joint round {index} complete', flush=True)
    for stage, count in [('selection', settings['selection_candidates']), ('confirmation', settings['confirmation_candidates'])]:
        bank = driver.base.shortlist(rows, case['required'], 2 if smoke else count)
        rows = driver.base.stage(settings, ROOT/settings['data'], output, {key: bank}, stage, smoke)[key]
    nominee = dict(case_id=case['case_id'], subject=case['subject'], model='COMPACT', compact_variant=case['variant'],
        candidate=rows[0], finalists=rows, unique_candidates=len(seen), reference_id=case['required']['id'])
    if case['kind'] == 'recovery':
        nominee['replicate'] = case['replicate']
    write(output/'nominees.json', dict(status='frozen_before_test' if case['kind']=='human' else 'frozen_before_truth',
        settings=settings, smoke=smoke, source_sha256=hashes, data=str((ROOT/settings['data']).resolve()),
        nominee=nominee, nominees=[nominee]))
    write(output/'completion.json', dict(status='complete', smoke=smoke, nomination_sha256=digest(output/'nominees.json'),
        test_choices_used=False, truths_opened=False))
    print(case['case_id']+': parameters frozen', flush=True)
    return nominee


def run(config: Path, output: Path, smoke: bool) -> None:
    if output.exists():
        raise FileExistsError('Use a new refit study directory')
    policy = yaml.safe_load(config.read_text())
    if policy['variants'] != ['COMPACT', 'TRANSFER'] or not 1 <= policy['parallel_budget'] <= 128:
        raise ValueError('Require the two registered variants and at most 128 workers')
    output.mkdir(parents=True)
    driver.base.archive(output, config, sources(driver.io))
    cases, prepared = prepare(policy, output, smoke)
    workers = 1 if smoke else min(policy['parallel_budget'], len(os.sched_getaffinity(0)))
    write(output/'protocol.json', dict(settings=policy, smoke=smoke, workers=workers, source_sha256=sources(driver.io),
        python=sys.version, versions={n: importlib.metadata.version(n) for n in ('numpy','scipy','pandas','PyYAML')},
        purpose='Equal-budget simplification comparison and bounded optimizer diagnosis; no oral fitting'))
    recovery_fit = output/'recovery_fit'
    recovery_fit.mkdir(exist_ok=True)
    write(recovery_fit/'protocol.json', dict(prepared=str(prepared.resolve()), smoke=smoke,
        workers=workers, source_sha256=sources(driver.io)))
    print(f'{len(cases)} cases; {workers} shared numeric workers; one thread per filter', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as numerical:
        configure_executor(numerical)
        with ThreadPoolExecutor(max_workers=1 if smoke else len(cases)) as controllers:
            futures = [controllers.submit(fit_case, case, smoke) for case in cases]
            nominees = [future.result() for future in futures]
        recovered = [r for r in nominees if r['case_id'] in {c['case_id'] for c in cases if c['kind']=='recovery'}]
        write(recovery_fit/'nominees.json', dict(status='frozen_before_truth', smoke=smoke, nominees=recovered))
        write(recovery_fit/'completion.json', dict(status='complete', smoke=smoke, truths_opened=False,
            cases=len(recovered), nomination_sha256=digest(recovery_fit/'nominees.json')))
        write(output/'human_nominees.json', dict(status='frozen_before_test', smoke=smoke,
            nominees=[r for r in nominees if r not in recovered]))
        # Truth uses the original COMPACT mechanism and the same numerical pool.
        recovery.score_truth(prepared, recovery_fit, output/'recovery_scores', numerical_pool=numerical)
    from ...evaluation.model_1021_simplification_recovery_states import summarize
    summarize(prepared, recovery_fit, output/'recovery_scores', output/'recovery_states')
    write(output/'completion.json', dict(status='complete', smoke=smoke, human_cases=len(nominees)-len(recovered),
        recovery_cases=len(recovered), human_evaluation='pending', original_results_preserved=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'output', 'worker'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.config and args.output:
        run(args.config, args.output, args.smoke)
    else:
        parser.error('Use --config and --output, or internal --worker')

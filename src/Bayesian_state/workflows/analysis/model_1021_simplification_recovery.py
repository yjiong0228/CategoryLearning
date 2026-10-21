"""Joint COMPACT recovery with support-only starts and a shared worker queue.

Prepare from a completed simplification --evaluation into a new --output.
Search the --prepared cases into another new --output [--smoke], then score
the --fit against its generating points with --phase score-truth. Two lowest
generation seeds at each of subjects 205 and 306 define four fixed cases. All
eight/nine variable coordinates are reestimated from synthetic training choices.
The optimizer does not read truths.json. Generating points are scored only
after estimates freeze, using the same independent confirmation seed bank.

Case controllers are waiting threads; all numerical subprocesses share one
128-worker queue. Faster cases advance without waiting for slower cases.
Post-freeze generating-point scores also share this queue across cases.
Each independent filter seed is one process task; its original seed order is
preserved when pooling predictions. Completion records count actual filters.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from . import model_1021_simplification as driver
from .model_0923_h_objective_recovery import fixture
from ...evaluation.model_0923_h_validation import checked_arrays, nll, jackknife_scores, jackknife_se
from ...optimization.search.adaptive_proposals import candidate, initial_points, propose_round, select_elites
from ...utils.seeding import stable_seed

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
CAPTURE = 'src/Bayesian_state/workflows/analysis/model_1021_simplification_recovery_capture.py'
PF = 'src/Bayesian_state/inference/backends/particle_filter.py'
COLLECTOR = 'src/Bayesian_state/workflows/analysis/model_0923_target_state_recovery.py'
read, write, digest = driver.io.read, driver.io.write, driver.io.digest


def neutral_starts(space: dict, count: int, seed: int) -> list[dict]:
    """Use only the declared support and random seed; no generating estimate."""
    anchor = {k: values[len(values)//2] for k, values in space.items()}
    points = initial_points(space, anchor, max(count, len(space[driver.H+'max_capacity'])+2), seed,
                            cover_key=driver.H+'max_capacity')
    ordered = ([anchor] + [p for p in points if p != anchor])[:count]
    if len(ordered) != count or len({candidate(p, 'check')['id'] for p in ordered}) != count:
        raise ValueError('Insufficient neutral candidates')
    return [candidate(p, 'support_only_start') for p in ordered]


def prepare(evaluation: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new recovery preparation directory')
    declared = read(evaluation/'protocol.json')
    fit = Path(declared['fit'])
    frozen = read(fit/'nominees.json')
    if (read(evaluation/'completion.json')['status'] != 'complete' or frozen['smoke']
            or declared['nomination_sha256'] != digest(fit/'nominees.json')):
        raise ValueError('Require complete scientific generation at frozen parameters')
    settings = deepcopy(frozen['settings'])
    if settings['subjects'] != [108, 205, 306]:
        raise ValueError('This registered recovery uses development points 205 and 306')
    output.mkdir(parents=True)
    cases, truths, bindings = [], [], {}
    for subject in (205, 306):
        chosen, = [r for r in frozen['nominees'] if (r['subject'], r['model']) == (subject, 'COMPACT')]
        row, = [r for r in read(evaluation/'autonomous/evaluation.json')['rows']
                if (r['subject'], r['model']) == (subject, 'COMPACT')]
        if row['candidate'] != chosen['candidate']:
            raise ValueError('Generation differs from the frozen COMPACT point')
        bank = checked_arrays(evaluation/'autonomous', row)
        if len(bank['seeds']) != settings['autonomous_paths'] or len(set(bank['seeds'])) != len(bank['seeds']):
            raise ValueError('Incomplete or duplicate generation seeds')
        original = Path(frozen['data'])/f'subject_{subject}'
        meta = read(original/'prepared.json')
        cut = meta['training_trials']
        if digest(original/'training.npz') != meta['array_sha256']['training.npz']:
            raise ValueError('Original training input changed')
        with np.load(original/'training.npz') as z:
            physical = {k: z[k].copy() for k in ('stimulus', 'categories')}
        np.testing.assert_array_equal(physical['categories'], bank['categories'][:cut])
        source_job = next(read(p) for p in (evaluation/'jobs').glob(f'AUTO_evaluate_S{subject}_COMPACT_*.json'))
        for name, sha in source_job['source_sha256'].items():
            if digest(ROOT/name) != sha:
                raise ValueError('Generating implementation changed')
        space = driver.model_space(settings, 'COMPACT', meta['condition'])
        point = chosen['candidate']['hyperparams']
        if set(point) != set(space) or any(point[k] not in levels for k, levels in space.items()):
            raise ValueError('Generating point is outside the registered COMPACT support')
        for replicate, index in enumerate(np.argsort(bank['seeds'])[:2]):
            case_id = f'S{subject}_R{replicate}'
            folder = output/case_id/'data'/f'subject_{subject}'
            fixture(original, folder, dict(physical, choices=bank['choices'][index, :cut],
                                          feedback=bank['feedback'][index, :cut]))
            write(folder.parent/'preparation_manifest.json', dict(status='complete', subjects=[subject],
                settings=dict(source_fit=str(Path(source_job['runtime']).parent)), data_role='synthetic_training_only'))
            own = deepcopy(settings)
            own.pop('warm_start_fits')
            own.update(subjects=[subject], arms=dict(COMPACT=own['arms']['COMPACT']), data=str(folder.parent.resolve()),
                protocol='model1021_compact_recovery_v1', scope='synthetic_joint_recovery',
                recovery_target_rule=int(row['target_rule']),
                seed=stable_seed(dict(base=2026102192, subject=subject, replicate=replicate, role='recovery_fit')))
            starts = neutral_starts(space, 40, stable_seed(dict(base=own['seed'], role='neutral_starts')))
            case = dict(case_id=case_id, subject=subject, condition=meta['condition'], replicate=replicate,
                settings=own, space=space, starts=starts, generating_parameters_used_for_starts=False,
                generation_seed=int(bank['seeds'][index]), trials=cut,
                arrays_sha256={n: digest(folder/n) for n in ('training.npz', 'sealed_test.npz', 'engine.json', 'prepared.json')})
            write(output/case_id/'case.json', case)
            cases.append(dict(case_id=case_id, file=str((output/case_id/'case.json').resolve()),
                              sha256=digest(output/case_id/'case.json')))
            state_file = output/case_id/'states_truth.npz'
            with state_file.open('xb') as stream:
                np.savez_compressed(stream, **{k: bank[k][index, :cut] for k in ('present', 'contribution', 'beta')})
            truths.append(dict(case_id=case_id, subject=subject, replicate=replicate, model='COMPACT',
                point=chosen['candidate']['hyperparams'], generation_seed=int(bank['seeds'][index]),
                states_truth=str(state_file.resolve()), states_truth_sha256=digest(state_file)))
        array = evaluation/'autonomous/evaluate'/f'subject_{subject}'/'COMPACT'/(row['candidate']['id']+'.npz')
        bindings[str(array.resolve())] = digest(array)
    write(output/'protocol.json', dict(cases=cases, generation_selection='Two lowest saved seeds per point; no outcome selection',
        subjects=[205, 306], estimate='All variable COMPACT coordinates jointly', oral_used=False,
        scope='Two fitted points, four synthetic sequences; limited recovery coverage'))
    write(output/'truths.json', dict(rows=truths, use='After estimates freeze only'))
    write(output/'completion.json', dict(status='complete', cases=4, inputs_sha256=bindings,
        protocol_sha256=digest(output/'protocol.json'), truths_sha256=digest(output/'truths.json'), new_generation_runs=0))


def recovery_hashes(v) -> dict:
    hashes = driver.source_hashes(v)
    for name in (ENTRY, CAPTURE, PF, COLLECTOR):
        hashes[name] = digest(ROOT/name)
    return hashes


def split_filter_jobs(jobs: list[dict]) -> tuple[list[dict], list[tuple[dict, list[dict]]]]:
    """Give each independent seed its own queue slot; retain original seed order."""
    expanded, groups = [], []
    for original in jobs:
        if not original['seeds']:
            raise ValueError('A scoring job requires at least one filter seed')
        if len(original['seeds']) == 1:
            expanded.append(original)
            continue
        children = []
        target = Path(original['target'])
        for index, seed in enumerate(original['seeds']):
            child = deepcopy(original)
            suffix = f'_F{index:03d}'
            child.update(seeds=[seed], name=original['name']+suffix,
                         target=str(target.parent/'filter_chunks'/(target.name+suffix)))
            children.append(child)
        expanded.extend(children)
        groups.append((original, children))
    return expanded, groups


def configure_executor(pool: ThreadPoolExecutor) -> None:
    """All cases dispatch into this one numerical process queue."""
    driver.base.source_hashes = recovery_hashes

    def launch(original: dict, folder: Path):
        job = deepcopy(original)
        job['mechanism']['capacity'] = 'evidence'
        if job['model'] != 'COMPACT' or job['phase'] != 'training':
            raise ValueError('Recovery fits use COMPACT synthetic training only')
        entry = driver.ENTRY
        if job['name'].startswith('confirmation_'):
            settings = read(folder/'protocol.json')['settings']
            job.update(target_rule=settings['recovery_target_rule'],
                state_capture_target=str((folder/'state_capture'/job['name']).resolve()),
                target_capture_sources={name: digest(ROOT/name) for name in (CAPTURE, PF, COLLECTOR)})
            entry = CAPTURE
        directory = folder/'jobs'
        directory.mkdir(exist_ok=True)
        file = directory/(job['name']+'.json')
        write(file, job)
        launcher = ('import runpy,sys;runtime,script=sys.argv[1:3];sys.path.insert(0,runtime);'
                    "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")
        with tempfile.TemporaryDirectory(prefix='compact_recovery_') as cache:
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', NUMBA_NUM_THREADS='1', NUMBA_CACHE_DIR=cache)
            with file.with_suffix('.log').open('x') as log:
                result = subprocess.run([sys.executable, '-c', launcher, job['runtime'], str(ROOT/entry),
                    '--worker', str(file)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('Recovery worker failed: '+str(file.with_suffix('.log')))

    def execute(jobs, folder, budget):
        from .model_0923_h_training_audit import combine
        expanded, groups = split_filter_jobs(jobs)
        for future in as_completed([pool.submit(launch, job, folder) for job in expanded]):
            future.result()
        for original, children in groups:
            combine(original, children)

    driver.base.execute = execute


def checked_cases(prepared: Path) -> list[dict]:
    done = read(prepared/'completion.json')
    if done['status'] != 'complete' or done['protocol_sha256'] != digest(prepared/'protocol.json'):
        raise ValueError('Recovery preparation changed')
    cases = []
    for row in read(prepared/'protocol.json')['cases']:
        file = Path(row['file'])
        if digest(file) != row['sha256']:
            raise ValueError('Case definition changed')
        case = read(file)
        if case['generating_parameters_used_for_starts'] is not False:
            raise ValueError('Recovery must start independently of generating parameters')
        folder = Path(case['settings']['data'])/f'subject_{case["subject"]}'
        for name, sha in case['arrays_sha256'].items():
            if digest(folder/name) != sha:
                raise ValueError('Synthetic data changed')
        cases.append(case)
    return cases


def fit_case(case: dict, output: Path, smoke: bool) -> dict:
    settings, subject = case['settings'], case['subject']
    bank = [stable_seed(dict(protocol=settings['protocol'], base=settings['seed'], stage=stage,
                            subject=subject, repeat=i))
            for stage in ('screen', 'refine', 'selection', 'confirmation')
            for i in range(2 if smoke else settings['budgets'][stage]['seeds'])]
    if len(set(bank)) != len(bank) or case['generation_seed'] in bank:
        raise ValueError('Recovery fitting and generation seed banks overlap')
    output.mkdir()
    hashes = recovery_hashes(driver.io)
    starts = case['starts'][:4] if smoke else case['starts']
    key = subject, 'COMPACT'
    required, seen = starts[0], {r['id'] for r in starts}
    write(output/'protocol.json', dict(settings=settings, smoke=smoke, case_id=case['case_id'],
        source_sha256=hashes, space=case['space'], initialization='support-only; generating point unopened'))
    rows = driver.base.stage(settings, Path(settings['data']), output, {key: starts}, 'screen', smoke)
    elites = select_elites(rows[key], 1 if smoke else settings['elites'], exclude_key=driver.H+'max_capacity')
    proposed = propose_round(elites, case['space'], seen, 1 if smoke else settings['proposals_per_elite'],
        stable_seed(dict(base=settings['seed'], subject=subject, role='joint_refine')), .5, .25)
    if len(proposed) != (1 if smoke else settings['elites']*settings['proposals_per_elite']):
        raise ValueError('Recovery refinement budget changed')
    seen.update(r['id'] for r in proposed)
    rows = driver.base.stage(settings, Path(settings['data']), output, {key: proposed+elites}, 'refine', smoke)
    for stage, count in [('selection', settings['selection_candidates']), ('confirmation', settings['confirmation_candidates'])]:
        bank = driver.base.shortlist(rows[key], required, 2 if smoke else count)
        rows = driver.base.stage(settings, Path(settings['data']), output, {key: bank}, stage, smoke)
    nominee = dict(case_id=case['case_id'], subject=subject, replicate=case['replicate'], model='COMPACT',
        candidate=rows[key][0], finalists=rows[key], unique_candidates=len(seen))
    write(output/'nominees.json', dict(status='frozen_before_truth', smoke=smoke, settings=settings, nominee=nominee,
                                     source_sha256=hashes))
    write(output/'completion.json', dict(status='complete', smoke=smoke, truths_opened=False,
                                        nomination_sha256=digest(output/'nominees.json')))
    print(case['case_id']+': all estimates frozen', flush=True)
    return nominee


def search(prepared: Path, output: Path, smoke: bool) -> None:
    if output.exists():
        raise FileExistsError('Use a new recovery fit directory')
    cases = checked_cases(prepared)
    cases = cases[:1] if smoke else cases
    workers = 1 if smoke else min(128, len(os.sched_getaffinity(0)))
    output.mkdir(parents=True)
    for name in recovery_hashes(driver.io):
        archived = output/'source'/name
        archived.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, archived)
    write(output/'protocol.json', dict(prepared=str(prepared.resolve()), smoke=smoke, cases=[c['case_id'] for c in cases],
        workers=workers, numeric_threads=1, source_sha256=recovery_hashes(driver.io),
        parallel_unit='One complete independent particle filter',
        estimation='All variable coordinates; support-only initialization; truth unopened'))
    with ThreadPoolExecutor(max_workers=workers) as numerical:
        configure_executor(numerical)
        with ThreadPoolExecutor(max_workers=len(cases)) as controllers:
            futures = [controllers.submit(fit_case, c, output/c['case_id'], smoke) for c in cases]
            nominees = [future.result() for future in futures]
    write(output/'nominees.json', dict(status='frozen_before_truth', smoke=smoke, nominees=nominees))
    write(output/'completion.json', dict(status='complete', smoke=smoke, cases=len(cases), truths_opened=False,
                                        executed_filter_processes=len(list(output.glob('*/jobs/*.json'))),
                                        nomination_sha256=digest(output/'nominees.json')))


def score_truth(prepared: Path, fit: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new recovery truth-scoring directory')
    completed = read(fit/'completion.json')
    if completed['status'] != 'complete' or completed['nomination_sha256'] != digest(fit/'nominees.json'):
        raise ValueError('Freeze all estimates before opening generating parameters')
    smoke = completed['smoke']
    cases = checked_cases(prepared)
    cases = cases[:1] if smoke else cases
    prep_done = read(prepared/'completion.json')
    if digest(prepared/'truths.json') != prep_done['truths_sha256']:
        raise ValueError('Generating parameters changed')
    truth = {r['case_id']: r for r in read(prepared/'truths.json')['rows']}
    fitted = {r['case_id']: r for r in read(fit/'nominees.json')['nominees']}
    output.mkdir(parents=True)
    estimates, comparisons = [], []

    def compare_case(case: dict) -> tuple[list[dict], dict]:
        own = output/case['case_id']
        own.mkdir()
        point = candidate(truth[case['case_id']]['point'], 'generating_point_after_freeze')
        settings, subject = case['settings'], case['subject']
        write(own/'protocol.json', dict(settings=settings, smoke=smoke, use='after_estimation_truth_score'))
        rows = driver.base.stage(settings, Path(settings['data']), own, {(subject, 'COMPACT'): [point]}, 'confirmation', smoke)
        generated = rows[subject, 'COMPACT'][0]
        selected = fitted[case['case_id']]['candidate']
        with np.load(Path(generated['score_file']).with_suffix('.npz')) as a, np.load(Path(selected['score_file']).with_suffix('.npz')) as b:
            np.testing.assert_array_equal(a['seeds'], b['seeds'])
            difference = nll(b['observed_probability'])-nll(a['observed_probability'])
            se = jackknife_se(jackknife_scores(b['observed_probability'])-jackknife_scores(a['observed_probability']))
        comparison = dict(case_id=case['case_id'], subject=subject, replicate=case['replicate'],
            estimate_nll=selected['mean_nll'], truth_nll=generated['mean_nll'], estimate_minus_truth=difference,
            paired_mc_se=se, search_shortfall=difference > 2*se)
        own_estimates = []
        for key, levels in case['space'].items():
            value, actual = selected['hyperparams'][key], point['hyperparams'][key]
            own_estimates.append(dict(case_id=case['case_id'], subject=subject, replicate=case['replicate'],
                parameter=key, estimated=value, generating=actual, fitted=len(levels)>1,
                normalized_grid_error=abs(levels.index(value)-levels.index(actual))/max(1, len(levels)-1)))
        return own_estimates, comparison

    with ThreadPoolExecutor(max_workers=1 if smoke else min(128, len(os.sched_getaffinity(0)))) as pool:
        configure_executor(pool)
        with ThreadPoolExecutor(max_workers=len(cases)) as controllers:
            futures = [controllers.submit(compare_case, case) for case in cases]
            for future in futures:
                own_estimates, comparison = future.result()
                estimates.extend(own_estimates)
                comparisons.append(comparison)
    pd.DataFrame(estimates).to_csv(output/'parameters.csv', index=False)
    pd.DataFrame(comparisons).to_csv(output/'likelihood_comparison.csv', index=False)
    write(output/'completion.json', dict(status='complete', smoke=smoke, cases=len(cases),
        executed_filter_processes=len(list(output.glob('*/jobs/*.json'))),
        search_shortfalls=sum(r['search_shortfall'] for r in comparisons),
        scope='Two generating points and two sequences each; finite-sequence joint recovery diagnostic',
        source_fit=str(fit.resolve()), prepared=str(prepared.resolve()), truths_opened_after_freeze=True,
        artifacts_sha256={p.name: digest(p) for p in output.glob('*.csv')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'search', 'score-truth'), required=True)
    for name in ('evaluation', 'prepared', 'fit', 'output'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.phase == 'prepare' and args.evaluation and args.output and not args.smoke:
        prepare(args.evaluation, args.output)
    elif args.phase == 'search' and args.prepared and args.output:
        search(args.prepared, args.output, args.smoke)
    elif args.phase == 'score-truth' and args.prepared and args.fit and args.output and not args.smoke:
        score_truth(args.prepared, args.fit, args.output)
    else:
        parser.error('Supply the required phase inputs and a new output directory')

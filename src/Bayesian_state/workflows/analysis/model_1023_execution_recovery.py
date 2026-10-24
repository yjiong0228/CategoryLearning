"""Cross-model, parameter and latent-state recovery for belief execution.

Prepare fresh fixed-seed synthetic cases from the original ADAPT training
anchors with --phase prepare --config YAML --data ORIGINAL --output NEW.
Fit all three execution arms using --phase fit --prepared CASES --output NEW.
Generating parameters and trajectories remain outside the fitting case files.
The bounded optimizer uses support-only starts and a single numerical queue.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import os
from pathlib import Path

import numpy as np
import yaml

from . import model_1023_execution_study as study
from . import model_1023_execution_validation as validation
from .model_0923_h_validation import training_boundary
from .model_0923_h_objective_recovery import save_arrays
from .model_1021_simplification_recovery import neutral_starts
from ...optimization.search.adaptive_proposals import candidate, propose_round, select_elites
from ...utils.seeding import stable_seed

ROOT, read, write, digest = study.ROOT, study.read, study.write, study.digest


def sources() -> dict:
    own = str(Path(__file__).resolve().relative_to(ROOT))
    return dict(study.sources(study.driver.io), **validation.sources(), **{own: digest(Path(__file__))})


def fixture(original: Path, target: Path, arrays: dict, runtime: str, fraction: float) -> dict:
    """Preserve the task/perception metadata and split synthetic choices in time."""
    target.mkdir(parents=True)
    meta = deepcopy(read(original / 'prepared.json'))
    if digest(original / 'engine.json') != meta['engine_sha256']:
        raise ValueError('Original engine changed')
    write(target / 'engine.json', read(original / 'engine.json'))
    n = len(arrays['choices']); cut = training_boundary(n, fraction)
    hashes = {}
    for name, part in [('training.npz', slice(0, cut)), ('sealed_test.npz', slice(cut, n))]:
        hashes[name] = save_arrays(target, name, {k: v[part] for k, v in arrays.items()})
    meta.update(trials=n, training_trials=cut, test_trials=n-cut, array_sha256=hashes,
        engine_sha256=digest(target / 'engine.json'), data_role='synthetic_chronological_recovery',
        original_prepared_sha256=digest(original / 'prepared.json'),
        split='approximately_70_percent_at_32_trial_boundary; synthetic_choices')
    write(target / 'prepared.json', meta)
    write(target.parent / 'preparation_manifest.json', dict(status='complete', subjects=[meta['subject']],
        settings=dict(source_fit=str(Path(runtime).parent))))
    return meta


def prepare(config: Path, data: Path, output: Path, smoke: bool = False) -> None:
    if output.exists():
        raise FileExistsError('Use a new recovery preparation destination')
    policy = yaml.safe_load(config.read_text())
    plan = policy['recovery']
    if smoke:
        policy['parallel_budget'] = 1
        plan.update(subjects=plan['subjects'][:1], replicates=1, max_trials=128)
    settings = yaml.safe_load((ROOT / policy['base_config']).read_text())
    precision = deepcopy(settings['arms']['COMPACT'])
    settings.update(arms={mode: deepcopy(precision) for mode in policy['arms']},
        budgets=deepcopy(plan['budgets']), parallel_budget=policy['parallel_budget'])
    anchors, bindings = {}, {}
    for file_name in policy['warm_nominations']:
        file = ROOT / file_name
        frozen = read(file)
        if frozen['smoke'] or frozen['later_choices_used'] or frozen['status'] != 'frozen_before_test':
            raise ValueError('Require original training-only ADAPT generation anchors')
        anchors.update({r['subject']: r['candidate']['hyperparams'] for r in frozen['nominees'] if r['arm'] == 'ADAPT'})
        bindings[file_name] = digest(file)
    runtime = str((ROOT / policy['preparation']['source_fit'] / 'source').resolve())
    output.mkdir(parents=True)
    study.base.archive(output, config, sources())
    write(output / 'generation_protocol.json', dict(policy=policy, smoke=smoke, source_sha256=digest(Path(__file__)),
        study_sources=study.sources(study.driver.io), inputs_sha256=bindings,
        generation='two fixed independent seeds at each declared point; no outcome selection',
        fitting='support-only starts; truths stored separately', scope='bounded coverage of three task schedules'))
    audits, generation = [], []
    for sid in plan['subjects']:
        original = data / f'subject_{sid}'
        meta = read(original / 'prepared.json')
        point = deepcopy(anchors[sid]); point[study.KAPPA] = 0.
        common = dict(subject=sid, folder=str(original.resolve()), runtime=runtime,
            mechanism=settings['mechanism'], source_sha256=study.sources(study.driver.io))
        audits.append(dict(common, phase='audit', name=f'audit_S{sid}', candidate=candidate(point, 'geometry'),
            geometry_beta=25., target=str((output / 'geometry' / f'subject_{sid}').resolve())))
        modes = [('MIX', 0.), ('MAP', 0.)] + [('STICKY', k) for k in plan['kappa_values']]
        if smoke:
            modes = [('STICKY', 2.)]
        for mode, kappa in modes:
            own = dict(point, **{study.KAPPA: kappa})
            space = study.parameter_space(settings, mode, meta['condition'], policy['kappa_values'])
            if set(space) != set(own) or any(own[k] not in values for k, values in space.items()):
                raise ValueError('Generating point outside fitted parameter support')
            identity = f'S{sid}_{mode}_K{kappa:g}'
            seeds = [stable_seed(dict(base=policy['seed'], role='execution_recovery_generation',
                subject=sid, mode=mode, kappa=kappa, replicate=r)) for r in range(plan['replicates'])]
            generation.append(dict(common, phase='autonomous', name=identity, model=mode,
                candidate=candidate(own, 'generating_point'), arm_settings=precision, seeds=seeds, smoke=False,
                generation_trials=plan['max_trials'], target=str((output / 'generated' / identity).resolve())))
    with ThreadPoolExecutor(max_workers=min(policy['parallel_budget'], len(os.sched_getaffinity(0)))) as pool:
        queue = validation.NumericalQueue(pool)
        queue.execute(audits, output, policy['parallel_budget'])
        queue.execute(generation, output, policy['parallel_budget'])
    cases, truths = [], []
    for job in generation:
        sid = job['subject']; original = Path(job['folder'])
        meta = read(original / 'prepared.json')
        parts = []
        for file_name in ('training.npz', 'sealed_test.npz'):
            if digest(original / file_name) != meta['array_sha256'][file_name]:
                raise ValueError('Original task arrays changed')
            with np.load(original / file_name) as z:
                parts.append({k: z[k] for k in ('stimulus', 'categories')})
        physical = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        source = Path(job['target']).with_suffix('.npz')
        if digest(source) != read(source.with_suffix('.json'))['predictions_sha256']:
            raise ValueError('Generated trajectory changed')
        with np.load(source) as z:
            bank = {k: z[k] for k in z.files}
        np.testing.assert_array_equal(bank['seeds'], job['seeds'])
        n = len(bank['categories'])
        np.testing.assert_array_equal(physical['categories'][:n], bank['categories'])
        geometry = read(output / 'geometry' / f'subject_{sid}' / 'audit.json')
        for replicate, seed in enumerate(job['seeds']):
            case_id = job['name'] + f'_R{replicate}'
            own = output / 'cases' / case_id
            folder = own / 'data' / f'subject_{sid}'
            arrays = {k: v[:n] for k, v in physical.items()}
            arrays.update(choices=bank['choices'][replicate], feedback=bank['feedback'][replicate])
            prepared = fixture(original, folder, arrays, runtime, plan['training_fraction'])
            case_settings = deepcopy(settings)
            case_settings.update(data=str(folder.parent.resolve()), subjects=[sid],
                protocol=policy['protocol'] + '_recovery', seed=stable_seed(dict(base=policy['seed'],
                    role='execution_recovery_fit', case=case_id)))
            spaces, starts = {}, {}
            for mode in policy['arms']:
                spaces[mode] = study.parameter_space(case_settings, mode, meta['condition'], policy['kappa_values'])
                starts[mode] = neutral_starts(spaces[mode], plan['initial_candidates'],
                    stable_seed(dict(base=case_settings['seed'], role='support_only_starts')))
                # This check never supplies truth to the proposal generator.
                # A coincidental exact anchor match aborts preparation.
                if any(r['hyperparams'] == job['candidate']['hyperparams'] for r in starts[mode]):
                    raise ValueError('A support-only start coincides with the generating point')
            case = dict(case_id=case_id, subject=sid, condition=meta['condition'], settings=case_settings,
                plan=plan, spaces=spaces, starts=starts, generating_parameters_used_for_starts=False,
                generation_seed=seed, training_trials=prepared['training_trials'], trials=n,
                target_rules=geometry['training_perfect_rules'],
                arrays_sha256={name: digest(folder / name) for name in
                    ('training.npz', 'sealed_test.npz', 'engine.json', 'prepared.json')})
            write(own / 'case.json', case)
            cases.append(dict(case_id=case_id, file=str((own / 'case.json').resolve()), sha256=digest(own / 'case.json')))
            truth_file = output / 'truth_states' / (case_id + '.npz')
            truth_file.parent.mkdir(exist_ok=True)
            with truth_file.open('xb') as stream:
                np.savez_compressed(stream, **{k: bank[k][replicate] for k in
                    ('rule', 'belief', 'precision', 'contribution', 'executed', 'switch', 'forced', 'search')})
            truths.append(dict(case_id=case_id, mode=job['model'], point=job['candidate']['hyperparams'],
                generation_seed=seed, states=str(truth_file.resolve()), states_sha256=digest(truth_file),
                generation_array_sha256=digest(source)))
    write(output / 'cases.json', dict(cases=cases, truths_used_for_fitting=False))
    write(output / 'truths.json', dict(rows=truths, role='opened after all parameter estimates freeze'))
    write(output / 'completion.json', dict(status='complete', smoke=smoke, cases=len(cases),
        cases_sha256=digest(output / 'cases.json'), truths_sha256=digest(output / 'truths.json')))


def fit_case(case: dict, output: Path, smoke: bool) -> None:
    settings = deepcopy(case['settings']); sid = case['subject']
    if smoke:
        settings['parallel_budget'] = 1
    data = Path(settings['data'])
    seed_bank = [stable_seed(dict(protocol=settings['protocol'], base=settings['seed'], stage=name,
        subject=sid, repeat=i)) for name, budget in settings['budgets'].items()
        for i in range(2 if smoke else budget['seeds'])]
    if len(set(seed_bank)) != len(seed_bank) or case['generation_seed'] in seed_bank:
        raise ValueError('Generation and evaluation random seed banks overlap')
    output.mkdir(parents=True)
    write(output / 'protocol.json', dict(settings=settings, smoke=smoke, case_id=case['case_id'],
        source_sha256=study.sources(study.driver.io), recovery_source_sha256=digest(Path(__file__)),
        initialization='support only; generating parameters unopened', spaces=case['spaces']))
    modes = list(settings['arms'])
    banks = {(sid, mode): case['starts'][mode][:8 if smoke else len(case['starts'][mode])] for mode in modes}
    stage = lambda bank, name: study.base.stage(settings, data, output, bank, name, smoke)
    screen = stage(banks, 'screen')
    refine = {}
    anchors = {key: rows[0] for key, rows in banks.items()}
    for key, rows in screen.items():
        elites = select_elites(rows, 1 if smoke else 2, exclude_key=study.driver.H + 'max_capacity')
        quota = 2 if smoke else case['plan']['refinement_candidates']
        proposed = propose_round(elites, case['spaces'][key[1]], {r['id'] for r in rows}, quota // len(elites),
            stable_seed(dict(base=settings['seed'], subject=sid, role='recovery_refine')), .4, .2)
        if len(proposed) != quota:
            raise ValueError('Recovery refinement quota not met')
        refine[key] = proposed + elites
    refined = stage(refine, 'refine_1')
    confirmation = study.confirmation_bank(screen, refined, anchors, [sid], modes,
        4 if smoke else case['plan']['confirmation_candidates'], 1)
    confirmed = stage(confirmation, 'confirmation')
    nominees = [dict(subject=sid, arm=key[1], candidate=rows[0], finalists=rows) for key, rows in confirmed.items()]
    write(output / 'nominees.json', dict(status='frozen_before_test', nominees=nominees, smoke=smoke,
        later_choices_used=False, oral_used=False, truths_opened=False, selection='independent_training_NLL'))
    if not smoke:
        stage({(sid, row['arm']): [row['candidate']] for row in nominees}, 'evaluate')
    write(output / 'completion.json', dict(status='complete', smoke=smoke, case_id=case['case_id'],
        nominees_sha256=digest(output / 'nominees.json'), truths_opened=False))
    print(case['case_id'] + ': estimates frozen', flush=True)


def fit(prepared: Path, output: Path, smoke: bool) -> None:
    if output.exists():
        raise FileExistsError('Use a new recovery fitting destination')
    done = read(prepared / 'completion.json')
    if done['status'] != 'complete' or done['cases_sha256'] != digest(prepared / 'cases.json'):
        raise ValueError('Recovery preparation changed')
    if done.get('smoke') and not smoke:
        raise ValueError('Smoke preparation is not a scientific recovery dataset')
    definitions = read(prepared / 'cases.json')['cases']
    definitions = definitions[:1] if smoke else definitions
    cases = []
    for row in definitions:
        path = Path(row['file'])
        if digest(path) != row['sha256']:
            raise ValueError('Recovery case definition changed')
        case = read(path)
        if case['generating_parameters_used_for_starts']:
            raise ValueError('Recovery starts must be independent of truths')
        folder = Path(case['settings']['data']) / f'subject_{case["subject"]}'
        for name, sha in case['arrays_sha256'].items():
            if digest(folder / name) != sha:
                raise ValueError('Synthetic fitting data changed')
        cases.append(case)
    output.mkdir(parents=True)
    # Earlier smoke fixtures may predate archival; their protocol still binds
    # the complete generating policy. Scientific preparation always archives.
    config = prepared / 'config.yaml'
    if config.exists():
        study.base.archive(output, config, sources())
    elif not smoke:
        raise ValueError('Scientific recovery preparation lacks its configuration archive')
    workers = 1 if smoke else min(128, len(os.sched_getaffinity(0)))
    write(output / 'protocol.json', dict(prepared=str(prepared.resolve()), cases=definitions, smoke=smoke,
        sources=study.sources(study.driver.io), recovery_source_sha256=digest(Path(__file__)),
        numeric_workers=workers, numeric_threads=1, truths_opened=False))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        queue = validation.NumericalQueue(pool)
        study.base.source_hashes, study.base.execute = study.sources, queue.execute
        with ThreadPoolExecutor(max_workers=len(cases)) as controllers:
            futures = [controllers.submit(fit_case, case, output / case['case_id'], smoke) for case in cases]
            for future in futures:
                future.result()
    write(output / 'completion.json', dict(status='complete', smoke=smoke, cases=len(cases), truths_opened=False,
        nominees_sha256={case['case_id']: digest(output / case['case_id'] / 'nominees.json') for case in cases}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'fit'), required=True)
    for key in ('config', 'data', 'prepared', 'output'):
        parser.add_argument('--' + key, type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.phase == 'prepare' and args.config and args.data and args.output:
        prepare(args.config, args.data, args.output, args.smoke)
    elif args.phase == 'fit' and args.prepared and args.output:
        fit(args.prepared, args.output, args.smoke)
    else:
        parser.error('Prepare requires --config --data --output; fit requires --prepared --output')

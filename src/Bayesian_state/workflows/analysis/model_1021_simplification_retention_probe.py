"""Cross baseline search and evidence retention at the frozen S205 COMPACT fit.

Use --plan PLAN_JSON --output NEW; --smoke verifies two saved path prefixes.
The registered four parameter vectors reuse two confirmation scores and one
autonomous bank. Three new banks and two training scores use the original
workers, seeds and trial boundaries. The original nominees remain unchanged.
After all four recovery refinement stages finish, at most 16 probe workers
fit within the shared 128-process allocation. This is a development diagnostic.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from . import model_1021_simplification as driver
from .model_0923_h_training_audit import combine
from ...evaluation.model_0923_h_validation import (
    block_paths, checked_arrays, distribution_scores, jackknife_scores, jackknife_se, nll,
)
from ...evaluation.model_0923_h_generation_uncertainty import path_jackknife
from ...optimization.search.adaptive_proposals import candidate

read, write, digest = driver.io.read, driver.io.write, driver.io.digest
SEARCH = driver.H + 'search_controller.baseline_probability'
DECAY = driver.H + 'intervention.decay'


def check_points(points: list[dict], nominee: dict) -> None:
    """Only the two declared coordinates may differ across the full factorial."""
    pairs = set()
    for point in points:
        params = point['hyperparams']
        if candidate(params, 'identity')['id'] != point['id']:
            raise ValueError('Candidate identity changed')
        if set(params) != set(nominee['hyperparams']) or any(
                params[k] != v for k, v in nominee['hyperparams'].items() if k not in (SEARCH, DECAY)):
            raise ValueError('Probe changes an undeclared coordinate')
        if (point['baseline_probability'], point['evidence_decay']) != (params[SEARCH], params[DECAY]):
            raise ValueError('Coordinate labels disagree with parameters')
        pairs.add((params[SEARCH], params[DECAY]))
    if len(points) != 4 or pairs != {(p, d) for p in (.02, .05) for d in (.95, .99)}:
        raise ValueError('Require the four registered search-retention combinations')


def resource_gate(plan: dict) -> int:
    """Bound ready primary, recovery and probe processes without overlapping grids."""
    gate = plan['resource_gate']
    folder = Path(gate['recovery_fit'])
    cases = sorted(folder.glob('S*'))
    if {p.name for p in cases} != {'S205_R0', 'S205_R1', 'S306_R0', 'S306_R1'}:
        raise ValueError('Require the four registered recovery cases')
    remaining = 0
    for case in cases:
        if read(case/'refine_completion.json')['status'] != 'complete':
            raise ValueError('Recovery refinement must finish before the probe')
        settings = read(case/'protocol.json')['settings']
        remaining += max(settings['selection_candidates'] * settings['budgets']['selection']['seeds'],
                         settings['confirmation_candidates'] * settings['budgets']['confirmation']['seeds'],
                         settings['budgets']['confirmation']['seeds'])
    primary = read(Path(plan['fit'])/'nominees.json')['settings']
    conditional = len(primary['subjects']) * len(primary['arms']) * primary['budgets']['evaluate']['seeds']
    workers = gate['diagnostic_worker_limit']
    if (remaining > gate['recovery_remaining_processes_upper_bound']
            or conditional > gate['primary_conditional_processes_upper_bound']
            or not 1 <= workers <= 16 or remaining + conditional + workers > 128):
        raise ValueError('Concurrent numerical budget exceeds the registered limit')
    return workers


def load_bank(file: Path) -> tuple[dict, dict]:
    record = read(file)
    if digest(file.with_suffix('.npz')) != record['predictions_sha256']:
        raise ValueError('Saved numerical bank changed')
    with np.load(file.with_suffix('.npz')) as z:
        arrays = {k: z[k] for k in z.files}
    return record, arrays


def retained_after_discovery(present: np.ndarray) -> np.ndarray:
    """Per-path presence from first discovery onward; undiscovered paths are undefined."""
    result = np.full(len(present), np.nan)
    for i, path in enumerate(present):
        found = np.flatnonzero(path)
        if len(found):
            result[i] = path[found[0]:].mean()
    return result


def summarize(plan: dict, frozen: dict, output: Path, generation: dict, training: dict) -> None:
    folder = driver.ROOT/frozen['settings']['data']/f"subject_{plan['subject']}"
    meta = read(folder/'prepared.json')
    human = []
    inputs = {}
    for name in ('training.npz', 'sealed_test.npz'):
        file = folder/name
        if digest(file) != meta['array_sha256'][name]:
            raise ValueError('Human trial boundary or data changed')
        inputs[str(file)] = digest(file)
        with np.load(file) as z:
            human.append(z['choices'] == z['categories'])
    human = np.concatenate(human)
    cut, length = meta['training_trials'], len(human)
    scores, paths, curves, jacks = [], [], [], {}
    baseline, = [p['id'] for p in plan['points'] if p['role'] == 'frozen_nominee']
    original_seeds = None
    for point in plan['points']:
        key = point['id']
        gr, a = load_bank(Path(generation[key]))
        tr, t = load_bank(Path(training[key]))
        for file in (Path(generation[key]), Path(training[key])):
            inputs[str(file)] = digest(file)
            inputs[str(file.with_suffix('.npz'))] = digest(file.with_suffix('.npz'))
        if gr['candidate']['hyperparams'] != point['hyperparams'] or tr['candidate']['hyperparams'] != point['hyperparams']:
            raise ValueError('Diagnostic bank differs from planned parameters')
        if tr['trials'] != cut or tr['particles'] != plan['training_rescore']['particles']:
            raise ValueError('Training score crossed its boundary or changed particle budget')
        np.testing.assert_array_equal(a['seeds'], plan['path_seeds'])
        if original_seeds is None:
            original_seeds = t['seeds']
        np.testing.assert_array_equal(t['seeds'], original_seeds)
        if len(original_seeds) != plan['training_rescore']['seeds']:
            raise ValueError('Training confirmation bank is incomplete')
        if a['choices'].shape != (plan['paths'], length) or t['observed_probability'].shape != (8, cut):
            raise ValueError('Diagnostic arrays have incorrect trial or seed counts')
        np.testing.assert_allclose(a['probabilities'].sum(-1), 1., atol=1e-12, rtol=0)
        if not np.isfinite(a['probabilities']).all() or np.any(a['probabilities'] < 0):
            raise ValueError('Invalid generation probabilities')
        generated = a['choices'] == a['categories'][None, :]
        retention = retained_after_discovery(a['present'])
        tag = dict(candidate=key, baseline_probability=point['baseline_probability'],
                   evidence_decay=point['evidence_decay'], role=point['role'])
        for i, seed in enumerate(a['seeds']):
            found = np.flatnonzero(a['present'][i])
            paths.append(dict(tag, seed=int(seed), discovered=bool(len(found)),
                first_discovery_trial=int(found[0]+1) if len(found) else None,
                retained_after_discovery=retention[i], late_present=a['present'][i, -128:].mean(),
                target_capacity_drops=int(a['target_capacity_dropped'][i].sum()),
                late_accuracy=generated[i, -128:].mean()))
        training_jk = jackknife_scores(t['observed_probability'])
        for phase, first, last in [('training', 0, cut), ('test', cut, length), ('whole', 0, length)]:
            x = block_paths(generated, first, last)
            y = block_paths(human[None, :], first, last)[0]
            jk = path_jackknife(x, y)
            jacks[key, phase] = dict(jk, training_nll=training_jk)
            scores.append(dict(tag, phase=phase, trials=last-first, **distribution_scores(x, y),
                energy_mc_se=jackknife_se(jk['energy']), training_nll=nll(t['observed_probability']),
                training_nll_mc_se=jackknife_se(training_jk),
                target_ever=float(a['present'][:, :last].any(1).mean()),
                target_capacity_drops=float(a['target_capacity_dropped'][:, first:last].sum(1).mean()),
                target_late_present=float(a['present'][:, max(first,last-128):last].mean()),
                mean_search=float(a['search'][:, first:last].mean()),
                late_accuracy=float(generated[:, max(first,last-128):last].mean())))
            if phase == 'whole':
                for i, observed in enumerate(y):
                    curves.append(dict(tag, trial=min((i+1)*32, length), training_trials=cut,
                        observed=observed, autonomous=float(x[:, i].mean()),
                        lower=float(np.quantile(x[:, i], .1)), upper=float(np.quantile(x[:, i], .9))))
    table = pd.DataFrame(scores)
    contrasts = []
    for phase, group in table.groupby('phase'):
        reference = group[group.candidate == baseline].iloc[0]
        for _, row in group.iterrows():
            contrast = dict(candidate=row.candidate, phase=phase)
            for metric in ('training_nll', 'energy', 'crps'):
                contrast['delta_'+metric] = row[metric] - reference[metric]
                contrast['delta_'+metric+'_mc_se'] = jackknife_se(
                    jacks[row.candidate, phase][metric] - jacks[baseline, phase][metric])
            contrasts.append(contrast)
    for name, table in dict(scores=table, contrasts=pd.DataFrame(contrasts),
                            paths=pd.DataFrame(paths), curves=pd.DataFrame(curves)).items():
        table.to_csv(output/(name+'.csv'), index=False)
    write(output/'completion.json', dict(status='complete', scope=plan['scope'],
        primary_nominee_unchanged=True, oral_used_for_selection=False, inputs_sha256=inputs,
        artifacts_sha256={p.name: digest(p) for p in output.glob('*.csv')}))


def run(plan_file: Path, output: Path, smoke: bool = False) -> None:
    if output.exists():
        raise FileExistsError('Use a new probe output directory')
    plan = read(plan_file)
    fit, source = Path(plan['fit']), Path(plan['generation_source'])
    if (digest(fit/'nominees.json') != plan['nomination_sha256']
            or digest(source/'completion.json') != plan['generation_completion_sha256']):
        raise ValueError('Frozen inputs changed')
    frozen = read(fit/'nominees.json')
    nominee, = [r for r in frozen['nominees'] if (r['subject'], r['model']) == (205, 'COMPACT')]
    check_points(plan['points'], nominee['candidate'])
    workers = resource_gate(plan)
    for name, sha in frozen['source_sha256'].items():
        if digest(driver.ROOT/name) != sha:
            raise ValueError('Frozen scientific source changed: '+name)
    baseline_row, = [r for r in read(source/'autonomous/evaluation.json')['rows']
                     if (r['subject'], r['model']) == (205, 'COMPACT')]
    baseline = nominee['candidate']['id']
    if baseline_row['candidate'] != nominee['candidate']:
        raise ValueError('Generation and fit use different nominees')
    baseline_arrays = checked_arrays(source/'autonomous', baseline_row)
    np.testing.assert_array_equal(baseline_arrays['seeds'], plan['path_seeds'])
    template = read(next((source/'jobs').glob(f'*S205_COMPACT_{baseline}_C000.json')))
    output.mkdir(parents=True)
    write(output/'protocol.json', dict(fit=str(fit.resolve()), plan=plan,
        plan_sha256=digest(plan_file), smoke=smoke, workers=1 if smoke else workers,
        source_sha256=frozen['source_sha256'], controller_sha256=digest(Path(__file__))))
    if smoke:
        job = deepcopy(template)
        job.update(name='saved_prefix_smoke', seeds=plan['path_seeds'][:2], smoke=True,
                   target=str((output/'prefix').resolve()))
        driver.execute([job], output, 1)
        _, arrays = load_bank(output/'prefix.json')
        for name, value in arrays.items():
            expected = baseline_arrays[name]
            expected = expected[:16] if name == 'categories' else expected[:2] if name == 'seeds' else expected[:2, :16]
            np.testing.assert_array_equal(value, expected)
        write(output/'completion.json', dict(status='complete', smoke=True, exact_saved_prefix=True,
                                            paths=2, trials=16, arrays=list(arrays)))
        return
    jobs, groups, aggregate_jobs = [], {}, {}
    training = {p['id']: p['score_file'] for p in nominee['finalists']}
    baseline_training = Path(training[baseline])
    tr, _ = load_bank(baseline_training)
    if (tr['particles'], len(tr['seeds'])) != (512, 8):
        raise ValueError('Require original 512-particle eight-seed confirmation')
    scoring_template = read(fit/'jobs'/f'confirmation_S205_COMPACT_{baseline}_C000.json')
    for point in plan['points']:
        point = {k: point[k] for k in ('id', 'hyperparams', 'sources', 'origin')}
        key = point['id']
        if key in training:
            continue
        target = output/'training'/key
        aggregate_jobs['train', key] = dict(scoring_template, candidate=point,
                                            seeds=tr['seeds'], target=str(target.resolve()))
        groups['train', key] = []
        for i, seed in enumerate(tr['seeds']):
            job = deepcopy(scoring_template)
            job.update(candidate=point, seeds=[seed], name=f'training_{key}_C{i:03d}',
                       target=str((output/'training/chunks'/f'{key}_C{i:03d}').resolve()))
            groups['train', key].append(job)
            jobs.append(job)
    generation = {baseline: str(source/'autonomous/evaluate/subject_205/COMPACT'/(baseline+'.json'))}
    for point in plan['points']:
        key = point['id']
        if key == baseline:
            continue
        point = {k: point[k] for k in ('id', 'hyperparams', 'sources', 'origin')}
        target = output/'autonomous/evaluate/subject_205/COMPACT'/key
        aggregate_jobs['auto', key] = dict(template, candidate=point, seeds=plan['path_seeds'],
                                           target=str(target.resolve()))
        groups['auto', key] = []
        for i in range(0, plan['paths'], plan['autonomous_chunk']):
            job = deepcopy(template)
            job.update(candidate=point, seeds=plan['path_seeds'][i:i+plan['autonomous_chunk']],
                       name=f'autonomous_{key}_C{i:03d}',
                       target=str((target.parent/'chunks'/f'{key}_C{i:03d}').resolve()))
            groups['auto', key].append(job)
            jobs.append(job)
    write(output/'jobs_plan.json', dict(jobs=jobs, training_reused=training, generation_reused=generation))
    driver.execute(jobs, output, workers)
    for (role, key), chunks in groups.items():
        job = aggregate_jobs[role, key]
        Path(job['target']).parent.mkdir(parents=True, exist_ok=True)
        if role == 'train':
            training[key] = combine(job, chunks)['score_file']
        else:
            driver.io.combine_evaluation_chunks(job, chunks, True)
            generation[key] = str(Path(job['target']).with_suffix('.json'))
    write(output/'banks.json', dict(training=training, generation=generation))
    summarize(plan, frozen, output, generation, training)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.plan, args.output, args.smoke)

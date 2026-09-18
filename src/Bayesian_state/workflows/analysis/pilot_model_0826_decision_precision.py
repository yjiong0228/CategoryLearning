"""Calibrate a fixed winner's numerical regret, then challenge an S103 plateau.

Reuses the shared model/evaluator. Only parameter discovery and numerical effort
change. Seed bootstrap is an approximate finite-particle diagnostic, not a
parameter confidence interval, integration convergence proof or global optimum.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter

from joblib import Parallel, delayed
import numpy as np
import yaml

from ...utils.parallel import parallel_job_count, single_threaded_processes
from .pilot_model_0826_adaptive_effort import (
    initial_points, load_arrays, propose_round, select_elites, verify_files,
    verify_reuse_context,
)
from .pilot_model_0826_search_stopping import (
    candidate, make_context, merge_bank, parameter_support, ray_proposals,
)
from .pilot_model_0826_simplified_fit import (
    mixture_nll, paired_difference, point_id, score_one, validation_seeds, write_json,
)


def ranking(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: (r['mean_nll'], r['id']))


def freeze_json(path: Path, value: object) -> None:
    """Resume only an identical plan; never overwrite existing observations."""
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Frozen plan differs: {path}')
    else:
        write_json(path, value)


def compact_one(context: dict, point: dict, particles: int, seed: int) -> dict:
    # Retain exactly the existing evaluator and probability extraction. This
    # pilot does not deliver states; avoiding their IPC/disk copies is lossless
    # for its declared score-only question.
    run = score_one(context, point, particles, seed)
    return {k: run[k] for k in ('probabilities', 'observed', 'mask', 'seconds')}


def score_group(specs: list[tuple[dict, list[dict]]], budget: dict, family: str,
                base_seed: int, jobs: int, directory: Path) -> dict:
    """One process layer across subjects x candidates x seeds; protected batches."""
    plan = {'budget': budget, 'family': family, 'base_seed': base_seed,
            'banks': {str(ctx['subject']): bank for ctx, bank in specs}}
    if (directory/'scores.json').exists():
        result = json.loads((directory/'scores.json').read_text())
        if result['plan'] != plan:
            raise ValueError('Completed batch configuration differs')
        return result
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory/'plan.json', plan)
    tasks = []
    for ctx, bank in specs:
        if not bank or len({r['id'] for r in bank}) != len(bank):
            raise ValueError('A score bank must be nonempty and unique')
        seeds = validation_seeds(base_seed, ctx['subject'], family, budget['filter_seed_count'])
        tasks.extend((ctx, row, seed) for row in bank for seed in seeds)
    began = perf_counter()
    workers = parallel_job_count(jobs, len(tasks))
    with single_threaded_processes():
        raw = Parallel(n_jobs=workers)(
            delayed(compact_one)(ctx, row['hyperparams'], budget['particle_count'], seed)
            for ctx, row, seed in tasks)
    subjects, offset = {}, 0
    for ctx, bank in specs:
        sid = ctx['subject']; target = directory/str(sid); target.mkdir()
        seeds = validation_seeds(base_seed, sid, family, budget['filter_seed_count'])
        rows = []; first = raw[offset]
        for row in bank:
            runs = raw[offset:offset+len(seeds)]; offset += len(seeds)
            for run in runs:
                np.testing.assert_array_equal(run['observed'], first['observed'])
                np.testing.assert_array_equal(run['mask'], first['mask'])
            p = np.stack([run['probabilities'] for run in runs])
            np.savez_compressed(target/f"{row['id']}.npz", probabilities=p, seeds=seeds,
                                observed=first['observed'], mask=first['mask'])
            rows.append({**row, 'mean_nll': mixture_nll(p, first['observed'], first['mask']),
                         'worker_seconds': sum(run['seconds'] for run in runs)})
        subjects[str(sid)] = {'rows': rows, 'seeds': seeds,
                              'trials': len(first['mask']), 'scored_trials': int(first['mask'].sum())}
    result = {'plan': plan, 'subjects': subjects, 'seconds': perf_counter()-began,
              'workers': workers, 'filter_runs': len(tasks)}
    write_json(directory/'scores.json', result)
    print(json.dumps({'batch': str(directory), 'seconds': result['seconds'],
                      'workers': workers, 'filter_runs': len(tasks)}), flush=True)
    return result


def decision_diagnostics(arrays: dict[str, dict], selected: str, tolerance: float,
                         alpha: float, replicates: int, seed: int) -> dict:
    """Upper numerical regret for a PRESELECTED point versus the whole bank.

    Resample complete PF seeds, preserving the temporal covariance and common
    random numbers across candidates. The maximum is formed inside each
    bootstrap draw: poor-vs-poor differences are never a stopping requirement.
    """
    if selected not in arrays or not 0 < alpha < 1 or tolerance <= 0:
        raise ValueError('Invalid selected candidate, alpha or tolerance')
    ids = sorted(arrays); first = arrays[ids[0]]
    y, mask, seeds = first['observed'], first['mask'], first['seeds']
    if len(seeds) < 2 or len(np.unique(seeds)) != len(seeds) or not mask.any():
        raise ValueError('Need distinct seeds and at least one scored trial')
    probabilities = []
    for pid in ids:
        a = arrays[pid]
        for key in ('observed', 'mask', 'seeds'):
            np.testing.assert_array_equal(a[key], first[key])
        p = a['probabilities']
        if p.ndim != 3 or p.shape[:2] != (len(seeds), len(y)) or not np.isfinite(p).all() or (p < 0).any():
            raise ValueError('Invalid probability array')
        np.testing.assert_allclose(p.sum(axis=-1), 1., atol=1e-10)
        probabilities.append(p[:, np.arange(len(y)), y][:, mask])
    weights = np.random.default_rng(seed).multinomial(
        len(seeds), np.full(len(seeds), 1/len(seeds)), size=replicates)/len(seeds)
    boot = np.stack([-np.log(np.clip(weights@q, 1e-12, 1.)).mean(axis=1) for q in probabilities])
    scores = np.array([-np.log(np.clip(q.mean(axis=0), 1e-12, 1.)).mean() for q in probabilities])
    index = ids.index(selected)
    differences = boot[index]-boot
    regret = differences.max(axis=0)  # Includes self, so regret >= 0.
    lower, upper = np.quantile(regret, [alpha, 1-alpha])
    pairs = {}
    for j, pid in enumerate(ids):
        if pid != selected:
            pairs[pid] = {'selected_minus_candidate': float(scores[index]-scores[j]),
                          'interval95': np.quantile(differences[j], [.025, .975]).tolist()}
    old_width = max((float(np.diff(np.quantile(boot[i]-boot[j], [.025, .975]))[0]/2)
                     for i in range(len(ids)) for j in range(i)), default=0.)
    # Old rule is reported for comparison only; this does not rewrite old reports.
    old_winner = int(scores.argmin())
    old_regret = float(np.quantile(boot[old_winner]-boot.min(axis=0), .95))
    status = ('acceptable_within_bank' if upper <= tolerance else
              'selected_point_inferior' if lower > tolerance else 'unresolved')
    return {'selected': selected, 'scores': dict(zip(ids, scores.tolist())),
            'point_regret': float(scores[index]-scores.min()),
            'regret_lower': float(lower), 'regret_upper': float(upper),
            'alpha': alpha, 'tolerance': tolerance, 'status': status,
            'selected_minus_candidates': pairs,
            'old_all_pair_halfwidth': old_width, 'old_winner_regret95': old_regret,
            'old_score_gate_pass': old_width <= tolerance and old_regret <= tolerance,
            'scope': 'Approximate numerical-seed bootstrap within a fixed bank at finite R. Not parameter uncertainty or integration convergence.'}


def archived_banks(config: dict) -> dict[int, list[dict]]:
    banks = {}
    for subject in config['subjects']:
        sid = subject['subject']
        path = (Path(config['previous_adaptive'])/'calibration_229/R128/confirmation/scores.json'
                if sid == 229 else Path(config['previous_stopping'])/f'search_{sid}/confirmation/scores.json')
        rows = json.loads(path.read_text())['rows']
        # Old scores select the point, but every test below uses fresh seeds.
        banks[sid] = ranking(rows)
    return banks


def independent_budget_difference(left: dict, right: dict, replicates: int, seed: int) -> dict:
    """Fresh families require independent resampling, not paired seed indices."""
    for key in ('observed', 'mask'):
        np.testing.assert_array_equal(left[key], right[key])
    if set(left['seeds']) & set(right['seeds']):
        raise ValueError('Independent budget comparison contains shared seeds')
    rng = np.random.default_rng(seed); scores, boot = [], []
    for a in (left, right):
        y, mask = a['observed'], a['mask']
        q = a['probabilities'][:, np.arange(len(y)), y][:, mask]
        b = len(q)
        w = rng.multinomial(b, np.full(b, 1/b), size=replicates)/b
        scores.append(float(-np.log(np.clip(q.mean(axis=0), 1e-12, 1.)).mean()))
        boot.append(-np.log(np.clip(w@q, 1e-12, 1.)).mean(axis=1))
    return {'audit_minus_decision_nll': scores[0]-scores[1],
            'independent_interval95': np.quantile(boot[0]-boot[1], [.025, .975]).tolist(),
            'choice_probability_rmse': float(np.sqrt(np.mean((
                left['probabilities'].mean(axis=0)-right['probabilities'].mean(axis=0))**2))),
            'scope': 'Both seed family and possibly R/B differ. This is sensitivity, not an estimate of particle bias.'}


def calibration(config: dict, contexts: dict[int, dict], banks: dict[int, list[dict]],
                output: Path) -> dict:
    directory = output/'calibration'; directory.mkdir(exist_ok=True)
    settings = config['calibration']; primary = {sid: rows[0]['id'] for sid, rows in banks.items()}
    freeze_json(directory/'frozen_banks.json', {'banks': {str(k): v for k, v in banks.items()},
                                               'primary': {str(k): v for k, v in primary.items()}})
    active = set(banks); records = {sid: [] for sid in banks}; selected_tiers = {}
    for i, budget in enumerate(settings['tiers']):
        if not active:
            break
        target = directory/f'tier_{i}'
        result = score_group([(contexts[sid], banks[sid]) for sid in sorted(active)], budget,
                             'decision_precision', config['base_seed'], config['parallel_budget'], target)
        for sid in sorted(active):
            arrays = {r['id']: load_arrays(target/str(sid), r['id']) for r in banks[sid]}
            d = decision_diagnostics(arrays, primary[sid], settings['tolerance'],
                                     settings['alpha']/len(settings['tiers']),
                                     settings['bootstrap_replicates'], config['base_seed']+sid)
            records[sid].append({'tier': i, 'budget': budget, 'diagnostic': d})
            if d['status'] != 'unresolved':
                selected_tiers[sid] = i
        active -= selected_tiers.keys()
        freeze_json(target/'decisions.json', {str(s): records[s][-1] for s in map(int, result['subjects'])})
    # Audit every original candidate, including apparently poor candidates, on
    # an independent family. Its result cannot change the earlier tier decision.
    target = directory/'audit'
    audit = score_group([(contexts[sid], banks[sid]) for sid in sorted(banks)], settings['audit'],
                        'decision_precision_audit', config['base_seed'], config['parallel_budget'], target)
    summaries = []
    for sid in sorted(banks):
        arrays = {r['id']: load_arrays(target/str(sid), r['id']) for r in banks[sid]}
        d = decision_diagnostics(arrays, primary[sid], settings['tolerance'], settings['alpha'],
                                 settings['bootstrap_replicates'], config['base_seed']+sid+10000)
        decision = records[sid][-1]
        previous_arrays = load_arrays(directory/f"tier_{decision['tier']}"/str(sid), primary[sid])
        sensitivity = independent_budget_difference(arrays[primary[sid]], previous_arrays,
                                                    settings['bootstrap_replicates'],
                                                    config['base_seed']+sid+20000)
        used = sum(len(banks[sid])*r['budget']['particle_count']*r['budget']['filter_seed_count'] for r in records[sid])
        ceiling = len(banks[sid])*sum(b['particle_count']*b['filter_seed_count'] for b in settings['tiers'])
        summaries.append({'subject': sid, 'candidate_count': len(banks[sid]), 'selected': primary[sid],
                          'trials': audit['subjects'][str(sid)]['trials'], 'tiers': records[sid],
                          'decision_status': decision['diagnostic']['status'],
                          'selected_budget': decision['budget'], 'independent_audit': d,
                          'selected_score_sensitivity': sensitivity,
                          'independent_audit_confirms_acceptance':
                              decision['diagnostic']['status'] == d['status'] == 'acceptable_within_bank',
                          'particle_seed_units': used, 'all_tiers_particle_seed_units': ceiling})
    summary = {'subjects': summaries, 'audit_budget': settings['audit'],
               'interpretation': 'Three planned looks use alpha/3 for adaptive regret bounds. Finite-seed bootstrap remains approximate. Audit uses fresh seeds and alpha=.05; it is neither behavioral holdout nor infinite-particle truth.'}
    freeze_json(directory/'summary.json', summary)
    return summary


def guide_shortlist(rows: list[dict], settings: dict) -> list[dict]:
    chosen = ranking(rows)[:settings['guide_top']] + select_elites(rows, settings['guide_diverse'])
    return list({r['id']: r for r in chosen}.values())


def advance_plateau(previous_best: float, new_best: float, stagnant: int,
                    min_improvement: float, patience: int) -> tuple[int, bool]:
    gain = previous_best-new_best
    if gain < -1e-10:
        raise ValueError('Cumulative guide best cannot worsen')
    stagnant = stagnant+1 if gain <= min_improvement else 0
    return stagnant, stagnant >= patience


def challenge_verdict(plateau: bool, status: str) -> str:
    if not plateau:
        return 'budget_cap_without_plateau'
    if status == 'acceptable_within_bank':
        return 'plateau_survived_tested_challenge'
    if status == 'selected_point_inferior':
        return 'plateau_selection_falsified_in_audit'
    return 'plateau_challenge_inconclusive'


def extend_search(config: dict, ctx: dict, output: Path, smoke: bool) -> dict:
    settings = config['extension']; sid = ctx['subject']; seed = config['base_seed']+sid
    directory = output/f'extension_{sid}'; directory.mkdir(exist_ok=True)
    previous = Path(config['previous_stopping'])/f'search_{sid}'
    old_context = json.loads((previous.parent/'context.json').read_text())
    space, anchor = parameter_support(config)
    old = []
    for path in sorted(previous.glob('round_*/search/scores.json')):
        old += json.loads(path.read_text())['rows']
    challenge_old = json.loads((previous/'challenge/search/scores.json').read_text())['rows']
    historical_confirm = json.loads((previous/'confirmation/scores.json').read_text())['rows']
    initial = merge_bank([('initial', ranking(old)[:settings['initial_top_per_source']]),
                          ('initial', ranking(challenge_old)[:settings['initial_top_per_source']]),
                          ('initial', historical_confirm)])
    discovery = old+challenge_old
    search_base_seed = old_context['config']['base_seed']+sid
    if smoke:
        initial = initial[:2]
        # Full-sequence archived scores cannot guide a truncated smoke.
        discovery = []
    freeze_json(directory/'initial_bank.json', initial)
    result = score_group([(ctx, initial)], settings['guide'], 'plateau_guide', seed,
                         config['parallel_budget'], directory/'initial_guide')
    guided = result['subjects'][str(sid)]['rows']
    initial_best = ranking(guided)[0]; best = initial_best['mean_nll']; stagnant = 0
    phases, plateau_round = [], None
    seen = {r['id'] for r in discovery} | {r['id'] for r in initial}

    def evaluate_proposals(bank: list[dict], target: Path) -> tuple[list[dict], dict]:
        target.mkdir(exist_ok=True)
        freeze_json(target/'proposals.json', bank)
        low = score_group([(ctx, bank)], config['search'], 'search', search_base_seed,
                          config['parallel_budget'], target/'discovery')
        rows = low['subjects'][str(sid)]['rows']
        shortlist = guide_shortlist(rows, settings)
        high = score_group([(ctx, shortlist)], settings['guide'], 'plateau_guide', seed,
                           config['parallel_budget'], target/'guide')
        discovery.extend(rows); seen.update(r['id'] for r in rows)
        return high['subjects'][str(sid)]['rows'], {
            'candidates': len(bank), 'guided_candidates': len(shortlist),
            'discovery_seconds': low['seconds'], 'guide_seconds': high['seconds']}

    for iteration in range(1, settings['max_rounds']+1):
        bank = ray_proposals(select_elites(guided, settings['elite_count']), space, seen,
                             settings['proposals_per_elite'], seed+iteration, 'extension')
        for point in initial_points(space, anchor, settings['fresh_global_count'], seed+100+iteration):
            pid = point_id(point)
            if pid not in seen and pid not in {r['id'] for r in bank}:
                bank.append(candidate(point, 'extension', kind='fresh_global'))
        if not bank:
            break
        added, timing = evaluate_proposals(bank, directory/f'round_{iteration}')
        guided += added; new_best = ranking(guided)[0]['mean_nll']
        stagnant, plateau = advance_plateau(best, new_best, stagnant,
                                            settings['min_improvement'], settings['patience'])
        phases.append({'round': iteration, 'best_guide_nll': new_best, 'gain': best-new_best,
                       'stagnant_rounds': stagnant, **timing})
        best = new_best
        freeze_json(directory/f'round_{iteration}/progress.json', phases[-1])
        if plateau:
            plateau_round = iteration
            break
    frozen_base = ranking(guided)[:settings['audit_top_per_source']]
    base_primary = frozen_base[0]['id']
    freeze_json(directory/'frozen_stop.json', {'plateau_round': plateau_round, 'primary': base_primary,
                                             'base': frozen_base, 'phases': phases})
    challenge_rows, challenge_phases = [], []
    for iteration in range(settings['challenge_rounds']):
        bank = propose_round(select_elites(guided+challenge_rows, settings['elite_count']), space,
                             seen, settings['proposals_per_elite'], seed+500+iteration,
                             settings['challenge_local_fraction'], settings['challenge_jump_fraction'])
        for row in bank:
            row['sources'] = ['challenge']
        if iteration == 0:
            for point in initial_points(space, anchor, settings['challenge_global_count'], seed+600):
                pid = point_id(point)
                if pid not in seen and pid not in {r['id'] for r in bank}:
                    bank.append(candidate(point, 'challenge', kind='fresh_global'))
        if not bank:
            continue
        added, timing = evaluate_proposals(bank, directory/f'challenge_{iteration}')
        challenge_rows += added; challenge_phases.append({'round': iteration, **timing})
    bank = merge_bank([('base', frozen_base),
                       ('challenge', ranking(challenge_rows)[:settings['audit_top_per_source']]),
                       ('initial', [initial_best])])
    freeze_json(directory/'audit_bank.json', bank)
    score_group([(ctx, bank)], settings['audit'], 'plateau_independent_audit', seed,
                config['parallel_budget'], directory/'audit')
    arrays = {r['id']: load_arrays(directory/'audit'/str(sid), r['id']) for r in bank}
    criteria = config['calibration']
    decision = decision_diagnostics(arrays, base_primary, criteria['tolerance'], criteria['alpha'],
                                    criteria['bootstrap_replicates'], seed+1000)
    improvement = paired_difference(arrays[initial_best['id']], arrays[base_primary], seed+1100,
                                    criteria['bootstrap_replicates'])
    summary = {'subject': sid, 'initial_primary': initial_best['id'], 'base_primary': base_primary,
               'candidate_sources': {r['id']: r['sources'] for r in bank},
               'initial_guide_nll': initial_best['mean_nll'], 'plateau_round': plateau_round,
               'phases': phases, 'challenge_phases': challenge_phases,
               'initial_minus_base_independent_audit': improvement, 'audit_decision': decision,
               'advice': challenge_verdict(plateau_round is not None, decision['status']),
               'search_candidate_count_new': sum(p['candidates'] for p in phases),
               'challenge_candidate_count': sum(p['candidates'] for p in challenge_phases),
               'guide_budget': settings['guide'], 'audit_budget': settings['audit'],
               'scope': 'One historically selected subject, independent numerical audit, same full behavior sequence. Cheap discovery can omit good candidates. Surviving this challenge is not global convergence or a population stopping error rate.'}
    freeze_json(directory/'summary.json', summary)
    return summary


def run(path: Path, output: Path, smoke: bool, resume: bool) -> None:
    path, output = path.resolve(), output.resolve()
    config = yaml.safe_load(path.read_text())
    for key in ('parameter_space', 'processed_dir', 'previous_stopping', 'previous_adaptive'):
        config[key] = str((path.parent/config[key]).resolve())
    for subject in config['subjects']:
        subject['engine'] = str((path.parent/subject['engine']).resolve())
    if smoke:
        config['parallel_budget'] = 1
        config['subjects'] = [config['subjects'][0], config['subjects'][-1]]
        for budget in [*config['calibration']['tiers'], config['calibration']['audit'],
                       config['extension']['guide'], config['extension']['audit'], config['search']]:
            budget.update(particle_count=2, filter_seed_count=2)
        config['calibration']['bootstrap_replicates'] = 100
        config['extension'].update(max_rounds=2, elite_count=1, proposals_per_elite=2,
                                   fresh_global_count=9, guide_top=1, guide_diverse=1,
                                   initial_top_per_source=1, challenge_rounds=1,
                                   challenge_global_count=10, audit_top_per_source=1)
    versions = {k: importlib.metadata.version(k) for k in ('numpy', 'scipy', 'pandas', 'numba', 'joblib')}
    previous = json.loads((Path(config['previous_stopping'])/'context.json').read_text())
    # No old score is reused unless all recorded original inputs/code/libraries
    # still match. Newly added workflow files do not alter original dependencies.
    verify_reuse_context(previous, versions)
    files = {path, *map(Path, previous['input_sha256'])}
    files.update(Path('src/Bayesian_state').rglob('*.py'))
    files.update(Path(config['previous_stopping']).rglob('*.json'))
    files.update(Path(s['engine']) for s in config['subjects'])
    context = {'config': config, 'smoke': smoke, 'versions': versions,
               'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}}
    if resume:
        if json.loads((output/'context.json').read_text()) != context:
            raise ValueError('Inputs/code/environment changed; use a new output directory')
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output/'context.json', context)
        (output/'workflow_source_at_run.py.txt').write_text(Path(__file__).read_text())
    if (output/'summary.json').exists():
        print('Completed output verified; no computation required.', flush=True)
        return
    contexts = {s['subject']: make_context(config, s, smoke) for s in config['subjects']}
    banks = archived_banks(config)
    if smoke:
        banks = {sid: rows[:2] for sid, rows in banks.items()}
    began = perf_counter()
    calibration_result = calibration(config, contexts, banks, output)
    extension = extend_search(config, contexts[config['extension']['subject']], output, smoke)
    verify_files(context['input_sha256'])
    write_json(output/'summary.json', {'calibration': calibration_result, 'extension': extension,
                                      'wall_seconds': perf_counter()-began, 'smoke': smoke,
                                      'production_configuration_changed': False})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Verify context and reuse complete batches')
    args = parser.parse_args()
    run(args.config, args.output_dir, args.smoke, args.resume)


if __name__ == '__main__':
    main()

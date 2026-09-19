"""Bounded search-coverage and fixed-candidate numerical-budget experiments.

Reuses the shared PF and the previous pilot's scoring contracts. This is an
experimental search policy, not a replacement cognitive model or production
optimizer. Candidate proposals never read archived fit parameters.
"""
from __future__ import annotations

from ...optimization.search.adaptive_proposals import initial_points, select_elites, neighbor_values, propose_round


import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter

from joblib import Parallel, delayed
import numpy as np
import yaml

from ...optimization.model_0826 import (
    WORKSPACE_PROFILE_KEY, build_model_0826_hyper_config,
    extract_model_0826_parameters,
)
from ...optimization.parameter_space import load_model_parameter_space
from ...simulation.data import SubjectTrialDataLoader
from ...utils.parallel import parallel_job_count, single_threaded_processes
from ...utils.seeding import stable_seed
from .pilot_model_0826_simplified_fit import (
    STATE_KEYS, compare_arrays, mixture_nll, paired_difference, point_id,
    read_rows, score_bank, score_one, validation_seeds, write_json,
)


def verify_files(fingerprints: dict) -> None:
    """A reused numerical result requires every recorded dependency unchanged."""
    changed = [str(p) for p, digest in fingerprints.items()
               if not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest() != digest]
    if changed:
        raise ValueError(f'Prior pilot dependencies changed; cannot reuse: {changed}')


def verify_reuse_context(previous: dict, versions: dict) -> None:
    """Check numerical libraries as well as file content before prefix reuse."""
    if previous.get('versions') != versions:
        raise ValueError('Prior pilot numerical library versions changed; cannot reuse')
    verify_files(previous['input_sha256'])


def load_arrays(directory: Path, pid: str) -> dict:
    with np.load(directory/f'{pid}.npz') as data:
        return dict(data)


def subset_arrays(arrays: dict, indices: object) -> dict:
    result = dict(arrays)
    for key in ('probabilities', 'seeds', *STATE_KEYS):
        result[key] = arrays[key][indices]
    return result


def budget_diagnostics(bank: list[dict], arrays: dict[str, dict], count: int,
                       criteria: dict, bootstrap_seed: int) -> dict:
    """Numerical precision within a frozen bank, not a convergence certificate."""
    ids = [r['id'] for r in bank]
    first = arrays[ids[0]]
    y, mask = first['observed'], first['mask']
    probabilities, splits, scores = [], {}, {}
    for pid in ids:
        a = subset_arrays(arrays[pid], slice(0, count))
        np.testing.assert_array_equal(a['seeds'], first['seeds'][:count])
        np.testing.assert_array_equal(a['observed'], y)
        np.testing.assert_array_equal(a['mask'], mask)
        probabilities.append(a['probabilities'][:, np.arange(len(y)), y][:, mask])
        scores[pid] = mixture_nll(a['probabilities'], y, mask)
        splits[pid] = compare_arrays(subset_arrays(a, slice(0, count//2)),
                                     subset_arrays(a, slice(count//2, count)))
    weights = np.random.default_rng(bootstrap_seed).multinomial(
        count, np.full(count, 1/count), size=criteria['bootstrap_replicates']) / count
    boot = np.stack([-np.log(np.clip(weights @ p, 1e-12, 1.)).mean(axis=1) for p in probabilities])
    winner = min(ids, key=lambda p: (scores[p], p))
    regret95 = float(np.quantile(boot[ids.index(winner)] - boot.min(axis=0), .95))
    halfwidth = max((float(np.diff(np.quantile(boot[i]-boot[j], [.025, .975]))[0]/2)
                     for i in range(len(ids)) for j in range(i)), default=0.)
    worst = {key: max(v[key] for v in splits.values()) for key in next(iter(splits.values()))}
    ranking_stable = (halfwidth <= criteria['max_pairwise_interval_halfwidth']
                      and regret95 <= criteria['max_bootstrap_regret95'])
    prediction_stable = (ranking_stable and
                         worst['choice_probability_rmse'] <= criteria['max_choice_probability_rmse'])
    state_stable = all(worst[key] <= criteria['max_'+key] for key in
                       ('mean_rule_total_variation', 'active_probability_mae', 'strategy_probability_mae'))
    group_winners = []
    available = len(first['seeds'])
    for start in range(0, available-count+1, count):
        group = {pid: mixture_nll(arrays[pid]['probabilities'][start:start+count], y, mask) for pid in ids}
        group_winners.append(min(ids, key=lambda pid: (group[pid], pid)))
    return {'seed_count': count, 'scores': scores, 'winner': winner,
            'disjoint_group_winners': group_winners, 'disjoint_group_count': len(group_winners),
            'winner_bootstrap_frequency': float(np.mean(boot.argmin(axis=0) == ids.index(winner))),
            'bootstrap_regret95': regret95, 'max_pairwise_interval_halfwidth': halfwidth,
            'worst_split_seed_difference': worst, 'split_seed_by_candidate': splits,
            'ranking_numerically_stable': bool(ranking_stable),
            'prediction_numerically_stable': bool(prediction_stable),
            'states_numerically_stable': bool(state_stable),
            'interpretation': 'Split metrics use B/2 vs B/2 seeds. A single full-B group is not an independent B-vs-B replication. Bootstrap covers numerical seeds in this bank only.'}


def effort_action(numerical_ok: bool, search_stalled: bool, gap_upper: float | None,
                  last_gain: float, tolerance: float, min_gain: float,
                  output_ok: bool = True) -> str:
    """Separate search-score precision from parameter coverage and output precision.

    ``numerical_ok`` describes score comparisons. Final prediction/state precision
    need not block additional parameter exploration when the score gap is clear.
    """
    if not numerical_ok:
        return 'calibrate_numerical_budget'
    if last_gain > min_gain or (gap_upper is not None and gap_upper > tolerance):
        return 'expand_search_coverage'
    if not output_ok:
        return 'calibrate_output_precision'
    if search_stalled and gap_upper is not None:
        return 'provisional_stop_within_tested_scope'
    return 'insufficient_search_diagnostics'


def subject_context(previous: dict, sid: int, smoke: bool) -> dict:
    subject = next(s for s in previous['config']['subjects'] if s['subject'] == sid)
    engine = yaml.safe_load(Path(subject['engine']).read_text())
    loader = SubjectTrialDataLoader(engine, previous['config']['processed_dir'])
    arrays = loader._extract_arrays(loader._get_subject_frame(sid, 1.), 32 if smoke else None)
    return {**subject, 'engine': engine, 'arrays': arrays,
            'processed_dir': Path(previous['config']['processed_dir'])}


def extend_confirmation(ctx: dict, bank: list[dict], budget: dict, previous_dir: Path,
                        base_seed: int, jobs: int, directory: Path) -> dict:
    """Read-only reuse of a verified prefix, computing only new independent seeds."""
    output = directory/'confirmation'
    if (output/'scores.json').exists():
        return json.loads((output/'scores.json').read_text())
    output.mkdir()
    old = json.loads((previous_dir/'scores.json').read_text())
    if old['budget']['particle_count'] != budget['particle_count']:
        raise ValueError('Particle count differs from reusable bank')
    seeds = validation_seeds(base_seed, ctx['subject'], 'confirmation', budget['filter_seed_count'])
    old_rows = {r['id']: r for r in old['rows']}
    cache = {}
    for row in bank:
        if old_rows[row['id']]['hyperparams'] != row['hyperparams']:
            raise ValueError('Candidate differs from reusable bank')
        cache[row['id']] = load_arrays(previous_dir, row['id'])
        np.testing.assert_array_equal(cache[row['id']]['seeds'], seeds[:len(old['seeds'])])
    count = len(old['seeds'])
    if count >= len(seeds):
        raise ValueError('Extension requires new seeds')
    began = perf_counter()
    new_seeds = seeds[count:]
    with single_threaded_processes():
        runs = Parallel(n_jobs=parallel_job_count(jobs, len(bank)*len(new_seeds)))(
            delayed(score_one)(ctx, row['hyperparams'], budget['particle_count'], seed)
            for row in bank for seed in new_seeds)
    rows = []
    for i, row in enumerate(bank):
        old_arrays = cache[row['id']]
        new = runs[i*len(new_seeds):(i+1)*len(new_seeds)]
        for run in new:
            np.testing.assert_array_equal(run['observed'], old_arrays['observed'])
            np.testing.assert_array_equal(run['mask'], old_arrays['mask'])
        arrays = {key: np.concatenate([old_arrays[key], np.stack([r[key] for r in new])])
                  for key in ('probabilities', *STATE_KEYS)}
        np.savez_compressed(output/f"{row['id']}.npz", **arrays, seeds=seeds,
                            observed=old_arrays['observed'], mask=old_arrays['mask'])
        rows.append({**row, 'mean_nll': mixture_nll(arrays['probabilities'], old_arrays['observed'], old_arrays['mask'])})
    report = {'budget': budget, 'seeds': seeds, 'rows': rows, 'seconds': perf_counter()-began,
              'reused_seed_count_per_candidate': count, 'new_seed_count_per_candidate': len(new_seeds),
              'reused_source': str(previous_dir)}
    write_json(output/'scores.json', report)
    return report


def run_calibration(config: dict, previous: dict, root: Path, smoke: bool) -> dict:
    settings = config['calibration']; sid = settings['subject']
    directory = root/f'calibration_{sid}'
    directory.mkdir(exist_ok=True)
    if (directory/'summary.json').exists():
        return json.loads((directory/'summary.json').read_text())
    original = Path(config['previous_pilot'])/f'subject_{sid}'
    screen = json.loads((original/'screen/scores.json').read_text())
    bank = []
    for source in ('simplified', 'reference'):
        ranked = sorted([r for r in screen['rows'] if source in r['sources']], key=lambda r: (r['mean_nll'], r['id']))
        bank += [{k: r[k] for k in ('id', 'hyperparams', 'sources')} for r in ranked[:settings['candidates_per_source']]]
    bank = list({r['id']: r for r in bank}.values())
    ctx = subject_context(previous, sid, smoke)
    if not (directory/'candidate_bank.json').exists():
        write_json(directory/'candidate_bank.json', bank)
    tables, timings, loaded = [], [], {}
    for particles in settings['particles']:
        target = directory/f'R{particles}'
        target.mkdir(exist_ok=True)
        budget = {'particle_count': particles, 'filter_seed_count': max(settings['seed_counts'])}
        can_reuse = (settings['reuse_previous_confirmation'] and not smoke
                     and particles == previous['config']['confirmation']['particle_count'])
        if can_reuse:
            result = extend_confirmation(ctx, bank, budget, original/'confirmation',
                                         previous['config']['base_seed'], config['parallel_budget'], target)
        else:
            result = score_bank(ctx, bank, budget, 'confirmation', previous['config']['base_seed'],
                                config['parallel_budget'], target)
        arrays = {r['id']: load_arrays(target/'confirmation', r['id']) for r in bank}
        loaded[particles] = arrays
        for count in settings['seed_counts']:
            tables.append({'particles': particles, **budget_diagnostics(bank, arrays, count, config['criteria'],
                            stable_seed({'role': 'adaptive_calibration_bootstrap', 'subject': sid, 'B': count}))})
        timings.append({'particles': particles, 'seconds': result['seconds'],
                        'reused_seed_count_per_candidate': result.get('reused_seed_count_per_candidate', 0)})
        print(json.dumps({'subject': sid, 'phase': 'calibration', 'R': particles,
                          'seconds': result['seconds'], 'last_diagnostics': {k:v for k,v in tables[-1].items()
                          if k not in ('scores', 'split_seed_by_candidate')}}), flush=True)
    changes = []
    for low, high in zip(settings['particles'][:-1], settings['particles'][1:]):
        for row in bank:
            a, b = loaded[low][row['id']], loaded[high][row['id']]
            changes.append({'low_R': low, 'high_R': high, 'candidate': row['id'], **compare_arrays(a,b),
                            **paired_difference(b,a,config['base_seed'],config['criteria']['bootstrap_replicates'])})
    summary = {'subject': sid, 'bank': bank, 'budgets': tables, 'timings': timings,
               'particle_changes_at_max_B': changes,
               'scope': 'Frozen four-candidate bank; prior confirmation prefixes reused only after source/input verification. No new parameter search; no production budget selected automatically.'}
    write_json(directory/'summary.json', summary)
    return summary


def run_search(config: dict, previous: dict, root: Path, smoke: bool) -> dict:
    settings = config['search']; sid = settings['subject']
    directory = root/f'search_{sid}'
    directory.mkdir(exist_ok=True)
    if (directory/'summary.json').exists():
        return json.loads((directory/'summary.json').read_text())
    ctx = subject_context(previous, sid, smoke)
    parameter_space = load_model_parameter_space(config['parameter_space'], expected_model_id='model_0826')
    hyper = build_model_0826_hyper_config(
        {'analysis_id': config['analysis_id'], 'subjects': [sid], 'hyper_base_seed': config['base_seed']},
        parameter_space, 'PMH', directory/'unused_base.yaml', directory/'unused_hyper',
        {'coarse': settings, 'fine': settings, 'final_rescore': {**config['confirmation'], 'seed_family': 'unused'}})
    space = {k: v['values'] for k,v in hyper['stages']['fine']['hyperparam_space'].items()}
    points = initial_points(space, hyper['cd']['initial_points'][0], settings['initial_count'], config['base_seed'])
    all_rows, phases, stagnant = [], [], 0
    best = None
    for iteration in range(settings['rounds']+1):
        target = directory/f'round_{iteration}'
        target.mkdir(exist_ok=True)
        if iteration == 0:
            bank = [{'id':point_id(p), 'hyperparams':p, 'sources':['new_search'],
                     'origin': {'kind':'initial'}} for p in points]
        else:
            elites = select_elites(all_rows, settings['elite_count'])
            bank = propose_round(elites, space, {r['id'] for r in all_rows}, settings['proposals_per_elite'],
                                 config['base_seed']+iteration, settings['local_fraction'], settings['jump_fraction'])
        if not (target/'proposal_plan.json').exists():
            write_json(target/'proposal_plan.json', bank)
        if not bank:
            break
        result = score_bank(ctx, bank, settings, 'search', config['base_seed'], config['parallel_budget'], target)
        all_rows += result['rows']
        value = min(r['mean_nll'] for r in all_rows)
        gain = None if best is None else best-value
        stagnant = stagnant+1 if gain is not None and gain <= settings['min_improvement'] else 0
        best = value
        phase = {'round':iteration, 'new_candidates':len(bank), 'best_mean_nll':best, 'gain':gain,
                 'seconds':result['seconds'], 'stagnant_rounds':stagnant}
        phases.append(phase)
        print(json.dumps({'subject':sid, 'phase':'bounded_search', **phase}), flush=True)
        if stagnant >= settings['patience']:
            break
    # Only after all new proposals are finished do archived parameters enter.
    previous_dir = Path(config['previous_pilot'])/f'subject_{sid}'
    archived = read_rows(Path(ctx['reference_finalists']))
    old_screen = json.loads((previous_dir/'screen/scores.json').read_text())
    baseline = min((r for r in old_screen['rows'] if 'simplified' in r['sources']), key=lambda r:r['mean_nll'])
    bank = {}
    groups = [('new_search', sorted(all_rows,key=lambda r:(r['mean_nll'],r['id']))[:settings['shortlist_size']]),
              ('reference', archived), ('previous_simplified', [baseline])]
    for source, rows in groups:
        for row in rows:
            pid = point_id(row['hyperparams'])
            bank.setdefault(pid, {'id':pid,'hyperparams':row['hyperparams'],'sources':[]})['sources'].append(source)
    screen = score_bank(ctx, list(bank.values()), config['screen'], 'screen', config['base_seed'],
                        config['parallel_budget'], directory)
    chosen, primary = {}, {}
    for source, _ in groups:
        ranked = sorted([r for r in screen['rows'] if source in r['sources']],key=lambda r:(r['mean_nll'],r['id']))
        primary[source] = ranked[0]['id']
        for row in ranked[:config['confirmation']['top_per_source']]:
            chosen[row['id']] = bank[row['id']]
    confirm = score_bank(ctx,list(chosen.values()),config['confirmation'],'confirmation',config['base_seed'],
                         config['parallel_budget'],directory)
    loaded = {pid:load_arrays(directory/'confirmation',pid) for pid in chosen}
    comparisons = {}
    for source in ('reference','previous_simplified'):
        left,right = loaded[primary['new_search']], loaded[primary[source]]
        comparisons[source] = {**compare_arrays(left,right), **paired_difference(left,right,config['base_seed'],
                               config['criteria']['bootstrap_replicates'])}
    numeric = budget_diagnostics(list(chosen.values()),loaded,config['confirmation']['filter_seed_count'],
                                 config['criteria'],config['base_seed'])
    stopped = stagnant >= settings['patience']
    summary = {'subject':sid, 'candidate_count':len(all_rows), 'phases':phases,
               'candidate_cap':settings['initial_count']+settings['rounds']*settings['elite_count']*settings['proposals_per_elite'],
               'primary_selected_on_screen':primary,
               'parameters':{source:extract_model_0826_parameters(bank[pid]['hyperparams']) for source,pid in primary.items()},
               'comparison':comparisons, 'numerical_diagnostics':numeric,
               'search_stalled':stopped, 'search_seconds':sum(r['seconds'] for r in phases),
               'screen_seconds':screen['seconds'], 'confirmation_seconds':confirm['seconds'],
               'recommended_next_action':effort_action(numeric['ranking_numerically_stable'],stopped,
                   comparisons['reference']['paired_numerical_interval95'][1], phases[-1]['gain'] or 0.,
                   config['criteria']['max_mean_nll_gap'],settings['min_improvement'],
                   output_ok=numeric['prediction_numerically_stable'] and
                   (numeric['states_numerically_stable'] or not config['criteria']['require_state_stability'])),
               'scope':'Fresh stratified starts and bounded local/joint proposals on existing fine support; historical parameters never used for proposals. New numerical seeds; one exploratory replicate, not a causal optimizer benchmark or global convergence proof.'}
    write_json(directory/'summary.json',summary)
    return summary


def run(config_path: Path, output: Path, smoke: bool, resume: bool) -> None:
    config_path, output = config_path.resolve(), output.resolve()
    config = yaml.safe_load(config_path.read_text())
    search = config['search']
    if not (0 <= search['local_fraction'] <= 1 and 0 <= search['jump_fraction'] <= 1
            and search['local_fraction'] + search['jump_fraction'] <= 1):
        raise ValueError('Proposal fractions must be nonnegative and sum to at most one')
    for key in ('elite_count','rounds','proposals_per_elite','patience','shortlist_size'):
        if int(search[key]) < 1:
            raise ValueError(f'search.{key} must be positive')
    for key in ('particles','seed_counts'):
        values = config['calibration'][key]
        if values != sorted(set(values)) or any(int(v) != v or v < 2 for v in values):
            raise ValueError(f'calibration.{key} must be increasing unique integers >= 2')
    if any(v % 2 for v in config['calibration']['seed_counts']):
        raise ValueError('Split-seed calibration requires even seed counts')
    for key in ('previous_pilot','parameter_space'):
        config[key] = str((config_path.parent/config[key]).resolve())
    previous_path = Path(config['previous_pilot'])/'context.json'
    previous = json.loads(previous_path.read_text())
    versions = {k:importlib.metadata.version(k) for k in ('numpy','scipy','pandas','numba','joblib')}
    verify_reuse_context(previous, versions)
    if smoke:
        config['parallel_budget'] = 1
        config['search'].update(initial_count=9,elite_count=2,rounds=1,proposals_per_elite=4,shortlist_size=1)
        for name in ('search','screen','confirmation'):
            config[name].update(particle_count=2,filter_seed_count=2)
        config['confirmation']['top_per_source'] = 1
        config['calibration'].update(particles=[2,4],seed_counts=[2,4],candidates_per_source=1)
        config['criteria']['bootstrap_replicates'] = 100
    files = {Path(p) for p in previous['input_sha256']}
    files.update([config_path,previous_path,Path(config['parameter_space'])])
    files.update(Path(config['previous_pilot']).rglob('*.json'))
    files.update(Path(config['previous_pilot']).rglob('*.npz'))
    files.update(Path('src/Bayesian_state').rglob('*.py'))
    context = {'config':config,'smoke':smoke,
               'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)},
               'versions':versions}
    if resume:
        if json.loads((output/'context.json').read_text()) != context:
            raise ValueError('Inputs/source/config/versions changed; use a new output directory')
    else:
        output.mkdir(parents=True,exist_ok=False)
        write_json(output/'context.json',context)
    if (output/'summary.json').exists():
        print((output/'summary.json').read_text(),flush=True)
        return
    search = run_search(config,previous,output,smoke)
    calibration = run_calibration(config,previous,output,smoke)
    verify_files(context['input_sha256'])
    write_json(output/'summary.json',{'smoke':smoke,'search':search,'calibration':calibration,
               'policy':'Use common numerical/scientific criteria, not identical compute per subject. These are diagnostic recommendations; production configuration was not changed.'})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',action='store_true',help='Reuse complete batches in an identical context')
    args = parser.parse_args()
    run(args.config,args.output_dir,args.smoke,args.resume)


if __name__ == '__main__':
    main()

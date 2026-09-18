"""Audit bounded search stopping and localize PF score variability.

No cognitive equations or production defaults change. Search challenges are
computed even after a provisional plateau, so they can falsify an early stop.
Historical S129 warm starts are a separate, explicitly labelled experiment.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import yaml

from ...optimization.model_0826 import WORKSPACE_PROFILE_KEY, build_model_0826_hyper_config
from ...optimization.model_0826 import extract_model_0826_parameters
from ...optimization.parameter_space import load_model_parameter_space
from ...simulation.data import SubjectTrialDataLoader
from ...simulation.execution import evaluate_state_model_run
from ...simulation.parameters import apply_fixed_hyperparams_to_engine_config
from ...utils.parallel import parallel_job_count, single_threaded_processes
from .pilot_model_0826_adaptive_effort import (
    budget_diagnostics, initial_points, load_arrays, neighbor_values,
    select_elites, subset_arrays, verify_files,
)
from .pilot_model_0826_simplified_fit import (
    STATE_KEYS, compare_arrays, mixture_nll, paired_difference, point_id,
    score_bank, write_json,
)


def candidate(point: dict, source: str, **origin: object) -> dict:
    return {'id': point_id(point), 'hyperparams': deepcopy(point),
            'sources': [source], 'origin': origin}


def merge_bank(groups: list[tuple[str, list[dict]]]) -> list[dict]:
    bank = {}
    for source, rows in groups:
        for row in rows:
            pid = row['id']
            item = bank.setdefault(pid, candidate(row['hyperparams'], source))
            if source not in item['sources']:
                item['sources'].append(source)
    return list(bank.values())


def axis_values(key: str, current: object, values: list) -> list[list]:
    """Primitive-coordinate rays; derived E_E moves with E_C / delta_E."""
    if not isinstance(current, dict) or key == WORKSPACE_PROFILE_KEY:
        return [values]
    named = [extract_model_0826_parameters({key: v}) for v in values]
    anchor = extract_model_0826_parameters({key: current})
    names = [k for k in anchor if k != 'E_E']
    return [[v for v, p in zip(values, named)
             if all(abs(p[k]-anchor[k]) < 1e-9 for k in names if k != axis)]
            for axis in names]


def ray_proposals(elites: list[dict], space: dict, seen: set[str],
                  per_elite: int, seed: int, source: str) -> list[dict]:
    """Round-robin axes with both long jumps and neighbors, without dense grids.

    Every primitive axis gets opportunities before extra points on any axis.
    This avoids spending most proposals in a large flattened joint profile.
    """
    rng = np.random.default_rng(seed)
    pending = {}
    for elite in elites:
        anchor = elite['hyperparams']
        pools = []
        for key, values in space.items():
            for ray in axis_values(key, anchor[key], values):
                ordered = [ray[int(i)] for i in np.unique(np.linspace(0, len(ray)-1, 3).astype(int))]
                ordered += [v for v in neighbor_values(key, anchor[key], values) if v in ray]
                ordered += [ray[int(i)] for i in rng.permutation(len(ray))]
                pools.append([{**deepcopy(anchor), key: deepcopy(v)} for v in ordered])
        added = 0
        while added < per_elite and any(pools):
            for pool in pools:
                while pool:
                    point = pool.pop(0)
                    pid = point_id(point)
                    if pid in seen or pid in pending:
                        continue
                    pending[pid] = candidate(point, source, kind='primitive_ray', anchor=elite['id'])
                    added += 1
                    break
                if added >= per_elite:
                    break
    return list(pending.values())


def stopping_advice(ranking_ok: bool, plateau: bool, challenger_gap_lower: float,
                    challenger_gap_upper: float, tolerance: float) -> str:
    """Gap is base minus challenger: positive means the challenger is better."""
    if not ranking_ok:
        return 'calibrate_score_precision'
    if challenger_gap_lower > tolerance:
        return 'expand_search_challenger_improved'
    if not plateau:
        return 'budget_cap_without_plateau'
    if challenger_gap_upper > tolerance:
        return 'challenge_inconclusive'
    return 'provisional_stop_within_tested_scope'


def make_context(config: dict, subject: dict, smoke: bool) -> dict:
    engine = yaml.safe_load(Path(subject['engine']).read_text())
    loader = SubjectTrialDataLoader(engine, config['processed_dir'])
    frame = loader._get_subject_frame(subject['subject'], 1.)
    if set(frame['condition']) != {subject['condition']}:
        raise ValueError('Subject condition differs from configuration')
    arrays = loader._extract_arrays(frame, 32 if smoke else None)
    return {**subject, 'engine': engine, 'arrays': arrays,
            'processed_dir': Path(config['processed_dir'])}


def parameter_support(config: dict) -> tuple[dict, dict]:
    space = load_model_parameter_space(config['parameter_space'], expected_model_id='model_0826')
    b = config['search']
    hyper = build_model_0826_hyper_config(
        {'analysis_id': config['analysis_id'], 'subjects': [129], 'hyper_base_seed': config['base_seed']},
        space, 'PMH', 'unused_base.yaml', 'unused_output',
        {'coarse': b, 'fine': b, 'final_rescore': {**config['confirmation'], 'seed_family': 'unused'}})
    return {k: v['values'] for k, v in hyper['stages']['fine']['hyperparam_space'].items()}, hyper['cd']['initial_points'][0]


def search_subject(config: dict, subject: dict, output: Path, smoke: bool) -> dict:
    sid = subject['subject']; directory = output/f'search_{sid}'
    directory.mkdir(exist_ok=True)
    if (directory/'summary.json').exists():
        return json.loads((directory/'summary.json').read_text())
    ctx = make_context(config, subject, smoke)
    space, anchor = parameter_support(config)
    settings = config['search']; seed = config['base_seed'] + sid
    points = initial_points(space, anchor, settings['initial_count'], seed)
    rows, phases, stagnant, best, first_plateau = [], [], 0, None, None
    plateau_rows = None
    for iteration in range(settings['rounds']+1):
        target = directory/f'round_{iteration}'; target.mkdir(exist_ok=True)
        bank = ([candidate(p, 'base', kind='initial') for p in points] if iteration == 0 else
                ray_proposals(select_elites(rows, settings['elite_count']), space,
                              {r['id'] for r in rows}, settings['proposals_per_elite'], seed+iteration, 'base'))
        if not bank:
            break
        if not (target/'proposals.json').exists():
            write_json(target/'proposals.json', bank)
        scored = score_bank(ctx, bank, settings, 'search', seed, config['parallel_budget'], target)
        rows += scored['rows']; new_best = min(r['mean_nll'] for r in rows)
        gain = None if best is None else best-new_best
        stagnant = stagnant+1 if gain is not None and gain <= settings['min_improvement'] else 0
        best = new_best
        if stagnant >= settings['patience'] and first_plateau is None:
            first_plateau = iteration
            plateau_rows = list(rows)
        phases.append({'round': iteration, 'candidates': len(bank), 'best_mean_nll': best,
                       'gain': gain, 'stagnant_rounds': stagnant, 'seconds': scored['seconds']})
        # Deliberately continue to the cap even if a plateau is reached: this is
        # a falsification experiment for stopping, not an automatic fitter.
    ranking = lambda values: sorted(values, key=lambda r: (r['mean_nll'], r['id']))
    base_rows = plateau_rows if plateau_rows is not None else rows
    base = ranking(base_rows)[:settings['shortlist_size']]
    write_path = directory/'frozen_base.json'
    if not write_path.exists():
        write_json(write_path, {'shortlist': base, 'first_plateau_round': first_plateau, 'phases': phases})
    seen = {r['id'] for r in rows}
    challenge = ray_proposals(select_elites(rows, 2), space, seen,
                             settings['challenge_local_count']//2, seed+100, 'challenge')
    for p in initial_points(space, anchor, settings['challenge_global_count'], seed+200):
        if point_id(p) not in seen and point_id(p) not in {r['id'] for r in challenge}:
            challenge.append(candidate(p, 'challenge', kind='fresh_global'))
    target = directory/'challenge'; target.mkdir(exist_ok=True)
    challenged = score_bank(ctx, challenge, settings, 'search', seed, config['parallel_budget'], target)
    after_stop = rows[len(base_rows):]
    groups = [('base', base), ('challenge', ranking(challenged['rows']+after_stop)[:settings['shortlist_size']])]
    warm_seconds = 0.
    if sid == 129:
        # Historical values enter only after cold search and challenge are frozen.
        old = json.loads((Path(config['previous_simple'])/'subject_129/screen/scores.json').read_text())
        refs = ranking([r for r in old['rows'] if 'reference' in r['sources']])
        warm = ray_proposals(refs[:2], space, {r['id'] for r in refs}, settings['proposals_per_elite'],
                             seed+300, 'warm')
        target = directory/'warm'; target.mkdir(exist_ok=True)
        warm_result = score_bank(ctx, warm, settings, 'search', seed, config['parallel_budget'], target)
        warm_seconds = warm_result['seconds']
        groups += [('reference', refs), ('warm', ranking(warm_result['rows'])[:settings['shortlist_size']])]
    bank = merge_bank(groups)
    screen = score_bank(ctx, bank, config['screen'], 'screen', seed, config['parallel_budget'], directory)
    primary, chosen = {}, {}
    for source, _ in groups:
        ranked = ranking([r for r in screen['rows'] if source in r['sources']])
        primary[source] = ranked[0]['id']
        for row in ranked[:config['confirmation']['top_per_source']]:
            chosen[row['id']] = next(r for r in bank if r['id'] == row['id'])
    confirmation = score_bank(ctx, list(chosen.values()), config['confirmation'], 'confirmation', seed,
                              config['parallel_budget'], directory)
    arrays = {pid: load_arrays(directory/'confirmation', pid) for pid in chosen}
    differences = {source: {**paired_difference(arrays[primary['base']], arrays[pid], seed,
                                  config['criteria']['bootstrap_replicates']),
                            **compare_arrays(arrays[primary['base']], arrays[pid])}
                   for source, pid in primary.items() if source != 'base'}
    numeric = budget_diagnostics(list(chosen.values()), arrays, config['confirmation']['filter_seed_count'],
                                 config['criteria'], seed)
    interval = differences['challenge']['paired_numerical_interval95']
    plateau = first_plateau is not None
    summary = {'subject': sid, 'condition': subject['condition'], 'trials': len(ctx['arrays'].feedback),
               'scored_trials': int(next(iter(arrays.values()))['mask'].sum()),
               'phases': phases, 'first_plateau_round': first_plateau,
               'plateau_at_cap': stagnant >= settings['patience'], 'proposed_early_stop': plateau,
               'base_candidates': len(base_rows), 'total_search_candidates': len(rows),
               'post_stop_candidates': len(after_stop), 'challenge_candidates': len(challenge),
               'primary_selected_on_independent_screen': primary,
               'confirmed_primary_scores': {k: next(r['mean_nll'] for r in confirmation['rows'] if r['id'] == pid)
                                             for k, pid in primary.items()},
               'parameters': {k: extract_model_0826_parameters(chosen[pid]['hyperparams']) for k, pid in primary.items()},
               'base_minus_comparator': differences, 'numerical_diagnostics': numeric,
               'advice': stopping_advice(numeric['ranking_numerically_stable'], plateau,
                                        interval[0], interval[1], config['criteria']['max_mean_nll_gap']),
               'seconds': {'search': sum(p['seconds'] for p in phases), 'challenge': challenged['seconds'],
                           'warm': warm_seconds, 'screen': screen['seconds'], 'confirmation': confirmation['seconds']},
               'scope': 'Full-sequence in-sample pilot. Primary points frozen before confirmation. No global optimum, population inference, or production stopping validation claimed.'}
    if sid == 129 and differences['reference']['paired_numerical_interval95'][0] > config['criteria']['max_mean_nll_gap']:
        summary['advice'] = 'expand_search_known_reference_better'
    write_json(directory/'summary.json', summary)
    print(json.dumps({'subject': sid, 'completed': True, 'advice': summary['advice']}), flush=True)
    return summary


def influence_diagnostics(arrays: dict, segment_length: int) -> tuple[dict, list[dict]]:
    """Delta-method seed influence; retain within-run temporal covariance.

    Segment covariance shares sum to one, but can be negative. Independent-trial
    variance is a diagnostic comparison, never the reported uncertainty estimate.
    """
    p = arrays['probabilities']; y = arrays['observed']; mask = arrays['mask']
    q = p[:, np.arange(len(y)), y][:, mask]; mu = q.mean(axis=0)
    influence = -(q-mu)/np.clip(mu, 1e-12, 1.)
    seed_score = influence.mean(axis=1); b, t = q.shape
    total_var = float(seed_score.var(ddof=1))
    variance = influence.var(axis=0, ddof=1)
    top = np.argsort(variance)[::-1]; n = max(1, int(np.ceil(.1*t)))
    trial_indices = np.flatnonzero(mask)
    segments = []
    for start in range(0, len(y), segment_length):
        use = (trial_indices >= start) & (trial_indices < start+segment_length)
        contribution = influence[:, use].sum(axis=1)/t
        share = float(np.cov(contribution, seed_score, ddof=1)[0, 1]/total_var) if total_var else 0.
        segments.append({'first_trial': start+1, 'last_trial': min(start+segment_length, len(y)),
                         'score_variance_covariance_share': share,
                         'mean_observed_probability': float(mu[use].mean()) if use.any() else None})
    summary = {'seed_count': b, 'mean_nll': mixture_nll(p, y, mask),
               'delta_method_mcse': float(np.sqrt(total_var/b)),
               'independent_trial_mcse_diagnostic': float(np.sqrt(variance.sum()/t**2/b)),
               'temporal_covariance_variance_ratio': float(total_var/(variance.sum()/t**2)) if variance.sum() else 0.,
               'top_decile_pointwise_variance_share': float(variance[top[:n]].sum()/variance.sum()) if variance.sum() else 0.,
               'highest_variance_trials': (trial_indices[top[:12]]+1).tolist(),
               'minimum_mean_observed_probability': float(mu.min()),
               'individual_seed_floor_count': int((q <= 1e-12).sum())}
    return summary, segments


def trace_one(ctx: dict, row: dict, particles: int, seed: int) -> dict:
    engine = apply_fixed_hyperparams_to_engine_config(ctx['engine'], row['hyperparams'])
    engine['inference']['particle_count'] = particles
    start = perf_counter()
    result = evaluate_state_model_run(
        subject_id=ctx['subject'], condition=ctx['condition'], arrays=ctx['arrays'], params={},
        engine_config_template=engine, processed_data_dir=ctx['processed_dir'], window_size=16,
        keep_logs=True, prediction_mode='prior_t', selection_prediction_mode='prior_t',
        loss_metric='choice_nll', trajectory_seed=seed)
    m = result.metrics_by_mode['prior_t']; log = result.state_log
    p = np.asarray(m['pred_category_probs'])
    np.testing.assert_allclose(p.sum(axis=1), 1., atol=1e-10)
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('Invalid prediction probabilities')
    keys = (*STATE_KEYS, 'pre_choice_ess', 'post_choice_ess', 'resampled')
    values = {k: np.asarray(log[k]) for k in keys}
    if not all(np.isfinite(a).all() for a in values.values()):
        raise ValueError('Non-finite state / PF diagnostic')
    return {'probabilities': p, 'observed': np.asarray(m['observed_choice_index']),
            'mask': np.asarray(m['valid_trial_mask'], dtype=bool),
            'seconds': perf_counter()-start, **values}


def noise_experiment(config: dict, output: Path, smoke: bool) -> dict:
    directory = output/'noise_229'; directory.mkdir(exist_ok=True)
    if (directory/'summary.json').exists():
        return json.loads((directory/'summary.json').read_text())
    if any(directory.glob('R*.npz')):
        raise ValueError('Partial diagnostic output preserved; use a new output directory')
    settings = config['noise']; ctx = make_context(config, settings, smoke)
    previous = Path(config['previous_adaptive'])/'calibration_229'
    bank = json.loads((previous/'candidate_bank.json').read_text())
    archival, segments = [], []
    for particles in (32, 64, 128):
        for row in bank:
            a = load_arrays(previous/f'R{particles}/confirmation', row['id'])
            summary, parts = influence_diagnostics(a, settings['segment_length'])
            archival.append({'candidate': row['id'], 'particles': particles, **summary})
            segments += [{'candidate': row['id'], 'particles': particles, **part} for part in parts]
    if not (directory/'archived_noise.json').exists():
        write_json(directory/'archived_noise.json', archival)
        pd.DataFrame(segments).to_csv(directory/'archived_segments.csv', index=False)
    old = json.loads((previous/'R128/confirmation/scores.json').read_text())
    seeds = old['seeds'][:settings['seed_count']]
    # Trace the archived seed prefix exactly, then increase R for the two noisy
    # reference candidates. These are paired diagnostic reruns, not new seeds.
    specs = [(settings['trace_particles'], row) for row in bank]
    specs += [(settings['larger_particles'], row) for row in bank if 'reference' in row['sources']]
    began = perf_counter()
    with single_threaded_processes():
        raw = Parallel(n_jobs=parallel_job_count(config['parallel_budget'], len(specs)*len(seeds)))(
            delayed(trace_one)(ctx, row, particles, seed) for particles, row in specs for seed in seeds)
    records, arrays_by_key, segment_rows = [], {}, []
    for i, (particles, row) in enumerate(specs):
        runs = raw[i*len(seeds):(i+1)*len(seeds)]; first = runs[0]
        for r in runs:
            np.testing.assert_array_equal(r['observed'], first['observed'])
            np.testing.assert_array_equal(r['mask'], first['mask'])
        a = {k: np.stack([r[k] for r in runs]) for k in
             ('probabilities', *STATE_KEYS, 'pre_choice_ess', 'post_choice_ess', 'resampled')}
        a.update(observed=first['observed'], mask=first['mask'], seeds=np.array(seeds))
        np.savez_compressed(directory/f"R{particles}_{row['id']}.npz", **a)
        arrays_by_key[particles, row['id']] = a
        replay = None
        if particles == settings['trace_particles'] and not smoke:
            historical = load_arrays(previous/'R128/confirmation', row['id'])
            for key in ('probabilities', *STATE_KEYS):
                np.testing.assert_array_equal(a[key], historical[key][:len(seeds)])
            replay = 'exact_array_equal'
        summary, parts = influence_diagnostics(a, settings['segment_length'])
        for part in parts:
            sl = slice(part['first_trial']-1, part['last_trial'])
            segment_rows.append({'candidate': row['id'], 'particles': particles, **part,
                                 'mean_post_ess_fraction': float(a['post_choice_ess'][:, sl].mean()/particles),
                                 'resample_fraction': float(a['resampled'][:, sl].mean())})
        records.append({'candidate': row['id'], 'particles': particles, 'parameters': extract_model_0826_parameters(row['hyperparams']),
                        **summary, 'archived_prefix_replay': replay,
                        'mean_pre_ess_fraction': float(a['pre_choice_ess'].mean()/particles),
                        'mean_post_ess_fraction': float(a['post_choice_ess'].mean()/particles),
                        'post_ess_fraction_q05': float(np.quantile(a['post_choice_ess']/particles, .05)),
                        'resample_fraction': float(a['resampled'].mean()),
                        'post_ess_below_tenth_fraction': float((a['post_choice_ess'] < particles*.1).mean())})
    comparisons = []
    for row in bank:
        key = settings['larger_particles'], row['id']
        if key in arrays_by_key:
            left, right = arrays_by_key[key], arrays_by_key[settings['trace_particles'], row['id']]
            comparisons.append({'candidate': row['id'], **compare_arrays(left, right),
                                **paired_difference(left, right, config['base_seed'], config['criteria']['bootstrap_replicates'])})
    pd.DataFrame(segment_rows).to_csv(directory/'trace_segments.csv', index=False)
    summary = {'records': records, 'higher_minus_lower_R': comparisons, 'seconds': perf_counter()-began,
               'scope': 'Archived seed-prefix replay with ESS traces and paired particle-count sensitivity. Parameter associations and ESS correlations do not establish a causal source of error. No cognitive parameter changed.'}
    write_json(directory/'summary.json', summary)
    return summary


def run(path: Path, output: Path, smoke: bool, resume: bool) -> None:
    path, output = path.resolve(), output.resolve()
    config = yaml.safe_load(path.read_text())
    for key in ('parameter_space', 'processed_dir', 'previous_adaptive', 'previous_simple'):
        config[key] = str((path.parent/config[key]).resolve())
    for subject in [*config['subjects'], config['noise']]:
        subject['engine'] = str((path.parent/subject['engine']).resolve())
    if smoke:
        config['parallel_budget'] = 1
        # Exercise both ordinary and condition-3 model branches on 32 trials.
        config['subjects'] = [config['subjects'][0], config['subjects'][-1]]
        config['search'].update(initial_count=9, rounds=1, elite_count=1, proposals_per_elite=4,
                                challenge_global_count=10, challenge_local_count=4, shortlist_size=1)
        for phase in ('search', 'screen', 'confirmation'):
            config[phase].update(particle_count=2, filter_seed_count=2)
        config['confirmation']['top_per_source'] = 1
        config['noise'].update(trace_particles=2, larger_particles=4, seed_count=2)
        config['criteria']['bootstrap_replicates'] = 100
    versions = {k: importlib.metadata.version(k) for k in ('numpy', 'scipy', 'pandas', 'numba', 'joblib')}
    files = {path, Path(config['parameter_space'])}
    files.update(Path('src/Bayesian_state').rglob('*.py'))
    files.update(Path(config['processed_dir']).glob('*.csv'))
    files.update(Path('src/Bayesian_state/hypothesis_space').rglob('*.npy'))
    files.update(Path(s['engine']) for s in [*config['subjects'], config['noise']])
    for key in ('previous_adaptive', 'previous_simple'):
        files.update(Path(config[key]).rglob('*.json'))
        files.update(Path(config[key]).rglob('*.npz'))
        previous = json.loads((Path(config[key])/'context.json').read_text())
        files.update(Path(p) for p in previous['input_sha256'])
    context = {'config': config, 'smoke': smoke, 'versions': versions,
               'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}}
    if resume:
        if json.loads((output/'context.json').read_text()) != context:
            raise ValueError('Inputs, code, or environment changed; use a new output directory')
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output/'context.json', context)
        (output/'workflow_source_at_run.py.txt').write_text(Path(__file__).read_text())
    if (output/'summary.json').exists():
        print('Completed output verified; no computation required.', flush=True)
        return
    searches = []
    for subject in config['subjects']:
        searches.append(search_subject(config, subject, output, smoke))
        if subject['subject'] == 129:
            noise = noise_experiment(config, output, smoke)
    verify_files(context['input_sha256'])
    write_json(output/'summary.json', {'smoke': smoke, 'searches': searches, 'noise': noise,
                                      'production_configuration_changed': False})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Verify context and reuse completed batches')
    args = parser.parse_args()
    run(args.config, args.output_dir, args.smoke, args.resume)


if __name__ == '__main__':
    main()

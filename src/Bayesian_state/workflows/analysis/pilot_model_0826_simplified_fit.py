"""Compare a bounded single-stage search with archived full-search finalists.

Uses the shared Hyper-CD and simulation evaluator. Archived points enter only
after a new search finishes, never its initialization or candidate proposals.
This changes search approximation, not cognitive equations. Confirmation in
this pilot is deliberately independent of both search and shortlist screening.
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
import yaml

from ...optimization.model_0826 import (
    build_model_0826_hyper_config, extract_model_0826_parameters,
)
from ...optimization.parameter_space import load_model_parameter_space
from ...optimization.search.cd_v2 import canonical_point_key
from ...optimization.search.coordinate_descent import HyperCDOptimizer
from ...simulation.data import SubjectTrialDataLoader
from ...simulation.execution import evaluate_state_model_run
from ...simulation.parameters import apply_fixed_hyperparams_to_engine_config
from ...utils.parallel import parallel_job_count, single_threaded_processes
from ...utils.seeding import stable_seed


STATE_KEYS = (
    'marginal_prior', 'marginal_active_probability',
    'predictive_strategy_exploit', 'predictive_strategy_local_explore',
    'predictive_strategy_global_explore',
)


def write_json(path: Path, value: object) -> None:
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def point_id(point: dict) -> str:
    return hashlib.sha256(canonical_point_key(point).encode()).hexdigest()[:16]


def select_diverse(rows: list[dict], top_k: int, per_workspace: bool) -> list[dict]:
    """Keep global leaders plus the best observed member of each M/chi cell."""
    ranked = sorted(rows, key=lambda row: (row['aggregated_error'], point_id(row['hyperparams'])))
    selected = {point_id(row['hyperparams']): row for row in ranked[:top_k]}
    if per_workspace:
        seen = set()
        for row in ranked:
            p = extract_model_0826_parameters(row['hyperparams'])
            cell = (p['M'], p['chi'])
            if cell not in seen:
                selected[point_id(row['hyperparams'])] = row
                seen.add(cell)
    return list(selected.values())


def mixture_nll(probabilities: np.ndarray, observed: np.ndarray, mask: np.ndarray) -> float:
    """Average seed probabilities before the log, on the original score mask."""
    mean = probabilities.mean(axis=0)
    selected = mean[np.arange(len(observed)), observed]
    return float(-np.log(np.clip(selected[mask], 1e-12, 1.)).mean())


def validation_seeds(base_seed: int, subject: int, phase: str, count: int) -> list[int]:
    return [stable_seed({'role': 'simplified_fit_pilot', 'base': base_seed,
                        'subject': subject, 'phase': phase, 'index': i}) for i in range(count)]


def score_one(context: dict, point: dict, particles: int, seed: int) -> dict:
    engine = apply_fixed_hyperparams_to_engine_config(context['engine'], point)
    engine['inference']['particle_count'] = particles
    began = perf_counter()
    result = evaluate_state_model_run(
        subject_id=context['subject'], condition=context['condition'], arrays=context['arrays'],
        params={}, engine_config_template=engine,
        processed_data_dir=context['processed_dir'], window_size=16, keep_logs=True,
        prediction_mode='prior_t', selection_prediction_mode='prior_t',
        loss_metric='choice_nll', trajectory_seed=seed,
    )
    metrics = result.metrics_by_mode['prior_t']
    p = np.asarray(metrics['pred_category_probs'])
    assert np.isfinite(p).all() and (p >= 0).all()
    np.testing.assert_allclose(p.sum(axis=1), 1., atol=1e-10)
    states = {key: np.asarray(result.state_log[key]) for key in STATE_KEYS}
    assert all(np.isfinite(value).all() for value in states.values())
    return {'probabilities': p, 'observed': np.asarray(metrics['observed_choice_index']),
            'mask': np.asarray(metrics['valid_trial_mask'], dtype=bool),
            'seconds': perf_counter() - began, **states}


def score_bank(context: dict, bank: list[dict], budget: dict, phase: str,
               base_seed: int, jobs: int, output: Path) -> dict:
    """Flatten candidate x seed work; checkpoint completed banks without overwriting."""
    directory = output / phase
    if (directory / 'scores.json').exists():
        return json.loads((directory / 'scores.json').read_text())
    directory.mkdir(exist_ok=False)
    seeds = validation_seeds(base_seed, context['subject'], phase, budget['filter_seed_count'])
    started = perf_counter()
    with single_threaded_processes():
        raw = Parallel(n_jobs=parallel_job_count(jobs, len(bank) * len(seeds)))(
            delayed(score_one)(context, row['hyperparams'], budget['particle_count'], seed)
            for row in bank for seed in seeds)
    rows = []
    observed, mask = raw[0]['observed'], raw[0]['mask']
    for index, row in enumerate(bank):
        runs = raw[index * len(seeds):(index + 1) * len(seeds)]
        for run in runs:
            np.testing.assert_array_equal(run['observed'], observed)
            np.testing.assert_array_equal(run['mask'], mask)
        arrays = {key: np.stack([run[key] for run in runs])
                  for key in ('probabilities', *STATE_KEYS)}
        np.savez_compressed(directory / f"{row['id']}.npz", **arrays,
                            observed=observed, mask=mask, seeds=seeds)
        rows.append({**row, 'mean_nll': mixture_nll(arrays['probabilities'], observed, mask),
                     'worker_seconds': sum(run['seconds'] for run in runs)})
    report = {'subject': context['subject'], 'phase': phase, 'budget': budget,
              'seeds': seeds, 'seconds': perf_counter() - started, 'rows': rows}
    write_json(directory / 'scores.json', report)
    print(json.dumps({'subject': context['subject'], 'phase': phase,
                      'candidates': len(rows), 'seconds': report['seconds']}), flush=True)
    return report


def compare_arrays(left: dict, right: dict) -> dict:
    """Pre-choice marginals, not reconstructed individual latent histories."""
    np.testing.assert_array_equal(left['observed'], right['observed'])
    np.testing.assert_array_equal(left['mask'], right['mask'])
    lp, rp = left['probabilities'].mean(axis=0), right['probabilities'].mean(axis=0)
    result = {'choice_probability_rmse': float(np.sqrt(np.mean((lp-rp)**2))),
              'mean_rule_total_variation': float(np.mean(.5*np.sum(np.abs(
                  left['marginal_prior'].mean(axis=0)-right['marginal_prior'].mean(axis=0)), axis=1))),
              'active_probability_mae': float(np.mean(np.abs(
                  left['marginal_active_probability'].mean(axis=0)-right['marginal_active_probability'].mean(axis=0))))}
    strategy = [float(np.mean(np.abs(left[k].mean(axis=0)-right[k].mean(axis=0)))) for k in STATE_KEYS[2:]]
    result['strategy_probability_mae'] = float(max(strategy))
    return result


def paired_difference(left: dict, right: dict, seed: int, count: int) -> dict:
    """Numerical seed bootstrap for fixed, independently selected candidates."""
    np.testing.assert_array_equal(left['seeds'], right['seeds'])
    np.testing.assert_array_equal(left['observed'], right['observed'])
    np.testing.assert_array_equal(left['mask'], right['mask'])
    y, mask = left['observed'], left['mask']
    l = left['probabilities'][:, np.arange(len(y)), y][:, mask]
    r = right['probabilities'][:, np.arange(len(y)), y][:, mask]
    b = len(l)
    weights = np.random.default_rng(seed).multinomial(b, np.full(b, 1/b), size=count) / b
    boot = -np.log(np.clip(weights @ l, 1e-12, 1)).mean(axis=1) + np.log(np.clip(weights @ r, 1e-12, 1)).mean(axis=1)
    return {'mean_nll_difference': float(-np.log(np.clip(l.mean(axis=0), 1e-12, 1)).mean()
                                       + np.log(np.clip(r.mean(axis=0), 1e-12, 1)).mean()),
            'paired_numerical_interval95': np.quantile(boot, [.025, .975]).tolist()}


def prepare_search(config: dict, subject: dict, root: Path, smoke: bool) -> tuple[dict, Path]:
    directory = root / f"subject_{subject['subject']}"
    directory.mkdir()
    base = {
        'engine_config_path': subject['engine'], 'subjects': [subject['subject']],
        'dataset': {'processed_dir': config['processed_dir'], 'learning_data': 'Task2_processed.csv',
                    'perception_summary': 'Task1b_errorsummary_24.csv',
                    'perception_summary_72': 'Task1b_errorsummary_72.csv',
                    'feature_order_data': 'Task2_processed.csv'},
        'output_dir': str(directory / 'unused_fixed_simulation'), 'n_jobs': config['parallel_budget'],
        'simulation_repeats': config['search']['filter_seed_count'], 'repeat_aggregation': 'mean_probability',
        'window_size': 16, 'stop_at': 1., 'max_trials': 32 if smoke else None, 'keep_logs': False,
        'prediction_mode': 'prior_t', 'selection_prediction_mode': 'prior_t',
        'loss_metric': 'choice_nll', 'evaluation_protocol': {'mode': 'all'},
        'statistics_config': {'enabled': False},
    }
    base_path = directory / 'base.yaml'
    base_path.write_text(yaml.safe_dump(base, sort_keys=False))
    space = load_model_parameter_space(config['parameter_space'], expected_model_id='model_0826')
    analysis = {'analysis_id': config['analysis_id'], 'subjects': [subject['subject']],
                'hyper_base_seed': config['base_seed'], 'max_trials': base['max_trials'],
                'cd': {**config['search'], 'parallel_budget': config['parallel_budget']}}
    budgets = {'coarse': config['search'], 'fine': config['search'],
               'final_rescore': {**config['confirmation'], 'seed_family': 'unused'}}
    hyper = build_model_0826_hyper_config(analysis, space, 'PMH', base_path, directory/'search', budgets)
    hyper['final_rescore']['enabled'] = False
    if smoke:
        hyper['cd']['n_restarts'] = 1
        hyper['cd']['initial_points'] = hyper['cd']['initial_points'][:1]
        hyper['cd']['max_outer_iters'] = 1
        anchor = hyper['cd']['initial_points'][0]
        for key in hyper['hyperparam_space']:
            values = [anchor[key]]
            if key.endswith('workspace_execution'):
                values.append(hyper['hyperparam_space'][key]['values'][-1])
            hyper['hyperparam_space'][key] = {'values': values}
            hyper['stages']['coarse']['hyperparam_space'][key] = {'values': values}
    path = directory / 'hyper.yaml'
    path.write_text(yaml.safe_dump(hyper, sort_keys=False))
    return hyper, path


def run(config_path: Path, output: Path, smoke: bool, resume: bool) -> None:
    config_path, output = config_path.resolve(), output.resolve()
    config = yaml.safe_load(config_path.read_text())
    for key in ('parameter_space', 'processed_dir'):
        config[key] = str((config_path.parent/config[key]).resolve())
    for subject in config['subjects']:
        for key in ('engine', 'reference_finalists'):
            subject[key] = str((config_path.parent/subject[key]).resolve())
    if smoke:
        config['subjects'] = config['subjects'][:1]
        config['parallel_budget'] = 1
        for key in ('search', 'screen', 'confirmation'):
            config[key].update(particle_count=2, filter_seed_count=2)
        config['search'].update(shortlist_top_k=1, retain_best_per_workspace=False)
        config['confirmation']['top_per_source'] = 1
    inputs = [config_path, Path(config['parameter_space']),
              *Path(config['processed_dir']).glob('*.csv'),
              *Path('src/Bayesian_state').rglob('*.py')]
    for subject in config['subjects']:
        inputs += [Path(subject['engine']), Path(subject['reference_finalists'])]
    fingerprint = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    context = {'config': config, 'input_sha256': fingerprint, 'smoke': smoke,
               'versions': {name: importlib.metadata.version(name)
                            for name in ('numpy', 'scipy', 'pandas', 'joblib', 'numba')}}
    if resume:
        if json.loads((output/'context.json').read_text()) != context:
            raise ValueError('Pilot resume inputs/config/source/versions changed; use a new directory')
        if (output/'summary.json').exists():
            print((output/'summary.json').read_text(), flush=True)
            return
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output/'context.json', context)
    summaries = []
    for subject in config['subjects']:
        sid = subject['subject']
        directory = output/f'subject_{sid}'
        if (directory/'comparison.json').exists():
            summaries.append(json.loads((directory/'comparison.json').read_text()))
            continue
        if not directory.exists():
            hyper, path = prepare_search(config, subject, output, smoke)
        else:
            path = directory/'hyper.yaml'
            hyper = yaml.safe_load(path.read_text())
        if not (directory/'search_timing.json').exists():
            began = perf_counter()
            optimizer = HyperCDOptimizer(hyper, path)
            checkpoint = directory/f'search/subject_{sid}/search_checkpoint.json'
            was_resumed = checkpoint.exists()
            optimizer.run(subjects=[sid], stage='coarse', resume=was_resumed)
            write_json(directory/'search_timing.json', {
                'seconds': perf_counter()-began, 'resumed': was_resumed})
        archive = directory/f'search/subject_{sid}'
        search_rows = read_rows(archive/'all_combinations.jsonl')
        shortlist = select_diverse(search_rows, config['search']['shortlist_top_k'],
                                    config['search']['retain_best_per_workspace'])
        # Deliberately read historical parameter values only after search completion.
        reference = read_rows(Path(subject['reference_finalists']))
        if smoke:
            reference = reference[:1]
        bank = {}
        for source, rows in (('simplified', shortlist), ('reference', reference)):
            for row in rows:
                pid = point_id(row['hyperparams'])
                record = bank.setdefault(pid, {'id': pid, 'hyperparams': row['hyperparams'], 'sources': []})
                record['sources'].append(source)
        engine = yaml.safe_load(Path(subject['engine']).read_text())
        loader = SubjectTrialDataLoader(engine, config['processed_dir'])
        frame = loader._get_subject_frame(sid, 1.)
        arrays = loader._extract_arrays(frame, 32 if smoke else None)
        ctx = {**subject, 'engine': engine, 'arrays': arrays, 'processed_dir': Path(config['processed_dir'])}
        screen = score_bank(ctx, list(bank.values()), config['screen'], 'screen', config['base_seed'],
                            config['parallel_budget'], directory)
        ranked = sorted(screen['rows'], key=lambda r: (r['mean_nll'], r['id']))
        chosen, primary = {}, {}
        for source in ('simplified', 'reference'):
            rows = [row for row in ranked if source in row['sources']]
            primary[source] = rows[0]['id']
            for row in rows[:config['confirmation']['top_per_source']]:
                chosen[row['id']] = bank[row['id']]
        low_winner = min(search_rows, key=lambda r: (r['aggregated_error'], point_id(r['hyperparams'])))
        chosen[point_id(low_winner['hyperparams'])] = bank[point_id(low_winner['hyperparams'])]
        confirmation = score_bank(ctx, list(chosen.values()), config['confirmation'], 'confirmation',
                                  config['base_seed'], config['parallel_budget'], directory)
        loaded = {}
        for source, pid in primary.items():
            with np.load(directory/'confirmation'/f'{pid}.npz') as npz:
                loaded[source] = dict(npz)
        metrics = compare_arrays(loaded['simplified'], loaded['reference'])
        metrics.update(paired_difference(loaded['simplified'], loaded['reference'],
                       stable_seed({'role': 'pilot_bootstrap', 'subject': sid}),
                       config['comparison']['bootstrap_replicates']))
        limits = config['comparison']
        prediction_pass = (metrics['paired_numerical_interval95'][1] <= limits['max_mean_nll_increase']
                           and metrics['choice_probability_rmse'] <= limits['max_choice_probability_rmse'])
        state_pass = all(metrics[k] <= limits['max_'+k] for k in (
            'mean_rule_total_variation', 'active_probability_mae', 'strategy_probability_mae'))
        own_noise = {}
        for source, value in loaded.items():
            half = len(value['probabilities']) // 2
            a, b = dict(value), dict(value)
            for key in ('probabilities', *STATE_KEYS):
                a[key], b[key] = value[key][:half], value[key][half:]
            own_noise[source] = compare_arrays(a, b)
        restarts = json.loads((archive/'restart_summary.json').read_text())
        search_timing = json.loads((directory/'search_timing.json').read_text())
        summary = {
            'subject': sid, 'condition': subject['condition'], 'trials': len(arrays.feedback),
            'search_candidates': len(search_rows), 'search_shortlist': len(shortlist),
            'search_seconds': search_timing['seconds'], 'screen_seconds': screen['seconds'],
            'confirmation_seconds': confirmation['seconds'], 'primary_selected_on_screen': primary,
            'parameters': {source: extract_model_0826_parameters(bank[pid]['hyperparams']) for source,pid in primary.items()},
            'comparison': metrics, 'same_parameter_split_seed_noise': own_noise,
            'prediction_screen_pass': bool(prediction_pass), 'state_screen_pass': bool(state_pass),
            'limits': limits, 'restarts': [{k: r[k] for k in ('outer_iters_completed','stopped_by','num_improvements')}
                                         for r in restarts['coarse']],
            'interpretation': 'Pilot only; fixed candidates selected with independent screen seeds. Numerical intervals are not participant/parameter confidence intervals. States are pre-choice filtering marginals, not recovered true latent paths. No production replacement.',
        }
        write_json(directory/'comparison.json', summary)
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    write_json(output/'summary.json', {'smoke': smoke, 'subjects': summaries})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true', help='32 trials, tiny search and one worker')
    parser.add_argument('--resume', action='store_true', help='Resume identical context; completed banks only')
    args = parser.parse_args()
    run(args.config, args.output_dir, args.smoke, args.resume)


if __name__ == '__main__':
    main()

"""Bounded, fixed-parameter state replay for the nine completed Model 0826 fits.

Uses the shared evaluator without changing scientific mechanisms or fitting.
Primary estimates preserve the original nominee, including unresolved fits.
One alternative per participant is a sensitivity check, not a confidence set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from joblib import Parallel, delayed

from .bottleneck_analysis import ROOT, sha256
from src.Bayesian_state.optimization.adaptive_fit import make_context
from src.Bayesian_state.optimization.adaptive_runtime import freeze_json, publish
from src.Bayesian_state.simulation.execution import evaluate_state_model_run
from src.Bayesian_state.simulation.parameters import apply_fixed_hyperparams_to_engine_config
from src.Bayesian_state.utils.parallel import parallel_job_count, single_threaded_processes
from src.Bayesian_state.utils.seeding import stable_seed

CONFIG = Path(__file__).with_name('nine_subject_config.json')
STATE_KEYS = (
    'marginal_prior', 'marginal_active_probability', 'marginal_executed_probability',
    'predictive_swap_probability', 'predictive_swap_event_probability',
    'predictive_search_range', 'predictive_replacement_fraction',
    'predictive_newcomer_distance', 'predictive_failure_pressure',
    'predictive_execution_switch_probability', 'predictive_execution_switch_event_probability',
    'predictive_executed_beta', 'removed_mass', 'pairing_prior',
    'pairing_prior_entropy', 'pairing_prior_confidence', 'pre_choice_ess', 'post_choice_ess',
)


def cohort(config: dict) -> list[dict]:
    """Resolve original nominees and a prespecified near-candidate alternative."""
    rows = {}
    for name in config['source_runs']:
        directory = ROOT / name
        fit = json.loads((directory / 'fit_results.json').read_text())
        manifest = json.loads((directory / 'manifest.json').read_text())
        for sid, result in fit['subjects'].items():
            if int(sid) in rows:
                raise ValueError('Participant appears in two source fits')
            specs = [s for s in manifest['subjects'] if s['subject'] == int(sid)]
            near = set(result['near_candidates'])
            options = [p for p in result['candidate_bank']
                       if p['id'] != result['selected'] and p['id'] in near]
            if not options:
                raise ValueError(f'No documented near candidate for S{sid}')
            alternative = min(options, key=lambda p: (p['mean_nll'], p['id']))
            rows[int(sid)] = {'fit': result, 'spec': specs[0], 'fit_config': manifest['config'],
                              'source': name, 'alternative': alternative,
                              'source_manifest': manifest}
    if set(rows) != set(config['subjects']):
        raise ValueError('Source cohort does not equal requested nine participants')
    return [rows[sid] for sid in config['subjects']]


def run_one(row: dict, config: dict, output: Path, variant: str, repeat: int,
            smoke: bool = False) -> dict:
    sid = row['fit']['subject']
    destination = output / f'S{sid}' / f'{variant}_{repeat:02d}.npz'
    if destination.exists():
        return {'subject': sid, 'variant': variant, 'repeat': repeat, 'cached': True}
    context = make_context(row['spec'], row['fit_config'], smoke)
    point = row['fit']['hyperparams'] if variant == 'selected' else row['alternative']['hyperparams']
    point_id = row['fit']['selected'] if variant == 'selected' else row['alternative']['id']
    engine = apply_fixed_hyperparams_to_engine_config(context['engine'], point)
    particles = 2 if smoke else config['particle_count']
    engine['inference']['particle_count'] = particles
    # Same seed schedule for two parameter candidates; no observed outcomes select seeds.
    seed = stable_seed({'role': 'nine_subject_figures', 'base': config['seed_base'],
                        'subject': sid, 'repeat': repeat})
    began = perf_counter()
    result = evaluate_state_model_run(
        subject_id=sid, condition=context['condition'], arrays=context['arrays'], params={},
        engine_config_template=engine, processed_data_dir=context['processed_dir'],
        window_size=16 if smoke else config['display_window'], keep_logs=True,
        prediction_mode='prior_t', selection_prediction_mode='prior_t',
        loss_metric='choice_nll', trajectory_seed=seed)
    state, metrics = result.state_log, result.metrics_by_mode['prior_t']
    arrays = {k: np.asarray(state[k]) for k in STATE_KEYS if k in state and state[k] is not None}
    for k in ('pred_category_probs', 'observed_choice', 'observed_feedback',
              'true_category_index', 'valid_trial_mask', 'score_trial_mask'):
        arrays[k] = np.asarray(metrics[k])
    arrays.update(subject=sid, condition=context['condition'], point_id=point_id,
                  seed=seed, particles=particles, seconds=perf_counter() - began)
    q, active = arrays['marginal_prior'], arrays['marginal_active_probability']
    assert q.shape == active.shape and np.isfinite(q).all() and np.isfinite(active).all()
    np.testing.assert_allclose(q.sum(axis=1), 1, atol=1e-9)
    assert np.all(q <= active + 1e-9)
    np.testing.assert_allclose(arrays['pred_category_probs'].sum(axis=1), 1, atol=1e-9)
    publish(destination, lambda handle: np.savez_compressed(handle, **arrays))
    receipt = {'subject': sid, 'variant': variant, 'repeat': repeat,
               'seconds': float(arrays['seconds']), 'trials': len(q)}
    print(json.dumps(receipt), flush=True)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    rows = cohort(config)
    inputs = {str(CONFIG.relative_to(ROOT)): sha256(CONFIG)}
    for row in rows:
        for name, expected in row['source_manifest']['input_sha256'].items():
            path = Path(name)
            # Validate the frozen mechanisms, catalogue and source data used by the fit.
            if path.suffix in ('.py', '.npy', '.csv') or path == Path(row['spec']['engine']):
                if sha256(path) != expected:
                    raise ValueError(f'Frozen input changed: {path}')
                inputs[str(path.relative_to(ROOT))] = expected
        for filename in ('fit_results.json', 'manifest.json'):
            path = ROOT / row['source'] / filename
            inputs[str(path.relative_to(ROOT))] = sha256(path)
    provenance = {'config': config, 'smoke': args.smoke, 'input_sha256': inputs,
                  'cohort': [{'subject': r['fit']['subject'], 'condition': r['fit']['condition'],
                              'source': r['source'], 'selected': r['fit']['selected'],
                              'alternative': r['alternative']['id'], 'issues': r['fit']['issues']}
                             for r in rows],
                  'scope': 'Fixed-parameter observed-history filtering; no refit or autonomous intervention.'}
    freeze_json(args.output / 'manifest.json', provenance)
    jobs = [(r, variant, repeat) for r in rows
            for variant, count in [('selected', config['selected_repeats']),
                                    ('alternative', config['alternative_repeats'])]
            for repeat in range(count)]
    if args.smoke:
        jobs = [(rows[0], 'selected', 0)]
    workers = 1 if args.smoke else parallel_job_count(config['parallel_budget'], len(jobs))
    with single_threaded_processes():
        receipts = Parallel(n_jobs=workers)(
            delayed(run_one)(r, config, args.output, variant, repeat, args.smoke)
            for r, variant, repeat in jobs)
    freeze_json(args.output / 'completion.json', {'runs': receipts, 'workers': workers})


if __name__ == '__main__':
    main()

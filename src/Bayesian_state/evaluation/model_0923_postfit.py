"""Fixed-parameter, observed-history analysis for Model 0923 full-v3.

Particle snapshots are taken by the public filter's analysis observer. The
observer receives detached arrays and records pre-feedback states at two
inference times. All quantiles condition on the selected parameter point.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..inference.backends.particle_filter import run_state_model_particle_filter
from ..metrics import TrialPrediction, choice_brier, choice_nll
from ..model import ModelContext, StateModel
from ..simulation.config import load_yaml, resolve_engine_config
from ..simulation.data import SubjectTrialDataLoader
from ..simulation.parameters import apply_fixed_hyperparams_to_engine_config
from ..utils.datasets import resolve_dataset_paths
from ..utils.seeding import stable_seed, derive_trajectory_seed
from ..utils.subjects import resolve_subject_config

PHASES = ('predictive', 'filtered')
SCALARS = ('candidate_count', 'effective_participation', 'choice_weighted_beta', 'belief_entropy')
RULE_FIELDS = ('belief', 'active', 'choice_weights', 'dominant')
PARAMETER_DOMAINS = {
    'baseline_probability': (None, None), 'error_gain': (0., None),
    'failure_decay': (0., None), 'global_search': (0., 1.),
    'global_search_failure_gain': (0., 1.), 'max_capacity': (2, None),
    'exit_rate': (0., 1.), 'local_structure_weight': (0., 1.),
    'transfer_scale': (None, None), 'execution_inertia': (0., None),
    'gamma': (0., 1.), 'beta_init': (.1, 25.), 'learning_rate': (0., 1.),
}


def weighted_quantiles(values: np.ndarray, weights: np.ndarray,
                       probabilities: Sequence[float] = (.1, .5, .9)) -> np.ndarray:
    """Invert the weighted empirical CDF; final axis indexes particle draws."""
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    if values.shape != weights.shape or not np.isfinite(values).all():
        raise ValueError('Quantile values and weights must be finite and aligned')
    if not np.isfinite(weights).all() or np.any(weights < 0) or np.any(weights.sum(axis=-1) <= 0):
        raise ValueError('Quantile weights must be finite, nonnegative, with positive mass')
    order = np.argsort(values, axis=-1, kind='stable')
    sorted_values = np.take_along_axis(values, order, axis=-1)
    sorted_weights = np.take_along_axis(weights, order, axis=-1)
    cdf = np.cumsum(sorted_weights, axis=-1)
    cdf /= cdf[..., -1:]
    outputs = []
    for probability in probabilities:
        if not 0 < probability < 1:
            raise ValueError('Quantile probabilities must lie in (0, 1)')
        index = np.argmax(cdf >= probability, axis=-1)
        outputs.append(np.take_along_axis(sorted_values, index[..., None], axis=-1)[..., 0])
    return np.stack(outputs, axis=-1)


class ParticleStateCollector:
    """Retain compact scalar distributions and weighted rule-state moments."""
    def __init__(self) -> None:
        self.rows: dict[str, list[dict[str, np.ndarray]]] = {phase: [] for phase in PHASES}

    def __call__(self, trial: int, phase: str, snapshot: Mapping[str, np.ndarray]) -> None:
        if phase not in self.rows or trial != len(self.rows[phase]):
            raise ValueError('Observer trial order is inconsistent')
        weights = np.asarray(snapshot['weights'])
        belief, active, contribution = [np.asarray(snapshot[k]) for k in ('belief', 'active', 'choice_weights')]
        count = active.sum(axis=1)
        participation = 1. / np.sum(contribution ** 2, axis=1)
        if np.any(count < 1) or np.any(participation > count + 1e-10):
            raise ValueError('Within-path candidate or participation invariant failed')
        log_belief = np.zeros_like(belief)
        np.log(belief, out=log_belief, where=belief > 0)
        row = {'weights': weights.copy(), 'candidate_count': count,
               'effective_participation': participation,
               'choice_weighted_beta': np.sum(contribution * snapshot['beta'], axis=1),
               'belief_entropy': -np.sum(belief * log_belief, axis=1)}
        for name in RULE_FIELDS:
            values = snapshot[name]
            row[name] = weights @ values
            row[name + '_second'] = weights @ (values ** 2)
        row['active_beta'] = weights @ (active * snapshot['beta'])
        for name in ('search', 'global_search', 'failure', 'search_event', 'admission_event'):
            row[name] = np.asarray(weights @ snapshot[name])
        if 'pairing' in snapshot:
            row['pairing'] = weights @ snapshot['pairing']
        self.rows[phase].append(row)

    def arrays(self) -> dict[str, np.ndarray]:
        if len(self.rows['predictive']) != len(self.rows['filtered']):
            raise ValueError('Observer phases have different lengths')
        return {phase + '_' + key: np.stack([row[key] for row in rows])
                for phase, rows in self.rows.items() for key in rows[0]}


def resolve_subject(fit_run: Path, subject: int, cache: Path,
                    max_trials: int | None = None) -> dict[str, Any]:
    """Resolve archived inputs and selected parameters without altering them."""
    base_path = fit_run / 'runtime/simulation.yaml'
    config = resolve_subject_config(load_yaml(base_path), subject)
    engine = resolve_engine_config(config, base_path.parent, subject_id=subject)
    best = json.loads((fit_run / 'hyper_cd' / f'subject_{subject}' / 'best_hyperparams.json').read_text())
    point = best['final_rescore_best']['hyperparams']
    engine = apply_fixed_hyperparams_to_engine_config(engine, point)
    engine['oral_readout'] = None
    engine['partition']['kwargs']['similarity_cache_dir'] = str(cache.resolve())
    engine['inference']['particle_blocks'] = 1
    paths = resolve_dataset_paths(config, base_path.parent)
    loader = SubjectTrialDataLoader(engine, dataset_paths=paths)
    loader.prepare_data(paths['learning_data'])
    frame = loader._get_subject_frame(subject, 1.)
    arrays = loader._extract_arrays(frame, max_trials)
    if 'miss' in frame:
        frame = frame.loc[frame['miss'] != 1]
    frame = frame.iloc[:len(arrays.choices)].copy()
    if len(frame) != len(arrays.choices):
        raise ValueError('Source trial identifiers and model arrays are not aligned')
    return dict(subject=subject, condition=loader._get_condition_value(frame),
                engine=engine, paths=paths, arrays=arrays, frame=frame,
                point=point, best=best)


def score_probabilities(probabilities: np.ndarray, choices: np.ndarray) -> dict[str, float]:
    """Use the shared score definitions, excluding the initialization trial."""
    mask = np.ones(len(choices), bool)
    mask[0] = False
    prediction = TrialPrediction(probabilities, choices - 1, mask)
    return {'mean_choice_nll': choice_nll(prediction).value,
            'choice_brier': choice_brier(prediction).value}


def run_particle_task(task: Mapping[str, Any]) -> str:
    """Run one independent fixed-parameter PF and publish a new seed artifact."""
    inputs = resolve_subject(Path(task['fit_run']), int(task['subject']), Path(task['cache']), task.get('max_trials'))
    collector = ParticleStateCollector() if task.get('record_states', True) else None
    np.random.seed(int(task['seed']))
    result = run_state_model_particle_filter(
        engine_config=inputs['engine'], subject_id=inputs['subject'], condition=inputs['condition'],
        stimulus=inputs['arrays'].stimulus, choices=inputs['arrays'].choices,
        feedback=inputs['arrays'].feedback, particle_count=int(task['particles']),
        filter_seed=int(task['seed']), choice_readout_power=1.,
        processed_data_dir=inputs['paths']['processed_dir'], dataset_paths=inputs['paths'],
        state_observer=collector,
    )
    probabilities = np.asarray(result.observation_probabilities['prior_t'])
    scores = score_probabilities(probabilities, inputs['arrays'].choices)
    if not np.isfinite(list(scores.values())).all():
        raise ValueError('Non-finite fixed-parameter score')
    values = {'probabilities': probabilities, 'seed': np.asarray(task['seed']),
              'pre_choice_ess': result.diagnostics['pre_choice_ess'],
              'post_choice_ess': result.diagnostics['post_choice_ess'],
              'resampled': result.diagnostics['resampled'],
              'mean_choice_nll': np.asarray(scores['mean_choice_nll'])}
    if collector is not None:
        values.update(collector.arrays())
        np.testing.assert_allclose(values['predictive_choice_weights'], result.state_probabilities['choice_weights'], atol=1e-15)
        np.testing.assert_allclose(values['filtered_choice_weights'], result.state_probabilities['filtered_choice_weights'], atol=1e-15)
        np.testing.assert_allclose(np.sum(values['predictive_effective_participation'] * values['predictive_weights'], axis=1), result.latent_summaries['effective_participation'], atol=1e-14)
    if inputs['condition'] == 3:
        values['pairing_post_feedback'] = result.state_probabilities['pairing_posterior']
    output = Path(task['output'])
    with output.open('xb') as stream:
        np.savez_compressed(stream, **values)
    return str(output)


def postfit_seed(base_seed: int, subject: int, repeat: int) -> int:
    return stable_seed({'seed_role': 'model0923_postfit_independent_pf',
                        'base_seed': base_seed, 'subject': subject, 'repeat': repeat})


def original_final_seed(fit_run: Path, subject: int, repeat: int = 0) -> int:
    search = load_yaml(fit_run / 'runtime/hyper.yaml')
    base = stable_seed({'seed_role': 'hyper_cd_stage_common_random_numbers',
                        'hyper_base_seed': search['hyper_base_seed'], 'stage': 'final_rescore',
                        'subject_id': subject, 'seed_family': search['final_rescore']['seed_family']})
    return derive_trajectory_seed(base, 'simulation', repeat)


def audit_search(fit_run: Path, subjects: Sequence[int], output: Path) -> pd.DataFrame:
    """Audit the saved candidate bank, stopping rules and parameter domains."""
    space = load_yaml(fit_run / 'runtime/hyper.yaml')['hyperparam_space']
    parameters, restarts, finalists, summaries = [], [], [], []
    for sid in subjects:
        directory = fit_run / 'hyper_cd' / f'subject_{sid}'
        best = json.loads((directory / 'best_hyperparams.json').read_text())['final_rescore_best']
        point = best['hyperparams']
        names = {key.split('.')[-1]: value for key, value in point.items()}
        inactive = {}
        if names['global_search'] == 1:
            inactive.update(global_search_failure_gain='global_search=1', local_structure_weight='global_search=1')
        if names['error_gain'] == 0 and (names['global_search'] == 1 or names['global_search_failure_gain'] == 0):
            inactive['failure_decay'] = 'failure has no effect on search probability or range'
        for key, value in point.items():
            name = key.split('.')[-1]
            grid = space[key]['values']
            edge = 'lower' if value == grid[0] else 'upper' if value == grid[-1] else 'interior'
            bound = PARAMETER_DOMAINS[name][0 if edge == 'lower' else 1] if edge != 'interior' else None
            parameters.append(dict(subject=sid, condition=sid // 100, parameter=name, path=key,
                                   estimate=value, grid_min=min(grid), grid_max=max(grid), edge=edge,
                                   boundary_kind=('mathematical' if bound == value else 'search_support') if edge != 'interior' else 'interior',
                                   inactive_at_selected_point=name in inactive, inactive_reason=inactive.get(name, ''),
                                   parameter_uncertainty='pending_profile_or_recovery'))
        stop = json.loads((directory / 'restart_summary.json').read_text())
        limited = False
        for stage, rows in stop.items():
            for row in rows:
                limited |= row['stopped_by'] == 'max_outer_iters'
                restarts.append(dict(subject=sid, stage=stage, restart=row['restart_id'],
                                     rounds=row['outer_iters_completed'], stopped_by=row['stopped_by'],
                                     start_nll=row['initial_error'], end_nll=row['best_error'],
                                     evaluated=row['num_new_evaluations'], cache_hits=row['num_cache_hits']))
        records = [json.loads(line) for line in (directory / 'final_rescore.jsonl').read_text().splitlines()]
        winner = next(row for row in records if row['hyperparams'] == point)
        errors = np.asarray(winner['subject_metrics'][str(sid)]['simulation']['sample_errors'])
        for rank, row in enumerate(sorted(records, key=lambda r: r['aggregated_error'])):
            sample = np.asarray(row['subject_metrics'][str(sid)]['simulation']['sample_errors'])
            delta = sample - errors
            finalists.append(dict(subject=sid, rank=rank + 1, selected=row['hyperparams'] == point,
                                   mean_probability_nll=row['aggregated_error'],
                                   gap_to_selected=row['aggregated_error'] - best['aggregated_error'],
                                   mean_paired_single_run_nll_delta=delta.mean(),
                                   sd_paired_single_run_nll_delta=delta.std(ddof=1),
                                   fraction_single_seeds_better=float(np.mean(delta < 0)),
                                   interpretation='paired PF seed diagnostics; parameter confidence pending'))
        gaps = sorted(row['aggregated_error'] for row in records)
        summaries.append(dict(subject=sid, condition=sid // 100, selected_fit_nll=best['aggregated_error'],
                              some_restart_at_round_cap=limited, finalist_gap=gaps[1] - gaps[0],
                              parameters_at_edge=sum(p['edge'] != 'interior' for p in parameters if p['subject'] == sid),
                              inactive_parameters=','.join(sorted(inactive))))
    for name, rows in [('parameters', parameters), ('restarts', restarts), ('finalists', finalists), ('search_summary', summaries)]:
        pd.DataFrame(rows).to_csv(output / f'{name}.csv', index=False)
    return pd.DataFrame(summaries)


def rule_catalog(inputs: Mapping[str, Any]) -> pd.DataFrame:
    model = StateModel(inputs['engine'], context=ModelContext(
        condition=inputs['condition'], subject_id=inputs['subject'],
        processed_data_dir=inputs['paths']['processed_dir'], dataset_paths=inputs['paths']))
    return pd.DataFrame([{'rule': spec.index, 'family': spec.family,
                          'parameters': json.dumps(dict(spec.parameters), default=str, sort_keys=True)}
                         for spec in model.partition_model.hypothesis_space])


def summarize_subject(inputs: Mapping[str, Any], files: Sequence[str], output: Path,
                      window: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Pool equal PF-seed mixtures and retain within-path scalar uncertainty."""
    runs = []
    for file in files:
        with np.load(file) as saved:
            runs.append({name: saved[name] for name in saved.files})
    probabilities = np.mean([r['probabilities'] for r in runs], axis=0)
    arrays = inputs['arrays']
    summary = dict(subject=inputs['subject'], condition=inputs['condition'],
                   modeled_trials=len(arrays.choices), scored_trials=len(arrays.choices)-1,
                   **score_probabilities(probabilities, arrays.choices))
    uniform = np.full_like(probabilities, 1 / probabilities.shape[1])
    summary['uniform_nll'] = score_probabilities(uniform, arrays.choices)['mean_choice_nll']
    summary['nll_gain_over_uniform'] = summary['uniform_nll'] - summary['mean_choice_nll']
    summary['original_selected_nll'] = inputs['best']['final_rescore_best']['aggregated_error']
    summary['fresh_minus_original_nll'] = summary['mean_choice_nll'] - summary['original_selected_nll']
    half = np.mean([r['probabilities'] for r in runs[:max(1,len(runs)//2)]], axis=0)
    summary['full_minus_half_repeats_nll'] = summary['mean_choice_nll'] - score_probabilities(half, arrays.choices)['mean_choice_nll']
    summaries = {'probabilities': probabilities}
    trial = inputs['frame'][[name for name in ('iSub', 'condition', 'iSession', 'iRun', 'iBlock', 'iTrial', 'choice', 'category', 'feedback') if name in inputs['frame']]].reset_index(drop=True)
    trial['valid_trial'] = np.arange(1, len(arrays.choices)+1)
    trial['scored'] = np.arange(len(arrays.choices)) > 0
    trial['observed_choice_probability'] = probabilities[np.arange(len(arrays.choices)), arrays.choices-1]
    trial['choice_nll'] = -np.log(np.clip(trial['observed_choice_probability'], 1e-12, 1))
    correct = probabilities[np.arange(len(arrays.choices)), arrays.categories-1]
    trial['predicted_correct_probability'] = correct
    trial['observed_correct'] = (arrays.feedback == 1).astype(int)
    for c in range(probabilities.shape[1]):trial[f'probability_choice_{c+1}'] = probabilities[:, c]
    for phase in PHASES:
        weights = np.concatenate([r[phase+'_weights'] for r in runs], axis=1) / len(runs)
        for scalar in SCALARS:
            samples = np.concatenate([r[phase+'_'+scalar] for r in runs], axis=1)
            mean = np.sum(weights * samples, axis=1)
            quantiles = weighted_quantiles(samples, weights)
            trial[phase+'_'+scalar] = mean
            summaries[phase+'_'+scalar] = np.column_stack((mean, quantiles))
            for j, label in enumerate(('q10','q50','q90')):trial[phase+'_'+scalar+'_'+label] = quantiles[:,j]
        for field in (*RULE_FIELDS, 'active_beta', 'search', 'global_search', 'failure', 'search_event', 'admission_event'):
            summaries[phase+'_'+field] = np.mean([r[phase+'_'+field] for r in runs], axis=0)
        numerator = summaries[phase+'_active_beta']
        denominator = summaries[phase+'_active']
        conditional_beta = np.full_like(numerator, np.nan)
        np.divide(numerator, denominator, out=conditional_beta, where=denominator>0)
        summaries[phase+'_beta_given_active'] = conditional_beta
        for field in ('search', 'global_search', 'failure', 'search_event', 'admission_event'):
            trial[phase+'_'+field] = summaries[phase+'_'+field]
        if inputs['condition']==3:
            summaries[phase+'_pairing'] = np.mean([r[phase+'_pairing'] for r in runs], axis=0)
    if inputs['condition']==3:
        summaries['pairing_post_feedback'] = np.mean([r['pairing_post_feedback'] for r in runs],axis=0)
        for k in range(3):trial[f'pairing_post_feedback_{k+1}'] = summaries['pairing_post_feedback'][:,k]
    summary['mean_pre_choice_candidates'] = float(trial['predictive_candidate_count'].mean())
    summary['mean_pre_choice_participation'] = float(trial['predictive_effective_participation'].mean())
    summary['mean_pre_choice_beta'] = float(trial['predictive_choice_weighted_beta'].mean())
    summary['median_post_choice_ess_fraction'] = float(np.median([r['post_choice_ess']/r['predictive_weights'].shape[1] for r in runs]))
    summary['resampling_fraction'] = float(np.mean([r['resampled'] for r in runs]))
    for phase in PHASES:
        for name in ('belief', 'choice_weights', 'dominant'):
            np.testing.assert_allclose(summaries[phase+'_'+name].sum(axis=1), 1., atol=1e-12)
        if np.any(summaries[phase+'_active'].sum(axis=1) > inputs['engine']['modules']['hypo_transitions_mod']['kwargs']['max_capacity'] + 1e-10):
            raise ValueError('Candidate count exceeds fitted capacity')
    trial.to_csv(output/'trials.csv',index=False)
    catalog = rule_catalog(inputs)
    catalog.to_csv(output/'rule_catalog.csv',index=False)
    rule_frame = pd.DataFrame({'valid_trial':np.repeat(np.arange(1,len(arrays.choices)+1),len(catalog)),
                               'rule':np.tile(catalog['rule'],len(arrays.choices))})
    for phase in PHASES:
        for field in RULE_FIELDS:rule_frame[phase+'_'+field] = summaries[phase+'_'+field].reshape(-1)
        rule_frame[phase+'_beta_given_active'] = summaries[phase+'_beta_given_active'].reshape(-1)
    rule_frame.to_csv(output/'rule_states.csv',index=False)
    with (output/'pooled_states.npz').open('xb') as stream:np.savez_compressed(stream,**summaries)
    return summary, summaries

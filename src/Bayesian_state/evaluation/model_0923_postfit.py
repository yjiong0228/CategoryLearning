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
from ..utils.seeding import stable_seed
from .model_0923_fit_source import (
    PARAMETER_DOMAINS, audit_fit_source, original_replay_spec, resolve_fit_subject,
)

PHASES = ('predictive', 'filtered')
SCALARS = ('candidate_count', 'effective_participation', 'choice_weighted_beta', 'belief_entropy')
RULE_FIELDS = ('belief', 'active', 'choice_weights', 'dominant')


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
    """Resolve either saved fit layout through the shared read-only adapter."""
    return resolve_fit_subject(fit_run, subject, cache, max_trials)


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
    if task.get('oral_readout') is not None:
        # Expression settings are fixed evaluation inputs. Reports only score
        # predictions and never contribute to fitting or particle weights.
        from ..model.modules.readout.oral import OralReadoutParameters
        OralReadoutParameters.from_config(task['oral_readout'])
        inputs['engine']['oral_readout'] = deepcopy(task['oral_readout'])
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
              'particles': np.asarray(task['particles']),
              'pre_choice_ess': result.diagnostics['pre_choice_ess'],
              'post_choice_ess': result.diagnostics['post_choice_ess'],
              'resampled': result.diagnostics['resampled'],
              'mean_choice_nll': np.asarray(scores['mean_choice_nll'])}
    if task.get('oral_readout') is not None:
        for name in ('oral_report_codes', 'oral_report_probabilities', 'oral_report_valid_mask'):
            values[name] = result.observation_probabilities[name]
        oral_metadata = dict(result.metadata['oral_readout'], parameters=task['oral_readout']['kwargs'])
        values['oral_readout_metadata_json'] = np.asarray(json.dumps(
            oral_metadata, ensure_ascii=False, allow_nan=False))
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
    """Return the recorded final scoring seed for either saved fit layout."""
    return original_replay_spec(fit_run, subject, repeat)['seed']


def audit_search(fit_run: Path, subjects: Sequence[int], output: Path) -> pd.DataFrame:
    """Write the stable saved-selection audit tables for either fit layout."""
    return audit_fit_source(fit_run, subjects, output)


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
    if not runs:
        raise ValueError('At least one saved PF seed is required')
    seeds = [int(run['seed']) for run in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError('Saved PF seeds must be distinct')
    seed_rows = []
    for run in runs:
        p = np.asarray(run['probabilities'])
        if p.ndim != 2 or len(p) != len(inputs['arrays'].choices) or p.shape != runs[0]['probabilities'].shape or not np.isfinite(p).all() or np.any(p < 0):
            raise ValueError('Saved seed probability arrays must be finite and aligned')
        np.testing.assert_allclose(p.sum(axis=1), 1., rtol=0, atol=1e-10)
        scores = score_probabilities(p, inputs['arrays'].choices)
        if not np.isclose(scores['mean_choice_nll'], float(run['mean_choice_nll']), rtol=0, atol=1e-12):
            raise ValueError('Saved seed NLL differs from its probability array')
        particles = int(run.get('particles', run['predictive_weights'].shape[1]))
        if run['predictive_weights'].shape[1] != particles:
            raise ValueError('Saved particle count differs from the state arrays')
        for phase in PHASES:
            weights = np.asarray(run[phase+'_weights'])
            if weights.shape != (len(p), particles) or not np.isfinite(weights).all() or np.any(weights < 0):
                raise ValueError('Saved PF state weights must be finite and aligned')
            np.testing.assert_allclose(weights.sum(axis=1), 1., rtol=0, atol=1e-10)
        if run['resampled'].shape != (len(p),) or not np.isin(run['resampled'], [False, True]).all():
            raise ValueError('Saved resampling indicators must be trial-aligned booleans')
        for name in ('pre_choice_ess', 'post_choice_ess'):
            ess = np.asarray(run[name])
            if ess.shape != (len(p),) or not np.isfinite(ess).all() or np.any(ess < 1-1e-10) or np.any(ess > particles+1e-10):
                raise ValueError('Saved ESS is outside its particle budget')
        seed_rows.append(dict(seed=int(run['seed']), particles=particles, **scores))
    if len({row['particles'] for row in seed_rows}) != 1:
        raise ValueError('Seed aggregation requires a common particle budget')
    pd.DataFrame(seed_rows).to_csv(output / 'seed_nll.csv', index=False)
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
    trial['predictive_ess_fraction'] = np.mean([r['pre_choice_ess']/n['particles'] for r,n in zip(runs,seed_rows)],axis=0)
    trial['filtered_ess_fraction'] = np.mean([r['post_choice_ess']/n['particles'] for r,n in zip(runs,seed_rows)],axis=0)
    trial['resampled_fraction'] = np.mean([r['resampled'] for r in runs],axis=0)
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

"""Compare saved pre-choice predictions with a subject's observed behavior.

Exact correctness uses the task category; the probability of the subject's
chosen response remains a separate choice-fit quantity. All calculations use
the saved scoring mask, and sequential comparisons stay within each session.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..metrics.behavior import (
    centered_curve_metrics, curve_discrepancy_metrics,
    history_kernel_metrics, switch_behavior_metrics,
)
from ..metrics.prediction import TrialPrediction, choice_brier, choice_nll
from ..metrics.task import category_learning_feedback, condition3_task_metric_summaries
from .autonomous_trajectories import rolling_binary_ensemble

ARTIFACTS = (
    'behavior_alignment.json', 'behavior_alignment.csv',
    'behavior_alignment_trials.csv', 'behavior_alignment_curves.csv',
)
TRIAL_KEYS = ('iSub', 'condition', 'iSession', 'iRun', 'iBlock', 'iTrial', 'valid_trial')
HISTORY_MAX_LAG = 8


def _json_values(value: Any) -> Any:
    """Represent unavailable numerical summaries as strict JSON null values."""
    if isinstance(value, Mapping):
        return {str(key): _json_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_values(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _discrepancy(observed: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    shape = curve_discrepancy_metrics(observed, predicted)
    centered = centered_curve_metrics(observed, predicted)
    # Pool shared shape summaries across contiguous evaluable segments so a
    # masked interval never becomes an artificial adjacent-trial change.
    finite = np.isfinite(observed) & np.isfinite(predicted)
    edges = np.diff(np.r_[False, finite, False].astype(int))
    pairs = slope_pairs = 0
    observed_change = predicted_change = slope_agreement = 0.
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if end - start < 2:
            continue
        segment = curve_discrepancy_metrics(observed[start:end], predicted[start:end])
        count = end - start - 1
        pairs += count
        observed_change += count * segment['true_vol']
        predicted_change += count * segment['pred_vol']
        slopes = int(np.sum((np.abs(np.diff(observed[start:end])) > 1e-12)
                            & (np.abs(np.diff(predicted[start:end])) > 1e-12)))
        if slopes:
            slope_pairs += slopes
            slope_agreement += slopes * segment['slope_agree']
    shape['true_vol'] = observed_change / pairs if pairs else float('nan')
    shape['pred_vol'] = predicted_change / pairs if pairs else float('nan')
    shape['vol_ratio'] = predicted_change / observed_change if observed_change > 0 else float('nan')
    shape['slope_agree'] = slope_agreement / slope_pairs if slope_pairs else float('nan')
    return dict(**shape, signed_bias_predicted_minus_observed=centered['level_bias'],
                centered_mae=centered['centered_mae'], centered_rmse=centered['centered_rmse'],
                n_observations=centered['n_observations'], adjacent_evaluable_pairs=pairs)


def _history_summary(observed: np.ndarray, predicted: np.ndarray, scored: np.ndarray,
                     sessions: np.ndarray) -> dict[str, Any]:
    """Pad session boundaries so the shared lag estimator cannot cross them."""
    boundaries = np.r_[0, np.flatnonzero(sessions[1:] != sessions[:-1]) + 1, len(sessions)]
    human, model, valid = [], [], []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        selected = scored[start:end]
        human.extend(np.where(selected, observed[start:end], np.nan))
        model.extend(np.where(selected, predicted[start:end], np.nan))
        valid.extend(selected)
        human.extend([np.nan] * HISTORY_MAX_LAG)
        model.extend([np.nan] * HISTORY_MAX_LAG)
        valid.extend([False] * HISTORY_MAX_LAG)
    result = history_kernel_metrics(
        dict(true_acc=human, pred_acc=model, valid_trial_mask=valid),
        max_lag=HISTORY_MAX_LAG, ridge=1e-3, standardize=True,
    )
    return dict(status='completed_descriptive' if result['n_rows'] > HISTORY_MAX_LAG else 'insufficient_trials',
                interpretation='Association with the previous eight observed exact-correctness outcomes within session',
                **result)


def _sequential_summary(trials: pd.DataFrame, probabilities: np.ndarray,
                        choices: np.ndarray, observed: np.ndarray,
                        predicted: np.ndarray, feedback: np.ndarray,
                        scored: np.ndarray, condition: int) -> tuple[dict[str, Any], np.ndarray]:
    pair_mask = np.zeros(len(scored), dtype=bool)
    if 'iSession' not in trials:
        return dict(status='unavailable', reason='Session identifiers are required for sequential comparisons'), pair_mask
    sessions = trials['iSession'].to_numpy()
    if pd.isna(sessions).any():
        raise ValueError('Session identifiers must be present on every behavior trial')
    pair_mask[1:] = scored[1:] & scored[:-1] & (sessions[1:] == sessions[:-1])
    metric_input = dict(pred_category_probs=probabilities, observed_choice_index=choices - 1,
                        true_acc=observed, valid_trial_mask=pair_mask)
    switches = switch_behavior_metrics(metric_input, min_trials=1)
    strata = [('full_success', 1.), ('failure', 0.)]
    if condition == 3:
        strata.insert(1, ('partial_success', .5))
    by_feedback = {}
    previous = np.r_[np.nan, feedback[:-1]]
    for label, value in strata:
        metrics = switch_behavior_metrics(
            dict(metric_input, valid_trial_mask=pair_mask & (previous == value)), min_trials=1,
        )
        by_feedback[label] = {key: metrics[key] for key in (
            'n_pairs', 'switch_human', 'switch_model', 'switch_abs_diff',
            'perseveration_human', 'perseveration_model', 'perseveration_abs_diff',
        )}
    return dict(status='completed_descriptive', scored_pairs=int(pair_mask.sum()),
                switch=dict(status='completed_descriptive' if pair_mask.any() else 'insufficient_pairs',
                            exact_error_includes_partial_success=condition == 3, **switches),
                by_previous_feedback=by_feedback,
                history=_history_summary(observed, predicted, scored, sessions)), pair_mask


def summarize_behavior_alignment(inputs: Mapping[str, Any], trials: pd.DataFrame,
                                 window: int, output: Path) -> dict[str, Any]:
    """Write four fixed artifacts from saved mean-probability trial predictions.

    Rolling windows start at index one and require every constituent trial to
    be scored. Missing windows retain their original ending trial coordinates.
    This helper adds no PF runs, predictive draws or acceptance thresholds.
    """
    if type(window) is not int or window < 1:
        raise ValueError('Behavior alignment window must be a positive integer')
    condition = int(inputs['condition'])
    if condition not in (1, 2, 3):
        raise ValueError('Behavior alignment supports conditions 1, 2 and 3')
    arrays = inputs['arrays']
    choices, categories, feedback = [np.asarray(getattr(arrays, name)) for name in
                                     ('choices', 'categories', 'feedback')]
    n = len(choices)
    if n < 1 or choices.shape != (n,) or categories.shape != (n,) or feedback.shape != (n,) or len(trials) != n:
        raise ValueError('Behavior observations and prediction table must align')
    subject = inputs.get('subject')
    if subject is None:
        if 'iSub' not in trials or trials['iSub'].nunique() != 1:
            raise ValueError('Behavior alignment requires one identified subject')
        subject = trials['iSub'].iloc[0]
    if 'iSub' in trials and not trials['iSub'].eq(subject).all():
        raise ValueError('Behavior table subject differs from its fixed-parameter input')
    if 'condition' in trials and not trials['condition'].eq(condition).all():
        raise ValueError('Behavior table condition differs from its fixed-parameter input')
    for name, values in [('choice', choices), ('category', categories), ('feedback', feedback)]:
        if name not in trials or not np.array_equal(trials[name].to_numpy(), values):
            raise ValueError(f'Behavior table differs from the observed {name} sequence')
    if 'scored' not in trials or not np.isin(trials['scored'], [True, False]).all():
        raise ValueError('Behavior alignment requires an explicit boolean scoring mask')
    scored = trials['scored'].to_numpy(dtype=bool)
    n_categories = 2 if condition == 1 else 4
    for name, values in [('choice', choices), ('category', categories)]:
        if (not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all()
                or np.any((values < 1) | (values > n_categories))):
            raise ValueError(f'Behavior {name} identifiers fall outside the task categories')
    choices, categories = choices.astype(int), categories.astype(int)
    expected_feedback = np.array([category_learning_feedback(y, c, condition=condition)
                                  for y, c in zip(choices, categories)])
    if not np.array_equal(feedback, expected_feedback):
        raise ValueError('Observed feedback differs from the task choice/category semantics')
    columns = [f'probability_choice_{category}' for category in range(1, n_categories + 1)]
    if any(column not in trials for column in columns):
        raise ValueError('Behavior alignment requires all category probability columns')
    probabilities = trials[columns].to_numpy(dtype=float)
    if (not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(axis=1), 1., rtol=0, atol=1e-10)):
        raise ValueError('Behavior category probabilities must be finite, nonnegative and normalized')
    observed = (choices == categories).astype(float)
    predicted = probabilities[np.arange(n), categories - 1]
    if ('predicted_correct_probability' in trials and not np.allclose(
            trials['predicted_correct_probability'], predicted, rtol=0, atol=1e-10)):
        raise ValueError('Saved correct probability differs from the true-category probability')
    prediction = TrialPrediction(probabilities, choices - 1, scored)
    uniform = TrialPrediction(np.full_like(probabilities, 1. / n_categories), choices - 1, scored)
    frequency = (np.bincount(choices[scored] - 1, minlength=n_categories) / scored.sum()
                 if scored.any() else np.full(n_categories, 1. / n_categories))
    constant = TrialPrediction(np.tile(frequency, (n, 1)), choices - 1, scored)

    source = trials[[key for key in TRIAL_KEYS if key in trials]].copy()
    source['valid_trial'] = np.arange(1, n + 1)
    for name, values in [('choice', choices), ('category', categories), ('feedback', feedback),
                         ('scored', scored), ('observed_exact_correct', observed),
                         ('conditional_correct_probability', predicted),
                         ('accuracy_residual', observed - predicted)]:
        source[name] = values
    source['conditional_chosen_response_probability'] = probabilities[np.arange(n), choices - 1]
    sequential, pair_mask = _sequential_summary(
        trials, probabilities, choices, observed, predicted, feedback, scored, condition,
    )
    source['scored_within_session_pair'] = pair_mask
    source['previous_feedback'] = np.where(pair_mask, np.r_[np.nan, feedback[:-1]], np.nan)

    curves = pd.DataFrame(columns=[*source.columns.intersection(TRIAL_KEYS),
        'window_start_valid_trial', 'scored_window', 'observed_rolling_exact_accuracy',
        'conditional_rolling_correct_probability', 'rolling_accuracy_residual'])
    if n > window:
        end_trial, rolling = rolling_binary_ensemble(np.stack((observed, predicted)), window_size=window)
        starts = end_trial - window
        complete = np.array([scored[start:start + window].all() for start in starts])
        rolling[:, ~complete] = np.nan
        curves = source.iloc[end_trial - 1][[key for key in TRIAL_KEYS if key in source]].reset_index(drop=True)
        curves['window_start_valid_trial'] = starts + 1
        curves['scored_window'] = complete
        curves['observed_rolling_exact_accuracy'] = rolling[0]
        curves['conditional_rolling_correct_probability'] = rolling[1]
        curves['rolling_accuracy_residual'] = rolling[0] - rolling[1]
    curve_summary = _discrepancy(
        curves['observed_rolling_exact_accuracy'].to_numpy(dtype=float),
        curves['conditional_rolling_correct_probability'].to_numpy(dtype=float),
    )
    condition3 = dict(status='not_applicable')
    if condition == 3:
        condition3 = dict(status='completed', **condition3_task_metric_summaries(
            probabilities=probabilities, choices=choices, categories=categories,
            feedback=feedback, valid_trial_mask=scored,
        ))
        family_probs = probabilities.reshape(n, 2, 2).sum(axis=2)
        observed_family = ((choices - 1) // 2 == (categories - 1) // 2).astype(float)
        predicted_family = family_probs[np.arange(n), (categories - 1) // 2]
        expected_reward = .5 * (predicted + predicted_family)
        source['observed_family_correct'] = observed_family
        source['conditional_family_probability'] = predicted_family
        source['conditional_expected_reward'] = expected_reward
        if n > window:
            _, additional = rolling_binary_ensemble(
                np.stack((observed_family, predicted_family, feedback, expected_reward)), window_size=window,
            )
            additional[:, ~curves['scored_window'].to_numpy(dtype=bool)] = np.nan
        else:
            additional = np.empty((4, 0))
        for name, values in zip(('observed_rolling_family_accuracy', 'conditional_rolling_family_probability',
                                 'observed_rolling_mean_reward', 'conditional_rolling_expected_reward'), additional):
            curves[name] = values
        curves['rolling_family_residual'] = additional[0] - additional[1]
        curves['rolling_reward_residual'] = additional[2] - additional[3]

    summary = _json_values(dict(
        status='completed_in_sample', subject=int(subject),
        condition=condition, scored_trials=int(scored.sum()), window_size=window, rolling_start_index=1,
        mean_choice_nll=choice_nll(prediction).value, choice_brier=choice_brier(prediction).value,
        uniform_nll=choice_nll(uniform).value,
        constant_choice_frequency_nll=choice_nll(constant).value,
        constant_choice_frequency=frequency.tolist(),
        constant_reference_scope='Choice frequencies estimated on these same scored trials; in-sample baseline',
        observed_accuracy=float(observed[scored].mean()) if scored.any() else None,
        conditional_correct_probability=float(predicted[scored].mean()) if scored.any() else None,
        trial_discrepancy=_discrepancy(np.where(scored, observed, np.nan), np.where(scored, predicted, np.nan)),
        rolling=dict(status='completed_descriptive' if curve_summary['n_observations'] else 'insufficient_scored_windows',
                     **curve_summary), sequential=sequential, condition3=condition3,
        behavioral_predictive_interval=dict(status='not_performed', reason='Predictive draw settings are not configured'),
        formal_sequential_residual_tests=dict(
            status='not_performed' if n_categories == 2 else 'unsupported_for_multicategory',
            reason='The legacy formal residual protocol is binary-only and is not part of this descriptive summary'),
        thresholds='Descriptive metrics; no behavioral acceptance thresholds',
        segments='Overall, early and late conditional/autonomous accuracy are retained in learning.csv',
        probability_timing='Before the current choice and feedback',
    ))
    output = Path(output)
    for name in ARTIFACTS:
        if (output / name).exists():
            raise FileExistsError(f'Behavior alignment artifact already exists: {output / name}')
    output.mkdir(parents=True, exist_ok=True)
    with (output / ARTIFACTS[0]).open('x') as stream:
        stream.write(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    flat = {key: value for key, value in summary.items() if not isinstance(value, dict)}
    for group in ('trial_discrepancy', 'rolling'):
        flat.update({f'{group}_{key}': value for key, value in summary[group].items()})
    pd.DataFrame([flat]).to_csv(output / ARTIFACTS[1], index=False, mode='x')
    source.to_csv(output / ARTIFACTS[2], index=False, mode='x')
    curves.to_csv(output / ARTIFACTS[3], index=False, mode='x')
    return summary

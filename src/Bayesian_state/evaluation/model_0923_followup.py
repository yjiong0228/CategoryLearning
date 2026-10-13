"""Fixed-candidate PF noise checks and saved autonomous learning summaries.

Seed jackknife errors measure numerical integration noise at fixed data and
parameters. Behavioral quantiles describe independent simulated trajectories.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .model_0923_postfit import score_probabilities


def compare_finalists(probabilities: np.ndarray, choices: np.ndarray,
                      selected_index: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare candidate × seed × trial × category probabilities using common seeds."""
    values = np.asarray(probabilities, dtype=float)
    choices = np.asarray(choices, dtype=int)
    if values.ndim != 4 or values.shape[2] != len(choices) or values.shape[1] < 2:
        raise ValueError('Need aligned candidate, seed, trial and category dimensions')
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Probabilities must be finite and nonnegative')
    if not np.allclose(values.sum(axis=-1), 1, rtol=0, atol=1e-10):
        raise ValueError('Category probabilities must sum to one')
    if not 0 <= selected_index < len(values):
        raise ValueError('Selected candidate index is out of range')
    if np.any((choices < 1) | (choices > values.shape[-1])):
        raise ValueError('Observed choices are outside the category range')

    def scores(batch: np.ndarray) -> np.ndarray:
        return np.array([score_probabilities(p, choices)['mean_choice_nll'] for p in batch])

    repeats = values.shape[1]
    mean = values.mean(axis=1)
    total = scores(mean)
    # Jackknife the mean-probability objective, preserving pairing between candidates.
    leave_one_out = np.stack([scores((values.sum(axis=1) - values[:, i]) / (repeats - 1))
                             for i in range(repeats)])
    deltas = leave_one_out - leave_one_out[:, selected_index, None]
    mc_se = np.sqrt((repeats - 1) / repeats * np.sum((deltas - deltas.mean(axis=0)) ** 2, axis=0))
    rank = np.argsort(np.argsort(total, kind='stable'), kind='stable') + 1
    half = repeats // 2
    first, second = scores(values[:, :half].mean(axis=1)), scores(values[:, half:].mean(axis=1))
    rows = []
    for i in range(len(values)):
        rows.append(dict(candidate=i, fresh_rank=int(rank[i]), selected=i == selected_index,
                         fresh_mean_probability_nll=float(total[i]),
                         fresh_gap_to_selected=float(total[i] - total[selected_index]),
                         paired_seed_jackknife_mc_se=float(mc_se[i]),
                         mean_abs_probability_difference_to_selected=float(np.mean(np.abs(mean[i, 1:] - mean[selected_index, 1:]))),
                         max_abs_probability_difference_to_selected=float(np.max(np.abs(mean[i, 1:] - mean[selected_index, 1:]))),
                         identical_all_seed_probabilities_to_selected=bool(np.array_equal(values[i], values[selected_index])),
                         first_half_nll=float(first[i]), second_half_nll=float(second[i])))
    diagnostic = dict(selected_candidate=int(selected_index), fresh_winner=int(np.argmin(total)),
                      selected_in_fresh_minimum=bool(np.isclose(total[selected_index], total.min(), rtol=0, atol=1e-12)),
                      first_half_winner=int(np.argmin(first)), second_half_winner=int(np.argmin(second)),
                      fresh_best_improvement=float(total[selected_index] - total.min()),
                      max_seed_jackknife_mc_se=float(mc_se.max()),
                      uncertainty='PF seed integration noise at fixed observed data; parameter intervals remain pending')
    return rows, diagnostic


def learning_summary(observed_feedback: np.ndarray, simulated_feedback: np.ndarray,
                     expected_correct: np.ndarray, conditional_correct: np.ndarray,
                     segment_size: int, late_window: int,
                     target_accuracy: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Summarize exact correctness; the first trial initializes the model and is excluded."""
    observed = np.asarray(observed_feedback, dtype=float)
    simulated = np.asarray(simulated_feedback, dtype=float)
    expected = np.asarray(expected_correct, dtype=float)
    conditional = np.asarray(conditional_correct, dtype=float)
    if simulated.ndim != 2 or expected.shape != simulated.shape:
        raise ValueError('Need aligned rollout × trial arrays')
    if observed.shape != (simulated.shape[1],) or conditional.shape != observed.shape or len(observed) < 2:
        raise ValueError('Observed and conditional trial dimensions must agree')
    if not np.isin(observed, [0, .5, 1]).all() or not np.isin(simulated, [0, .5, 1]).all():
        raise ValueError('Feedback must use the task values 0, 0.5 and 1')
    if any(not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) for p in (expected, conditional)):
        raise ValueError('Correct probabilities must be finite and lie in [0, 1]')
    if segment_size < 1 or late_window < 1 or not 0 <= target_accuracy <= 1:
        raise ValueError('Invalid segment or target settings')

    def segment(start: int, end: int) -> dict[str, Any]:
        accuracy = (simulated[:, start:end] == 1).mean(axis=1)
        q10, median, q90 = np.quantile(accuracy, [.1, .5, .9])
        expected_means = expected[:, start:end].mean(axis=1)
        eq10, emed, eq90 = np.quantile(expected_means, [.1, .5, .9])
        return dict(first_valid_trial=start + 1, last_valid_trial=end, scored_trials=end - start,
                    observed_accuracy=float(np.mean(observed[start:end] == 1)),
                    conditional_correct_probability=float(conditional[start:end].mean()),
                    autonomous_accuracy_q10=float(q10), autonomous_accuracy_median=float(median),
                    autonomous_accuracy_q90=float(q90), autonomous_expected_q10=float(eq10),
                    autonomous_expected_median=float(emed), autonomous_expected_q90=float(eq90),
                    fraction_rollouts_at_or_above_target=float(np.mean(accuracy >= target_accuracy)))

    n = len(observed)
    rows = [segment(start, min(n, start + segment_size)) for start in range(1, n, segment_size)]
    overall, early, late = segment(1, n), segment(1, min(n, 1 + late_window)), segment(max(1, n - late_window), n)
    summary = {f'{phase}_{key}': value for phase, row in [('overall', overall), ('early', early), ('late', late)]
               for key, value in row.items()}
    summary['overall_observed_above_autonomous_q90'] = overall['observed_accuracy'] > overall['autonomous_accuracy_q90']
    summary['late_observed_above_autonomous_q90'] = late['observed_accuracy'] > late['autonomous_accuracy_q90']
    summary['late_expected_median_minus_early'] = late['autonomous_expected_median'] - early['autonomous_expected_median']
    summary['target_accuracy'] = target_accuracy
    return rows, summary

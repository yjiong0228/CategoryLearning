"""Audit how particle-filter strategy state reaches observable choices.

The audit is invoked by the shared :mod:`run_model_evaluation` entry point. It
replays the fitted dynamic model with common PF seeds and computes three
off-policy readouts from every pre-choice particle state while leaving the
actual filtering weights and cognitive-state updates unchanged:

``hypothesis_map``
    Each particle follows only its highest-prior active hypothesis. This tests
    whether within-particle hypothesis averaging suppresses dynamics.

``adaptive_sharpening``
    The fitted sharpening power relaxes toward one as exploration rises. This
    tests a hypothesis-level reduction in commitment during exploration.

``exploration_lapse``
    The fitted choice probability is mixed toward chance in proportion to the
    exploration event probability. This is a diagnostic upper bound on a
    direct strategy-to-choice coupling, not a fitted model variant.

The current fitted readout remains the only readout used to update particle
weights. Consequently all four curves share identical latent particle paths.
"""

from __future__ import annotations

from copy import deepcopy
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.Bayesian_state.metrics import sliding_binary_metrics
from src.Bayesian_state.evaluation.particle_filter.strategy import (
    _causal_recent_accuracy,
    _curve_volatility,
    _event_indices,
    _run_variant,
    _subject_runtime,
)
from src.Bayesian_state.simulation.config import load_yaml, resolve_subjects


LOGGER = logging.getLogger(__name__)

CURRENT = "current_marginal"
HYPOTHESIS_MAP = "hypothesis_map"
ADAPTIVE_SHARPENING = "adaptive_sharpening"
EXPLORATION_LAPSE = "exploration_lapse"
READOUT_ORDER = (
    CURRENT,
    HYPOTHESIS_MAP,
    ADAPTIVE_SHARPENING,
    EXPLORATION_LAPSE,
)
READOUT_LABELS = {
    CURRENT: "Current marginal",
    HYPOTHESIS_MAP: "Within-particle MAP",
    ADAPTIVE_SHARPENING: "Exploration-adaptive sharpening",
    EXPLORATION_LAPSE: "Exploration-gated choice uncertainty",
}
READOUT_SHORT_LABELS = {
    CURRENT: "Current",
    HYPOTHESIS_MAP: "Hypothesis MAP",
    ADAPTIVE_SHARPENING: "Adaptive power",
    EXPLORATION_LAPSE: "Choice coupling",
}
READOUT_COLORS = {
    CURRENT: "#0072B2",
    HYPOTHESIS_MAP: "#D55E00",
    ADAPTIVE_SHARPENING: "#7A5195",
    EXPLORATION_LAPSE: "#CC6677",
}
ANCESTRAL_FIELDS = (
    "correct_probability",
    "strategy_exploit",
    "strategy_local_explore",
    "strategy_global_explore",
    "swap_event",
    "transition_rate",
    "search_range",
    "failure_pressure",
    "mastery_evidence",
)
OPTIONAL_ANCESTRAL_FIELDS = (
    "executed_hypothesis",
    "execution_switch_event",
    "execution_dwell_trials",
)
PLOT_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
}
PHASE_ORDER = ("warmup", "low", "middle_recovery", "mastery")
PHASE_LABELS = {
    "warmup": "Warm-up",
    "low": "Low performance",
    "middle_recovery": "Learning / recovery",
    "mastery": "Mastery",
}
PHASE_COLORS = {
    "warmup": "#999999",
    "low": "#D55E00",
    "middle_recovery": "#E69F00",
    "mastery": "#0072B2",
}
GAIN_SCREEN_STRATA = (
    "overall",
    "low",
    "middle_recovery",
    "mastery",
    "deep_valley",
)
GAIN_SCREEN_STRATUM_LABELS = {
    "overall": "All scored trials",
    "low": "Low performance",
    "middle_recovery": "Learning / recovery",
    "mastery": "Mastery",
    "deep_valley": "Deep valley",
}
STRATEGY_CONFIDENCE_GAIN_PATH = (
    "engine.choice_readout.kwargs.strategy_confidence_gain"
)


def _strategy_confidence_gain_values(
    values: Sequence[float],
) -> tuple[float, ...]:
    """Validate a coarse gain screen that includes the disabled baseline."""

    parsed = tuple(float(value) for value in values)
    if not parsed:
        raise ValueError("Strategy-confidence gain screen cannot be empty.")
    if not all(np.isfinite(value) and 0.0 <= value <= 10.0 for value in parsed):
        raise ValueError("Strategy-confidence gains must be finite and lie in [0, 10].")
    if len(set(parsed)) != len(parsed):
        raise ValueError("Strategy-confidence gain screen cannot contain duplicates.")
    if 0.0 not in parsed:
        raise ValueError("Strategy-confidence gain screen must include gain=0 as ablation.")
    return tuple(sorted(parsed))


def _engine_with_strategy_confidence_gain(
    engine_config: Mapping[str, Any],
    gain: float,
) -> dict[str, Any]:
    """Return a copy with only the strategy-confidence gain changed."""

    resolved = deepcopy(dict(engine_config))
    choice_readout = resolved.setdefault("choice_readout", {})
    if not isinstance(choice_readout, dict):
        raise TypeError("choice_readout must be a mapping")
    kwargs = choice_readout.setdefault("kwargs", {})
    if not isinstance(kwargs, dict):
        raise TypeError("choice_readout.kwargs must be a mapping")
    kwargs["strategy_confidence_gain"] = float(gain)
    return resolved


def _performance_phase(
    causal_accuracy: np.ndarray,
    *,
    low_threshold: float,
    mastery_threshold: float = 0.85,
) -> np.ndarray:
    """Label performance using only accuracy available before the current trial."""

    if not 0.0 <= float(low_threshold) < float(mastery_threshold) <= 1.0:
        raise ValueError(
            "Performance thresholds must satisfy 0 <= low < mastery <= 1."
        )
    accuracy = np.asarray(causal_accuracy, dtype=float).reshape(-1)
    phase = np.full(accuracy.size, "warmup", dtype=object)
    finite = np.isfinite(accuracy)
    phase[finite & (accuracy <= float(low_threshold))] = "low"
    phase[
        finite
        & (accuracy > float(low_threshold))
        & (accuracy < float(mastery_threshold))
    ] = "middle_recovery"
    phase[finite & (accuracy >= float(mastery_threshold))] = "mastery"
    return phase


def _correct_probabilities(
    category_probabilities: np.ndarray,
    true_category_index: np.ndarray,
) -> np.ndarray:
    probabilities = np.asarray(category_probabilities, dtype=float)
    true_index = np.asarray(true_category_index, dtype=int).reshape(-1)
    if probabilities.ndim != 3 or probabilities.shape[2] != 2:
        raise ValueError(
            "choice-transmission probability runs must have shape "
            f"(seeds, trials, 2), got {probabilities.shape}."
        )
    if true_index.size != probabilities.shape[1]:
        raise ValueError("true-category indices do not align with probability trials.")
    if np.any((true_index < 0) | (true_index >= probabilities.shape[2])):
        raise ValueError("true-category indices fall outside the probability width.")
    selected = np.take_along_axis(
        probabilities,
        true_index[None, :, None],
        axis=2,
    )
    return selected[:, :, 0]


def _rolling_curve(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    window_size: int,
    score_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    observed_curve, predicted_curve, _ = sliding_binary_metrics(
        observed,
        predicted,
        window_size=int(window_size),
        score_trial_mask=score_mask,
    )
    return observed_curve, predicted_curve


def _low_entry_delta(
    values: np.ndarray,
    events: Sequence[int],
    *,
    pre_offsets: Sequence[int] = (-3, -2, -1),
    post_offsets: Sequence[int] = (0, 1, 2),
) -> tuple[float, int]:
    series = np.asarray(values, dtype=float).reshape(-1)
    differences: list[float] = []
    for event in events:
        pre = np.asarray([int(event) + int(offset) for offset in pre_offsets], dtype=int)
        post = np.asarray([int(event) + int(offset) for offset in post_offsets], dtype=int)
        if np.any(pre < 0) or np.any(post >= series.size):
            continue
        before = series[pre]
        after = series[post]
        if np.all(np.isfinite(before)) and np.all(np.isfinite(after)):
            differences.append(float(np.mean(after) - np.mean(before)))
    return (
        float(np.mean(differences)) if differences else float("nan"),
        int(len(differences)),
    )


def _choice_nll(
    category_probabilities: np.ndarray,
    observed_choice_index: np.ndarray,
    score_mask: np.ndarray,
) -> float:
    probabilities = np.asarray(category_probabilities, dtype=float)
    choice = np.asarray(observed_choice_index, dtype=int).reshape(-1)
    mask = np.asarray(score_mask, dtype=bool).reshape(-1).copy()
    mask &= (choice >= 0) & (choice < probabilities.shape[1])
    mask &= np.all(np.isfinite(probabilities), axis=1)
    if mask.size:
        mask[0] = False
    rows = np.flatnonzero(mask)
    if not rows.size:
        return float("nan")
    selected = probabilities[rows, choice[rows]]
    return float(np.mean(-np.log(np.clip(selected, 1e-12, 1.0))))


def _normalized_weights(values: np.ndarray) -> np.ndarray:
    weights = np.asarray(values, dtype=float).reshape(-1)
    if (
        weights.size == 0
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
        or float(np.sum(weights)) <= 0.0
    ):
        raise ValueError("Ancestral path weights must be finite and non-negative.")
    return weights / float(np.sum(weights))


def _weighted_path_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: Sequence[float] = (0.10, 0.50, 0.90),
) -> np.ndarray:
    """Compute weighted pointwise quantiles across complete latent paths."""

    paths = np.asarray(values, dtype=float)
    probability = _normalized_weights(weights)
    requested = np.asarray(quantiles, dtype=float).reshape(-1)
    if paths.ndim != 2 or paths.shape[0] != probability.size:
        raise ValueError("Ancestral values must have shape (paths, trials).")
    if not np.all(np.isfinite(paths)):
        raise ValueError("Ancestral path quantiles require finite values.")
    if np.any((requested < 0.0) | (requested > 1.0)):
        raise ValueError("Ancestral path quantiles must lie in [0, 1].")
    output = np.empty((requested.size, paths.shape[1]), dtype=float)
    for trial_index in range(paths.shape[1]):
        sample = paths[:, trial_index]
        order = np.argsort(sample, kind="stable")
        cumulative = np.cumsum(probability[order])
        cumulative[-1] = 1.0
        indices = np.searchsorted(cumulative, requested, side="left")
        output[:, trial_index] = sample[order[np.clip(indices, 0, order.size - 1)]]
    return output


def _posterior_medoid_index(
    path_features: np.ndarray,
    weights: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Choose the observed posterior path closest to the weighted ensemble."""

    strategy = np.asarray(path_features, dtype=float)
    probability = _normalized_weights(weights)
    if strategy.ndim != 3 or strategy.shape[0] != probability.size:
        raise ValueError(
            "Strategy paths must have shape (paths, trials, components)."
        )
    if not np.all(np.isfinite(strategy)):
        raise ValueError("Strategy paths contain non-finite values.")
    flattened = strategy.reshape(strategy.shape[0], -1)
    scores = np.empty(flattened.shape[0], dtype=float)
    for path_index, candidate in enumerate(flattened):
        distance = np.mean(np.abs(flattened - candidate[None, :]), axis=1)
        scores[path_index] = float(np.sum(probability * distance))
    order = np.lexsort((-probability, scores))
    return int(order[0]), scores


def _rolling_path_means(values: np.ndarray, window_size: int) -> np.ndarray:
    paths = np.asarray(values, dtype=float)
    if paths.ndim != 2:
        raise ValueError("Rolling ancestral paths must be a 2-D matrix.")
    window = int(window_size)
    if window <= 0 or paths.shape[1] <= window:
        raise ValueError("Rolling window must be shorter than the path.")
    starts = np.arange(1, paths.shape[1] - window + 1, dtype=int)
    return np.column_stack(
        [np.mean(paths[:, start : start + window], axis=1) for start in starts]
    )


def _summarize_ancestral_runs(
    runs: Sequence[Mapping[str, Any]],
    *,
    subject_id: int,
    window_size: int,
) -> dict[str, Any]:
    """Combine equal-seed PF genealogies into one posterior path ensemble."""

    optional_fields = tuple(
        field
        for field in OPTIONAL_ANCESTRAL_FIELDS
        if all(
            f"audit_ancestral_{field}" in (run.get("state_log") or {})
            for run in runs
        )
    )
    path_fields = ANCESTRAL_FIELDS + optional_fields
    combined: dict[str, list[np.ndarray]] = {
        field: [] for field in path_fields
    }
    particle_indices: list[np.ndarray] = []
    path_weights: list[np.ndarray] = []
    seed_values: list[np.ndarray] = []
    terminal_particles: list[np.ndarray] = []
    resampling_counts: list[int] = []
    start_ancestor_counts: list[int] = []
    true_accuracy = None
    marginal_correct_runs: list[np.ndarray] = []

    for seed_index, run in enumerate(runs):
        state_log = run.get("state_log") or {}
        indices = np.asarray(
            state_log.get("audit_ancestral_particle_indices"), dtype=int
        )
        weights = _normalized_weights(
            np.asarray(state_log.get("audit_ancestral_weights"), dtype=float)
        )
        if indices.ndim != 2 or indices.shape[0] != weights.size:
            raise ValueError("Ancestral particle indices do not align with weights.")
        n_paths, n_trials = indices.shape
        particle_indices.append(indices)
        path_weights.append(weights / float(len(runs)))
        seed = int(run.get("trajectory_seed", seed_index))
        seed_values.append(np.full(n_paths, seed, dtype=int))
        terminal_particles.append(np.arange(n_paths, dtype=int))
        start_ancestor_counts.append(int(np.unique(indices[:, 0]).size))
        resampled = np.asarray(state_log.get("resampled"), dtype=bool).reshape(-1)
        resampling_counts.append(int(np.sum(resampled)))
        for field in path_fields:
            values = np.asarray(
                state_log.get(f"audit_ancestral_{field}"), dtype=float
            )
            if values.shape != (n_paths, n_trials):
                raise ValueError(
                    f"Ancestral field {field!r} has shape {values.shape}; "
                    f"expected {(n_paths, n_trials)}."
                )
            combined[field].append(values)

        metrics = (run.get("metrics_by_mode") or {}).get("prior_t") or {}
        category_probabilities = np.asarray(
            metrics.get("pred_category_probs"), dtype=float
        )
        category_index = np.asarray(
            metrics.get("true_category_index"), dtype=int
        ).reshape(-1)
        marginal_correct_runs.append(
            _correct_probabilities(
                category_probabilities[None, :, :], category_index
            )[0]
        )
        observed = np.asarray(metrics.get("true_acc"), dtype=float).reshape(-1)
        if true_accuracy is None:
            true_accuracy = observed
        elif not np.array_equal(true_accuracy, observed, equal_nan=True):
            raise ValueError("Ancestral common-seed runs disagree on true accuracy.")

    weights = _normalized_weights(np.concatenate(path_weights))
    paths = {field: np.concatenate(values, axis=0) for field, values in combined.items()}
    indices = np.concatenate(particle_indices, axis=0)
    seeds = np.concatenate(seed_values)
    terminal = np.concatenate(terminal_particles)
    strategy = np.stack(
        [
            paths["strategy_exploit"],
            paths["strategy_local_explore"],
            paths["strategy_global_explore"],
        ],
        axis=2,
    )
    if not np.allclose(np.sum(strategy, axis=2), 1.0, atol=1e-8):
        raise ValueError("Ancestral strategy components do not sum to one.")
    medoid_features = np.concatenate(
        [strategy, paths["correct_probability"][:, :, None]],
        axis=2,
    )
    representative_index, medoid_scores = _posterior_medoid_index(
        medoid_features, weights
    )
    exploration = (
        paths["strategy_local_explore"] + paths["strategy_global_explore"]
    )
    posterior_swap_event_probability = np.sum(
        weights[:, None] * paths["swap_event"], axis=0
    )
    correct_quantiles = _weighted_path_quantiles(
        paths["correct_probability"], weights
    )
    exploration_quantiles = _weighted_path_quantiles(exploration, weights)
    rolling_paths = _rolling_path_means(
        paths["correct_probability"], int(window_size)
    )
    rolling_quantiles = _weighted_path_quantiles(rolling_paths, weights)
    assert true_accuracy is not None
    marginal_correct = np.mean(np.stack(marginal_correct_runs, axis=0), axis=0)
    observed_rolling = _rolling_path_means(
        true_accuracy[None, :], int(window_size)
    )[0]
    marginal_rolling = _rolling_path_means(
        marginal_correct[None, :], int(window_size)
    )[0]
    representative_rolling = rolling_paths[representative_index]
    return {
        "subject_id": int(subject_id),
        "n_trials": int(indices.shape[1]),
        "window_size": int(window_size),
        "weights": weights,
        "seeds": seeds,
        "terminal_particles": terminal,
        "particle_indices": indices,
        "paths": paths,
        "path_fields": path_fields,
        "strategy": strategy,
        "representative_index": int(representative_index),
        "representative_seed": int(seeds[representative_index]),
        "representative_terminal_particle": int(terminal[representative_index]),
        "representative_weight": float(weights[representative_index]),
        "representative_medoid_score": float(medoid_scores[representative_index]),
        "effective_path_count": float(1.0 / np.sum(np.square(weights))),
        "mean_start_ancestor_count": float(np.mean(start_ancestor_counts)),
        "mean_resampling_count": float(np.mean(resampling_counts)),
        "true_accuracy": true_accuracy,
        "marginal_correct": marginal_correct,
        "correct_quantiles": correct_quantiles,
        "exploration_quantiles": exploration_quantiles,
        "posterior_swap_event_probability": posterior_swap_event_probability,
        "rolling_trial": np.arange(
            int(window_size) + 1, indices.shape[1] + 1, dtype=int
        ),
        "observed_rolling": observed_rolling,
        "marginal_rolling": marginal_rolling,
        "representative_rolling": representative_rolling,
        "rolling_quantiles": rolling_quantiles,
    }


def _summarize_subject_runs(
    runs: Sequence[Mapping[str, Any]],
    *,
    subject_id: int,
    window_size: int,
    low_accuracy_threshold: float,
) -> dict[str, Any]:
    probability_runs: dict[str, list[np.ndarray]] = {
        readout: [] for readout in READOUT_ORDER
    }
    q10_runs: list[np.ndarray] = []
    q50_runs: list[np.ndarray] = []
    q90_runs: list[np.ndarray] = []
    exploration_runs: list[np.ndarray] = []
    unsharpened_runs: list[np.ndarray] = []
    sharpened_no_lapse_runs: list[np.ndarray] = []
    strategy_confidence_no_lapse_runs: list[np.ndarray] = []
    persistent_execution_no_lapse_runs: list[np.ndarray] = []
    correct_available_runs: list[np.ndarray] = []
    correct_prior_mass_runs: list[np.ndarray] = []
    best_active_correct_runs: list[np.ndarray] = []
    failure_pressure_runs: list[np.ndarray] = []
    mastery_evidence_runs: list[np.ndarray] = []
    confidence_signal_runs: list[np.ndarray] = []
    choice_precision_runs: list[np.ndarray] = []
    execution_switch_probability_runs: list[np.ndarray] = []
    execution_switch_event_runs: list[np.ndarray] = []
    execution_dwell_runs: list[np.ndarray] = []
    persistent_execution_flags: list[bool] = []
    true_accuracy = None
    true_category_index = None
    observed_choice_index = None
    score_mask = None

    audit_keys = {
        HYPOTHESIS_MAP: "audit_hypothesis_map",
        ADAPTIVE_SHARPENING: "audit_adaptive_sharpening",
        EXPLORATION_LAPSE: "audit_exploration_lapse",
    }
    for run in runs:
        metrics = (run.get("metrics_by_mode") or {}).get("prior_t") or {}
        state_log = run.get("state_log") or {}
        probability_runs[CURRENT].append(
            np.asarray(metrics["pred_category_probs"], dtype=float)
        )
        for readout, key in audit_keys.items():
            if key not in state_log:
                raise ValueError(
                    f"Choice-transmission audit field {key!r} was not persisted."
                )
            probability_runs[readout].append(np.asarray(state_log[key], dtype=float))
        q10_runs.append(np.asarray(state_log["audit_particle_correct_q10"], dtype=float))
        q50_runs.append(np.asarray(state_log["audit_particle_correct_q50"], dtype=float))
        q90_runs.append(np.asarray(state_log["audit_particle_correct_q90"], dtype=float))
        exploration_runs.append(
            1.0
            - np.asarray(state_log["predictive_strategy_exploit"], dtype=float)
        )
        unsharpened_runs.append(
            np.asarray(state_log["audit_unsharpened_expectation"], dtype=float)
        )
        sharpened_no_lapse_runs.append(
            np.asarray(state_log["audit_sharpened_no_lapse"], dtype=float)
        )
        strategy_confidence_no_lapse_runs.append(
            np.asarray(
                state_log["audit_strategy_confidence_no_lapse"],
                dtype=float,
            )
        )
        persistent_execution_no_lapse_runs.append(
            np.asarray(
                state_log.get(
                    "audit_persistent_execution_no_lapse",
                    state_log["audit_strategy_confidence_no_lapse"],
                ),
                dtype=float,
            )
        )
        persistent_execution_flags.append(
            "audit_persistent_execution_no_lapse" in state_log
        )
        correct_available_runs.append(
            np.asarray(
                state_log["audit_correct_predicting_available_probability"],
                dtype=float,
            )
        )
        correct_prior_mass_runs.append(
            np.asarray(
                state_log["audit_correct_predicting_prior_mass"],
                dtype=float,
            )
        )
        best_active_correct_runs.append(
            np.asarray(
                state_log["audit_best_active_correct_probability"],
                dtype=float,
            )
        )
        failure_pressure_runs.append(
            np.asarray(state_log["predictive_failure_pressure"], dtype=float)
        )
        mastery_evidence_runs.append(
            np.asarray(state_log["predictive_mastery_evidence"], dtype=float)
        )
        confidence_signal_runs.append(
            np.asarray(
                state_log["predictive_choice_confidence_signal"], dtype=float
            )
        )
        choice_precision_runs.append(
            np.asarray(
                state_log["predictive_strategy_choice_precision"], dtype=float
            )
        )
        execution_switch_probability_runs.append(
            np.asarray(
                state_log.get(
                    "predictive_execution_switch_probability",
                    np.zeros_like(state_log["predictive_failure_pressure"]),
                ),
                dtype=float,
            )
        )
        execution_switch_event_runs.append(
            np.asarray(
                state_log.get(
                    "predictive_execution_switch_event_probability",
                    np.zeros_like(state_log["predictive_failure_pressure"]),
                ),
                dtype=float,
            )
        )
        execution_dwell_runs.append(
            np.asarray(
                state_log.get(
                    "predictive_execution_dwell_trials",
                    np.full_like(
                        state_log["predictive_failure_pressure"],
                        np.nan,
                        dtype=float,
                    ),
                ),
                dtype=float,
            )
        )
        if true_accuracy is None:
            true_accuracy = np.asarray(metrics["true_acc"], dtype=float)
            true_category_index = np.asarray(
                metrics["true_category_index"], dtype=int
            )
            observed_choice_index = np.asarray(
                metrics["observed_choice_index"], dtype=int
            )
            raw_mask = metrics.get("score_trial_mask")
            score_mask = (
                np.ones(true_accuracy.size, dtype=bool)
                if raw_mask is None
                else np.asarray(raw_mask, dtype=bool)
            )

    if any(
        value is None
        for value in (
            true_accuracy,
            true_category_index,
            observed_choice_index,
            score_mask,
        )
    ):
        raise RuntimeError("Choice-transmission audit did not receive trial metrics.")
    assert true_accuracy is not None
    assert true_category_index is not None
    assert observed_choice_index is not None
    assert score_mask is not None
    if len(set(persistent_execution_flags)) != 1:
        raise ValueError(
            "Choice-transmission runs disagree on persistent-execution state."
        )
    persistent_execution_enabled = bool(persistent_execution_flags[0])

    category_runs = {
        readout: np.stack(values, axis=0)
        for readout, values in probability_runs.items()
    }
    correct_runs = {
        readout: _correct_probabilities(values, true_category_index)
        for readout, values in category_runs.items()
    }
    expected_correct = {
        readout: np.mean(values, axis=0)
        for readout, values in correct_runs.items()
    }
    mean_category = {
        readout: np.mean(values, axis=0)
        for readout, values in category_runs.items()
    }
    rolling = {}
    observed_curve = None
    for readout in READOUT_ORDER:
        observed_curve, predicted_curve = _rolling_curve(
            true_accuracy,
            expected_correct[readout],
            window_size=int(window_size),
            score_mask=score_mask,
        )
        rolling[readout] = predicted_curve
    assert observed_curve is not None

    causal_accuracy = _causal_recent_accuracy(true_accuracy, int(window_size))
    event_indices = _event_indices(
        true_accuracy,
        causal_accuracy,
        low_threshold=float(low_accuracy_threshold),
    )
    low_events = event_indices["low_performance_entry"]
    q10 = np.mean(np.vstack(q10_runs), axis=0)
    q50 = np.mean(np.vstack(q50_runs), axis=0)
    q90 = np.mean(np.vstack(q90_runs), axis=0)
    exploration = np.mean(np.vstack(exploration_runs), axis=0)
    unsharpened_category = np.mean(
        np.stack(unsharpened_runs, axis=0), axis=0
    )
    sharpened_no_lapse_category = np.mean(
        np.stack(sharpened_no_lapse_runs, axis=0), axis=0
    )
    unsharpened_correct = _correct_probabilities(
        unsharpened_category[None, :, :], true_category_index
    )[0]
    sharpened_no_lapse_correct = _correct_probabilities(
        sharpened_no_lapse_category[None, :, :], true_category_index
    )[0]
    strategy_confidence_no_lapse_category = np.mean(
        np.stack(strategy_confidence_no_lapse_runs, axis=0), axis=0
    )
    strategy_confidence_no_lapse_correct = _correct_probabilities(
        strategy_confidence_no_lapse_category[None, :, :],
        true_category_index,
    )[0]
    persistent_execution_no_lapse_category = np.mean(
        np.stack(persistent_execution_no_lapse_runs, axis=0), axis=0
    )
    persistent_execution_no_lapse_correct = _correct_probabilities(
        persistent_execution_no_lapse_category[None, :, :],
        true_category_index,
    )[0]
    correct_available = np.mean(np.vstack(correct_available_runs), axis=0)
    correct_prior_mass = np.mean(np.vstack(correct_prior_mass_runs), axis=0)
    best_active_correct = np.mean(
        np.vstack(best_active_correct_runs), axis=0
    )
    confidence_signal = np.mean(np.vstack(confidence_signal_runs), axis=0)
    choice_precision = np.mean(np.vstack(choice_precision_runs), axis=0)
    failure_pressure = np.mean(np.vstack(failure_pressure_runs), axis=0)
    mastery_evidence = np.mean(np.vstack(mastery_evidence_runs), axis=0)
    execution_switch_probability = np.mean(
        np.vstack(execution_switch_probability_runs), axis=0
    )
    execution_switch_event = np.mean(
        np.vstack(execution_switch_event_runs), axis=0
    )
    execution_dwell_trials = np.nanmean(
        np.vstack(execution_dwell_runs), axis=0
    )
    phase = _performance_phase(
        causal_accuracy,
        low_threshold=float(low_accuracy_threshold),
    )
    low_mask = np.isfinite(causal_accuracy) & (
        causal_accuracy <= float(low_accuracy_threshold)
    )
    mastery_mask = np.isfinite(causal_accuracy) & (causal_accuracy >= 0.85)

    def phase_mean(values: np.ndarray, mask: np.ndarray) -> float:
        selected = np.asarray(values, dtype=float)[mask]
        selected = selected[np.isfinite(selected)]
        return float(np.mean(selected)) if selected.size else float("nan")

    return {
        "subject_id": int(subject_id),
        "window_size": int(window_size),
        "n_trials": int(true_accuracy.size),
        "n_seeds": int(len(runs)),
        "true_accuracy": true_accuracy,
        "true_category_index": true_category_index,
        "observed_choice_index": observed_choice_index,
        "score_mask": score_mask,
        "causal_accuracy": causal_accuracy,
        "observed_curve": observed_curve,
        "category_runs": category_runs,
        "mean_category": mean_category,
        "correct_runs": correct_runs,
        "expected_correct": expected_correct,
        "rolling": rolling,
        "exploration": exploration,
        "phase": phase,
        "correct_available": correct_available,
        "correct_prior_mass": correct_prior_mass,
        "best_active_correct": best_active_correct,
        "unsharpened_correct": unsharpened_correct,
        "sharpened_no_lapse_correct": sharpened_no_lapse_correct,
        "strategy_confidence_no_lapse_correct": (
            strategy_confidence_no_lapse_correct
        ),
        "persistent_execution_no_lapse_correct": (
            persistent_execution_no_lapse_correct
        ),
        "persistent_execution_enabled": persistent_execution_enabled,
        "failure_pressure": failure_pressure,
        "mastery_evidence": mastery_evidence,
        "choice_confidence_signal": confidence_signal,
        "strategy_choice_precision": choice_precision,
        "execution_switch_probability": execution_switch_probability,
        "execution_switch_event": execution_switch_event,
        "execution_dwell_trials": execution_dwell_trials,
        "particle_q10": q10,
        "particle_q50": q50,
        "particle_q90": q90,
        "particle_q80_width": q90 - q10,
        "mean_particle_q80_width": float(np.mean(q90 - q10)),
        "low_particle_q80_width": phase_mean(q90 - q10, low_mask),
        "mastery_particle_q80_width": phase_mean(q90 - q10, mastery_mask),
        "events": event_indices,
        "low_events": low_events,
    }


def _trial_rows(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    n_trials = int(profile["n_trials"])
    curve_x = np.arange(
        int(profile["window_size"]) + 1,
        n_trials + 1,
        dtype=int,
    )
    rows: list[dict[str, Any]] = []
    for readout in READOUT_ORDER:
        rolling_full = np.full(n_trials, np.nan, dtype=float)
        curve = np.asarray(profile["rolling"][readout], dtype=float)
        n_curve = min(curve.size, curve_x.size)
        rolling_full[curve_x[:n_curve] - 1] = curve[:n_curve]
        for index in range(n_trials):
            rows.append(
                {
                    "subject_id": int(profile["subject_id"]),
                    "readout": readout,
                    "readout_label": READOUT_LABELS[readout],
                    "trial": int(index + 1),
                    "true_correct": float(profile["true_accuracy"][index]),
                    "causal_recent_accuracy": float(
                        profile["causal_accuracy"][index]
                    ),
                    "strategy_explore": float(profile["exploration"][index]),
                    "expected_correct_probability": float(
                        profile["expected_correct"][readout][index]
                    ),
                    "rolling_expected_accuracy": float(rolling_full[index]),
                    "particle_correct_q10": float(profile["particle_q10"][index]),
                    "particle_correct_q50": float(profile["particle_q50"][index]),
                    "particle_correct_q90": float(profile["particle_q90"][index]),
                }
            )
    return rows


def _error_transmission_trial_rows(
    profile: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    final_correct = np.asarray(
        profile["expected_correct"][CURRENT], dtype=float
    )
    unsharpened = np.asarray(profile["unsharpened_correct"], dtype=float)
    sharpened = np.asarray(
        profile["sharpened_no_lapse_correct"], dtype=float
    )
    strategy_confidence = np.asarray(
        profile["strategy_confidence_no_lapse_correct"], dtype=float
    )
    persistent_execution = np.asarray(
        profile["persistent_execution_no_lapse_correct"], dtype=float
    )
    for index in range(int(profile["n_trials"])):
        rows.append(
            {
                "subject_id": int(profile["subject_id"]),
                "trial": int(index + 1),
                "persistent_execution_enabled": bool(profile["persistent_execution_enabled"]),
                "performance_phase": str(profile["phase"][index]),
                "performance_phase_label": PHASE_LABELS[
                    str(profile["phase"][index])
                ],
                "observed_correct": float(profile["true_accuracy"][index]),
                "causal_recent_accuracy": float(
                    profile["causal_accuracy"][index]
                ),
                "correct_predicting_rule_available": float(
                    profile["correct_available"][index]
                ),
                "best_active_correct_probability": float(
                    profile["best_active_correct"][index]
                ),
                "belief_mass_on_correct_predicting_rules": float(
                    profile["correct_prior_mass"][index]
                ),
                "belief_only_correct_probability": float(unsharpened[index]),
                "sharpened_correct_probability": float(sharpened[index]),
                "strategy_confidence_correct_probability": float(
                    strategy_confidence[index]
                ),
                "persistent_execution_correct_probability": float(
                    persistent_execution[index]
                ),
                "final_correct_probability": float(final_correct[index]),
                "sharpening_effect": float(
                    sharpened[index] - unsharpened[index]
                ),
                "strategy_confidence_effect": float(
                    strategy_confidence[index] - sharpened[index]
                ),
                "persistent_execution_effect": float(
                    persistent_execution[index] - strategy_confidence[index]
                ),
                "output_noise_effect": float(
                    final_correct[index] - persistent_execution[index]
                ),
                "accuracy_residual": float(
                    profile["true_accuracy"][index] - final_correct[index]
                ),
                "strategy_explore": float(profile["exploration"][index]),
                "failure_pressure": float(profile["failure_pressure"][index]),
                "mastery_evidence": float(profile["mastery_evidence"][index]),
                "execution_switch_probability": float(
                    profile["execution_switch_probability"][index]
                ),
                "execution_switch_event_probability": float(
                    profile["execution_switch_event"][index]
                ),
                "execution_dwell_trials": float(
                    profile["execution_dwell_trials"][index]
                ),
            }
        )
    return rows


def _error_transmission_phase_rows(
    profile: Mapping[str, Any],
) -> list[dict[str, Any]]:
    trial_rows = pd.DataFrame(_error_transmission_trial_rows(profile))
    value_columns = (
        "observed_correct",
        "causal_recent_accuracy",
        "correct_predicting_rule_available",
        "best_active_correct_probability",
        "belief_mass_on_correct_predicting_rules",
        "belief_only_correct_probability",
        "sharpened_correct_probability",
        "strategy_confidence_correct_probability",
        "persistent_execution_correct_probability",
        "final_correct_probability",
        "sharpening_effect",
        "strategy_confidence_effect",
        "persistent_execution_effect",
        "output_noise_effect",
        "accuracy_residual",
        "strategy_explore",
        "failure_pressure",
        "mastery_evidence",
        "execution_switch_probability",
        "execution_switch_event_probability",
        "execution_dwell_trials",
    )
    rows: list[dict[str, Any]] = []
    for phase in PHASE_ORDER:
        selected = trial_rows[trial_rows["performance_phase"] == phase]
        if selected.empty:
            continue
        row: dict[str, Any] = {
            "subject_id": int(profile["subject_id"]),
            "performance_phase": phase,
            "performance_phase_label": PHASE_LABELS[phase],
            "n_trials": int(len(selected)),
            "persistent_execution_enabled": bool(profile["persistent_execution_enabled"]),
        }
        for column in value_columns:
            values = pd.to_numeric(selected[column], errors="coerce")
            row[column] = float(values.mean())
        rows.append(row)
    return rows


def _summary_rows(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    current_curve = np.asarray(profile["rolling"][CURRENT], dtype=float)
    observed_curve = np.asarray(profile["observed_curve"], dtype=float)
    rows: list[dict[str, Any]] = []
    for readout in READOUT_ORDER:
        curve = np.asarray(profile["rolling"][readout], dtype=float)
        finite = np.isfinite(curve) & np.isfinite(observed_curve)
        current_finite = np.isfinite(curve) & np.isfinite(current_curve)
        low_delta, low_event_count = _low_entry_delta(
            profile["expected_correct"][readout],
            profile["low_events"],
        )
        rows.append(
            {
                "subject_id": int(profile["subject_id"]),
                "readout": readout,
                "readout_label": READOUT_LABELS[readout],
                "n_trials": int(profile["n_trials"]),
                "n_common_seeds": int(profile["n_seeds"]),
                "expected_curve_mae_to_subject": (
                    float(np.mean(np.abs(curve[finite] - observed_curve[finite])))
                    if np.any(finite)
                    else float("nan")
                ),
                "expected_curve_volatility": _curve_volatility(curve),
                "choice_nll": _choice_nll(
                    profile["mean_category"][readout],
                    profile["observed_choice_index"],
                    profile["score_mask"],
                ),
                "paired_curve_mae_from_current": (
                    float(
                        np.mean(
                            np.abs(curve[current_finite] - current_curve[current_finite])
                        )
                    )
                    if np.any(current_finite)
                    else float("nan")
                ),
                "mean_absolute_trial_effect_from_current": float(
                    np.mean(
                        np.abs(
                            profile["expected_correct"][readout]
                            - profile["expected_correct"][CURRENT]
                        )
                    )
                ),
                "low_entry_probability_delta": low_delta,
                "low_entry_event_count": low_event_count,
                "mean_particle_q80_width": profile["mean_particle_q80_width"],
                "low_particle_q80_width": profile["low_particle_q80_width"],
                "mastery_particle_q80_width": profile[
                    "mastery_particle_q80_width"
                ],
            }
        )
    return rows


def _gain_screen_masks(
    profile: Mapping[str, Any],
    *,
    deep_valley_threshold: float,
) -> dict[str, np.ndarray]:
    """Return causal evaluation strata; deep valleys are a strict low subset."""

    if not 0.0 <= float(deep_valley_threshold) <= 0.5:
        raise ValueError("deep_valley_threshold must lie in [0, 0.5].")
    n_trials = int(profile["n_trials"])
    score_mask = np.asarray(profile["score_mask"], dtype=bool).reshape(-1).copy()
    if score_mask.size != n_trials:
        raise ValueError("Gain-screen score mask does not align with trials.")
    if score_mask.size:
        score_mask[0] = False
    choices = np.asarray(profile["observed_choice_index"], dtype=int).reshape(-1)
    score_mask &= (choices >= 0) & (choices < 2)
    phase = np.asarray(profile["phase"], dtype=object).reshape(-1)
    causal_accuracy = np.asarray(
        profile["causal_accuracy"], dtype=float
    ).reshape(-1)
    return {
        "overall": score_mask,
        "low": score_mask & (phase == "low"),
        "middle_recovery": score_mask & (phase == "middle_recovery"),
        "mastery": score_mask & (phase == "mastery"),
        "deep_valley": (
            score_mask
            & np.isfinite(causal_accuracy)
            & (causal_accuracy <= float(deep_valley_threshold))
        ),
    }


def _gain_screen_trial_rows(
    profile: Mapping[str, Any],
    *,
    gain: float,
    deep_valley_threshold: float,
) -> list[dict[str, Any]]:
    """Build trial-level source data for one subject/gain replay."""

    n_trials = int(profile["n_trials"])
    mean_category = np.asarray(profile["mean_category"][CURRENT], dtype=float)
    choices = np.asarray(profile["observed_choice_index"], dtype=int).reshape(-1)
    expected_correct = np.asarray(
        profile["expected_correct"][CURRENT], dtype=float
    ).reshape(-1)
    masks = _gain_screen_masks(
        profile,
        deep_valley_threshold=float(deep_valley_threshold),
    )
    score_mask = masks["overall"]
    observed_choice_probability = np.full(n_trials, np.nan, dtype=float)
    valid_choice = (choices >= 0) & (choices < mean_category.shape[1])
    rows_index = np.flatnonzero(valid_choice)
    observed_choice_probability[rows_index] = mean_category[
        rows_index, choices[rows_index]
    ]
    nll = np.full(n_trials, np.nan, dtype=float)
    nll[score_mask] = -np.log(
        np.clip(observed_choice_probability[score_mask], 1e-12, 1.0)
    )
    rows: list[dict[str, Any]] = []
    for index in range(n_trials):
        rows.append(
            {
                "subject_id": int(profile["subject_id"]),
                "strategy_confidence_gain": float(gain),
                "trial": int(index + 1),
                "score_trial": bool(score_mask[index]),
                "performance_phase": str(profile["phase"][index]),
                "performance_phase_label": PHASE_LABELS[
                    str(profile["phase"][index])
                ],
                "deep_valley": bool(masks["deep_valley"][index]),
                "deep_valley_threshold": float(deep_valley_threshold),
                "observed_correct": float(profile["true_accuracy"][index]),
                "expected_correct_probability": float(expected_correct[index]),
                "accuracy_residual": float(
                    profile["true_accuracy"][index] - expected_correct[index]
                ),
                "observed_choice_index": int(choices[index]),
                "predicted_choice_one_probability": float(mean_category[index, 1]),
                "observed_choice_probability": float(
                    observed_choice_probability[index]
                ),
                "choice_nll_contribution": float(nll[index]),
                "causal_recent_accuracy": float(
                    profile["causal_accuracy"][index]
                ),
                "choice_confidence_signal": float(
                    profile["choice_confidence_signal"][index]
                ),
                "strategy_choice_precision": float(
                    profile["strategy_choice_precision"][index]
                ),
                "strategy_explore": float(profile["exploration"][index]),
                "failure_pressure": float(profile["failure_pressure"][index]),
                "mastery_evidence": float(profile["mastery_evidence"][index]),
            }
        )
    return rows


def _gain_screen_summary_rows(
    profile: Mapping[str, Any],
    *,
    gain: float,
    deep_valley_threshold: float,
) -> list[dict[str, Any]]:
    """Summarize proper choice fit separately from visible accuracy phases."""

    trial_df = pd.DataFrame(
        _gain_screen_trial_rows(
            profile,
            gain=float(gain),
            deep_valley_threshold=float(deep_valley_threshold),
        )
    )
    rows: list[dict[str, Any]] = []
    for stratum in GAIN_SCREEN_STRATA:
        if stratum == "overall":
            selected = trial_df[trial_df["score_trial"]]
        elif stratum == "deep_valley":
            selected = trial_df[trial_df["score_trial"] & trial_df["deep_valley"]]
        else:
            selected = trial_df[
                trial_df["score_trial"]
                & (trial_df["performance_phase"] == stratum)
            ]
        if selected.empty:
            continue
        rows.append(
            {
                "subject_id": int(profile["subject_id"]),
                "strategy_confidence_gain": float(gain),
                "stratum": stratum,
                "stratum_label": GAIN_SCREEN_STRATUM_LABELS[stratum],
                "n_trials": int(len(selected)),
                "n_common_seeds": int(profile["n_seeds"]),
                "choice_nll": float(selected["choice_nll_contribution"].mean()),
                "observed_accuracy": float(selected["observed_correct"].mean()),
                "expected_accuracy": float(
                    selected["expected_correct_probability"].mean()
                ),
                "expected_minus_observed_accuracy": float(
                    (
                        selected["expected_correct_probability"]
                        - selected["observed_correct"]
                    ).mean()
                ),
                "mean_confidence_signal": float(
                    selected["choice_confidence_signal"].mean()
                ),
                "mean_choice_precision": float(
                    selected["strategy_choice_precision"].mean()
                ),
                "deep_valley_threshold": float(deep_valley_threshold),
            }
        )
    return rows


def _event_rows(
    profile: Mapping[str, Any],
    *,
    relative_start: int,
    relative_end: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event_type, events in profile["events"].items():
        for event_number, event_index in enumerate(events, start=1):
            for readout in READOUT_ORDER:
                values = profile["expected_correct"][readout]
                for relative_trial in range(
                    int(relative_start), int(relative_end) + 1
                ):
                    index = int(event_index) + int(relative_trial)
                    if not 0 <= index < int(profile["n_trials"]):
                        continue
                    rows.append(
                        {
                            "subject_id": int(profile["subject_id"]),
                            "event_type": event_type,
                            "event_number": int(event_number),
                            "event_trial": int(event_index + 1),
                            "readout": readout,
                            "readout_label": READOUT_LABELS[readout],
                            "relative_trial": int(relative_trial),
                            "expected_correct_probability": float(values[index]),
                            "strategy_explore": float(
                                profile["exploration"][index]
                            ),
                            "causal_recent_accuracy": float(
                                profile["causal_accuracy"][index]
                            ),
                        }
                    )
    return rows


def _ancestral_trial_rows(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    representative = int(profile["representative_index"])
    paths = profile["paths"]
    correct_q = np.asarray(profile["correct_quantiles"], dtype=float)
    explore_q = np.asarray(profile["exploration_quantiles"], dtype=float)
    rows: list[dict[str, Any]] = []
    for trial_index in range(int(profile["n_trials"])):
        rows.append(
            {
                "subject_id": int(profile["subject_id"]),
                "trial": int(trial_index + 1),
                "observed_accuracy": float(profile["true_accuracy"][trial_index]),
                "marginal_correct_probability": float(
                    profile["marginal_correct"][trial_index]
                ),
                "path_correct_q10": float(correct_q[0, trial_index]),
                "path_correct_q50": float(correct_q[1, trial_index]),
                "path_correct_q90": float(correct_q[2, trial_index]),
                "representative_correct_probability": float(
                    paths["correct_probability"][representative, trial_index]
                ),
                "path_exploration_q10": float(explore_q[0, trial_index]),
                "path_exploration_q50": float(explore_q[1, trial_index]),
                "path_exploration_q90": float(explore_q[2, trial_index]),
                "representative_strategy_exploit": float(
                    paths["strategy_exploit"][representative, trial_index]
                ),
                "representative_strategy_local_explore": float(
                    paths["strategy_local_explore"][representative, trial_index]
                ),
                "representative_strategy_global_explore": float(
                    paths["strategy_global_explore"][representative, trial_index]
                ),
                "representative_swap_event": float(
                    paths["swap_event"][representative, trial_index]
                ),
                "posterior_swap_event_probability": float(
                    profile["posterior_swap_event_probability"][trial_index]
                ),
                "representative_transition_rate": float(
                    paths["transition_rate"][representative, trial_index]
                ),
                "representative_search_range": float(
                    paths["search_range"][representative, trial_index]
                ),
                "representative_failure_pressure": float(
                    paths["failure_pressure"][representative, trial_index]
                ),
                "representative_mastery_evidence": float(
                    paths["mastery_evidence"][representative, trial_index]
                ),
            }
        )
    return rows


def _ancestral_path_rows(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    paths = profile["paths"]
    rows: list[dict[str, Any]] = []
    representative = int(profile["representative_index"])
    n_paths = int(np.asarray(profile["weights"]).size)
    n_trials = int(profile["n_trials"])
    for path_index in range(n_paths):
        seed = int(profile["seeds"][path_index])
        terminal = int(profile["terminal_particles"][path_index])
        for trial_index in range(n_trials):
            rows.append(
                {
                    "subject_id": int(profile["subject_id"]),
                    "path_index": int(path_index),
                    "filter_seed": seed,
                    "terminal_particle": terminal,
                    "path_weight": float(profile["weights"][path_index]),
                    "is_representative": bool(path_index == representative),
                    "trial": int(trial_index + 1),
                    "particle_index": int(
                        profile["particle_indices"][path_index, trial_index]
                    ),
                    **{
                        field: float(paths[field][path_index, trial_index])
                        for field in profile["path_fields"]
                    },
                }
            )
    return rows


def _ancestral_summary_row(profile: Mapping[str, Any]) -> dict[str, Any]:
    rolling_q = np.asarray(profile["rolling_quantiles"], dtype=float)
    return {
        "subject_id": int(profile["subject_id"]),
        "n_trials": int(profile["n_trials"]),
        "n_paths": int(np.asarray(profile["weights"]).size),
        "effective_path_count": float(profile["effective_path_count"]),
        "mean_start_ancestor_count": float(profile["mean_start_ancestor_count"]),
        "mean_resampling_count": float(profile["mean_resampling_count"]),
        "representative_filter_seed": int(profile["representative_seed"]),
        "representative_terminal_particle": int(
            profile["representative_terminal_particle"]
        ),
        "representative_path_weight": float(profile["representative_weight"]),
        "representative_medoid_score": float(
            profile["representative_medoid_score"]
        ),
        "marginal_curve_mae_to_subject": float(
            np.mean(
                np.abs(profile["marginal_rolling"] - profile["observed_rolling"])
            )
        ),
        "representative_curve_mae_to_subject": float(
            np.mean(
                np.abs(
                    profile["representative_rolling"]
                    - profile["observed_rolling"]
                )
            )
        ),
        "representative_curve_mae_from_marginal": float(
            np.mean(
                np.abs(
                    profile["representative_rolling"]
                    - profile["marginal_rolling"]
                )
            )
        ),
        "mean_rolling_correct_q80_width": float(
            np.mean(rolling_q[2] - rolling_q[0])
        ),
        "mean_exploration_q80_width": float(
            np.mean(
                profile["exploration_quantiles"][2]
                - profile["exploration_quantiles"][0]
            )
        ),
        "mean_posterior_swap_event_probability": float(
            np.mean(profile["posterior_swap_event_probability"])
        ),
        "fraction_trials_with_path_event_disagreement": float(
            np.mean(
                np.ptp(np.asarray(profile["paths"]["swap_event"]), axis=0) > 0.0
            )
        ),
    }


def _save_figure(fig: plt.Figure, output_path: Path) -> Path:
    output_path = output_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _save_error_transmission_figure(
    phase_df: pd.DataFrame,
    base_path: Path,
) -> Path:
    """Show where prediction quality changes and how the controller responds."""

    required = {
        "subject_id",
        "performance_phase",
        "correct_predicting_rule_available",
        "belief_mass_on_correct_predicting_rules",
        "belief_only_correct_probability",
        "sharpened_correct_probability",
        "strategy_confidence_correct_probability",
        "final_correct_probability",
        "observed_correct",
        "strategy_explore",
        "failure_pressure",
        "mastery_evidence",
    }
    missing = sorted(required.difference(phase_df.columns))
    if missing:
        raise ValueError(
            "Error-transmission phase summary is missing columns: "
            + ", ".join(missing)
        )

    subjects = sorted(int(value) for value in phase_df["subject_id"].unique())
    if not subjects:
        raise ValueError("Error-transmission figure requires at least one subject.")
    has_persistent_execution = (
        "persistent_execution_correct_probability" in phase_df
        and "persistent_execution_enabled" in phase_df
        and bool(phase_df["persistent_execution_enabled"].astype(bool).any())
    )
    has_execution_switch = (
        "execution_switch_probability" in phase_df and has_persistent_execution
    )
    stage_columns = (
        "correct_predicting_rule_available",
        "belief_mass_on_correct_predicting_rules",
        "belief_only_correct_probability",
        "sharpened_correct_probability",
        "strategy_confidence_correct_probability",
        *(
            ("persistent_execution_correct_probability",)
            if has_persistent_execution
            else ()
        ),
        "final_correct_probability",
        "observed_correct",
    )
    stage_labels = (
        "Correct rule\navailable",
        "Belief on\ncorrect rules",
        "Belief-only\np(correct)",
        "After hypothesis\nsharpening",
        "After strategy\nconfidence",
        *(("After persistent\nexecution",) if has_persistent_execution else ()),
        "Final model\np(correct)",
        "Observed\naccuracy",
    )
    controller_columns = (
        "strategy_explore",
        "failure_pressure",
        "mastery_evidence",
        *(("execution_switch_probability",) if has_execution_switch else ()),
    )
    controller_labels = (
        "Exploration tendency",
        "Failure pressure",
        "Mastery evidence",
        *(("Execution switch probability",) if has_execution_switch else ()),
    )
    controller_colors = ("#7A5195", "#D55E00", "#009E73") + (
        ("#CC6677",) if has_execution_switch else ()
    )
    phase_x = np.arange(len(PHASE_ORDER), dtype=float)
    phase_tick_labels = tuple(PHASE_LABELS[phase] for phase in PHASE_ORDER)

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(
            len(subjects),
            2,
            figsize=(7.2, max(3.0, 1.8 * len(subjects) + 0.9)),
            squeeze=False,
            sharey=True,
            gridspec_kw={"width_ratios": (1.25, 1.0)},
        )
        for row_index, subject_id in enumerate(subjects):
            subject = phase_df[phase_df["subject_id"] == subject_id]
            by_phase = {
                str(row["performance_phase"]): row
                for _, row in subject.iterrows()
            }

            pipeline_ax = axes[row_index, 0]
            for phase in PHASE_ORDER:
                row = by_phase.get(phase)
                if row is None:
                    continue
                values = np.asarray([row[column] for column in stage_columns], dtype=float)
                pipeline_ax.plot(
                    np.arange(len(stage_columns)),
                    values,
                    color=PHASE_COLORS[phase],
                    marker="o",
                    markersize=3.4,
                    linewidth=1.25,
                    label=PHASE_LABELS[phase],
                )
            pipeline_ax.set_xticks(np.arange(len(stage_columns)))
            if row_index == len(subjects) - 1:
                pipeline_ax.set_xticklabels(
                    stage_labels,
                    fontsize=5.4,
                    rotation=28,
                    ha="right",
                    rotation_mode="anchor",
                )
            else:
                pipeline_ax.set_xticklabels([])
            pipeline_ax.set_ylabel(f"Subject {subject_id}\nProbability")
            pipeline_ax.set_ylim(0.0, 1.02)
            pipeline_ax.grid(axis="y", alpha=0.16)
            if row_index == 0:
                pipeline_ax.set_title(
                    "A  Prediction transmitted through model layers",
                    loc="left",
                    fontsize=8,
                )
                pipeline_ax.legend(
                    loc="lower left",
                    fontsize=5.7,
                    ncol=2,
                    handlelength=1.6,
                )

            controller_ax = axes[row_index, 1]
            for column, label, color in zip(
                controller_columns,
                controller_labels,
                controller_colors,
            ):
                values = np.full(len(PHASE_ORDER), np.nan, dtype=float)
                for phase_index, phase in enumerate(PHASE_ORDER):
                    row = by_phase.get(phase)
                    if row is not None:
                        values[phase_index] = float(row[column])
                controller_ax.plot(
                    phase_x,
                    values,
                    color=color,
                    marker="o",
                    markersize=3.4,
                    linewidth=1.25,
                    label=label,
                )
            controller_ax.set_xticks(phase_x)
            if row_index == len(subjects) - 1:
                controller_ax.set_xticklabels(
                    phase_tick_labels,
                    fontsize=5.6,
                    rotation=18,
                    ha="right",
                    rotation_mode="anchor",
                )
            else:
                controller_ax.set_xticklabels([])
            controller_ax.set_ylim(0.0, 1.02)
            controller_ax.grid(axis="y", alpha=0.16)
            if row_index == 0:
                controller_ax.set_title(
                    "B  Controller response across performance phases",
                    loc="left",
                    fontsize=8,
                )
                controller_ax.legend(
                    loc="upper right",
                    fontsize=5.7,
                    handlelength=1.6,
                )

        fig.suptitle(
            "Dynamic-continuous strategy-to-choice transmission",
            fontsize=9,
            y=0.995,
        )
        fig.text(
            0.5,
            0.006,
            (
                "Phases use only preceding rolling accuracy. Strategy confidence is a "
                "pre-choice readout stage (it overlaps hypothesis sharpening when gain=0). "
                "Persistent execution is the rule carried forward for overt choice. "
                "‘Correct rule’ is trial-local, not a claim of global rule truth."
            ),
            ha="center",
            va="bottom",
            fontsize=5.8,
            color="#444444",
        )
        fig.tight_layout(rect=(0.0, 0.065, 1.0, 0.96), h_pad=0.7, w_pad=1.0)
        return _save_figure(fig, base_path)


def _save_subject_curves(
    trial_df: pd.DataFrame,
    profiles: Mapping[int, Mapping[str, Any]],
    base_path: Path,
) -> Path:
    subjects = sorted(profiles)
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(
            len(subjects),
            1,
            figsize=(7.2, 2.0 * len(subjects) + 1.0),
            sharey=True,
            squeeze=False,
        )
        legend_items: dict[str, Any] = {}
        for ax, subject_id in zip(axes[:, 0], subjects):
            profile = profiles[subject_id]
            subject = trial_df[trial_df["subject_id"] == subject_id]
            x = np.arange(1, int(profile["n_trials"]) + 1)
            window = int(profile["window_size"])
            q10_curve = np.convolve(
                profile["particle_q10"][1:],
                np.ones(window) / float(window),
                mode="valid",
            )
            q90_curve = np.convolve(
                profile["particle_q90"][1:],
                np.ones(window) / float(window),
                mode="valid",
            )
            curve_x = np.arange(window + 1, int(profile["n_trials"]) + 1)
            envelope = ax.fill_between(
                curve_x,
                q10_curve,
                q90_curve,
                color="#9DB9D8",
                alpha=0.22,
                linewidth=0,
                label="Current particle 10–90% range",
            )
            legend_items.setdefault("Current particle 10–90% range", envelope)
            observed = ax.plot(
                curve_x,
                profile["observed_curve"],
                color="#111111",
                linewidth=1.6,
                label="Subject accuracy",
                zorder=7,
            )[0]
            legend_items.setdefault("Subject accuracy", observed)
            for readout in READOUT_ORDER:
                rows = subject[subject["readout"] == readout]
                finite_curve = np.isfinite(
                    rows["rolling_expected_accuracy"].to_numpy(dtype=float)
                )
                rows = rows.loc[finite_curve]
                line = ax.plot(
                    rows["trial"],
                    rows["rolling_expected_accuracy"],
                    color=READOUT_COLORS[readout],
                    linewidth=1.5 if readout == CURRENT else 1.1,
                    linestyle="--" if readout == EXPLORATION_LAPSE else "-",
                    label=READOUT_SHORT_LABELS[readout],
                    zorder=6 if readout == CURRENT else 5,
                )[0]
                legend_items.setdefault(READOUT_SHORT_LABELS[readout], line)
            ax.set(
                title=f"Subject {subject_id}",
                xlabel="Trial",
                ylabel="Rolling accuracy",
                xlim=(1, int(x[-1])),
                ylim=(0.0, 1.0),
            )
            ax.grid(axis="y", alpha=0.16)
        fig.suptitle(
            "Choice-transmission audit: identical PF states, alternative readouts",
            fontsize=9,
            y=0.995,
        )
        fig.legend(
            list(legend_items.values()),
            list(legend_items.keys()),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),
            ncol=3,
            fontsize=6.5,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
        return _save_figure(fig, base_path)


def _save_summary_figure(
    summary_df: pd.DataFrame,
    event_df: pd.DataFrame,
    base_path: Path,
) -> Path:
    subjects = sorted(int(value) for value in summary_df["subject_id"].unique())
    x = np.arange(len(subjects), dtype=float)
    offsets = np.linspace(-0.21, 0.21, len(READOUT_ORDER))
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))

        ax = axes[0, 0]
        for offset, readout in zip(offsets, READOUT_ORDER):
            rows = summary_df[summary_df["readout"] == readout].set_index(
                "subject_id"
            )
            values = [rows.loc[sid, "expected_curve_mae_to_subject"] for sid in subjects]
            ax.scatter(
                x + offset,
                values,
                s=25,
                color=READOUT_COLORS[readout],
                label=READOUT_SHORT_LABELS[readout],
            )
        ax.set(
            title="a  Fit to subject accuracy curve",
            ylabel="Expected-curve MAE",
            xticks=x,
            xticklabels=subjects,
        )
        ax.grid(axis="y", alpha=0.16)

        ax = axes[0, 1]
        alternatives = READOUT_ORDER[1:]
        alt_offsets = np.linspace(-0.16, 0.16, len(alternatives))
        for offset, readout in zip(alt_offsets, alternatives):
            rows = summary_df[summary_df["readout"] == readout].set_index(
                "subject_id"
            )
            values = [rows.loc[sid, "paired_curve_mae_from_current"] for sid in subjects]
            ax.scatter(
                x + offset,
                values,
                s=25,
                color=READOUT_COLORS[readout],
            )
        current_rows = summary_df[summary_df["readout"] == CURRENT].set_index(
            "subject_id"
        )
        particle_width = [
            current_rows.loc[sid, "mean_particle_q80_width"] for sid in subjects
        ]
        ax.scatter(
            x,
            particle_width,
            marker="D",
            facecolor="none",
            edgecolor="#333333",
            s=32,
            label="Particle 10–90% width",
        )
        ax.set(
            title="b  Averaging-layer effect size",
            ylabel="Probability / curve difference",
            xticks=x,
            xticklabels=subjects,
        )
        ax.grid(axis="y", alpha=0.16)

        ax = axes[1, 0]
        for offset, readout in zip(offsets, READOUT_ORDER):
            rows = summary_df[summary_df["readout"] == readout].set_index(
                "subject_id"
            )
            values = [rows.loc[sid, "low_entry_probability_delta"] for sid in subjects]
            ax.scatter(
                x + offset,
                values,
                s=25,
                color=READOUT_COLORS[readout],
            )
        ax.axhline(0.0, color="#777777", linewidth=0.8, linestyle=":")
        ax.set(
            title="c  Response at low-performance entry",
            ylabel="Post-minus-pre correct probability",
            xlabel="Subject",
            xticks=x,
            xticklabels=subjects,
        )
        ax.grid(axis="y", alpha=0.16)

        ax = axes[1, 1]
        low = event_df[event_df["event_type"] == "low_performance_entry"]
        grouped = (
            low.groupby(["readout", "relative_trial"], as_index=False)[
                "expected_correct_probability"
            ]
            .mean()
        )
        for readout in READOUT_ORDER:
            rows = grouped[grouped["readout"] == readout]
            ax.plot(
                rows["relative_trial"],
                rows["expected_correct_probability"],
                color=READOUT_COLORS[readout],
                linewidth=1.5 if readout == CURRENT else 1.1,
                linestyle="--" if readout == EXPLORATION_LAPSE else "-",
            )
        ax.axvline(0.0, color="#777777", linewidth=0.8, linestyle=":")
        ax.set(
            title="d  Event-aligned low-performance entry",
            xlabel="Trials relative to entry",
            ylabel="Predicted correct probability",
            ylim=(0.35, 0.75),
        )
        ax.grid(axis="y", alpha=0.16)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        particle_handles, particle_labels = axes[0, 1].get_legend_handles_labels()
        fig.legend(
            handles + particle_handles,
            labels + particle_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.975),
            ncol=3,
            fontsize=6.5,
        )
        fig.suptitle(
            "Where does dynamic strategy lose behavioral impact?",
            fontsize=9,
            y=0.998,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
        return _save_figure(fig, base_path)


def _save_gain_screen_figure(
    summary_df: pd.DataFrame,
    base_path: Path,
) -> Path:
    """Compare gain candidates overall and in the causally defined valleys."""

    required = {
        "subject_id",
        "strategy_confidence_gain",
        "stratum",
        "choice_nll",
        "choice_nll_improvement_from_gain0",
        "deep_valley_threshold",
        "n_common_seeds",
    }
    missing = sorted(required.difference(summary_df.columns))
    if missing:
        raise ValueError(
            "Strategy-confidence gain summary is missing columns: "
            + ", ".join(missing)
        )
    gains = np.sort(
        summary_df["strategy_confidence_gain"].dropna().unique().astype(float)
    )
    overall = summary_df[summary_df["stratum"] == "overall"]
    subjects = sorted(int(value) for value in overall["subject_id"].unique())
    macro_overall = (
        overall.groupby("strategy_confidence_gain", as_index=False)["choice_nll"]
        .mean()
        .sort_values("strategy_confidence_gain")
    )
    best_row = macro_overall.loc[macro_overall["choice_nll"].idxmin()]
    best_gain = float(best_row["strategy_confidence_gain"])
    threshold = float(summary_df["deep_valley_threshold"].dropna().iloc[0])
    n_seeds = int(summary_df["n_common_seeds"].dropna().max())

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))

        ax = axes[0]
        for subject_id in subjects:
            subject = overall[overall["subject_id"] == subject_id].sort_values(
                "strategy_confidence_gain"
            )
            ax.plot(
                subject["strategy_confidence_gain"],
                subject["choice_nll"],
                color="#B7B7B7",
                linewidth=0.9,
                marker="o",
                markersize=3.0,
                label=f"Subject {subject_id}",
            )
        ax.plot(
            macro_overall["strategy_confidence_gain"],
            macro_overall["choice_nll"],
            color="#0072B2",
            linewidth=2.0,
            marker="o",
            markersize=4.0,
            label="Subject-macro mean",
            zorder=5,
        )
        ax.scatter(
            [best_gain],
            [float(best_row["choice_nll"])],
            marker="*",
            s=65,
            color="#D55E00",
            edgecolor="white",
            linewidth=0.4,
            zorder=7,
            label=f"Best screened gain = {best_gain:g}",
        )
        ax.set(
            title="a  Overall observed-choice fit",
            xlabel="Strategy-confidence gain",
            ylabel="Choice NLL (lower is better)",
            xticks=gains,
        )
        ax.grid(axis="y", alpha=0.16)
        ax.legend(fontsize=6.0, loc="best")

        ax = axes[1]
        stratum_styles = {
            "low": ("#E69F00", "Low performance", "-"),
            "middle_recovery": ("#7A5195", "Learning / recovery", "-"),
            "mastery": ("#0072B2", "Mastery", "-"),
            "deep_valley": (
                "#D55E00",
                f"Deep valley (prior accuracy <= {threshold:.2f})",
                "--",
            ),
        }
        for stratum, (color, label, linestyle) in stratum_styles.items():
            selected = summary_df[summary_df["stratum"] == stratum]
            if selected.empty:
                continue
            macro = (
                selected.groupby("strategy_confidence_gain", as_index=False)[
                    "choice_nll_improvement_from_gain0"
                ]
                .mean()
                .sort_values("strategy_confidence_gain")
            )
            ax.plot(
                macro["strategy_confidence_gain"],
                macro["choice_nll_improvement_from_gain0"],
                color=color,
                linewidth=1.6,
                marker="o",
                markersize=3.5,
                linestyle=linestyle,
                label=label,
            )
        ax.axhline(0.0, color="#777777", linewidth=0.8, linestyle=":")
        ax.set(
            title="b  Where the gain helps",
            xlabel="Strategy-confidence gain",
            ylabel="NLL improvement from gain = 0",
            xticks=gains,
        )
        ax.grid(axis="y", alpha=0.16)
        ax.legend(fontsize=6.0, loc="best")

        fig.suptitle(
            "Common-seed strategy-confidence gain screen",
            fontsize=9,
            y=0.995,
        )
        fig.text(
            0.5,
            0.01,
            (
                f"Same {n_seeds} PF seeds at every gain. Deep valleys use only "
                "accuracy from the preceding window; positive values in b indicate "
                "better observed-choice prediction than the disabled baseline."
            ),
            ha="center",
            va="bottom",
            fontsize=5.8,
            color="#444444",
        )
        fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.95), w_pad=1.3)
        return _save_figure(fig, base_path)


def _save_ancestral_figure(
    profiles: Mapping[int, Mapping[str, Any]],
    base_path: Path,
) -> Path:
    subjects = sorted(profiles)
    panel_letters = iter("abcdefghijklmnopqrstuvwxyz")
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(
            len(subjects),
            2,
            figsize=(7.2, 2.0 * len(subjects) + 1.45),
            squeeze=False,
        )
        legend_items: dict[str, Any] = {}
        for row_index, subject_id in enumerate(subjects):
            profile = profiles[subject_id]
            representative = int(profile["representative_index"])
            trial = np.arange(1, int(profile["n_trials"]) + 1)
            rolling_trial = np.asarray(profile["rolling_trial"], dtype=int)

            ax = axes[row_index, 0]
            rolling_q = np.asarray(profile["rolling_quantiles"], dtype=float)
            band = ax.fill_between(
                rolling_trial,
                rolling_q[0],
                rolling_q[2],
                color="#E69F00",
                alpha=0.18,
                linewidth=0,
                label="Ancestral paths 10–90%",
            )
            legend_items.setdefault("Ancestral paths 10–90%", band)
            for values, label, color, width in (
                (profile["observed_rolling"], "Subject accuracy", "#111111", 1.5),
                (profile["marginal_rolling"], "PF marginal", "#0072B2", 1.4),
                (
                    profile["representative_rolling"],
                    "Representative ancestral path",
                    "#D55E00",
                    1.2,
                ),
            ):
                line = ax.plot(
                    rolling_trial,
                    values,
                    color=color,
                    linewidth=width,
                    label=label,
                )[0]
                legend_items.setdefault(label, line)
            ax.set(
                title=f"{next(panel_letters)}  Subject {subject_id}: behavioral expression",
                xlabel="Trial",
                ylabel="Rolling accuracy",
                xlim=(1, int(profile["n_trials"])),
                ylim=(0.0, 1.0),
            )
            ax.grid(axis="y", alpha=0.16)

            ax = axes[row_index, 1]
            explore_q = np.asarray(profile["exploration_quantiles"], dtype=float)
            if float(np.max(explore_q[2] - explore_q[0])) > 1e-8:
                band = ax.fill_between(
                    trial,
                    explore_q[0],
                    explore_q[2],
                    color="#999999",
                    alpha=0.18,
                    linewidth=0,
                    label="Exploration paths 10–90%",
                )
                legend_items.setdefault("Exploration paths 10–90%", band)
            paths = profile["paths"]
            for field, label, color in (
                ("strategy_exploit", "Exploit", "#0072B2"),
                ("strategy_local_explore", "Local explore", "#E69F00"),
                ("strategy_global_explore", "Global explore", "#CC6677"),
            ):
                line = ax.plot(
                    trial,
                    paths[field][representative],
                    color=color,
                    linewidth=1.15,
                    label=label,
                )[0]
                legend_items.setdefault(label, line)
            posterior_events = ax.plot(
                trial,
                profile["posterior_swap_event_probability"],
                color="#444444",
                linewidth=0.9,
                linestyle=":",
                alpha=0.85,
                label="Posterior replacement frequency",
            )[0]
            legend_items.setdefault(
                "Posterior replacement frequency", posterior_events
            )
            event_trials = trial[paths["swap_event"][representative] >= 0.5]
            if event_trials.size:
                ticks = ax.scatter(
                    event_trials,
                    np.full(event_trials.size, 1.01),
                    marker="|",
                    s=20,
                    color="#222222",
                    clip_on=False,
                    label="Realized replacement",
                )
                legend_items.setdefault("Realized replacement", ticks)
            ax.set(
                title=f"{next(panel_letters)}  Representative dynamic strategy",
                xlabel="Trial",
                ylabel="Strategy tendency",
                xlim=(1, int(profile["n_trials"])),
                ylim=(0.0, 1.04),
            )
            ax.grid(axis="y", alpha=0.16)

        fig.suptitle(
            "Coherent posterior particle genealogies",
            fontsize=9,
            y=0.997,
        )
        fig.legend(
            list(legend_items.values()),
            list(legend_items.keys()),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.962),
            ncol=3,
            fontsize=6.2,
        )
        fig.text(
            0.5,
            0.018,
            (
                "Representative = posterior-weighted latent-path medoid "
                "(strategy + choice probability); orange band = complete-path 10–90%, "
                "not uncertainty of the PF mean."
            ),
            ha="center",
            va="bottom",
            fontsize=6.2,
            color="#444444",
        )
        fig.text(
            0.5,
            0.005,
            (
                "Zero-width controller bands mean shared tendencies; dotted line and ticks "
                "show posterior and representative realized replacements."
            ),
            ha="center",
            va="bottom",
            fontsize=6.2,
            color="#444444",
        )
        fig.tight_layout(rect=(0.0, 0.052, 1.0, 0.86))
        return _save_figure(fig, base_path)


def run_particle_filter_choice_transmission_audit(
    results: Mapping[int, Mapping[str, Any]],
    *,
    simulation_config_path: str | Path,
    output_dir: str | Path,
    subjects: Sequence[int] | None = None,
    common_seeds: Sequence[int] = (20260821, 20260822, 20260823, 20260824),
    n_jobs: int = 1,
    particle_count: int = 32,
    low_accuracy_threshold: float = 0.60,
    event_relative_start: int = -8,
    event_relative_end: int = 16,
    strategy_confidence_gain_values: Sequence[float] | None = None,
    deep_valley_threshold: float = 0.40,
) -> dict[str, Any]:
    """Run common-state alternative-readout diagnostics and save source data."""

    config_path = Path(simulation_config_path).resolve()
    cfg = load_yaml(config_path)
    configured_subjects = resolve_subjects(subjects, None, cfg)
    selected_subjects = [sid for sid in configured_subjects if sid in results]
    if not selected_subjects:
        raise ValueError("No choice-audit subjects overlap the loaded results.")
    seeds = tuple(int(seed) for seed in common_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Choice-audit common seeds must be non-empty and unique.")
    if int(particle_count) < 2:
        raise ValueError("Choice-audit particle_count must be at least two.")
    gain_values = (
        None
        if strategy_confidence_gain_values is None
        else _strategy_confidence_gain_values(strategy_confidence_gain_values)
    )
    if gain_values is not None and not (
        0.0 <= float(deep_valley_threshold) <= 0.5
    ):
        raise ValueError(
            "Choice-audit deep_valley_threshold must lie in [0, 0.5]."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles: dict[int, Mapping[str, Any]] = {}
    ancestral_profiles: dict[int, Mapping[str, Any]] = {}
    trial_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    ancestral_trial_rows: list[dict[str, Any]] = []
    ancestral_path_rows: list[dict[str, Any]] = []
    ancestral_summary_rows: list[dict[str, Any]] = []
    error_trial_rows: list[dict[str, Any]] = []
    error_phase_rows: list[dict[str, Any]] = []
    gain_screen_trial_rows: list[dict[str, Any]] = []
    gain_screen_summary_rows: list[dict[str, Any]] = []

    for subject_id in selected_subjects:
        info = results[int(subject_id)]
        if str(info.get("state_distribution_kind", "")) != "particle_marginal":
            raise ValueError(f"Subject {subject_id} is not a particle-marginal result.")
        LOGGER.info("Choice-transmission audit subject %s", subject_id)
        runtime = _subject_runtime(cfg, config_path, subject_id, selected_subjects)
        engine_config = deepcopy(runtime["engine_config"])
        inference = engine_config.setdefault("inference", {})
        inference["particle_count"] = int(particle_count)
        inference["choice_transmission_audit"] = True
        runs = _run_variant(
            runtime,
            engine_config,
            subject_id=int(subject_id),
            seeds=seeds,
            n_jobs=int(n_jobs),
        )
        profile = _summarize_subject_runs(
            runs,
            subject_id=int(subject_id),
            window_size=int(runtime["window_size"]),
            low_accuracy_threshold=float(low_accuracy_threshold),
        )
        if gain_values is not None:
            readout_config = engine_config.get("choice_readout") or {}
            readout_kwargs = (
                readout_config.get("kwargs") or {}
                if isinstance(readout_config, Mapping)
                else {}
            )
            configured_gain = float(
                readout_kwargs.get("strategy_confidence_gain", 0.0)
            )
            for gain in gain_values:
                if np.isclose(float(gain), configured_gain, atol=0.0, rtol=0.0):
                    gain_profile = profile
                else:
                    LOGGER.info(
                        "Choice-transmission gain screen subject %s gain %.6g",
                        subject_id,
                        gain,
                    )
                    gain_engine_config = _engine_with_strategy_confidence_gain(
                        engine_config,
                        float(gain),
                    )
                    gain_runtime = dict(runtime)
                    gain_fixed_hyperparams = dict(runtime["fixed_hyperparams"])
                    gain_fixed_hyperparams[
                        STRATEGY_CONFIDENCE_GAIN_PATH
                    ] = float(gain)
                    gain_runtime["fixed_hyperparams"] = gain_fixed_hyperparams
                    gain_runs = _run_variant(
                        gain_runtime,
                        gain_engine_config,
                        subject_id=int(subject_id),
                        seeds=seeds,
                        n_jobs=int(n_jobs),
                    )
                    gain_profile = _summarize_subject_runs(
                        gain_runs,
                        subject_id=int(subject_id),
                        window_size=int(runtime["window_size"]),
                        low_accuracy_threshold=float(low_accuracy_threshold),
                    )
                gain_screen_trial_rows.extend(
                    _gain_screen_trial_rows(
                        gain_profile,
                        gain=float(gain),
                        deep_valley_threshold=float(deep_valley_threshold),
                    )
                )
                gain_screen_summary_rows.extend(
                    _gain_screen_summary_rows(
                        gain_profile,
                        gain=float(gain),
                        deep_valley_threshold=float(deep_valley_threshold),
                    )
                )
        profiles[int(subject_id)] = profile
        trial_rows.extend(_trial_rows(profile))
        summary_rows.extend(_summary_rows(profile))
        error_trial_rows.extend(_error_transmission_trial_rows(profile))
        error_phase_rows.extend(_error_transmission_phase_rows(profile))
        event_rows.extend(
            _event_rows(
                profile,
                relative_start=int(event_relative_start),
                relative_end=int(event_relative_end),
            )
        )
        ancestral_profile = _summarize_ancestral_runs(
            runs,
            subject_id=int(subject_id),
            window_size=int(runtime["window_size"]),
        )
        ancestral_profiles[int(subject_id)] = ancestral_profile
        ancestral_trial_rows.extend(_ancestral_trial_rows(ancestral_profile))
        ancestral_path_rows.extend(_ancestral_path_rows(ancestral_profile))
        ancestral_summary_rows.append(_ancestral_summary_row(ancestral_profile))

    trial_df = pd.DataFrame(trial_rows)
    summary_df = pd.DataFrame(summary_rows)
    event_df = pd.DataFrame(event_rows)
    ancestral_trial_df = pd.DataFrame(ancestral_trial_rows)
    ancestral_path_df = pd.DataFrame(ancestral_path_rows)
    ancestral_summary_df = pd.DataFrame(ancestral_summary_rows)
    error_trial_df = pd.DataFrame(error_trial_rows)
    error_phase_df = pd.DataFrame(error_phase_rows)
    gain_trial_df = pd.DataFrame(gain_screen_trial_rows)
    gain_summary_df = pd.DataFrame(gain_screen_summary_rows)
    if gain_values is not None:
        seed_text = ",".join(str(seed) for seed in seeds)
        gain_text = ",".join(f"{gain:g}" for gain in gain_values)
        for frame in (gain_trial_df, gain_summary_df):
            frame["common_pf_seeds"] = seed_text
            frame["particle_count"] = int(particle_count)
            frame["screened_gain_values"] = gain_text
        baseline = gain_summary_df[
            gain_summary_df["strategy_confidence_gain"] == 0.0
        ][["subject_id", "stratum", "choice_nll"]].rename(
            columns={"choice_nll": "gain0_choice_nll"}
        )
        gain_summary_df = gain_summary_df.merge(
            baseline,
            on=["subject_id", "stratum"],
            how="left",
            validate="many_to_one",
        )
        gain_summary_df["choice_nll_improvement_from_gain0"] = (
            gain_summary_df["gain0_choice_nll"] - gain_summary_df["choice_nll"]
        )
        macro_overall = (
            gain_summary_df[gain_summary_df["stratum"] == "overall"]
            .groupby("strategy_confidence_gain")["choice_nll"]
            .mean()
        )
        best_gain = float(macro_overall.idxmin())
        gain_summary_df["macro_overall_choice_nll"] = gain_summary_df[
            "strategy_confidence_gain"
        ].map(macro_overall)
        gain_summary_df["selected_by_macro_overall_nll"] = np.isclose(
            gain_summary_df["strategy_confidence_gain"],
            best_gain,
            atol=0.0,
            rtol=0.0,
        )
    event_summary_df = (
        event_df.groupby(
            ["event_type", "readout", "readout_label", "relative_trial"],
            as_index=False,
        )
        .agg(
            n_subjects=("subject_id", "nunique"),
            n_events=("event_number", "count"),
            expected_correct_probability_mean=(
                "expected_correct_probability",
                "mean",
            ),
            expected_correct_probability_sem=(
                "expected_correct_probability",
                "sem",
            ),
            strategy_explore_mean=("strategy_explore", "mean"),
            causal_recent_accuracy_mean=("causal_recent_accuracy", "mean"),
        )
    )

    trial_csv = output_dir / "choice_transmission_trial_data.csv"
    summary_csv = output_dir / "choice_transmission_summary.csv"
    event_csv = output_dir / "choice_transmission_event_data.csv"
    event_summary_csv = output_dir / "choice_transmission_event_summary.csv"
    ancestral_trial_csv = output_dir / "ancestral_trajectory_trial_data.csv"
    ancestral_path_csv = output_dir / "ancestral_trajectory_paths.csv"
    ancestral_summary_csv = output_dir / "ancestral_trajectory_summary.csv"
    error_trial_csv = output_dir / "error_transmission_trial_data.csv"
    error_phase_csv = output_dir / "error_transmission_phase_summary.csv"
    trial_df.to_csv(trial_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    event_df.to_csv(event_csv, index=False)
    event_summary_df.to_csv(event_summary_csv, index=False)
    ancestral_trial_df.to_csv(ancestral_trial_csv, index=False)
    ancestral_path_df.to_csv(ancestral_path_csv, index=False)
    ancestral_summary_df.to_csv(ancestral_summary_csv, index=False)
    error_trial_df.to_csv(error_trial_csv, index=False)
    error_phase_df.to_csv(error_phase_csv, index=False)
    gain_screen_outputs: list[Path] = []
    if gain_values is not None:
        gain_trial_csv = (
            output_dir / "strategy_confidence_gain_screen_trial_data.csv"
        )
        gain_summary_csv = (
            output_dir / "strategy_confidence_gain_screen_summary.csv"
        )
        gain_trial_df.to_csv(gain_trial_csv, index=False)
        gain_summary_df.to_csv(gain_summary_csv, index=False)
        gain_figure = _save_gain_screen_figure(
            gain_summary_df,
            output_dir / "strategy_confidence_gain_screen",
        )
        gain_screen_outputs.extend(
            [gain_trial_csv, gain_summary_csv, gain_figure]
        )
    figure_outputs = [
        _save_subject_curves(
            trial_df,
            profiles,
            output_dir / "choice_transmission_curves",
        ),
        _save_summary_figure(
            summary_df,
            event_df,
            output_dir / "choice_transmission_summary",
        ),
        _save_ancestral_figure(
            ancestral_profiles,
            output_dir / "ancestral_strategy_trajectories",
        ),
        _save_error_transmission_figure(
            error_phase_df,
            output_dir / "error_transmission_layers",
        ),
    ]
    return {
        "subjects": selected_subjects,
        "common_seeds": list(seeds),
        "readouts": list(READOUT_ORDER),
        "outputs": [
            trial_csv,
            summary_csv,
            event_csv,
            event_summary_csv,
            ancestral_trial_csv,
            ancestral_path_csv,
            ancestral_summary_csv,
            error_trial_csv,
            error_phase_csv,
            *gain_screen_outputs,
            *figure_outputs,
        ],
        "summary": summary_df,
        "event_summary": event_summary_df,
        "ancestral_summary": ancestral_summary_df,
        "error_transmission_phase_summary": error_phase_df,
        "strategy_confidence_gain_screen_summary": gain_summary_df,
        "selected_strategy_confidence_gain": (
            best_gain if gain_values is not None else None
        ),
    }


__all__ = [
    "ADAPTIVE_SHARPENING",
    "CURRENT",
    "EXPLORATION_LAPSE",
    "HYPOTHESIS_MAP",
    "READOUT_ORDER",
    "_engine_with_strategy_confidence_gain",
    "_gain_screen_summary_rows",
    "_performance_phase",
    "_save_error_transmission_figure",
    "_save_gain_screen_figure",
    "_strategy_confidence_gain_values",
    "run_particle_filter_choice_transmission_audit",
]

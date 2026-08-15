"""Trial-aligned accuracy curves and standard prediction metric bundles."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .behavior import exponential_smooth_curve
from .numeric import nanmean_or_nan


def family_correct(
    categories: Sequence[int] | np.ndarray,
    choices: Sequence[int] | np.ndarray,
    n_categories: int,
) -> np.ndarray:
    """Return trial correctness under the established two-family grouping."""
    category_array = np.asarray(categories, dtype=int)
    choice_array = np.asarray(choices, dtype=int)
    if int(n_categories) >= 4:
        category_family = np.where(np.isin(category_array, [1, 2]), 0, 1)
        choice_family = np.where(np.isin(choice_array, [1, 2]), 0, 1)
        return (category_family == choice_family).astype(float)
    return (category_array == choice_array).astype(float)


def family_indices(category: int, n_categories: int) -> np.ndarray:
    """Return zero-based category indices belonging to one category family."""
    category_index = int(category) - 1
    if int(n_categories) >= 4:
        return (
            np.asarray([0, 1], dtype=int)
            if category_index in (0, 1)
            else np.asarray([2, 3], dtype=int)
        )
    return np.asarray([category_index], dtype=int)


def target_majority_indices(target_probabilities: Any) -> np.ndarray | None:
    """Return the unique target-probability maximum, with ``-1`` for ties."""
    if target_probabilities is None:
        return None
    probabilities = np.asarray(target_probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[0] == 0:
        return None
    finite = np.all(np.isfinite(probabilities), axis=1)
    maximum = np.full(probabilities.shape[0], np.nan, dtype=float)
    maximum[finite] = np.max(probabilities[finite], axis=1)
    is_maximum = np.isclose(probabilities, maximum[:, None], rtol=0.0, atol=1e-12)
    unique = finite & (np.sum(is_maximum, axis=1) == 1)
    majority = np.full(probabilities.shape[0], -1, dtype=int)
    majority[unique] = np.argmax(probabilities[unique], axis=1)
    return majority


def safe_pearson(first: Any, second: Any, *, min_observations: int = 2) -> float:
    first_array = np.asarray(first, dtype=float).reshape(-1)
    second_array = np.asarray(second, dtype=float).reshape(-1)
    if first_array.shape != second_array.shape:
        return float("nan")
    finite = np.isfinite(first_array) & np.isfinite(second_array)
    if int(np.sum(finite)) < int(min_observations):
        return float("nan")
    first_array = first_array[finite]
    second_array = second_array[finite]
    if float(np.std(first_array)) <= 0.0 or float(np.std(second_array)) <= 0.0:
        return float("nan")
    return float(np.corrcoef(first_array, second_array)[0, 1])


def sliding_binary_metrics(
    observed: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
    *,
    window_size: int,
    score_trial_mask: Sequence[bool] | np.ndarray | None = None,
    start_index: int = 1,
    std_denominator: str = "window",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct aligned rolling means and Bernoulli aggregate uncertainty."""
    observed_array = np.asarray(observed, dtype=float).reshape(-1)
    predicted_array = np.asarray(predicted, dtype=float).reshape(-1)
    if observed_array.shape != predicted_array.shape:
        raise ValueError(
            f"observed and predicted lengths differ: {observed_array.shape} vs "
            f"{predicted_array.shape}"
        )
    window_size = int(window_size)
    start_index = int(start_index)
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if std_denominator not in {"window", "finite"}:
        raise ValueError("std_denominator must be 'window' or 'finite'")
    if score_trial_mask is None:
        score_mask = np.ones(observed_array.size, dtype=bool)
    else:
        score_mask = np.asarray(score_trial_mask, dtype=bool).reshape(-1)
        if score_mask.shape != observed_array.shape:
            raise ValueError(
                "score_trial_mask length does not match trial values: "
                f"{score_mask.shape[0]} vs {observed_array.shape[0]}"
            )

    observed_curve: list[float] = []
    predicted_curve: list[float] = []
    predicted_std: list[float] = []
    stop = observed_array.size - window_size + 1
    for start in range(start_index, stop):
        end = start + window_size
        if not bool(np.all(score_mask[start:end])):
            observed_curve.append(float("nan"))
            predicted_curve.append(float("nan"))
            predicted_std.append(float("nan"))
            continue
        observed_window = observed_array[start:end]
        predicted_window = predicted_array[start:end]
        observed_curve.append(nanmean_or_nan(observed_window))
        predicted_curve.append(nanmean_or_nan(predicted_window))
        finite = predicted_window[np.isfinite(predicted_window)]
        denominator = window_size if std_denominator == "window" else max(1, finite.size)
        predicted_std.append(
            float(np.sqrt(np.sum(finite * (1.0 - finite))) / denominator)
            if finite.size
            else float("nan")
        )
    return (
        np.asarray(observed_curve, dtype=float),
        np.asarray(predicted_curve, dtype=float),
        np.asarray(predicted_std, dtype=float),
    )


def accuracy_metrics_from_info(
    info: Mapping[str, Any], *, window_size: int | None = None
) -> dict[str, Any]:
    """Build the legacy ModelEvaluator rolling-accuracy mapping from saved arrays."""
    if info.get("true_acc") is None or info.get("pred_acc") is None:
        return {}
    try:
        observed = np.asarray(info.get("true_acc"), dtype=float).reshape(-1)
        predicted = np.asarray(info.get("pred_acc"), dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return {}
    n_trials = min(observed.size, predicted.size)
    resolved_window = _resolve_window_size(window_size, info)
    if n_trials <= 1 or resolved_window <= 0 or n_trials < resolved_window + 1:
        return {}
    observed_curve, predicted_curve, predicted_std = sliding_binary_metrics(
        observed[:n_trials], predicted[:n_trials], window_size=resolved_window
    )
    return {
        "sliding_true_acc": observed_curve,
        "sliding_pred_acc": predicted_curve,
        "sliding_pred_acc_std": predicted_std,
        "window_size": int(resolved_window),
    }


def target_majority_accuracy_metrics_from_info(
    info: Mapping[str, Any], *, window_size: int | None = None
) -> dict[str, Any]:
    """Compute observed and predicted choices of the unique target majority."""
    target = info.get("target_probs")
    probabilities = info.get("pred_category_probs")
    choices = info.get("observed_choice_index")
    if target is None or probabilities is None or choices is None:
        return {}
    try:
        target = np.asarray(target, dtype=float)
        probabilities = np.asarray(probabilities, dtype=float)
        choices = np.asarray(choices, dtype=int).reshape(-1)
    except (TypeError, ValueError):
        return {}
    if target.ndim != 2 or probabilities.ndim != 2:
        return {}
    n_trials = min(target.shape[0], probabilities.shape[0], choices.size)
    n_categories = min(target.shape[1], probabilities.shape[1])
    resolved_window = _resolve_window_size(window_size, info)
    if (
        n_trials <= 1
        or n_categories <= 0
        or resolved_window <= 0
        or n_trials < resolved_window + 1
    ):
        return {}
    target = target[:n_trials, :n_categories]
    probabilities = probabilities[:n_trials, :n_categories]
    choices = choices[:n_trials]
    majority = target_majority_indices(target)
    if majority is None:
        return {}
    observed = np.full(n_trials, np.nan, dtype=float)
    predicted = np.full(n_trials, np.nan, dtype=float)
    valid_choice = (
        (majority >= 0) & (choices >= 0) & (choices < n_categories)
    )
    observed[valid_choice] = (choices[valid_choice] == majority[valid_choice]).astype(float)
    finite_probability = np.all(np.isfinite(probabilities), axis=1)
    valid_predicted = (majority >= 0) & (majority < n_categories) & finite_probability
    rows = np.flatnonzero(valid_predicted)
    predicted[rows] = probabilities[rows, majority[rows]]
    observed_curve, predicted_curve, predicted_std = sliding_binary_metrics(
        observed,
        predicted,
        window_size=resolved_window,
        std_denominator="finite",
    )
    return {
        "target_majority_acc": observed,
        "pred_target_majority_acc": predicted,
        "target_majority_index": majority,
        "sliding_target_majority_acc": observed_curve,
        "sliding_pred_target_majority_acc": predicted_curve,
        "sliding_pred_target_majority_acc_std": predicted_std,
    }


def validate_exp_accuracy_alpha(alpha: Any) -> float | None:
    if alpha is None:
        return None
    try:
        value = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"exp_accuracy_alpha must be a number in (0, 1], got {alpha!r}"
        ) from exc
    if not np.isfinite(value) or value <= 0.0 or value > 1.0:
        raise ValueError(f"exp_accuracy_alpha must be in (0, 1], got {alpha!r}")
    return value


def exponential_accuracy_metrics_from_info(
    info: Mapping[str, Any], *, exp_accuracy_alpha: float | None = None
) -> dict[str, Any]:
    """Build exponentially smoothed accuracy curves from a saved result mapping."""
    alpha = validate_exp_accuracy_alpha(exp_accuracy_alpha)
    if alpha is None:
        alpha = validate_exp_accuracy_alpha(info.get("exp_accuracy_alpha"))
    if alpha is None:
        try:
            alpha = validate_exp_accuracy_alpha(2.0 / (float(info.get("window_size")) + 1.0))
        except (TypeError, ValueError):
            alpha = None
    if alpha is None:
        return {}
    n_categories = 2 if int(info.get("condition", 1)) == 1 else 4
    chance = 1.0 / float(max(1, n_categories))
    output: dict[str, Any] = {"exp_accuracy_alpha": float(alpha)}
    if info.get("true_acc") is not None and info.get("pred_acc") is not None:
        observed = np.asarray(info.get("true_acc"), dtype=float).reshape(-1)
        predicted = np.asarray(info.get("pred_acc"), dtype=float).reshape(-1)
        n_trials = min(observed.size, predicted.size)
        if n_trials:
            output["exp_true_acc"] = exponential_smooth_curve(
                observed[:n_trials], alpha=alpha, init_value=chance
            )
            output["exp_pred_acc"] = exponential_smooth_curve(
                predicted[:n_trials], alpha=alpha, init_value=chance
            )
    target_values = info.get("target_probs")
    try:
        has_target_values = bool(
            target_values is not None
            and np.asarray(target_values, dtype=float).size
            and np.any(np.isfinite(np.asarray(target_values, dtype=float)))
        )
    except (TypeError, ValueError):
        has_target_values = False
    target = (
        target_majority_accuracy_metrics_from_info(info, window_size=1)
        if has_target_values
        else {}
    )
    if target:
        observed = np.asarray(target["target_majority_acc"], dtype=float).reshape(-1)
        predicted = np.asarray(target["pred_target_majority_acc"], dtype=float).reshape(-1)
        n_trials = min(observed.size, predicted.size)
        if n_trials:
            output["exp_target_majority_acc"] = exponential_smooth_curve(
                observed[:n_trials], alpha=alpha, init_value=chance
            )
            output["exp_pred_target_majority_acc"] = exponential_smooth_curve(
                predicted[:n_trials], alpha=alpha, init_value=chance
            )
    return output


def choice_brier_curve_metrics_from_info(
    info: Mapping[str, Any], *, window_size: int | None = None
) -> dict[str, Any]:
    """Return trial and rolling multiclass Brier scores for observed choices."""
    probabilities = info.get("pred_category_probs")
    choices = info.get("observed_choice_index")
    if probabilities is None or choices is None:
        return {}
    try:
        probabilities = np.asarray(probabilities, dtype=float)
        choices = np.asarray(choices, dtype=int).reshape(-1)
    except (TypeError, ValueError):
        return {}
    if probabilities.ndim != 2 or probabilities.shape[0] == 0 or probabilities.shape[1] == 0:
        return {}
    n_trials = min(probabilities.shape[0], choices.size)
    probabilities = probabilities[:n_trials]
    choices = choices[:n_trials]
    n_categories = int(probabilities.shape[1])
    valid = _coerce_optional_mask(info.get("valid_trial_mask"), n_trials, pad=True)
    per_trial = np.full(n_trials, np.nan, dtype=float)
    keep = (
        valid
        & np.all(np.isfinite(probabilities), axis=1)
        & (choices >= 0)
        & (choices < n_categories)
    )
    rows = np.flatnonzero(keep)
    if rows.size:
        one_hot = np.zeros((rows.size, n_categories), dtype=float)
        one_hot[np.arange(rows.size), choices[rows]] = 1.0
        per_trial[rows] = np.sum(np.square(probabilities[rows] - one_hot), axis=1)
    resolved_window = _resolve_window_size(window_size, info)
    rolling: list[float] = []
    if resolved_window > 0 and n_trials >= resolved_window + 1:
        for start in range(1, n_trials - resolved_window + 1):
            rolling.append(nanmean_or_nan(per_trial[start : start + resolved_window]))
    return {
        "choice_brier": per_trial,
        "sliding_choice_brier": np.asarray(rolling, dtype=float),
        "choice_brier_window_size": int(resolved_window),
        "choice_brier_chance": float(1.0 - 1.0 / n_categories),
    }


def build_prediction_metric_bundle(
    category_probabilities: np.ndarray,
    *,
    choices: Sequence[int] | np.ndarray,
    feedback: Sequence[float] | np.ndarray,
    categories: Sequence[int] | np.ndarray | None,
    target_probabilities: np.ndarray | None,
    window_size: int,
    score_trial_mask: Sequence[bool] | np.ndarray | None = None,
    valid_trial_mask: Sequence[bool] | np.ndarray | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    mask_choice_prediction_by_validity: bool = False,
    target_std_denominator: str = "finite",
) -> dict[str, Any]:
    """Build the canonical optimizer/evaluator metric mapping from probabilities.

    Choices and categories use the repository's one-based data convention;
    emitted index arrays are zero-based. Trial zero remains an initialization
    trial and is excluded from predictive scoring.
    """
    probabilities = np.asarray(category_probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] == 0:
        raise ValueError(
            f"category_probabilities must be a non-empty 2-D array, got {probabilities.shape}"
        )
    observed_choices = np.asarray(choices, dtype=int).reshape(-1)
    observed_feedback = np.asarray(feedback, dtype=float).reshape(-1)
    n_trials, n_categories = probabilities.shape
    if observed_choices.size != n_trials or observed_feedback.size != n_trials:
        raise ValueError("probabilities, choices, and feedback must align")
    window_size = int(window_size)
    if window_size <= 0 or n_trials < window_size + 1:
        raise ValueError(
            "Not enough trials for sliding metrics: "
            f"need at least {window_size + 1}, got {n_trials}"
        )
    score_mask = _coerce_optional_mask(score_trial_mask, n_trials, pad=False)
    choice_index = observed_choices - 1
    if valid_trial_mask is None:
        valid = (
            score_mask
            & (choice_index >= 0)
            & (choice_index < n_categories)
            & np.all(np.isfinite(probabilities), axis=1)
        )
        valid[0] = False
    else:
        valid = _coerce_optional_mask(valid_trial_mask, n_trials, pad=False)

    true_accuracy = (observed_feedback == 1.0).astype(float)
    if categories is None:
        category_values = None
        true_category_index = np.full(n_trials, -1, dtype=int)
        true_family_accuracy = np.full(n_trials, np.nan, dtype=float)
        predicted_family_accuracy = np.full(n_trials, np.nan, dtype=float)
        predicted_accuracy = np.full(n_trials, np.nan, dtype=float)
        rows = np.flatnonzero(
            (choice_index >= 0)
            & (choice_index < n_categories)
            & np.all(np.isfinite(probabilities), axis=1)
        )
        if mask_choice_prediction_by_validity:
            rows = rows[valid[rows]]
        predicted_accuracy[rows] = probabilities[rows, choice_index[rows]]
    else:
        category_values = np.asarray(categories, dtype=int).reshape(-1)
        if category_values.size != n_trials:
            raise ValueError("categories length does not match probability trials")
        true_category_index = category_values - 1
        true_family_accuracy = family_correct(
            category_values, observed_choices, n_categories
        )
        predicted_accuracy = np.full(n_trials, np.nan, dtype=float)
        valid_category = (
            (true_category_index >= 0)
            & (true_category_index < n_categories)
            & np.all(np.isfinite(probabilities), axis=1)
        )
        rows = np.flatnonzero(valid_category)
        predicted_accuracy[rows] = probabilities[rows, true_category_index[rows]]
        predicted_family_accuracy = np.full(n_trials, np.nan, dtype=float)
        for trial_index in rows:
            members = family_indices(int(category_values[trial_index]), n_categories)
            members = members[(members >= 0) & (members < n_categories)]
            if members.size:
                predicted_family_accuracy[trial_index] = float(
                    np.sum(probabilities[trial_index, members])
                )

    if target_probabilities is None:
        target_matrix = np.full((n_trials, n_categories), np.nan, dtype=float)
    else:
        target_matrix = np.asarray(target_probabilities, dtype=float)
        if target_matrix.shape != probabilities.shape:
            raise ValueError(
                "target probabilities shape does not match predictions: "
                f"{target_matrix.shape} vs {probabilities.shape}"
            )
    majority_index = target_majority_indices(target_matrix)
    if majority_index is None:
        majority_index = np.full(n_trials, -1, dtype=int)
    observed_majority_accuracy = np.full(n_trials, np.nan, dtype=float)
    predicted_majority_accuracy = np.full(n_trials, np.nan, dtype=float)
    valid_majority = (
        (majority_index >= 0)
        & (majority_index < n_categories)
        & (choice_index >= 0)
        & (choice_index < n_categories)
    )
    majority_rows = np.flatnonzero(valid_majority)
    observed_majority_accuracy[majority_rows] = (
        choice_index[majority_rows] == majority_index[majority_rows]
    ).astype(float)
    finite_majority_rows = majority_rows[
        np.all(np.isfinite(probabilities[majority_rows]), axis=1)
    ]
    predicted_majority_accuracy[finite_majority_rows] = probabilities[
        finite_majority_rows, majority_index[finite_majority_rows]
    ]

    sliding_true, sliding_predicted, sliding_std = sliding_binary_metrics(
        true_accuracy,
        predicted_accuracy,
        window_size=window_size,
        score_trial_mask=score_mask,
    )
    sliding_true_family, sliding_predicted_family, sliding_family_std = (
        sliding_binary_metrics(
            true_family_accuracy,
            predicted_family_accuracy,
            window_size=window_size,
            score_trial_mask=score_mask,
        )
    )
    sliding_true_majority, sliding_predicted_majority, sliding_majority_std = (
        sliding_binary_metrics(
            observed_majority_accuracy,
            predicted_majority_accuracy,
            window_size=window_size,
            score_trial_mask=score_mask,
            std_denominator=target_std_denominator,
        )
    )

    alpha = float(2.0 / (float(window_size) + 1.0))
    chance = 1.0 / float(max(1, n_categories))
    family_error = np.abs(sliding_true_family - sliding_predicted_family)
    family_error = family_error[np.isfinite(family_error)]
    target_finite = (
        valid
        & np.all(np.isfinite(probabilities), axis=1)
        & np.all(np.isfinite(target_matrix), axis=1)
    )
    if np.any(target_finite):
        target_brier = float(
            np.mean(
                np.sum(
                    np.square(probabilities[target_finite] - target_matrix[target_finite]),
                    axis=1,
                )
            )
        )
        target_correlation = np.asarray(
            [
                safe_pearson(
                    probabilities[target_finite, category_index],
                    target_matrix[target_finite, category_index],
                )
                for category_index in range(n_categories)
            ],
            dtype=float,
        )
    else:
        target_brier = float("nan")
        target_correlation = np.full(n_categories, np.nan, dtype=float)

    output: dict[str, Any] = {
        "true_acc": true_accuracy,
        "pred_acc": predicted_accuracy,
        "true_family_acc": true_family_accuracy,
        "pred_family_acc": predicted_family_accuracy,
        "sliding_true_acc": sliding_true,
        "sliding_pred_acc": sliding_predicted,
        "sliding_pred_acc_std": sliding_std,
        "sliding_true_family_acc": sliding_true_family,
        "sliding_pred_family_acc": sliding_predicted_family,
        "sliding_pred_family_acc_std": sliding_family_std,
        "exp_true_acc": exponential_smooth_curve(
            true_accuracy, alpha=alpha, init_value=chance
        ),
        "exp_pred_acc": exponential_smooth_curve(
            predicted_accuracy, alpha=alpha, init_value=chance
        ),
        "exp_true_family_acc": exponential_smooth_curve(
            true_family_accuracy, alpha=alpha, init_value=chance
        ),
        "exp_pred_family_acc": exponential_smooth_curve(
            predicted_family_accuracy, alpha=alpha, init_value=chance
        ),
        "exp_accuracy_alpha": alpha,
        "target_majority_acc": observed_majority_accuracy,
        "pred_target_majority_acc": predicted_majority_accuracy,
        "sliding_target_majority_acc": sliding_true_majority,
        "sliding_pred_target_majority_acc": sliding_predicted_majority,
        "sliding_pred_target_majority_acc_std": sliding_majority_std,
        "exp_target_majority_acc": exponential_smooth_curve(
            observed_majority_accuracy, alpha=alpha, init_value=chance
        ),
        "exp_pred_target_majority_acc": exponential_smooth_curve(
            predicted_majority_accuracy, alpha=alpha, init_value=chance
        ),
        "family_mean_error": (
            float(np.mean(family_error)) if family_error.size else float("nan")
        ),
        "pred_category_probs": probabilities,
        "target_probs": target_matrix,
        "target_prob_brier": target_brier,
        "target_prob_corr_by_cat": target_correlation,
        "target_prob_corr_cat1": (
            float(target_correlation[0]) if target_correlation.size else float("nan")
        ),
        "true_category_index": true_category_index,
        "observed_choice_index": choice_index,
        # Retain the source observations so independent stochastic repeats can
        # be combined at the probability level and all derived metrics rebuilt
        # without reverse-engineering feedback/category codes.
        "observed_choice": observed_choices,
        "observed_feedback": observed_feedback,
        "observed_category": category_values,
        "target_majority_index": majority_index,
        "valid_trial_mask": valid,
        "score_trial_mask": score_mask,
        "window_size": int(window_size),
    }
    output.update(dict(diagnostics or {}))
    _attach_diagnostic_summaries(output, valid)
    return output


def predictive_accuracy_band_metrics(
    prediction_curves: np.ndarray,
    observed_curve: Sequence[float] | np.ndarray,
) -> dict[str, Any]:
    """Compute latent-run/Monte-Carlo bands across expected-accuracy curves.

    This helper does not add observation-level behavioral variability.  It is
    appropriate for trajectory ensembles or numerical stability diagnostics,
    but not for a particle-filter behavioral predictive interval.
    """
    predictions = np.asarray(prediction_curves, dtype=float)
    observed = np.asarray(observed_curve, dtype=float).reshape(-1)
    if predictions.ndim != 2 or predictions.shape[1] != observed.size:
        raise ValueError(
            "prediction_curves must have shape (run, curve_point) aligned to observed_curve"
        )
    if predictions.shape[0] == 0:
        raise ValueError("prediction_curves cannot be empty")
    q00 = np.nanmin(predictions, axis=0)
    q05, q25, q50, q75, q95 = np.nanquantile(
        predictions, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0
    )
    q100 = np.nanmax(predictions, axis=0)
    true_volatility = (
        float(np.mean(np.abs(np.diff(observed)))) if observed.size > 1 else float("nan")
    )
    median_volatility = (
        float(np.mean(np.abs(np.diff(q50)))) if q50.size > 1 else float("nan")
    )
    return {
        "q00": q00,
        "q05": q05,
        "q25": q25,
        "q50": q50,
        "q75": q75,
        "q95": q95,
        "q100": q100,
        "median_curve_mae": float(np.mean(np.abs(q50 - observed))),
        "coverage_50": float(np.mean((observed >= q25) & (observed <= q75))),
        "coverage_90": float(np.mean((observed >= q05) & (observed <= q95))),
        "median_vol_ratio": (
            float(median_volatility / true_volatility)
            if true_volatility > 0
            else float("nan")
        ),
    }


def conditional_behavioral_accuracy_band_metrics(
    prediction_probabilities: np.ndarray,
    observed_curve: Sequence[float] | np.ndarray,
    *,
    window_size: int,
    n_draws: int = 5000,
    seed: int = 20260810,
    score_trial_mask: Sequence[bool] | np.ndarray | None = None,
    start_index: int = 1,
) -> dict[str, Any]:
    """Simulate a rolling-accuracy band from trialwise correctness probabilities.

    Rows of ``prediction_probabilities`` are independent particle-filter
    repeats for the same observed history and fitted parameter setting.  They
    are averaged first to reduce finite-particle Monte-Carlo error.  Bernoulli
    correctness sequences are then sampled from that marginal probability
    curve and transformed with the same rolling window as the observed curve.

    The resulting intervals are pointwise, observed-history-conditional
    behavioral predictive intervals.  They are not autonomous rollouts and do
    not include fitted-parameter uncertainty.
    """
    probabilities = np.asarray(prediction_probabilities, dtype=float)
    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(1, -1)
    if probabilities.ndim != 2 or probabilities.shape[0] == 0:
        raise ValueError(
            "prediction_probabilities must have shape (run, trial) with at least one run"
        )
    n_runs, n_trials = probabilities.shape
    window_size = int(window_size)
    start_index = int(start_index)
    n_draws = int(n_draws)
    if window_size <= 0 or n_trials < window_size + start_index:
        raise ValueError(
            "Not enough trials for the requested rolling behavioral interval: "
            f"n_trials={n_trials}, window_size={window_size}, start_index={start_index}"
        )
    if n_draws < 2:
        raise ValueError("n_draws must be at least 2")

    finite = np.isfinite(probabilities)
    finite_values = probabilities[finite]
    if finite_values.size and (
        np.any(finite_values < 0.0) or np.any(finite_values > 1.0)
    ):
        raise ValueError("prediction probabilities must lie in [0, 1]")
    finite_count = np.sum(finite, axis=0)
    expected_trial = np.divide(
        np.nansum(probabilities, axis=0),
        finite_count,
        out=np.full(n_trials, np.nan, dtype=float),
        where=finite_count > 0,
    )

    if score_trial_mask is None:
        score_mask = np.ones(n_trials, dtype=bool)
    else:
        score_mask = np.asarray(score_trial_mask, dtype=bool).reshape(-1)
        if score_mask.size != n_trials:
            raise ValueError(
                "score_trial_mask length does not match prediction trials: "
                f"{score_mask.size} vs {n_trials}"
            )

    starts = np.arange(
        start_index,
        n_trials - window_size + 1,
        dtype=int,
    )
    observed = np.asarray(observed_curve, dtype=float).reshape(-1)
    if observed.size != starts.size:
        raise ValueError(
            "observed_curve length does not match rolling prediction length: "
            f"{observed.size} vs {starts.size}"
        )

    valid_trial = score_mask & np.isfinite(expected_trial)
    valid_window = np.asarray(
        [bool(np.all(valid_trial[start : start + window_size])) for start in starts],
        dtype=bool,
    )
    expected_curve = np.full(starts.size, np.nan, dtype=float)
    for curve_index, start in enumerate(starts):
        if valid_window[curve_index]:
            expected_curve[curve_index] = float(
                np.mean(expected_trial[start : start + window_size])
            )

    rng = np.random.default_rng(int(seed))
    draw_probability = np.where(np.isfinite(expected_trial), expected_trial, 0.0)
    binary_draws = rng.random((n_draws, n_trials)) < draw_probability[None, :]
    rolling_draws = np.full((n_draws, starts.size), np.nan, dtype=float)
    for curve_index, start in enumerate(starts):
        if valid_window[curve_index]:
            rolling_draws[:, curve_index] = np.mean(
                binary_draws[:, start : start + window_size],
                axis=1,
            )

    quantile_arrays = {
        key: np.full(starts.size, np.nan, dtype=float)
        for key in ("q05", "q25", "q50", "q75", "q95")
    }
    if np.any(valid_window):
        values = np.quantile(
            rolling_draws[:, valid_window],
            [0.05, 0.25, 0.50, 0.75, 0.95],
            axis=0,
        )
        for row, key in enumerate(("q05", "q25", "q50", "q75", "q95")):
            quantile_arrays[key][valid_window] = values[row]

    evaluable = (
        np.isfinite(observed)
        & np.isfinite(quantile_arrays["q05"])
        & np.isfinite(quantile_arrays["q95"])
    )
    if np.any(evaluable):
        coverage_50 = float(
            np.mean(
                (observed[evaluable] >= quantile_arrays["q25"][evaluable])
                & (observed[evaluable] <= quantile_arrays["q75"][evaluable])
            )
        )
        coverage_90 = float(
            np.mean(
                (observed[evaluable] >= quantile_arrays["q05"][evaluable])
                & (observed[evaluable] <= quantile_arrays["q95"][evaluable])
            )
        )
        mean_width_50 = float(
            np.mean(
                quantile_arrays["q75"][evaluable]
                - quantile_arrays["q25"][evaluable]
            )
        )
        mean_width_90 = float(
            np.mean(
                quantile_arrays["q95"][evaluable]
                - quantile_arrays["q05"][evaluable]
            )
        )
        expected_curve_mae = float(
            np.mean(np.abs(expected_curve[evaluable] - observed[evaluable]))
        )
    else:
        coverage_50 = float("nan")
        coverage_90 = float("nan")
        mean_width_50 = float("nan")
        mean_width_90 = float("nan")
        expected_curve_mae = float("nan")

    return {
        "band_type": "observed_history_conditional_behavioral",
        "n_runs": int(n_runs),
        "n_draws": int(n_draws),
        "seed": int(seed),
        "expected_trial_probability": expected_trial,
        "expected_curve": expected_curve,
        "coverage_50": coverage_50,
        "coverage_90": coverage_90,
        "mean_width_50": mean_width_50,
        "mean_width_90": mean_width_90,
        "expected_curve_mae": expected_curve_mae,
        **quantile_arrays,
    }


def _resolve_window_size(window_size: int | None, info: Mapping[str, Any]) -> int:
    value = window_size if window_size is not None else info.get("window_size")
    try:
        return int(value)
    except (TypeError, ValueError):
        return 16


def _coerce_optional_mask(values: Any, n_trials: int, *, pad: bool) -> np.ndarray:
    if values is None:
        return np.ones(n_trials, dtype=bool)
    try:
        mask = np.asarray(values, dtype=bool).reshape(-1)
    except (TypeError, ValueError):
        return np.ones(n_trials, dtype=bool)
    if mask.size == n_trials:
        return mask.copy()
    if not pad:
        raise ValueError(
            f"mask length does not match number of trials: {mask.size} vs {n_trials}"
        )
    output = np.zeros(n_trials, dtype=bool)
    output[: min(mask.size, n_trials)] = mask[:n_trials]
    return output


def _attach_diagnostic_summaries(output: dict[str, Any], valid: np.ndarray) -> None:
    n_trials = valid.size
    output_lapse = np.asarray(
        output.get("output_lapse", np.zeros(n_trials, dtype=float)), dtype=float
    ).reshape(-1)
    latent_volatility = np.asarray(
        output.get("latent_volatility", np.zeros(n_trials, dtype=float)), dtype=float
    ).reshape(-1)
    lapse_valid = valid[: min(valid.size, output_lapse.size)]
    lapse_values = output_lapse[: lapse_valid.size][lapse_valid]
    volatility_valid = valid[: min(valid.size, latent_volatility.size)]
    volatility_values = latent_volatility[: volatility_valid.size][volatility_valid]
    output["output_lapse"] = output_lapse
    output["output_lapse_mean"] = nanmean_or_nan(lapse_values)
    output["output_lapse_max"] = (
        float(np.nanmax(output_lapse)) if output_lapse.size else float("nan")
    )
    output["latent_volatility"] = latent_volatility
    output["latent_volatility_mean"] = nanmean_or_nan(volatility_values)
    output["latent_volatility_max"] = (
        float(np.nanmax(latent_volatility)) if latent_volatility.size else float("nan")
    )


__all__ = [
    "accuracy_metrics_from_info",
    "build_prediction_metric_bundle",
    "choice_brier_curve_metrics_from_info",
    "conditional_behavioral_accuracy_band_metrics",
    "exponential_accuracy_metrics_from_info",
    "family_correct",
    "family_indices",
    "predictive_accuracy_band_metrics",
    "safe_pearson",
    "sliding_binary_metrics",
    "target_majority_accuracy_metrics_from_info",
    "target_majority_indices",
    "validate_exp_accuracy_alpha",
]

"""Group-level paired summaries for frozen model comparisons."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def paired_metric_summary(
    candidate: Sequence[float] | np.ndarray,
    reference: Sequence[float] | np.ndarray,
    *,
    lower_is_better: bool,
    bootstrap_repeats: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Summarize candidate-minus-reference differences by independent unit."""
    candidate_array = np.asarray(candidate, dtype=float).reshape(-1)
    reference_array = np.asarray(reference, dtype=float).reshape(-1)
    if candidate_array.shape != reference_array.shape:
        raise ValueError(
            f"paired metric shapes differ: {candidate_array.shape} vs {reference_array.shape}"
        )
    bootstrap_repeats = int(bootstrap_repeats)
    if bootstrap_repeats < 1:
        raise ValueError("bootstrap_repeats must be positive")
    valid = np.isfinite(candidate_array) & np.isfinite(reference_array)
    difference = candidate_array[valid] - reference_array[valid]
    if difference.size == 0:
        return {
            "difference_direction": "candidate_minus_reference",
            "lower_is_better": bool(lower_is_better),
            "n_pairs": 0,
            "mean_difference": float("nan"),
            "median_difference": float("nan"),
            "ci025": float("nan"),
            "ci975": float("nan"),
            "candidate_win_count": 0,
            "candidate_win_fraction": float("nan"),
            "tie_count": 0,
        }
    generator = np.random.default_rng(int(seed))
    samples = generator.choice(
        difference,
        size=(bootstrap_repeats, difference.size),
        replace=True,
    ).mean(axis=1)
    ties = np.isclose(difference, 0.0, rtol=0.0, atol=1e-12)
    if lower_is_better:
        wins = (difference < 0.0) & ~ties
    else:
        wins = (difference > 0.0) & ~ties
    return {
        "difference_direction": "candidate_minus_reference",
        "lower_is_better": bool(lower_is_better),
        "n_pairs": int(difference.size),
        "mean_difference": float(np.mean(difference)),
        "median_difference": float(np.median(difference)),
        "ci025": float(np.quantile(samples, 0.025)),
        "ci975": float(np.quantile(samples, 0.975)),
        "candidate_win_count": int(np.sum(wins)),
        "candidate_win_fraction": float(np.mean(wins)),
        "tie_count": int(np.sum(ties)),
    }


def benjamini_hochberg(p_values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted p values, preserving NaNs."""
    values = np.asarray(p_values, dtype=float).reshape(-1)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite_positions = np.flatnonzero(np.isfinite(values))
    if finite_positions.size == 0:
        return adjusted
    finite_values = values[finite_positions]
    if np.any((finite_values < 0.0) | (finite_values > 1.0)):
        raise ValueError("finite p values must lie in [0, 1]")
    order = np.argsort(finite_values, kind="mergesort")
    ranked = finite_values[order]
    count = ranked.size
    raw_adjusted = ranked * count / np.arange(1, count + 1, dtype=float)
    monotone = np.minimum.accumulate(raw_adjusted[::-1])[::-1]
    reordered = np.empty_like(monotone)
    reordered[order] = np.clip(monotone, 0.0, 1.0)
    adjusted[finite_positions] = reordered
    return adjusted


__all__ = ["benjamini_hochberg", "paired_metric_summary"]

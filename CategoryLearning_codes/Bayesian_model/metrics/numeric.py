"""Internal numerical helpers shared by metric modules."""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def finite_array(values: Iterable[Any]) -> np.ndarray:
    array = np.asarray([safe_float(value) for value in values], dtype=float)
    return array[np.isfinite(array)]


def nanmean_or_nan(values: Sequence[Any]) -> float:
    array = finite_array(values)
    return float(np.mean(array)) if array.size else float("nan")


def nanmedian_or_nan(values: Sequence[Any]) -> float:
    array = finite_array(values)
    return float(np.median(array)) if array.size else float("nan")


def nanquantile_or_nan(values: Sequence[Any], quantile: float) -> float:
    array = finite_array(values)
    return float(np.quantile(array, float(quantile))) if array.size else float("nan")


def minimize_rank01(values: Sequence[float]) -> np.ndarray:
    """Return stable average ranks scaled to [0, 1] for a minimization metric."""

    array = np.asarray(values, dtype=float)
    ranks = np.ones(array.shape, dtype=float)
    finite_positions = np.flatnonzero(np.isfinite(array))
    if finite_positions.size == 0:
        return ranks
    if finite_positions.size == 1:
        ranks[finite_positions[0]] = 0.0
        return ranks

    finite_values = array[finite_positions]
    order = np.argsort(finite_values, kind="mergesort")
    sorted_positions = finite_positions[order]
    sorted_values = finite_values[order]
    denominator = float(max(1, sorted_positions.size - 1))
    start = 0
    while start < sorted_positions.size:
        end = start + 1
        while end < sorted_positions.size and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = ((start + end - 1) / 2.0) / denominator
        ranks[sorted_positions[start:end]] = float(rank)
        start = end
    return ranks


__all__ = [
    "finite_array",
    "minimize_rank01",
    "nanmean_or_nan",
    "nanmedian_or_nan",
    "nanquantile_or_nan",
    "safe_float",
]

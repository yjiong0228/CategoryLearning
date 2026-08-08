"""Small numerical helpers shared by metric modules."""
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


__all__ = [
    "finite_array",
    "nanmean_or_nan",
    "nanmedian_or_nan",
    "nanquantile_or_nan",
    "safe_float",
]

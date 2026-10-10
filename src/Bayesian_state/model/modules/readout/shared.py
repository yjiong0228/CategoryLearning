"""Probability and configuration helpers shared by readout channels."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def _mapping_get_path(root: Mapping[str, Any] | None, path: str) -> Any:
    current: Any = root
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _float_from_mapping(
    values: Mapping[str, Any],
    key: str,
    default: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    context: str = "readout",
) -> float:
    raw = values.get(key, default)
    try:
        out = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}.{key} must be numeric, got {raw!r}") from exc
    if not np.isfinite(out):
        raise ValueError(f"{context}.{key} must be finite, got {raw!r}")
    if min_value is not None and out < min_value:
        raise ValueError(f"{context}.{key} must be >= {min_value}, got {out!r}")
    if max_value is not None and out > max_value:
        raise ValueError(f"{context}.{key} must be <= {max_value}, got {out!r}")
    return out


def _bool_from_mapping(
    values: Mapping[str, Any], key: str, default: bool, *, context: str
) -> bool:
    raw = values.get(key, default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    if isinstance(raw, (int, float, np.integer, np.floating)):
        return bool(raw)
    raise ValueError(f"{context}.{key} must be boolean-like, got {raw!r}")


def normalize_probability_vector(
    values: Sequence[float] | np.ndarray,
    size: int | None = None,
    *,
    strict: bool = False,
) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float).reshape(-1)
    expected = probabilities.size if size is None else int(size)
    if probabilities.size != expected:
        raise ValueError(
            f"Probability vector width mismatch: expected {expected}, got {probabilities.size}"
        )
    valid = (
        probabilities.size > 0
        and np.all(np.isfinite(probabilities))
        and np.all(probabilities >= 0.0)
        and float(np.sum(probabilities)) > 0.0
    )
    if not valid:
        if strict:
            raise ValueError("Probability vector must have finite non-negative positive mass.")
        return np.full(expected, 1.0 / max(1, expected), dtype=float)
    return probabilities / float(np.sum(probabilities))

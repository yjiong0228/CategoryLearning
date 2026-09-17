"""Validate response encodings before any lossy integer conversion."""
from __future__ import annotations

import numpy as np


def response_ids(values, *, context: str = "choices", n_categories: int | None = None) -> np.ndarray:
    """Return positive integer IDs, rejecting missing, fractional or invalid IDs."""
    message = f"{context} must be finite integers"
    message += f" in [1, {n_categories}]" if n_categories is not None else " greater than zero"
    try:
        raw = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    valid = np.isfinite(raw) & (raw == np.floor(raw)) & (raw >= 1)
    # Prevent overflow during conversion even when the category count is unknown.
    valid &= raw < float(np.iinfo(np.int64).max)
    if n_categories is not None:
        valid &= raw <= n_categories
    if not np.all(valid):
        raise ValueError(message)
    return raw.astype(np.int64)

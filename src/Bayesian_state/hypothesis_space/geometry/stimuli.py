"""Stimulus validation shared by hypothesis-geometry implementations."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def as_stimuli(stimuli: np.ndarray | Iterable, n_dims: int) -> np.ndarray:
    values = np.asarray(stimuli, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != int(n_dims):
        raise ValueError(f"Expected stimuli[n, {n_dims}], got {values.shape}.")
    return values

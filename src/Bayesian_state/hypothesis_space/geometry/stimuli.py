"""Stimulus validation shared by hypothesis-geometry implementations."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def as_stimuli(stimuli: np.ndarray | Iterable, n_dims: int) -> np.ndarray:
    """Validate that stimuli are finite unit-cube points of the right width.

    Category regions are only defined inside the unit cube: the perception
    module clips every sampled stimulus into ``[0, 1]``, so a stimulus outside
    it is a caller error rather than a value the model has to interpret.
    Rejecting it here keeps the solver contract exact instead of extending a
    projection result past the region the geometry describes.
    """

    values = np.asarray(stimuli, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != int(n_dims):
        raise ValueError(f"Expected stimuli[n, {n_dims}], got {values.shape}.")
    if values.size and not np.all(np.isfinite(values)):
        raise ValueError("Stimuli must be finite.")
    if values.size and (values.min() < 0.0 or values.max() > 1.0):
        raise ValueError(
            "Stimuli must lie inside the unit cube [0, 1]; got values in "
            f"[{values.min()}, {values.max()}]."
        )
    return values

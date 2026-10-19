"""A fixed-geometry prototype with independent region/response associations.

Initialize each region uniformly, predict before feedback, then take one
gradient step on the observed binary-feedback event. This class supplies the
association update only. H search, capacity, rule entry/re-entry, geometry
precision learning and condition-3 pairing are not integrated. Existing model
configuration and production behavior remain unchanged.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from ..hypothesis_space.observation_model.region_response import (
    region_response_feedback_gradient, region_response_probabilities,
)


class RegionResponseLearner:
    """Maintain one response-association row per fixed geometrical region."""

    def __init__(self, n_regions: int, n_responses: int, *, learning_rate: float) -> None:
        if (type(n_regions) is not int or type(n_responses) is not int
                or n_regions < 1 or n_responses < 2
                or not np.isfinite(learning_rate) or learning_rate < 0):
            raise ValueError('Need region/response counts and a finite nonnegative learning rate')
        self.learning_rate = float(learning_rate)
        self._logits = np.zeros((n_regions, n_responses), dtype=float)

    @property
    def associations(self) -> np.ndarray:
        """Return an independent copy of the normalized association rows."""
        return np.exp(self._logits - logsumexp(self._logits, axis=1, keepdims=True))

    def predict(self, geometry: np.ndarray) -> np.ndarray:
        """Use the state from previous feedback; prediction does not mutate it."""
        return region_response_probabilities(geometry, self._logits)

    def update(self, geometry: np.ndarray, choice: int, feedback: float) -> float:
        """Update from observed feedback and return its pre-update log probability."""
        log_event, gradient = region_response_feedback_gradient(geometry, self._logits, choice, feedback)
        candidate = self._logits + self.learning_rate * gradient
        # A common row offset has no effect on response probabilities.
        candidate -= candidate.mean(axis=1, keepdims=True)
        if not np.isfinite(candidate).all():
            raise ValueError('Association update produced nonfinite row logits')
        self._logits = candidate
        return float(log_event)

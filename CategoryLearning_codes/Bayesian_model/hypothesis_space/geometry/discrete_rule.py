"""Exact parity-rule evaluation over binary stimuli."""

from __future__ import annotations

from itertools import product
from typing import Iterable

import numpy as np

from ..spaces import DiscreteHypothesisSpace


class DiscreteRuleGeometry:
    """Evaluate fixed-label parity rules and convert predictions to probabilities."""

    def __init__(self, space: DiscreteHypothesisSpace) -> None:
        self.space = space
        self.stimulus_space = np.asarray(
            list(product([-1, 1], repeat=space.n_dims)),
            dtype=int,
        )
        self.prediction_table = np.asarray(
            [
                self.category_assignments(hypothesis.index, self.stimulus_space)
                for hypothesis in space
            ],
            dtype=int,
        )
        self._similarity_matrix: np.ndarray | None = None

    @property
    def similarity_matrix(self) -> np.ndarray:
        if self._similarity_matrix is None:
            predictions = self.prediction_table
            self._similarity_matrix = (
                predictions[:, None, :] == predictions[None, :, :]
            ).mean(axis=2)
        return self._similarity_matrix

    def coerce_stimuli(self, stimuli: np.ndarray | Iterable) -> np.ndarray:
        """Map positive values to ``1`` and all other values to ``-1``."""
        values = np.asarray(stimuli, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != self.space.n_dims:
            raise ValueError(
                f"Expected stimuli[n, {self.space.n_dims}], got {values.shape}."
            )
        return np.where(values > 0, 1, -1).astype(int)

    def category_assignments(
        self,
        hypothesis_index: int,
        stimuli: np.ndarray | Iterable,
    ) -> np.ndarray:
        """Return zero-based hard category labels for one rule."""
        values = self.coerce_stimuli(stimuli)
        hypothesis = self.space[int(hypothesis_index)]
        if not hypothesis.dims:
            signed_value = np.full(
                values.shape[0],
                hypothesis.polarity,
                dtype=int,
            )
        else:
            signed_value = hypothesis.polarity * np.prod(
                values[:, list(hypothesis.dims)],
                axis=1,
            )
        return np.where(signed_value > 0, 0, 1)

    def category_probabilities(
        self,
        hypothesis_index: int,
        stimuli: np.ndarray | Iterable,
        beta: float,
    ) -> np.ndarray:
        """Return ``[n_categories, n_trials]`` soft rule predictions."""
        values = self.coerce_stimuli(stimuli)
        predicted = self.category_assignments(hypothesis_index, values)
        scores = np.zeros((values.shape[0], self.space.n_cats), dtype=float)
        scores[np.arange(values.shape[0]), predicted] = float(beta)
        scores -= np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return (exp_scores / np.sum(exp_scores, axis=1, keepdims=True)).T


__all__ = ["DiscreteRuleGeometry"]

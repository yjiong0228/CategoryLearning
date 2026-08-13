"""Model-facing continuous partition over one space and two geometries.

The actual continuous space is :class:`ContinuousHypothesisSpace`.
Prototype and boundary implementations consume that same object.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..spaces import ContinuousHypothesisSpace, build_continuous_hypothesis_space
from ..geometry import BoundaryGeometry, PrototypeGeometry
from .base_partition import BasePartition
from ..similarity import ContinuousSimilarity, prototype_boundary_agreement


class ContinuousPartition(BasePartition):
    """Join one continuous space to prototype and boundary geometry."""

    EPS = 1e-7
    DISTANCE_MODE_PROTOTYPE = "prototype"
    DISTANCE_MODE_BOUNDARY = "boundary"
    DEFAULT_DISTANCE_MODE = DISTANCE_MODE_PROTOTYPE
    VALID_DISTANCE_MODES = (DISTANCE_MODE_PROTOTYPE, DISTANCE_MODE_BOUNDARY)

    def __init__(
        self,
        n_dims: int,
        n_cats: int,
        pairwise_similarity_tolerance: float = 0.10,
        center_band_tolerance: float = 0.10,
        prototype_method: str = PrototypeGeometry.METHOD_COMPONENT_VOLUME_CENTROID,
        similarity_n_samples: int = ContinuousSimilarity.DEFAULT_N_SAMPLES,
        similarity_cache_dir: str | Path | None = None,
    ) -> None:
        pair_tolerance = float(pairwise_similarity_tolerance)
        center_tolerance = float(center_band_tolerance)
        self.hypothesis_space: ContinuousHypothesisSpace = (
            build_continuous_hypothesis_space(
                n_dims,
                n_cats,
                pairwise_similarity_tolerance=pair_tolerance,
                center_band_tolerance=center_tolerance,
            )
        )
        super().__init__(n_dims, n_cats)
        self.pairwise_similarity_tolerance = pair_tolerance
        self.center_band_tolerance = center_tolerance

        self.boundary_geometry = BoundaryGeometry(self.hypothesis_space)
        self.prototype_geometry = PrototypeGeometry(
            self.hypothesis_space,
            method=prototype_method,
        )
        self.prototype_method = self.prototype_geometry.method
        self.connectivity_map = self._compute_connectivity_map()
        self.similarity = ContinuousSimilarity(
            self.hypothesis_space,
            self.boundary_geometry,
            n_samples=similarity_n_samples,
            runtime_cache_dir=similarity_cache_dir,
        )

    @property
    def length(self) -> int:
        return len(self.hypothesis_space)

    @property
    def similarity_matrix(self) -> np.ndarray:
        return self.similarity.matrix

    def get_category_prototypes(self, hypo: int, category: int) -> np.ndarray:
        """Return all automatically derived prototypes for one category."""
        return self.prototype_geometry.get_category_prototypes(hypo, category)

    def get_category_probabilities(
        self,
        hypo: int,
        data: list | tuple,
        beta: float,
        distance_mode: str | None = None,
        **kwargs,
    ) -> np.ndarray:
        mode = self._resolve_distance_mode(distance_mode)
        if mode == self.DISTANCE_MODE_PROTOTYPE:
            return self.prototype_geometry.category_probabilities(hypo, data[0], beta)
        return self.boundary_geometry.category_probabilities(hypo, data[0], beta)

    def prototype_boundary_agreement(
        self,
        *,
        n_samples: int = 10000,
        random_state: int = 0,
    ) -> np.ndarray:
        """Return per-hypothesis hard-label agreement between the two modes."""
        return prototype_boundary_agreement(
            self.prototype_geometry,
            self.boundary_geometry,
            n_samples=n_samples,
            random_state=random_state,
        )

    def _compute_connectivity_map(self) -> dict[int, dict[int, list[int]]]:
        return {
            hypothesis.index: {
                category: list(neighbors)
                for category, neighbors in enumerate(hypothesis.feedback_neighbors)
            }
            for hypothesis in self.hypothesis_space
        }

    def _category_feedback_likelihood(
        self,
        hypo: int,
        prob: np.ndarray,
        choices: np.ndarray,
        responses: np.ndarray,
    ) -> np.ndarray:
        """Map Task2's correct/related/incorrect codes to probabilities."""
        n_trials = len(choices)
        p_category = prob[choices, np.arange(n_trials)]
        family_probability = np.zeros(n_trials)
        mask = np.zeros_like(prob, dtype=bool)
        for trial_index in range(n_trials):
            alternatives = self.connectivity_map[hypo][choices[trial_index]]
            mask[alternatives, trial_index] = True
        family_probability = (prob * mask).sum(axis=0)
        return np.where(
            responses == 1,
            p_category,
            np.where(responses == 0.5, family_probability, 1.0 - p_category),
        )


__all__ = [
    "ContinuousPartition",
]

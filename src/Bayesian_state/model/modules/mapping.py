"""Minimal binary category-orientation learning for mapping sensitivity tests.

The module keeps geometry identity separate from the arbitrary task labels on
the two sides of that geometry.  It analytically marginalizes the two possible
orientations and therefore adds latent state but no sampling dimension or free
parameter.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .base_module import BaseModule, ModulePhase, ModuleRole


class BinaryOrientationMappingModule(BaseModule):
    """Rao--Blackwellized belief over the two labels of each binary geometry.

    ``orientation_probability[h]`` is the probability that region/category 1
    under the fixed-label geometry ``h`` maps to observed category A.  The
    complementary orientation is obtained by reversing the two category
    probabilities.  New workspace hypotheses restart from the neutral prior.
    """

    phase = ModulePhase.POST_CHOICE
    role = ModuleRole.MAPPING

    def __init__(
        self,
        engine: Any,
        *,
        initial_probability: float = 0.5,
        numerical_floor: float = 1e-12,
    ) -> None:
        super().__init__(engine)
        n_categories = int(getattr(engine.partition, "n_cats", 0))
        if n_categories != 2:
            raise ValueError(
                "BinaryOrientationMappingModule requires exactly two categories."
            )
        initial = float(initial_probability)
        if not np.isfinite(initial) or not 0.0 < initial < 1.0:
            raise ValueError("initial_probability must lie strictly between 0 and 1.")
        floor = float(numerical_floor)
        if not np.isfinite(floor) or not 0.0 < floor < 0.5:
            raise ValueError("numerical_floor must lie strictly between 0 and 0.5.")

        self.initial_probability = initial
        self.numerical_floor = floor
        self.orientation_probability = np.full(
            int(engine.set_size), initial, dtype=float
        )
        # Beta updating occurs after this module.  It must use the orientation
        # that generated the current choice, not the just-updated posterior.
        self.predictive_orientation_probability = self.orientation_probability.copy()
        self.orientation_log: list[np.ndarray] = []

    @staticmethod
    def _normalize_binary(probabilities: Sequence[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(probabilities, dtype=float).reshape(-1)
        if values.shape != (2,) or not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("binary category probabilities must be finite and non-negative.")
        total = float(np.sum(values))
        if total <= 0.0:
            raise ValueError("binary category probabilities have zero mass.")
        return values / total

    def initialize_orientation_for_hypotheses(
        self,
        indices: Sequence[int] | np.ndarray,
    ) -> None:
        """Reset newly proposed workspace hypotheses to the neutral prior."""

        resolved = np.asarray(indices, dtype=int).reshape(-1)
        if resolved.size == 0:
            return
        if np.any((resolved < 0) | (resolved >= self.orientation_probability.size)):
            raise ValueError("orientation initialization index falls outside the hypothesis space.")
        self.orientation_probability[resolved] = self.initial_probability
        self.predictive_orientation_probability[resolved] = self.initial_probability

    def condition_on_orientation_probability(
        self,
        probabilities: Sequence[float] | np.ndarray,
    ) -> None:
        """Clamp the current belief vector for an explicit oracle diagnostic.

        This method is deliberately separate from ordinary learning.  The
        particle filter uses it only when a caller supplies a full
        trial-by-hypothesis truth schedule for an identifiability check.
        Setting both arrays makes the supplied state the pre-feedback belief
        for the current trial and for the downstream beta update.
        """

        values = np.asarray(probabilities, dtype=float).reshape(-1)
        expected = (int(self.engine.set_size),)
        if values.shape != expected:
            raise ValueError("conditioned orientation state has the wrong shape.")
        if not np.all(np.isfinite(values)) or np.any(
            (values < self.numerical_floor)
            | (values > 1.0 - self.numerical_floor)
        ):
            raise ValueError(
                "conditioned orientation probabilities fall outside the "
                "module's numerical bounds."
            )
        self.orientation_probability = values.copy()
        self.predictive_orientation_probability = values.copy()

    def orient_category_probabilities(
        self,
        hypothesis: int,
        probabilities: Sequence[float] | np.ndarray,
        *,
        use_predictive_snapshot: bool = False,
    ) -> np.ndarray:
        """Marginalize a fixed-label binary emission over orientation belief."""

        fixed = self._normalize_binary(probabilities)
        index = int(hypothesis)
        source = (
            self.predictive_orientation_probability
            if use_predictive_snapshot
            else self.orientation_probability
        )
        if not 0 <= index < source.size:
            raise ValueError("hypothesis index falls outside the orientation state.")
        mapping_probability = float(source[index])
        return self._normalize_binary(
            mapping_probability * fixed
            + (1.0 - mapping_probability) * fixed[::-1]
        )

    def process(self, **kwargs: Any) -> None:
        """Marginalize geometry evidence and update active orientation beliefs."""

        del kwargs
        fixed_likelihood = np.asarray(self.engine.likelihood, dtype=float).reshape(-1)
        if fixed_likelihood.shape != self.orientation_probability.shape:
            raise ValueError("likelihood width does not match orientation state.")
        if not np.all(np.isfinite(fixed_likelihood)) or np.any(
            (fixed_likelihood < 0.0) | (fixed_likelihood > 1.0)
        ):
            raise ValueError("binary feedback likelihood must lie in [0, 1].")

        mask = getattr(self.engine, "hypotheses_mask", None)
        if mask is None:
            active = np.ones_like(fixed_likelihood, dtype=bool)
        else:
            active = np.asarray(mask, dtype=float).reshape(-1) > 0.0
        if active.shape != fixed_likelihood.shape or not np.any(active):
            raise ValueError("orientation learning requires a non-empty active mask.")

        predictive = self.orientation_probability.copy()
        reverse_likelihood = 1.0 - fixed_likelihood
        marginal = (
            predictive * fixed_likelihood
            + (1.0 - predictive) * reverse_likelihood
        )
        marginal = np.clip(marginal, self.numerical_floor, 1.0)
        updated = predictive.copy()
        updated[active] = (
            predictive[active] * fixed_likelihood[active] / marginal[active]
        )
        updated[active] = np.clip(
            updated[active], self.numerical_floor, 1.0 - self.numerical_floor
        )

        self.predictive_orientation_probability = predictive
        self.orientation_probability = updated
        self.engine.likelihood = marginal
        self.orientation_log.append(updated.copy())

    def state_dict(self) -> dict[str, Any]:
        return {
            "orientation_probability": self.orientation_probability.copy(),
            "predictive_orientation_probability": (
                self.predictive_orientation_probability.copy()
            ),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        current = np.asarray(state["orientation_probability"], dtype=float).reshape(-1)
        predictive = np.asarray(
            state["predictive_orientation_probability"], dtype=float
        ).reshape(-1)
        expected = (int(self.engine.set_size),)
        if current.shape != expected or predictive.shape != expected:
            raise ValueError("restored orientation state has the wrong shape.")
        if not np.all(np.isfinite(current)) or np.any((current <= 0.0) | (current >= 1.0)):
            raise ValueError("restored orientation probabilities must lie in (0, 1).")
        if not np.all(np.isfinite(predictive)) or np.any(
            (predictive <= 0.0) | (predictive >= 1.0)
        ):
            raise ValueError("restored predictive orientation must lie in (0, 1).")
        self.orientation_probability = current.copy()
        self.predictive_orientation_probability = predictive.copy()

    def clear_logs(self) -> None:
        self.orientation_log.clear()


__all__ = ["BinaryOrientationMappingModule"]

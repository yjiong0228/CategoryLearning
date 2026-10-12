"""Observation likelihood used by the Bayesian inference lifecycle.

This object is deliberately not an inference module.  It is a stateless
adapter from one model observation to the likelihood vector over the current
hypothesis space; the inference engine decides when that mandatory calculation
is executed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .pairing import PAIRING_SIBLINGS, pairing_feedback_kernel


class ObservationLikelihood:
    """Process ``p(observation | hypothesis)`` through a runtime partition."""

    DEFAULT_BETA = 10.0
    BETA_SOURCE_ACTION = "action"
    BETA_SOURCE_FIXED = "fixed"
    VALID_BETA_SOURCES = (BETA_SOURCE_ACTION, BETA_SOURCE_FIXED)

    def __init__(
        self,
        partition: Any,
        *,
        distance_mode: str | None = None,
        default_beta: float = DEFAULT_BETA,
        beta_source: str = BETA_SOURCE_ACTION,
        feedback_likelihood_mode: str = "category_feedback",
        feedback_lapse: float = 0.0,
    ) -> None:
        if partition is None or not callable(
            getattr(partition, "calc_likelihood", None)
        ):
            raise TypeError("partition must provide calc_likelihood().")

        valid_modes = tuple(getattr(partition, "VALID_DISTANCE_MODES", ()))
        resolved_mode = distance_mode
        if resolved_mode is None and len(valid_modes) == 1:
            resolved_mode = valid_modes[0]
        if resolved_mode is None:
            raise ValueError(
                "distance_mode must be explicitly configured when a partition "
                "supports multiple encoding modes."
            )
        resolved_mode = str(resolved_mode)
        if valid_modes and resolved_mode not in valid_modes:
            raise ValueError(
                f"Unsupported distance_mode '{resolved_mode}'. "
                f"Expected one of: {valid_modes}."
            )

        beta_value = float(default_beta)
        if not np.isfinite(beta_value):
            raise ValueError("default_beta must be finite.")

        resolved_beta_source = str(beta_source).strip().lower()
        beta_source_aliases = {
            "choice": self.BETA_SOURCE_ACTION,
            "dynamic": self.BETA_SOURCE_ACTION,
            "dynamic_action": self.BETA_SOURCE_ACTION,
            "evidence": self.BETA_SOURCE_FIXED,
            "fixed_evidence": self.BETA_SOURCE_FIXED,
        }
        resolved_beta_source = beta_source_aliases.get(
            resolved_beta_source,
            resolved_beta_source,
        )
        if resolved_beta_source not in self.VALID_BETA_SOURCES:
            raise ValueError(
                f"Unsupported beta_source '{beta_source}'. Expected one of: "
                f"{self.VALID_BETA_SOURCES}."
            )

        lapse_value = float(feedback_lapse)
        if not np.isfinite(lapse_value) or lapse_value < 0.0 or lapse_value >= 1.0:
            raise ValueError(
                f"feedback_lapse must be in [0, 1), got {feedback_lapse!r}."
            )

        feedback_mode = str(feedback_likelihood_mode)
        resolve_feedback_mode = getattr(
            partition,
            "_resolve_feedback_likelihood_mode",
            None,
        )
        if feedback_mode == "hierarchical_pairing":
            if int(getattr(partition, "n_cats", 0)) != 4:
                raise ValueError("hierarchical_pairing requires four response categories.")
            if lapse_value != 0.:
                raise ValueError("hierarchical_pairing currently requires feedback_lapse=0.")
        elif callable(resolve_feedback_mode):
            feedback_mode = resolve_feedback_mode(feedback_mode)

        self.partition = partition
        self.distance_mode = resolved_mode
        self.default_beta = beta_value
        self.beta_source = resolved_beta_source
        self.feedback_likelihood_mode = feedback_mode
        self.feedback_lapse = lapse_value

    def process(
        self,
        observation: Sequence[Any] | None,
        hypotheses: Sequence[int],
        beta: float | Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the one-dimensional likelihood vector for one completed trial."""

        if self.feedback_likelihood_mode == "hierarchical_pairing":
            raise ValueError("hierarchical_pairing requires joint memory; use pairing_kernel().")

        if observation is None or len(observation) != 3:
            raise ValueError(
                "observation must be (stimulus, one-indexed choice, feedback)."
            )
        stimulus, choice, feedback = observation
        if choice is None or feedback is None:
            raise ValueError(
                "choice and feedback must be available before likelihood evaluation."
            )

        hypothesis_args = tuple(int(hypothesis) for hypothesis in hypotheses)
        if not hypothesis_args:
            raise ValueError("at least one hypothesis is required.")

        # ``fixed`` gives rule evidence its own stationary scale.  The engine
        # may still pass the action-policy beta vector, but it is deliberately
        # ignored so changes in response confidence cannot also sharpen the
        # evidence used to rank hypotheses.  ``action`` preserves the legacy
        # coupled behavior for existing configurations.
        beta_value = (
            self.default_beta
            if self.beta_source == self.BETA_SOURCE_FIXED or beta is None
            else beta
        )
        single_trial_data = ([stimulus], [choice], [feedback])
        matrix = np.asarray(
            self.partition.calc_likelihood(
                hypos=hypothesis_args,
                data=single_trial_data,
                beta=beta_value,
                distance_mode=self.distance_mode,
                normalized=True,
                feedback_likelihood_mode=self.feedback_likelihood_mode,
                feedback_lapse=self.feedback_lapse,
            ),
            dtype=float,
        )
        expected_shape = (1, len(hypothesis_args))
        if matrix.shape != expected_shape:
            raise ValueError(
                "partition.calc_likelihood() returned shape "
                f"{matrix.shape}; expected {expected_shape}."
            )
        likelihood = matrix[0].copy()
        if not np.all(np.isfinite(likelihood)) or np.any(likelihood < 0.0):
            raise ValueError("likelihood must contain finite, non-negative values.")
        if float(np.sum(likelihood)) <= 0.0:
            raise ValueError("likelihood must contain positive total mass.")
        return likelihood

    def pairing_kernel(
        self, observation: Sequence[Any], hypotheses: Sequence[int],
        beta: float | Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute stateless absolute feedback probabilities for three pairings."""
        if self.feedback_likelihood_mode != "hierarchical_pairing":
            raise ValueError("pairing_kernel requires hierarchical_pairing mode.")
        if observation is None or len(observation) != 3:
            raise ValueError("observation must contain stimulus, choice and feedback.")
        stimulus, choice, feedback = observation
        beta_value = (self.default_beta if beta is None or self.beta_source == self.BETA_SOURCE_FIXED else beta)
        beta_values = np.broadcast_to(np.asarray(beta_value, dtype=float), (len(hypotheses),))
        data = ([stimulus], [choice], [feedback])
        probabilities = np.asarray([
            self.partition.get_category_probabilities(
                int(h), data, float(b), distance_mode=self.distance_mode,
            )[:, 0]
            for h, b in zip(hypotheses, beta_values)
        ])
        return pairing_feedback_kernel(probabilities, choice, feedback)

    def binary_feedback_likelihood(
        self,
        observation: Sequence[Any],
        hypotheses: Sequence[int],
        beta: float | Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Return absolute binary-feedback probabilities without an EPS floor.

        The 0923 gradient profile shares these emissions with precision learning.
        The historical ``process`` path keeps its existing partition clipping.
        """
        if self.feedback_likelihood_mode != "category_feedback" or self.feedback_lapse != 0.:
            raise ValueError("exact binary feedback requires category_feedback and feedback_lapse=0.")
        if observation is None or len(observation) != 3:
            raise ValueError("observation must contain stimulus, choice and feedback.")
        stimulus, choice, feedback = observation
        n_categories = int(self.partition.n_cats)
        if (choice is None or not np.isfinite(choice) or choice != int(choice)
                or not 1 <= int(choice) <= n_categories):
            raise ValueError("choice must be an integer within the category range.")
        if feedback not in (0., 1.):
            raise ValueError("binary feedback must be 0 or 1.")
        indices = tuple(int(h) for h in hypotheses)
        if not indices:
            raise ValueError("at least one hypothesis is required.")
        beta_value = self.default_beta if beta is None or self.beta_source == self.BETA_SOURCE_FIXED else beta
        betas = np.broadcast_to(np.asarray(beta_value, dtype=float), (len(indices),))
        if not np.all(np.isfinite(betas)) or np.any(betas < 0.):
            raise ValueError("predictive precisions must be finite and non-negative.")
        data = ([stimulus], [choice], [feedback])
        probabilities = np.asarray([
            self.partition.get_category_probabilities(h, data, float(b), distance_mode=self.distance_mode)[:, 0]
            for h, b in zip(indices, betas)
        ], dtype=float)
        if (probabilities.shape != (len(indices), n_categories)
                or not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.)
                or not np.allclose(probabilities.sum(axis=1), 1., rtol=1e-10, atol=1e-12)):
            raise ValueError("predictive category probabilities must be finite and normalized.")
        if feedback == 1.:
            return probabilities[:, int(choice) - 1].copy()
        # A sum retains tiny alternatives even when p(choice) rounds to one.
        return probabilities[:, np.arange(n_categories) != int(choice) - 1].sum(axis=1)

    def feedback_precision_gradient(
        self,
        observation: Sequence[Any],
        hypotheses: Sequence[int],
        beta: float | Sequence[float] | np.ndarray,
        *,
        pairing_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the per-rule log-feedback derivative at predictive precision.

        Distances, the perceived stimulus and pairing weights stay fixed during
        differentiation. Callers pass the pre-feedback pairing distribution;
        memory may already have updated its joint posterior. The geometry's
        exact distance cache reuses the distances used for the choice emission.
        """
        if (self.beta_source != self.BETA_SOURCE_ACTION
                or self.distance_mode != "boundary" or self.feedback_lapse != 0.):
            raise ValueError(
                "feedback_gradient requires boundary/action emissions and feedback_lapse=0."
            )
        if self.feedback_likelihood_mode not in ("category_feedback", "hierarchical_pairing"):
            raise ValueError("feedback_gradient requires binary or hierarchical category feedback.")
        if observation is None or len(observation) != 3:
            raise ValueError("observation must contain stimulus, choice and feedback.")
        stimulus, choice, feedback = observation
        n_categories = int(self.partition.n_cats)
        if (choice is None or not np.isfinite(choice) or choice != int(choice)
                or not 1 <= int(choice) <= n_categories):
            raise ValueError("choice must be an integer within the category range.")
        values = np.asarray(stimulus, dtype=float)
        if (values.shape != (4,) or not np.all(np.isfinite(values))
                or np.any(values < 0.) or np.any(values > 1.)):
            raise ValueError("feedback_gradient requires a finite stimulus in [0, 1]^4.")
        indices = tuple(int(h) for h in hypotheses)
        if not indices:
            raise ValueError("feedback_gradient requires at least one active hypothesis.")
        betas = np.broadcast_to(np.asarray(beta, dtype=float), (len(indices),))
        if not np.all(np.isfinite(betas)) or np.any(betas <= 0.):
            raise ValueError("feedback_gradient requires finite positive predictive precisions.")
        geometry = getattr(self.partition, "boundary_geometry", None)
        if not callable(getattr(geometry, "category_distances", None)):
            raise TypeError("feedback_gradient requires boundary category distances.")
        data = ([values], [choice], [feedback])
        distances = np.asarray([
            geometry.category_distances(h, [values])[0] for h in indices
        ], dtype=float)
        probabilities = np.asarray([
            self.partition.get_category_probabilities(
                h, data, float(b), distance_mode=self.distance_mode,
            )[:, 0] for h, b in zip(indices, betas)
        ], dtype=float)
        expected_shape = (len(indices), n_categories)
        if (distances.shape != expected_shape or probabilities.shape != expected_shape
                or not np.all(np.isfinite(distances)) or np.any(distances < 0.)
                or np.any(distances > 2. + 1e-8)):
            raise ValueError("category distances must have shape (rules, categories) and lie in [0, 2].")
        if (not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.)
                or not np.allclose(probabilities.sum(axis=1), 1., rtol=1e-10, atol=1e-12)):
            raise ValueError("predictive category probabilities must be finite and normalized.")

        chosen = int(choice) - 1
        if self.feedback_likelihood_mode == "hierarchical_pairing":
            if pairing_weights is None:
                raise ValueError("hierarchical feedback_gradient requires pre-feedback pairing weights.")
            weights = np.asarray(pairing_weights, dtype=float)
            if (weights.shape != (len(indices), 3) or not np.all(np.isfinite(weights))
                    or np.any(weights < 0.)
                    or not np.allclose(weights.sum(axis=1), 1., rtol=1e-10, atol=1e-12)):
                raise ValueError("pairing weights must be normalized with shape (rules, 3).")
            if feedback not in (0., .5, 1.):
                raise ValueError("hierarchical feedback must be 0, 0.5 or 1.")
            categories = np.arange(n_categories)[None, :]
            siblings = PAIRING_SIBLINGS[:, chosen, None]
            if feedback == 1.:
                compatible = np.broadcast_to(categories == chosen, (3, n_categories))
            elif feedback == .5:
                compatible = categories == siblings
            else:
                compatible = (categories != chosen) & (categories != siblings)
            compatibility = weights @ compatible
        else:
            if pairing_weights is not None:
                raise ValueError("binary feedback_gradient does not use pairing weights.")
            if feedback not in (0., 1.):
                raise ValueError("binary feedback must be 0 or 1.")
            compatible = np.arange(n_categories) == chosen
            if feedback == 0.:
                compatible = ~compatible
            compatibility = np.broadcast_to(compatible, expected_shape)

        joint = probabilities * compatibility
        evidence = joint.sum(axis=1, keepdims=True)
        if np.any(evidence <= 0.):
            raise ValueError("feedback_gradient requires positive feedback probability for each active rule.")
        # Direct sums retain tiny error-event mass that 1 - p(choice) can lose.
        conditioned = joint / evidence
        return np.sum(probabilities * distances, axis=1) - np.sum(conditioned * distances, axis=1)


__all__ = ["ObservationLikelihood"]

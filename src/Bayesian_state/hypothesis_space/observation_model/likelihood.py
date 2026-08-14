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
        if resolved_mode is None:
            resolved_mode = getattr(partition, "DEFAULT_DISTANCE_MODE", None)
        if resolved_mode is None and len(valid_modes) == 1:
            resolved_mode = valid_modes[0]
        if resolved_mode is None:
            raise ValueError("distance_mode is required for this partition.")
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
        if callable(resolve_feedback_mode):
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


__all__ = ["ObservationLikelihood"]

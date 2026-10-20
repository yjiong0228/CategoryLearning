"""Per-particle association memory for a finite active rule set.

Callers supply the currently active rule IDs. Evaluation reads the old state;
commit changes only those rules. Absent rules retain their last associations,
and unseen rules have uniform rows. Candidate recall and H selection are owned
by their respective modules; evaluating a candidate here never trains it.

The two-phase API lets choice evidence, rule belief and precision consume the
same pre-feedback associations before an association update is committed.
Snapshots are independent copies and include counts used to reject stale or
repeated feedback commits, including zero-learning-rate updates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.special import logsumexp

from ..hypothesis_space.observation_model.region_precision import region_response_feedback_derivatives
from ..hypothesis_space.observation_model.region_response import (
    region_response_feedback_gradient, region_response_probabilities,
)


def _readonly(values: np.ndarray) -> np.ndarray:
    """Use an immutable buffer so a caller cannot re-enable array writes."""
    values = np.asarray(values)
    return np.frombuffer(values.tobytes(), dtype=values.dtype).reshape(values.shape)


@dataclass(frozen=True)
class AssociationFeedback:
    """One state's uncommitted feedback evidence; arrays cannot be edited."""
    rules: np.ndarray
    log_event: np.ndarray
    association_gradient: np.ndarray
    precision_gradient: np.ndarray | None
    _owner: object = field(repr=False, compare=False)
    _generation: object = field(repr=False, compare=False)
    _old_logits: np.ndarray = field(repr=False, compare=False)
    _old_counts: np.ndarray = field(repr=False, compare=False)


class ActiveRuleAssociations:
    """Store independent response rows by stable catalogue rule identity."""

    VERSION = 'active-region-associations-v1'

    def __init__(self, n_rules: int, n_regions: int, n_responses: int, *, learning_rate: float) -> None:
        if (any(type(x) is not int for x in (n_rules, n_regions, n_responses))
                or n_rules < 1 or n_regions < 1 or n_responses < 2
                or not np.isfinite(learning_rate) or learning_rate < 0):
            raise ValueError('Require positive dimensions and a finite nonnegative learning rate')
        self.learning_rate = float(learning_rate)
        self._logits = np.zeros((n_rules, n_regions, n_responses))
        self._counts = np.zeros(n_rules, dtype=np.int64)
        self._generation = object()

    def _rules(self, rules: Sequence[int]) -> np.ndarray:
        indices = np.asarray(rules)
        if (indices.ndim != 1 or indices.size == 0
                or not np.issubdtype(indices.dtype, np.integer)
                or np.unique(indices).size != indices.size
                or np.any(indices < 0) or np.any(indices >= len(self._logits))):
            raise ValueError('Require distinct catalogue rule IDs in range')
        return indices.astype(np.int64, copy=True)

    def associations(self, rules: Sequence[int]) -> np.ndarray:
        """Read any rules, including absent candidates, without learning."""
        logits = self._logits[self._rules(rules)]
        return np.exp(logits - logsumexp(logits, axis=-1, keepdims=True))

    def predict(self, rules: Sequence[int], geometry: np.ndarray) -> np.ndarray:
        """Return each requested rule's response vector before new feedback."""
        return region_response_probabilities(geometry, self._logits[self._rules(rules)])

    def evaluate_feedback(
        self, rules: Sequence[int], geometry: np.ndarray, choice: int, feedback: float,
        *, distances: np.ndarray | None = None,
    ) -> AssociationFeedback:
        """Read evidence and gradients at the same state, without updating it.

        Supply region distances when beta learns from feedback. Geometry then
        must use the corresponding pre-feedback beta and distance softmax.
        """
        indices = self._rules(rules)
        logits = self._logits[indices]
        if distances is None:
            log_event, gradient = region_response_feedback_gradient(geometry, logits, choice, feedback)
            precision = None
        else:
            log_event, gradient, precision = region_response_feedback_derivatives(
                geometry, logits, distances, choice, feedback)
        return AssociationFeedback(
            _readonly(indices), _readonly(log_event), _readonly(gradient),
            None if precision is None else _readonly(precision), self, self._generation,
            _readonly(logits), _readonly(self._counts[indices]),
        )

    def commit_feedback(self, evidence: AssociationFeedback) -> None:
        """Apply one evaluated update; reject another particle's or stale state."""
        if (not isinstance(evidence, AssociationFeedback) or evidence._owner is not self
                or evidence._generation is not self._generation):
            raise ValueError('Feedback belongs to another association state')
        indices = evidence.rules
        if (not np.array_equal(self._counts[indices], evidence._old_counts)
                or not np.array_equal(self._logits[indices], evidence._old_logits)):
            raise ValueError('Feedback is stale or has already been committed')
        if np.any(self._counts[indices] == np.iinfo(np.int64).max):
            raise ValueError('Association feedback counts would overflow')
        candidate = evidence._old_logits + self.learning_rate * evidence.association_gradient
        candidate -= candidate.mean(axis=-1, keepdims=True)
        if not np.isfinite(candidate).all():
            raise ValueError('Association update produced invalid logits')
        self._logits[indices] = candidate
        self._counts[indices] += 1

    def state_dict(self) -> dict[str, Any]:
        """Export a self-contained state for particle cloning or checkpointing."""
        return dict(version=self.VERSION, learning_rate=self.learning_rate,
                    logits=self._logits.copy(), feedback_counts=self._counts.copy())

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Validate the entire snapshot before changing any live state."""
        if (set(state) != {'version', 'learning_rate', 'logits', 'feedback_counts'}
                or state['version'] != self.VERSION or state['learning_rate'] != self.learning_rate):
            raise ValueError('Association snapshot configuration differs')
        logits, counts = np.asarray(state['logits']), np.asarray(state['feedback_counts'])
        if (logits.shape != self._logits.shape or not np.issubdtype(logits.dtype, np.floating)
                or not np.isfinite(logits).all() or counts.shape != self._counts.shape
                or not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0)
                or np.any(counts > np.iinfo(np.int64).max)
                or np.any(logits[counts == 0] != 0)):
            raise ValueError('Invalid association snapshot arrays')
        self._logits = logits.astype(float, copy=True)
        self._counts = counts.astype(np.int64, copy=True)
        self._generation = object()

"""Controlled fixed-catalogue learner for response-association diagnostics.

All geometries are available throughout a sequence. Rule beliefs use the shared
0923 exact-power memory module; the conditional association/pairing distribution
uses the shared association kernel. Prediction precedes the current choice and
feedback. This diagnostic has no H search, capacity, perception noise, adaptive
precision or oral readout; it is not an installed StateModel variant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .modules.memory import DualMemoryModule
from ..hypothesis_space.observation_model.response_association import (
    association_choice_probabilities, association_feedback_kernel, update_associations,
)


@dataclass
class _MemoryContext:
    """The existing memory module's engine fields for an always-active catalogue."""

    set_size: int
    prior: np.ndarray
    hypotheses_mask: np.ndarray
    posterior: np.ndarray = field(init=False)
    state: dict[str, Any] = field(default_factory=dict)


class FixedCatalogueAssociationLearner:
    """Learn rule and association beliefs from actual choice/feedback pairs.

    Each geometry receives equal initial mass, regardless of the number of
    allowed label associations. ``learn_associations=False`` fixes the identity
    correspondence. Both modes retain three unknown response-family pairings
    for hierarchical feedback. Retention acts at post-feedback update, matching
    the existing exact-power memory timing; pre-choice prediction uses the
    preceding posterior. Rates are explicit experimental settings.
    """

    def __init__(
        self, n_rules: int, *, rule_retention: float, association_retention: float,
        learn_associations: bool, hierarchical: bool,
    ) -> None:
        if type(n_rules) is not int or n_rules < 1:
            raise ValueError('n_rules must be a positive integer.')
        if type(learn_associations) is not bool or type(hierarchical) is not bool:
            raise ValueError('Association and feedback modes must be Boolean.')
        if not np.isfinite(association_retention) or not 0. <= association_retention <= 1.:
            raise ValueError('Association retention must lie in [0, 1].')
        self.hierarchical = hierarchical
        self.association_retention = float(association_retention)
        self._allowed = np.ones(24, dtype=bool) if learn_associations else np.arange(24) == 0
        shape = (n_rules, 24, 3 if hierarchical else 1)
        self._associations = np.broadcast_to(self._allowed[None, :, None], shape).astype(float).copy()
        self._associations /= self._associations.sum(axis=(1, 2), keepdims=True)
        self._context = _MemoryContext(n_rules, np.full(n_rules, 1/n_rules), np.ones(n_rules))
        self._memory = DualMemoryModule(self._context, gamma=rule_retention,
            forgetting_mode='exact_power', w0=0., feedback_gain=1.)

    @property
    def rule_weights(self) -> np.ndarray:
        return self._context.prior.copy()

    @property
    def association_weights(self) -> np.ndarray:
        return self._associations.copy()

    def predict(self, geometry_probabilities: np.ndarray) -> np.ndarray:
        """Predict all four response probabilities from preceding feedback only."""
        per_rule = association_choice_probabilities(geometry_probabilities, self._associations)
        return self._context.prior @ per_rule

    def update(self, geometry_probabilities: np.ndarray, choice: int, feedback: float) -> None:
        """Consume one feedback event; update each factor with the same evidence."""
        kernel = association_feedback_kernel(geometry_probabilities, choice, feedback,
                                             hierarchical=self.hierarchical)
        conditional, evidence = update_associations(self._associations, kernel,
            retention=self.association_retention, association_mask=self._allowed)
        # Delegate the geometry marginal to the maintained 0923 memory update.
        # Commit the association state only after the marginal update succeeds.
        self._memory.state_update(evidence)
        self._context.prior = self._context.posterior.copy()
        self._associations = conditional

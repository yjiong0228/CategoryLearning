"""Experimental context-dependent use of learned geometric rules.

All catalogue rules remain available. The maintained catalogue learner owns
association learning and global evidence. LOCAL weights past feedback evidence
by similarity between past and present region-activation vectors. GLOBAL uses
the original learner exactly. This isolates rule use before changing finite-H
proposal, admission or retention. It has no oral input or added free parameter.
"""
from __future__ import annotations

import numpy as np
from scipy.special import softmax

from .region_catalogue_learner import FixedCatalogueRegionLearner


class RegionContextEvidence:
    """Discount completed feedback evidence in normalized region coordinates.

For each rule h, v_h = q_h / ||q_h||_2. Completed feedback with log evidence
ell_h relative to chance contributes v_h * ell_h to a gamma-discounted vector.
The current score is its inner product with the current v_h. Thus an old event
has weight gamma**age times the cosine similarity of its old and current q.
Identical region activations have similarity one, disjoint activations zero.
"""

    def __init__(self, n_rules: int, *, retention: float) -> None:
        if (type(n_rules) is not int or n_rules < 1 or not np.isfinite(retention)
                or not 0 <= retention <= 1):
            raise ValueError('Require a positive rule count and retention in [0,1]')
        self.retention = float(retention)
        self._evidence = np.zeros((n_rules, 4))

    def _features(self, geometry: np.ndarray) -> np.ndarray:
        q = np.asarray(geometry, float)
        if (q.shape != self._evidence.shape or not np.isfinite(q).all() or np.any(q < 0)
                or not np.allclose(q.sum(-1), 1., atol=1e-12, rtol=0)):
            raise ValueError('Require normalized four-region activations for each rule')
        return q / np.linalg.norm(q, axis=-1, keepdims=True)

    def scores(self, geometry: np.ndarray) -> np.ndarray:
        """Read accumulated evidence before any feedback from the current trial."""
        return (self._features(geometry) * self._evidence).sum(-1)

    def update(self, geometry: np.ndarray, log_evidence: np.ndarray) -> None:
        """Add one completed event's pre-update log likelihood relative to chance."""
        features = self._features(geometry)
        evidence = np.asarray(log_evidence, float)
        if evidence.shape != features.shape[:1] or not np.isfinite(evidence).all():
            raise ValueError('Require one finite log-evidence value per rule')
        candidate = self.retention*self._evidence + features*evidence[:, None]
        if not np.isfinite(candidate).all():
            raise ValueError('Context evidence overflow')
        self._evidence = candidate

    @property
    def evidence(self) -> np.ndarray:
        return self._evidence.copy()


class ContextCatalogueRegionLearner:
    """Use global or context-dependent evidence with the same association learner."""

    def __init__(self, n_rules: int, *, rule_retention: float,
                 association_learning_rate: float, mode: str) -> None:
        if mode not in ('GLOBAL', 'LOCAL'):
            raise ValueError('Declare GLOBAL or LOCAL evidence readout')
        if association_learning_rate is None:
            raise ValueError('Context comparison requires learned independent response rows')
        self.mode = mode
        self._base = FixedCatalogueRegionLearner(n_rules, rule_retention=rule_retention,
            association_learning_rate=association_learning_rate)
        self._context = RegionContextEvidence(n_rules, retention=rule_retention)

    @property
    def associations(self) -> np.ndarray:
        return self._base.associations

    def rule_weights(self, geometry: np.ndarray) -> np.ndarray:
        """Return current stimulus-specific rule contributions before feedback."""
        scores = self._context.scores(geometry)
        return self._base.rule_weights if self.mode == 'GLOBAL' else softmax(scores)

    def predict(self, geometry: np.ndarray) -> np.ndarray:
        weights = self.rule_weights(geometry)
        if self.mode == 'GLOBAL':
            return self._base.predict(geometry)
        return weights @ np.einsum('hg,hgc->hc', geometry, self.associations)

    def update(self, geometry: np.ndarray, choice: int, feedback: float) -> None:
        # Capture the likelihood before the shared learner changes A. Computing
        # the local state never conditions it on current oral content.
        self._context._features(geometry)
        if (isinstance(choice, (bool, np.bool_)) or not isinstance(choice, (int, np.integer))
                or not 1 <= choice <= 4 or feedback not in (0., 1.)):
            raise ValueError('Require a one-based response and binary feedback')
        a = self.associations
        probabilities = np.einsum('hg,hgc->hc', geometry, a)
        event = probabilities[:, choice-1] if feedback else np.delete(probabilities, choice-1, axis=1).sum(1)
        chance = .25 if feedback else .75
        evidence = np.log(np.maximum(event, np.finfo(float).tiny))-np.log(chance)
        evidence[np.all(a == .25, axis=(1, 2))] = 0.
        self._base.update(geometry, choice, feedback)
        self._context.update(geometry, evidence)

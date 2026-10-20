"""Controlled fixed-catalogue comparison of fixed labels and independent rows.

Every geometry remains available. Rule beliefs use the maintained exact-power
memory update. Learned rows start uniformly and update from each geometry's
pre-update binary-feedback likelihood gradient. The fixed arm keeps the original
identity labels. This diagnostic has no H search, capacity or perceptual noise.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from scipy.special import logsumexp

from .modules.memory import DualMemoryModule
from ..hypothesis_space.observation_model.region_response import (
    region_response_feedback_gradient, region_response_probabilities,
)


class FixedCatalogueRegionLearner:
    """Separate rule credibility from persistent region-response associations."""

    def __init__(self, n_rules: int, *, rule_retention: float,
                 association_learning_rate: float | None) -> None:
        if type(n_rules) is not int or n_rules < 1:
            raise ValueError('Require a positive rule count')
        if association_learning_rate is not None and (
                not np.isfinite(association_learning_rate) or association_learning_rate < 0):
            raise ValueError('Association rate must be finite and nonnegative, or None for fixed labels')
        self.rate = association_learning_rate
        self._logits = np.zeros((n_rules, 4, 4))
        self._context = SimpleNamespace(set_size=n_rules, prior=np.full(n_rules, 1/n_rules),
                                        hypotheses_mask=np.ones(n_rules), state={})
        self._memory = DualMemoryModule(self._context, gamma=rule_retention,
            forgetting_mode='exact_power', w0=0., feedback_gain=1.)

    @property
    def rule_weights(self) -> np.ndarray:
        return self._context.prior.copy()

    @property
    def associations(self) -> np.ndarray:
        if self.rate is None:
            return np.broadcast_to(np.eye(4), self._logits.shape).copy()
        return np.exp(self._logits - logsumexp(self._logits, axis=-1, keepdims=True))

    def _geometry(self, geometry: np.ndarray) -> np.ndarray:
        q = np.asarray(geometry, float)
        if (q.shape != self._logits.shape[:-1] or not np.isfinite(q).all() or np.any(q < 0)
                or not np.allclose(q.sum(-1), 1., atol=1e-12, rtol=0)):
            raise ValueError('Need normalized probabilities for every declared geometry')
        return q

    def predict(self, geometry: np.ndarray) -> np.ndarray:
        q = self._geometry(geometry)
        per_rule = q if self.rate is None else region_response_probabilities(q, self._logits)
        return self._context.prior @ per_rule

    def update(self, geometry: np.ndarray, choice: int, feedback: float) -> None:
        q = self._geometry(geometry)
        if (isinstance(choice, (bool, np.bool_)) or not isinstance(choice, (int, np.integer))
                or not 1 <= choice <= 4 or feedback not in (0., 1.)):
            raise ValueError('Require a one-based response and binary feedback')
        if self.rate is None:
            evidence = q[:, choice-1] if feedback else np.delete(q, choice-1, axis=1).sum(1)
            candidate = self._logits
        else:
            log_event, gradient = region_response_feedback_gradient(q, self._logits, choice, feedback)
            candidate = self._logits + self.rate * gradient
            candidate -= candidate.mean(-1, keepdims=True)
            if not np.isfinite(candidate).all():
                raise ValueError('Association update produced invalid logits')
            # A shared factor cancels in the normalized rule update. Scaling
            # avoids underflow while retaining the pre-update event evidence.
            evidence = np.exp(log_event - log_event.max())
        self._memory.state_update(evidence)
        self._context.prior = self._context.posterior.copy()
        self._logits = candidate

"""Precision derivative through independent region-to-response associations.

Geometry must be q(g) = softmax(-beta * distance(g)). Associations stay fixed
while differentiating beta. The association gradient and event probability
come from the shared binary-feedback kernel, evaluated at the same old state.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .region_response import _inputs, region_response_feedback_gradient


def region_response_feedback_derivatives(
    geometry: np.ndarray, logits: np.ndarray, distances: np.ndarray,
    choice: int, feedback: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return old log evidence and gradients for association logits and beta.

The beta derivative is the prior mean region distance minus its mean given
the observed feedback event. Incorrect feedback integrates every unchosen
response. Leading axes identify independent rules or batches of rules.
"""
    q, log_association = _inputs(geometry, logits)
    d = np.asarray(distances, dtype=float)
    if d.shape != q.shape or not np.isfinite(d).all() or np.any(d < 0):
        raise ValueError('Distances must be finite, nonnegative and match region probabilities')
    log_event, association_gradient = region_response_feedback_gradient(q, logits, choice, feedback)
    allowed = np.arange(log_association.shape[-1]) == choice - 1
    if feedback == 0.:
        allowed = ~allowed
    log_compatibility = logsumexp(log_association[..., allowed], axis=-1)
    with np.errstate(divide='ignore'):
        log_joint = np.log(q) + log_compatibility
    conditional_regions = np.exp(log_joint - log_event[..., None])
    beta_gradient = np.sum(q*d, axis=-1) - np.sum(conditional_regions*d, axis=-1)
    return log_event, association_gradient, beta_gradient

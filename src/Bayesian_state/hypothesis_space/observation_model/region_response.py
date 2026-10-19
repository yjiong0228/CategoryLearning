"""Feedback gradients for independent region-to-response associations.

Each region has a normalized row of response probabilities. Rows share no
one-to-one assignment constraint. Geometry probabilities are supplied by the
existing observation model. Binary feedback identifies the chosen response or
its complement; neither oral reports nor the true category enter this API.
The returned gradient is for log feedback likelihood with respect to row
logits. It is an associative gradient update, not a posterior over mappings.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def _inputs(geometry: np.ndarray, logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    geometry, logits = np.asarray(geometry, float), np.asarray(logits, float)
    if (logits.ndim < 2 or logits.shape[-2] < 1 or logits.shape[-1] < 2
            or geometry.shape != logits.shape[:-1]
            or not np.isfinite(logits).all() or not np.isfinite(geometry).all()
            or np.any(geometry < 0)
            or not np.allclose(geometry.sum(-1), 1., rtol=0, atol=1e-12)):
        raise ValueError('Need normalized region probabilities and finite region/response logits')
    log_association = logits - logsumexp(logits, axis=-1, keepdims=True)
    return geometry, log_association


def region_response_probabilities(geometry: np.ndarray, logits: np.ndarray) -> np.ndarray:
    """Mix row associations over geometry; leading axes identify fixed rules."""
    geometry, log_association = _inputs(geometry, logits)
    return np.sum(geometry[..., None] * np.exp(log_association), axis=-2)


def region_response_feedback_gradient(
    geometry: np.ndarray, logits: np.ndarray, choice: int, feedback: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return log event probability and its row-logit gradient before updating.

    Correct feedback selects one response; incorrect feedback sums the other
    responses. Log-space event normalization remains finite for very unlikely
    feedback. Zero-probability regions receive zero gradient. Feedback 0.5
    requires a separate hierarchical adapter and is rejected here.
    """
    geometry, log_association = _inputs(geometry, logits)
    categories = log_association.shape[-1]
    if (isinstance(choice, (bool, np.bool_)) or not isinstance(choice, (int, np.integer))
            or not 1 <= choice <= categories or feedback not in (0., 1.)):
        raise ValueError('Require a one-based response and binary feedback 0 or 1')
    allowed = np.arange(categories) == choice - 1
    if feedback == 0.:
        allowed = ~allowed
    with np.errstate(divide='ignore'):
        log_geometry = np.log(geometry)
    log_joint = log_geometry[..., None] + log_association
    event_joint = np.where(allowed, log_joint, -np.inf)
    log_event = logsumexp(event_joint, axis=(-2, -1))
    responsibilities = np.exp(event_joint - log_event[..., None, None])
    gradient = responsibilities - responsibilities.sum(-1, keepdims=True) * np.exp(log_association)
    return log_event, gradient

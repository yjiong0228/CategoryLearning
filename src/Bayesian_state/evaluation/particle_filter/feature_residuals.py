"""Training-prefix stimulus probes for frozen binary choice predictions.

The probe reuses the shared offset-logistic solver. It estimates auxiliary
diagnostic coefficients from a declared prefix and predicts all supplied rows;
it does not update the cognitive model, its parameters, or its state history.
Feature means and scales use the same prefix. Callers retain the task's actual
choice coordinates and distinguish development extrapolation from untouched
validation. Oral data are not an input to this component.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, logit

from ...metrics.residuals import _fit_offset_logistic


@dataclass(frozen=True)
class PrefixFeatureProbe:
    """Coefficients and fixed-prefix scaling for one diagnostic fit."""

    coefficient: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    probability: np.ndarray
    training_trials: int
    ridge: float
    gradient_max_abs: float


def fit_prefix_feature_probe(
    predicted: np.ndarray,
    observed: np.ndarray,
    features: np.ndarray,
    *,
    training_trials: int,
    ridge: float = 1.0,
) -> PrefixFeatureProbe:
    """Fit a ridge-penalized offset logit using only the declared prefix.

The log odds of ``predicted`` are a fixed offset. An intercept and the supplied
feature columns provide an additive correction. Non-intercept coefficients
have penalty ``ridge * sum(coefficient**2)`` on the summed training NLL.
Input probabilities must be strictly inside (0, 1); no silent clipping occurs.
The returned probability is an auxiliary prediction, not a new model state.
"""
    p = np.asarray(predicted, dtype=float)
    y = np.asarray(observed, dtype=float)
    x = np.asarray(features, dtype=float)
    if (p.ndim != 1 or y.shape != p.shape or x.ndim != 2 or x.shape[0] != p.size
            or not np.isfinite(p).all() or not np.isfinite(x).all()
            or not np.isfinite(y).all() or not np.isin(y, [0., 1.]).all()
            or np.any((p <= 0) | (p >= 1))):
        raise ValueError('Require aligned finite binary outcomes, features and interior probabilities')
    if type(training_trials) is not int or not 1 < training_trials < len(p):
        raise ValueError('Require a nonempty fitting prefix and later evaluation rows')
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError('Use a positive finite ridge penalty')
    prefix = slice(0, training_trials)
    mean = x[prefix].mean(axis=0)
    scale = x[prefix].std(axis=0)
    # A constant prefix column carries no fitted slope; retain its identity.
    scale = np.where(scale > 0, scale, 1.)
    standardized = (x - mean) / scale
    offset = logit(p)
    coefficient = _fit_offset_logistic(
        offset, standardized, y, np.arange(training_trials),
        include_state=True, ridge=float(ridge))
    design = np.column_stack([np.ones(len(p)), standardized])
    probability = expit(offset + design @ coefficient)
    gradient = design[prefix].T @ (probability[prefix] - y[prefix])
    gradient[1:] += 2 * ridge * coefficient[1:]
    maximum = float(np.max(np.abs(gradient)))
    if not np.isfinite(maximum) or maximum > 1e-4:
        raise RuntimeError('Prefix feature probe failed its gradient convergence check')
    return PrefixFeatureProbe(
        coefficient.copy(), mean.copy(), scale.copy(), probability.copy(),
        training_trials, float(ridge), maximum)

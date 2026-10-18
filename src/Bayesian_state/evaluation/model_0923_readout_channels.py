"""Allocate expected errors in a saved linear mixture of rule predictions.

This is a descriptive decomposition at fixed cognitive states. Its components
are not causal effects of modifying discovery, competition or precision.
"""
from __future__ import annotations

import numpy as np


def error_channels(present: np.ndarray, contribution: np.ndarray,
                   target_correct: np.ndarray, correct: np.ndarray) -> dict[str, np.ndarray]:
    """Separate missing-target, target-emission and competing-rule errors.

Arrays have the same (path, trial) shape. The target's emission may be NaN
where it is absent. ``force_present_target_correct`` replaces the current
mixture by that emission only on trials where the target already exists;
future states and feedback remain fixed in this instantaneous calculation.
"""
    p, c, q, y = [np.asarray(x, dtype=float) for x in
                  (present, contribution, target_correct, correct)]
    if p.ndim != 2 or any(x.shape != p.shape for x in (c, q, y)):
        raise ValueError('Need matching path-by-trial arrays')
    if (not all(np.isfinite(x).all() for x in (p, c, y))
            or np.any((p != 0) & (p != 1))
            or any(np.any((x < 0) | (x > 1)) for x in (c, y))
            or np.any(c[p == 0] != 0)):
        raise ValueError('Invalid presence, probability or absent-rule contribution')
    active = p.astype(bool)
    if not np.isfinite(q[active]).all() or np.any((q[active] < 0) | (q[active] > 1)):
        raise ValueError('Active target emissions must be probabilities')
    target_success = c * np.where(active, q, 0.)
    other_success = y - target_success
    if np.any(other_success < -1e-12) or np.any(other_success > 1 - c + 1e-12):
        raise ValueError('Target emission is incompatible with the saved linear mixture')
    target_error = c - target_success
    other_error = np.where(active, (1 - c) - other_success, 0.)
    missing_error = (1 - p) * (1 - y)
    np.testing.assert_allclose(missing_error + target_error + other_error, 1 - y,
                               atol=2e-12, rtol=0)
    return dict(missing_target_error=missing_error, target_emission_error=target_error,
                other_rules_error=other_error, expected_error=1-y,
                force_present_target_correct=np.where(active, q, y))

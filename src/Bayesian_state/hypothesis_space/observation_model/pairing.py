"""Feedback events over opaque response IDs; no true family enters this kernel."""

from __future__ import annotations

import numpy as np


PAIRING_LABELS = ("12|34", "13|24", "14|23")
PAIRING_SIBLINGS = np.array([[1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]])
PAIRING_SIBLINGS.setflags(write=False)


def pairing_feedback_kernel(probabilities: np.ndarray, choice: int, feedback: float) -> np.ndarray:
    """Return absolute event probabilities, shaped (rules, three pairings)."""
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] != 4 or p.shape[0] == 0:
        raise ValueError("pairing feedback requires probabilities shaped (rules, 4).")
    if (not np.all(np.isfinite(p)) or np.any(p < 0.)
            or not np.allclose(p.sum(axis=1), 1., atol=1e-12, rtol=1e-10)):
        raise ValueError("category probabilities must be finite, non-negative and normalized.")
    if not np.isfinite(choice) or choice != int(choice) or int(choice) not in (1, 2, 3, 4):
        raise ValueError("condition 3 choice must be an integer in 1..4.")
    if feedback not in (0., .5, 1.):
        raise ValueError("condition 3 feedback must be 0, 0.5 or 1.")
    y = int(choice) - 1
    own = p[:, y, None]
    sibling = p[:, PAIRING_SIBLINGS[:, y]]
    if feedback == 1.:
        return np.broadcast_to(own, sibling.shape).copy()
    if feedback == .5:
        return sibling.copy()
    # Sum the complementary events directly: 1-own-sibling can round a real
    # tiny cross-family probability to zero when the rule is very confident.
    other_family = (np.arange(4)[None, :] != y) & (
        np.arange(4)[None, :] != PAIRING_SIBLINGS[:, y, None]
    )
    return p @ other_family.T

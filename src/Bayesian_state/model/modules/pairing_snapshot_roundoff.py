"""Narrow experimental guard for roundoff in cached pairing evidence.

A normalized pairing mixture can sum to one plus one machine epsilon even
after its individual kernel entries are bounded. Particle restoration validates
this cached probability independently. The guard changes only overflowing
cached evidence on load, without changing the source snapshot or any valid
probability. Default memory classes remain untouched outside the context.
"""
from contextlib import contextmanager

import numpy as np

from .hypothesis_transition.feedback_guided import probability_roundoff


@contextmanager
def pairing_snapshot_roundoff_guard(cls):
    """Clip at most eight epsilons above one; preserve every other validation."""
    original = cls.load_state_dict
    corrections = []

    def load(self, state):
        evidence = state.get('feedback_evidence')
        if evidence is not None:
            values = np.asarray(evidence, dtype=float)
            bounded = probability_roundoff(values)
            if not np.array_equal(values, bounded):
                corrections.append(dict(phase='pairing_snapshot_evidence', maximum=float(values.max()),
                                        entries=int(np.count_nonzero(values > 1.))))
                state = dict(state, feedback_evidence=bounded)
        return original(self, state)

    try:
        cls.load_state_dict = load
        yield corrections
    finally:
        cls.load_state_dict = original

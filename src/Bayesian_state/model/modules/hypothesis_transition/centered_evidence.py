"""Experimental relative support for newly admitted COMPACT rules.

After completed feedback, center active log evidence so its base-prior-weighted
mean exponential support is one. A new rule's zero log evidence then represents
that current reference level. Incumbent odds and capacity rankings are preserved
by this common shift. Subsequent feedback updates the same single support value;
the intervention adds no free parameter, history buffer or protected duration.
Install inside compact_evidence in an isolated worker.
"""
from contextlib import contextmanager
from hashlib import sha256

import numpy as np
from scipy.special import logsumexp


def center_support(base: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, float]:
    """Center a finite active vector while preserving all pairwise score differences."""
    base, scores = np.asarray(base, dtype=float), np.asarray(scores, dtype=float)
    if (base.ndim != 1 or base.shape != scores.shape or not len(base)
            or np.any(base <= 0) or not np.isfinite(base).all() or not np.isfinite(scores).all()):
        raise ValueError('Require finite active scores and positive base weights')
    offset = float(logsumexp(np.log(base)+scores) - logsumexp(np.log(base)))
    return scores-offset, offset


@contextmanager
def centered_evidence(cls):
    """Change the entry reference level within the existing COMPACT transition."""
    original_init = cls.__init__
    original_prepare = cls._prepare_hypothesis_transition
    audit = dict(centered_feedback_updates=0, maximum_absolute_shift=0.)

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.configuration_signature = sha256(
            f'{self.configuration_signature}:centered-active-evidence-v1'.encode()).hexdigest()

    def prepare(self, **kwargs):
        pending = self.outcome_pending
        original_prepare(self, **kwargs)
        if pending:
            active = np.asarray(self.active, dtype=int)
            centered, shift = center_support(self.base_prior[active], self.retention_evidence[active])
            self.retention_evidence[active] = centered
            audit['centered_feedback_updates'] += 1
            audit['maximum_absolute_shift'] = max(audit['maximum_absolute_shift'], abs(shift))

    try:
        cls.__init__, cls._prepare_hypothesis_transition = initialize, prepare
        yield audit
    finally:
        cls.__init__, cls._prepare_hypothesis_transition = original_init, original_prepare

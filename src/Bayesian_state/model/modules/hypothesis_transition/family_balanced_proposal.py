"""Experimental two-level global discovery over the existing rule catalogue.

Each family with at least one inactive rule receives equal proposal mass; rules
within that family share its mass. This tests the effect of catalogue family
sizes on discovery. It keeps initial-rule selection, search occurrence, belief
priors, evidence, admission and execution unchanged. Existing family labels are
an explicit modelling assumption, not established psychological categories.

Install outside finite_region_learning so its proposal-dependent rehearsal and
admission wrappers continue to run. Supports full-v3, global-only search with a
uniform base prior. No oral data or new fitted parameter enters the proposal.
"""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256

import numpy as np


def family_proposal(families: tuple[str, ...], inactive: np.ndarray) -> np.ndarray:
    """Normalize equally across represented families, then within each family."""
    ids = np.asarray(inactive)
    if (not families or any(not isinstance(f, str) or not f for f in families)
            or ids.ndim != 1 or not len(ids) or not np.issubdtype(ids.dtype, np.integer)
            or len(np.unique(ids)) != len(ids) or np.any((ids < 0) | (ids >= len(families)))):
        raise ValueError('Require named families and unique valid inactive rule indices')
    _, inverse, counts = np.unique(np.asarray(families)[ids], return_inverse=True, return_counts=True)
    return 1. / (len(counts) * counts[inverse])


@contextmanager
def balanced_global_proposal(cls, *, enabled: bool = False):
    """Change genuine proposal draws while preserving all other H operations."""
    if type(enabled) is not bool: raise ValueError('Explicitly enable or disable family discovery')
    if not enabled:
        yield
        return
    initialize, draw = cls.__init__, cls._draw_v3_proposal

    def initialize_balanced(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        if (not self.continuous_choice_enabled or self.global_search != 1.
                or hasattr(self, '_proposal_families')
                or not np.allclose(self.base_prior, 1./self.total_hypo, atol=1e-15, rtol=0)):
            raise ValueError('Family diagnostic requires full-v3 global search and a uniform base prior')
        self._proposal_families = tuple(h.family for h in self.engine.partition.hypothesis_space)
        if len(self._proposal_families) != self.total_hypo:
            raise ValueError('Family labels differ from the cognitive catalogue')
        self.configuration_signature = sha256((self.configuration_signature +
            ':equal-available-family-proposal-v1:' + repr(self._proposal_families)).encode()).hexdigest()

    def draw_balanced(self, before, remaining, gamma, diagnostics):
        if self.current_g != 1.: raise ValueError('Family diagnostic requires an entirely global proposal')
        inactive = np.setdiff1d(np.arange(self.total_hypo), before)
        q = family_proposal(self._proposal_families, inactive)
        # Preserve the route draw and one catalogue draw used by the reference.
        # Sampling once from the factorized probabilities couples path seeds.
        self.trial_rng.random()
        newcomer = int(self.trial_rng.choice(inactive, p=q))
        diagnostics.update(proposal_source='global_equal_family', proposed_hypothesis=newcomer,
            proposed_family=self._proposal_families[newcomer],
            proposed_rule_probability=float(q[np.flatnonzero(inactive == newcomer)[0]]))
        return newcomer

    try:
        cls.__init__, cls._draw_v3_proposal = initialize_balanced, draw_balanced
        yield
    finally:
        cls.__init__, cls._draw_v3_proposal = initialize, draw

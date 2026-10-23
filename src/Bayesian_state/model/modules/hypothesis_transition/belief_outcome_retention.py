"""Gate rule-belief forgetting by the current observed feedback.

Within the compact belief framework, use delta_t = delta + (1-delta)*r_t
for the rule marginal's power update. Correct feedback preserves prior odds;
zero feedback uses the fitted retention; half credit uses their midpoint.
Conditional pairing retention, feedback likelihood and precision learning keep
their existing definitions. This is an outcome-dependent learning hypothesis,
not an exact Bayesian update with a feedback-independent prior.

Install after compact_belief (and optionally belief_slot_transfer) in an
isolated numerical worker. No parameter or accumulated memory state is added.
"""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256

import numpy as np

from ..base_module import ModuleRole
from ..memory import DualMemoryModule
from ..pairing_memory import HierarchicalPairingMemoryModule


def effective_retention(retention: float, feedback: float) -> float:
    """Preserve prior odds on success and discount them on graded failure."""
    if (not np.isfinite(retention) or not 0 <= retention <= 1
            or feedback not in (0., .5, 1.)):
        raise ValueError('Require retention in [0, 1] and feedback in {0, .5, 1}')
    return float(retention + (1 - retention) * feedback)


@contextmanager
def belief_outcome_retention(cls):
    """Temporarily change the rule exponent during feedback learning only."""
    initialize = cls.__init__
    process = DualMemoryModule.process
    prepare = HierarchicalPairingMemoryModule.prepare_feedback
    audit = dict(updates=0, error=0, partial=0, correct=0)

    def init(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        if not getattr(self, 'belief_framework', False):
            raise ValueError('Outcome retention requires normalized rule beliefs')
        self.configuration_signature = sha256(
            f'{self.configuration_signature}:belief-outcome-retention-v1'.encode()).hexdigest()
        self.belief_outcome_retention = True

    @contextmanager
    def learning(memory):
        transition = memory.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        if not getattr(transition, 'belief_outcome_retention', False):
            yield
            return
        feedback = float(memory.engine.observation[2])
        original = transition.intervention['decay']
        value = effective_retention(float(original), feedback)
        transition.intervention['decay'] = value
        try:
            yield
        finally:
            transition.intervention['decay'] = original
        audit['updates'] += 1
        audit[{0.: 'error', .5: 'partial', 1.: 'correct'}[feedback]] += 1

    def update(memory, **kwargs):
        with learning(memory):
            return process(memory, **kwargs)

    def pairing(memory, kernel):
        with learning(memory):
            return prepare(memory, kernel)

    try:
        cls.__init__ = init
        DualMemoryModule.process = update
        HierarchicalPairingMemoryModule.prepare_feedback = pairing
        yield audit
    finally:
        cls.__init__ = initialize
        DualMemoryModule.process = process
        HierarchicalPairingMemoryModule.prepare_feedback = prepare

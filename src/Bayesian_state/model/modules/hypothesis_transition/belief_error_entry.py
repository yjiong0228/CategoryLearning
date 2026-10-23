"""Allocate a replacement's belief using the last completed feedback.

At full capacity K, a new rule receives r*pi_removed + (1-r)/K, where r is
the previous outcome. Survivors share the remaining mass in their posterior
proportions. Success retains SLOT admission; failure gives the newcomer a
uniform trial share. Workspace growth retains compact belief admission.
No parameter, score or persistent memory state is added. Install inside
compact_belief and outside belief_slot_transfer in an isolated worker.
"""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256

import numpy as np

from ..base_module import ModuleRole
from ..pairing_memory import HierarchicalPairingMemoryModule


def newcomer_share(removed: float, capacity: int, feedback: float) -> float:
    """Interpolate slot transfer and a uniform opportunity after an error."""
    if (not isinstance(capacity, (int, np.integer)) or capacity < 1
            or not np.isfinite(removed) or not 0 <= removed <= 1 / capacity + 1e-12
            or feedback not in (0., .5, 1.)):
        raise ValueError('Require the least-believed slot and valid completed feedback')
    return float(feedback * removed + (1 - feedback) / capacity)


@contextmanager
def belief_error_entry(cls):
    """Use feedback-dependent entry mass while preserving survivor odds."""
    initialize, assign = cls.__init__, cls._assign_v3_prior
    audit = dict(replacements=0, zero_mass_entries=0, error=0, partial=0, correct=0)

    def init(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        if not getattr(self, 'belief_framework', False):
            raise ValueError('Error-dependent entry requires normalized rule beliefs')
        self.configuration_signature = sha256(
            f'{self.configuration_signature}:belief-error-entry-v1'.encode()).hexdigest()

    def prior(self, context, selection):
        if not len(selection.dropped) or not len(selection.newcomers):
            return assign(self, context, selection)
        if (len(selection.dropped) != 1 or len(selection.newcomers) != 1
                or len(context.active_before) != self.capacity
                or len(selection.active_after) != self.capacity):
            raise ValueError('Error-dependent entry requires one full-workspace replacement')
        feedback = self.previous_feedback
        mass = newcomer_share(float(context.posterior[selection.dropped].sum()), self.capacity, feedback)
        if feedback == 1.:
            result = assign(self, context, selection)
        else:
            result = np.zeros(self.total_hypo)
            result[selection.newcomers] = mass
            if len(selection.survivors):
                values = context.posterior[selection.survivors]
                if values.sum() <= 0:
                    raise ValueError('Surviving rules must have positive total belief')
                result[selection.survivors] = (1 - mass) * values / values.sum()
            memory = self.engine.get_module(ModuleRole.MEMORY)
            if isinstance(memory, HierarchicalPairingMemoryModule):
                memory.assign_relation_prior(result, selection)
        if (not np.isfinite(result).all() or np.any(result < 0)
                or not np.isclose(result.sum(), 1., atol=1e-12)):
            raise ValueError('Error-dependent entry must preserve normalized beliefs')
        selection.diagnostics.update(prior_initialization='belief_error_entry',
            belief_newcomer_mass=mass, entry_completed_feedback=feedback)
        audit['replacements'] += 1
        audit['zero_mass_entries'] += int(mass == 0.)
        audit[{0.: 'error', .5: 'partial', 1.: 'correct'}[feedback]] += 1
        return result

    try:
        cls.__init__, cls._assign_v3_prior = init, prior
        yield audit
    finally:
        cls.__init__, cls._assign_v3_prior = initialize, assign

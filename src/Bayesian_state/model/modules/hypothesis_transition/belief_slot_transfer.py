"""Conserve surviving rule beliefs when a full workspace replaces one rule.

This opt-in admission hypothesis transfers the removed rule's probability to
its replacement. Initial workspace growth uses the compact belief adapter's
uniform entry prior. There are no additional fitted parameters or memory
states. Install inside compact_belief in an isolated numerical worker.
"""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256

import numpy as np

from ..base_module import ModuleRole
from ..pairing_memory import HierarchicalPairingMemoryModule


@contextmanager
def belief_slot_transfer(cls):
    """Give a replacement its predecessor's posterior probability mass."""
    initialize = cls.__init__
    assign = cls._assign_v3_prior
    audit = dict(replacements=0, zero_mass_entries=0)

    def init(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        if not getattr(self, 'belief_framework', False):
            raise ValueError('Slot transfer requires the compact normalized-belief framework')
        self.configuration_signature = sha256(
            f'{self.configuration_signature}:belief-slot-transfer-v1'.encode()).hexdigest()

    def prior(self, context, selection):
        if not len(selection.dropped) or not len(selection.newcomers):
            return assign(self, context, selection)
        if (len(selection.dropped) != 1 or len(selection.newcomers) != 1
                or len(context.active_before) != self.capacity
                or len(selection.active_after) != self.capacity):
            raise ValueError('Slot transfer expects one replacement in a full workspace')
        result = np.zeros(self.total_hypo)
        result[selection.survivors] = context.posterior[selection.survivors]
        mass = float(context.posterior[selection.dropped].sum())
        result[selection.newcomers] = mass
        if not np.isfinite(result).all() or np.any(result < 0) or not np.isclose(result.sum(), 1., atol=1e-12):
            raise ValueError('Replacement must conserve a normalized posterior')
        memory = self.engine.get_module(ModuleRole.MEMORY)
        if isinstance(memory, HierarchicalPairingMemoryModule):
            memory.assign_relation_prior(result, selection)
        selection.diagnostics.update(prior_initialization='belief_slot_transfer',
                                     belief_newcomer_mass=mass)
        audit['replacements'] += 1
        audit['zero_mass_entries'] += int(mass == 0.)
        return result

    try:
        cls.__init__, cls._assign_v3_prior = init, prior
        yield audit
    finally:
        cls.__init__, cls._assign_v3_prior = initialize, assign

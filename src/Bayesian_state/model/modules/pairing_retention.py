"""Experimental separation of rule and same-family pairing retention.

The original factorized memory supplies the discounted conditional pairings,
absolute feedback evidence and validation. Its joint update is recombined with
the original rule-retention weights. Choice emissions, precision learning and
H algorithms retain their existing implementations. The override is fixed by
the diagnostic protocol and is encoded in particle snapshots.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np

from .pairing_memory import _normalize_joint


@contextmanager
def pairing_retention(cls, *, rate: float | None = None):
    """Override conditional pairing retention while preserving rule gamma.

    ``None`` leaves methods and state formats unchanged. A numeric rate applies
    only to factorized memory and marks snapshots so incompatible continuations
    fail. Returned counters describe computational updates, not participant data.
    """
    counts = dict(updates=0, recombined_updates=0)
    if rate is None:
        yield counts
        return
    if isinstance(rate, (bool, str)) or not np.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError('Pairing retention must be None or a finite rate in [0,1]')
    rate = float(rate)
    if hasattr(cls, '_pairing_retention_diagnostic'):
        raise ValueError('Pairing retention is already overridden')
    names = ('prepare_feedback', 'state_dict', 'load_state_dict')
    original = {name:getattr(cls, name) for name in names}
    tag = 'factorized_separate_pairing_v1'

    def prepare(self, kernel):
        if self.forgetting_mode != 'factorized':
            raise ValueError('Separate pairing retention requires factorized memory')
        rule_gamma = self.gamma
        try:
            self.gamma = rate
            evidence = original['prepare_feedback'](self, kernel)
        finally:
            self.gamma = rule_gamma
        if rate != rule_gamma:
            prior = np.asarray(self.engine.prior, dtype=float)
            mask = getattr(self.engine, 'hypotheses_mask', None)
            active = prior > 0 if mask is None else np.asarray(mask) > 0
            weights = np.zeros_like(prior)
            weights[active] = 1. if rule_gamma == 0. else prior[active]**rule_gamma
            weights /= weights.sum()
            # Only the rule marginal is replaced; the production feedback
            # cache and precision gradient share the discounted pairing state.
            self._pending_joint = _normalize_joint(
                weights[:, None] * self.feedback_pairing_weights * np.asarray(kernel))
            counts['recombined_updates'] += 1
        counts['updates'] += 1
        return evidence

    def snapshot(self):
        if self.forgetting_mode != 'factorized':
            raise ValueError('Separate pairing snapshots require factorized memory')
        state = original['state_dict'](self)
        state.update(forgetting_mode=tag, pairing_retention_override=rate)
        return state

    def restore(self, state):
        if (self.forgetting_mode != 'factorized' or state.get('forgetting_mode') != tag
                or state.get('pairing_retention_override') != rate):
            raise ValueError('Snapshot pairing-retention protocol differs')
        payload = dict(state, forgetting_mode='factorized')
        original['load_state_dict'](self, payload)

    try:
        cls._pairing_retention_diagnostic = rate
        for name, method in zip(names, (prepare, snapshot, restore)):
            setattr(cls, name, method)
        yield counts
    finally:
        for name, method in original.items():
            setattr(cls, name, method)
        delattr(cls, '_pairing_retention_diagnostic')

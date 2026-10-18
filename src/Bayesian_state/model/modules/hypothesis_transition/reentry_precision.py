"""Experimental retention of learned precision across workspace absences.

Every previously active rule keeps its last active precision. Absent rules
receive no precision updates. On reentry, their saved precision replaces the
ordinary initialization, while new rules use the existing initialization.
The cache persists indefinitely in this parameter-free structural diagnostic.
Workspace membership, admission evidence and belief transport keep their
existing algorithms. Cached precision is part of each particle's H state.
"""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256

import numpy as np

from ..base_module import ModuleRole


@contextmanager
def reentry_precision(cls, *, enabled: bool = True):
    """Retain per-rule precision; yield one diagnostic record per transition."""
    if type(enabled) is not bool:
        raise ValueError('Precision retention must be explicitly enabled or disabled')
    records = []
    if not enabled:
        yield records
        return
    names = ('__init__', '_finish_hypothesis_transition', 'load_state_dict')
    original = {name: getattr(cls, name) for name in names}

    def initialize(self, *args, **kwargs):
        original['__init__'](self, *args, **kwargs)
        if not self.continuous_choice_enabled:
            raise ValueError('Reentry precision requires the full-v3 choice mechanism')
        if 'reentry_precision_seen' in self._STATE_FIELDS:
            raise ValueError('Precision retention has already been installed')
        self._STATE_FIELDS += ('reentry_precision_seen', 'reentry_precision_values')
        self.reentry_precision_seen = np.zeros(self.total_hypo, dtype=bool)
        self.reentry_precision_values = np.zeros(self.total_hypo, dtype=float)
        self.configuration_signature = sha256(
            f'{self.configuration_signature}:last-active-precision-v1'.encode()).hexdigest()

    def finish(self, context, selection, prior, **kwargs):
        beta = self.engine.get_module(ModuleRole.BETA)
        if beta is None or beta.beta_update_mode != 'feedback_gradient':
            raise ValueError('Reentry precision requires feedback-gradient precision learning')
        before = np.asarray(context.active_before, dtype=int)
        values = beta.beta[before]
        if (not np.isfinite(values).all() or np.any(values < beta.beta_min)
                or np.any(values > beta.beta_max)):
            raise ValueError('Active rules must have valid learned precision before transition')
        # Completed feedback has already updated active precision. Save it
        # before ordinary transition initialization and inactive zeroing.
        self.reentry_precision_values[before] = values
        self.reentry_precision_seen[before] = True
        newcomers = np.asarray(selection.newcomers, dtype=int)
        restored = newcomers[self.reentry_precision_seen[newcomers]]
        restored_values = self.reentry_precision_values[restored].copy()
        result = original['_finish_hypothesis_transition'](self, context, selection, prior, **kwargs)
        beta.beta[restored] = restored_values
        self.engine.beta = beta.beta.copy()
        details = dict(precision_restored_rules=restored.tolist(),
                       precision_restored_values=restored_values.tolist())
        result.update(details)
        self.transition_log[-1].update(details)
        self.transition_rate_log[-1].update(details)
        records.append(dict(trial=int(context.trial_index), **details))
        return result

    def restore(self, state):
        seen = np.asarray(state.get('reentry_precision_seen'))
        values = np.asarray(state.get('reentry_precision_values'))
        beta = self.engine.get_module(ModuleRole.BETA)
        if (seen.shape != (self.total_hypo,) or seen.dtype != np.dtype(bool)
                or values.shape != (self.total_hypo,) or not np.issubdtype(values.dtype, np.number)
                or not np.isfinite(values).all() or np.any(values[~seen] != 0)
                or beta is None or np.any(values[seen] < beta.beta_min)
                or np.any(values[seen] > beta.beta_max)):
            raise ValueError('Invalid retained-precision snapshot')
        original['load_state_dict'](self, state)

    try:
        for name, method in zip(names, (initialize, finish, restore)):
            setattr(cls, name, method)
        yield records
    finally:
        for name, method in original.items():
            setattr(cls, name, method)

"""Experimental transfer of current precision when a new COMPACT rule enters.

The newcomer inherits the evidence-weighted mean precision of the active rules
after completed feedback. First-trial initialization uses the original beta_init.
Each active rule then continues its existing feedback learning. The adapter
uses current state only and adds no parameter or inactive-rule cache. Install
inside compact_evidence, optionally with centered_evidence.
"""
from contextlib import contextmanager
from hashlib import sha256

import numpy as np

from ..base_module import ModuleRole
from .compact_evidence import evidence_prior


@contextmanager
def entry_precision(cls):
    original_init, original_finish = cls.__init__, cls._finish_hypothesis_transition
    audit = dict(transferred_entries=0)

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.configuration_signature = sha256(
            f'{self.configuration_signature}:active-mean-entry-precision-v1'.encode()).hexdigest()

    def finish(self, context, selection, prior, **kwargs):
        beta = self.engine.get_module(ModuleRole.BETA)
        active = np.asarray(context.active_before, dtype=int)
        incoming = np.asarray(selection.newcomers, dtype=int)
        transferred = None
        if context.trial_index > 0 and len(incoming):
            support = evidence_prior(self.base_prior, active, self.retention_evidence)
            values = beta.beta[active]
            if (not np.isfinite(values).all() or np.any(values < beta.beta_min)
                    or np.any(values > beta.beta_max)):
                raise ValueError('Active precision is outside the model bounds')
            # Keep a floating-point weighted mean inside the observed convex hull.
            transferred = float(np.clip(np.average(values, weights=support[active]), values.min(), values.max()))
        result = original_finish(self, context, selection, prior, **kwargs)
        if transferred is not None:
            beta.beta[incoming] = transferred
            self.engine.beta = beta.beta.copy()
            audit['transferred_entries'] += len(incoming)
            result['entry_precision_transferred'] = transferred
            self.transition_log[-1]['entry_precision_transferred'] = transferred
            self.transition_rate_log[-1]['entry_precision_transferred'] = transferred
        return result

    try:
        cls.__init__, cls._finish_hypothesis_transition = initialize, finish
        yield audit
    finally:
        cls.__init__, cls._finish_hypothesis_transition = original_init, original_finish

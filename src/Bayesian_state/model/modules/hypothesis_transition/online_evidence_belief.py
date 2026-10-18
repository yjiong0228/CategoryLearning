"""Experimental active-set belief from the existing online retention evidence.

Incumbent evidence already accumulates feedback likelihood at the precision
used before learning from that feedback. An admitted rule receives the common
retrospective evidence computed by POOL admission. Normalize base rule weights
times exp(evidence) after final selection, then use ordinary execution inertia.
This replaces the previous rule prior without multiplying its evidence again.

The experiment changes how the existing H evidence affects choice. Its decay,
newcomer initialization and reset on removal remain those of episodic admission.
It therefore differs from common-precision retrospective recall in both the
precision and the history supporting incumbents. No new coefficient is fitted.
The intervening M posterior still supplies search parents and exit support;
gamma remains relevant there. The affinity-transfer channel is closed.
"""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256

import numpy as np

from ..base_module import ModuleRole
from ..pairing_memory import HierarchicalPairingMemoryModule
from .episodic_proposal import recalled_prior


@contextmanager
def online_evidence_belief(cls, *, enabled: bool = False):
    """Install after POOL admission; use completed-feedback evidence for choice."""
    events = []
    if not enabled:
        yield events
        return
    names = ('__init__', '_assign_v3_prior', '_finish_hypothesis_transition')
    original = {name: getattr(cls, name) for name in names}

    def initialize(self, *args, **kwargs):
        original['__init__'](self, *args, **kwargs)
        if (not hasattr(self, 'admission_history') or not self.continuous_choice_enabled
                or self.intervention_kind != 'E'):
            raise ValueError('Online evidence belief requires POOL and full-v3 E retention')
        self.configuration_signature = sha256(
            f'{self.configuration_signature}:online-evidence-belief-v1'.encode()).hexdigest()

    def assign(self, context, selection):
        if self.admission_history is None:
            return original['_assign_v3_prior'](self, context, selection)
        scores = self.retention_evidence.copy()
        if selection.newcomers.size:
            if (len(selection.newcomers) != 1 or self._recall is None
                    or self._recall['rules'][-1] != selection.newcomers[0]):
                raise ValueError('Online belief needs the admitted rule\'s recalled evidence')
            # POOL writes this value after _finish. Use the same value now,
            # before its inherited implementation resets newcomer evidence.
            scores[selection.newcomers] = self._recall['total'][-1]
        prior = recalled_prior(self.base_prior, selection.active_after,
                               np.arange(self.total_hypo), scores)
        memory = self.engine.get_module(ModuleRole.MEMORY)
        if isinstance(memory, HierarchicalPairingMemoryModule):
            memory.assign_relation_prior(prior, selection)
        selection.diagnostics.update(prior_initialization='online_retention_evidence',
            online_belief_scores=scores[selection.active_after].tolist())
        return prior

    def finish(self, context, selection, prior, **kwargs):
        result = original['_finish_hypothesis_transition'](self, context, selection, prior, **kwargs)
        if 'online_belief_scores' in selection.diagnostics:
            # Check the initialized newcomer and surviving rules against the
            # evidence actually retained for the next completed feedback.
            used = np.asarray(selection.diagnostics['online_belief_scores'])
            if not np.array_equal(used, self.retention_evidence[selection.active_after]):
                raise RuntimeError('Choice belief and retained H evidence disagree')
            events.append(dict(trial=int(context.trial_index),
                               newcomer_count=len(selection.newcomers),
                               active_count=len(selection.active_after),
                               evidence_matches_retained_state=True))
        return result

    try:
        for name, method in zip(names, (initialize, assign, finish)):
            setattr(cls, name, method)
        yield events
    finally:
        for name, method in original.items():
            setattr(cls, name, method)

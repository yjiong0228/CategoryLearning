"""Experimental belief reconstruction from completed episodes on each trial.

After H selects the final active set, use base rule weights and the common
discounted retrospective evidence to assign its prior. Every rule uses the
existing beta_init reference precision and retention_decay. The history sum
replaces the previous rule prior at this step; it is never multiplied by an
already accumulated posterior. The next ordinary feedback update still affects
the posterior used by subsequent search and exit decisions.

The entry schedule reproduces the earlier admission-only hypothesis. The
every_trial schedule also reconstructs belief when no newcomer enters. No
new coefficient or fixed window is introduced. transfer_scale has no channel
at reconstructed steps; gamma still affects the intervening posterior used by
search and exit. This retrospective cognitive belief uses a common reference
precision; its relation to the model's evolving rule precision is an explicit
experimental assumption. The outer particle filter keeps its existing role.
"""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256

import numpy as np

from ..base_module import ModuleRole
from ..pairing_memory import HierarchicalPairingMemoryModule
from .episodic_proposal import RecallCache, advance_recall, recalled_prior


@contextmanager
def continuous_recalled_belief(cls, *, schedule: str = 'off'):
    """Install after POOL admission and restore every patched method on exit."""
    if schedule not in ('off', 'entry', 'every_trial'):
        raise ValueError('Belief recall schedule must be off, entry or every_trial')
    events = []
    if schedule == 'off':
        yield events
        return
    names = ('__init__', '_assign_v3_prior', 'load_state_dict')
    original = {name: getattr(cls, name) for name in names}

    def initialize(self, *args, **kwargs):
        original['__init__'](self, *args, **kwargs)
        if not hasattr(self, 'admission_history') or not self.continuous_choice_enabled:
            raise ValueError('Belief recall requires POOL admission and full-v3 choice')
        self._STATE_FIELDS += ('belief_recall_cache',)
        self.belief_recall_cache = None
        self.configuration_signature = sha256(
            f'{self.configuration_signature}:continuous-recalled-belief-v1:{schedule}'.encode()).hexdigest()

    def assign(self, context, selection):
        if self.admission_history is None or (schedule == 'entry' and not selection.newcomers.size):
            return original['_assign_v3_prior'](self, context, selection)
        previous = self.belief_recall_cache
        cache, maximum = advance_recall(self.admission_history, previous,
            decay=self.retention_decay, likelihood=self.engine.observation_likelihood,
            rules=np.arange(self.total_hypo), beta=self.engine.get_module(ModuleRole.BETA).beta_init)
        self.belief_recall_cache = cache
        result = recalled_prior(self.base_prior, selection.active_after,
                                np.arange(self.total_hypo), np.asarray(cache.total))
        memory = self.engine.get_module(ModuleRole.MEMORY)
        if isinstance(memory, HierarchicalPairingMemoryModule):
            memory.assign_relation_prior(result, selection)
        record = dict(prior_initialization='continuous_common_recalled_evidence',
                      belief_recall_schedule=schedule, belief_recalled_episodes=cache.count,
                      belief_recalled_exposure=cache.exposure,
                      belief_new_episodes=cache.count - (previous.count if previous else 0),
                      belief_feedback_roundoff_max=max(0., maximum - 1.),
                      recalled_newcomer_mass=float(result[selection.newcomers].sum()))
        selection.diagnostics.update(record)
        events.append(dict(record, trial=int(context.trial_index),
                           newcomer_count=len(selection.newcomers), active_count=len(selection.active_after)))
        return result

    def restore(self, state):
        cache = state.get('belief_recall_cache')
        if cache is not None:
            if (not isinstance(cache, RecallCache) or len(cache.total) != self.total_hypo
                    or cache.decay != self.retention_decay
                    or cache.beta != self.engine.get_module(ModuleRole.BETA).beta_init):
                raise ValueError('Invalid recalled-belief cache in snapshot')
            node = state.get('admission_history')
            while node is not None and node is not cache.head:
                node = node.previous
            if node is None:
                raise ValueError('Recalled-belief snapshot history does not contain its cache')
        original['load_state_dict'](self, state)

    try:
        for name, replacement in zip(names, (initialize, assign, restore)):
            setattr(cls, name, replacement)
        yield events
    finally:
        for name, method in original.items():
            setattr(cls, name, method)

"""Opt-in single-rule execution on the normalized feedback-belief framework.

MIX preserves the existing model exactly. STICKY samples the next execution
rule from normalized belief times exp(kappa) for the previous rule. MAP
samples uniformly among maximum-belief rules, with no tie preference for the
previous rule. The existing execution_inertia configuration coordinate is
the participant's kappa; only STICKY permits a nonzero value.

Install after compact_belief, belief_slot_transfer, belief_outcome_retention
and belief_error_entry, in an isolated numerical worker. Learning, search,
capacity replacement and precision inheritance reuse those shared modules.
Execution receives no protection from capacity replacement. Its identity,
selection distribution and random stream survive particle cloning.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256

import numpy as np


@contextmanager
def belief_execution(cls, *, mode: str = 'MIX'):
    """Install MIX, MAP or STICKY execution without changing default models."""
    if mode not in ('MIX', 'MAP', 'STICKY'):
        raise ValueError('Execution mode must be MIX, MAP or STICKY')
    audit = dict(trials=0, switches=0, forced_switches=0, voluntary_switches=0)
    if mode == 'MIX':
        yield audit
        return
    names = ('__init__', '_finish_hypothesis_transition', 'load_state_dict')
    original = {name: getattr(cls, name) for name in names}

    def initialize(self, *args, **kwargs):
        kappa = kwargs.get('execution_inertia', 0.)
        if (isinstance(kappa, bool) or not np.isfinite(kappa) or kappa < 0
                or mode == 'MAP' and kappa != 0):
            raise ValueError('Require finite nonnegative kappa, and zero for MAP')
        # Belief initialization validates its own unchanged learning settings.
        original['__init__'](self, *args, **dict(kwargs, execution_inertia=0.))
        if not getattr(self, 'belief_framework', False):
            raise ValueError('Execution extension requires normalized rule beliefs')
        self.belief_execution_mode = mode
        self.belief_execution_kappa = float(kappa)
        self.execution_selection_probabilities = self.execution_probabilities.copy()
        self.execution_forced_switch = False
        self._STATE_FIELDS += ('execution_selection_probabilities', 'execution_forced_switch')
        self.configuration_signature = sha256(
            f'{self.configuration_signature}:belief-execution-v1:{mode}:{float(kappa)}'.encode()).hexdigest()

    def finish(self, context, selection, prior, **kwargs):
        previous = int(self.executed_hypothesis)
        old_count, old_dwell = self.execution_switch_count, self.execution_dwell_trials
        old_rng = deepcopy(self.execution_rng.bit_generator.state)
        # The shared transition already implements log-belief + a stay bonus
        # and categorical sampling. Only its execution branch changes here;
        # candidate selection and belief assignment have already completed.
        old_continuous, old_inertia = self.continuous_choice_enabled, self.execution_inertia
        self.continuous_choice_enabled = mode != 'STICKY'
        self.execution_inertia = self.belief_execution_kappa
        try:
            event = original['_finish_hypothesis_transition'](self, context, selection, prior, **kwargs)
        finally:
            self.continuous_choice_enabled, self.execution_inertia = old_continuous, old_inertia
        distribution = self.execution_probabilities.copy()
        if mode == 'MAP':
            # Uniform ties prevent an implicit stay bonus in the MAP control.
            self.execution_rng.bit_generator.state = old_rng
            tied = np.flatnonzero(prior == np.max(prior))
            distribution = np.zeros(self.total_hypo)
            distribution[tied] = 1. / len(tied)
            self.executed_hypothesis = int(self.execution_rng.choice(tied))
        selected = int(self.executed_hypothesis)
        switched = selected != previous
        forced = previous not in selection.active_after
        self.execution_selection_probabilities = distribution
        self.execution_forced_switch = bool(forced)
        self.current_execution_switch_probability = float(1. - distribution[previous])
        self.current_execution_switch_event = bool(switched)
        self.execution_switch_count = old_count + int(switched)
        self.execution_dwell_trials = 1 if switched else old_dwell + 1
        self.execution_probabilities = np.zeros(self.total_hypo)
        self.execution_probabilities[selected] = 1.
        record = dict(belief_execution_mode=mode, belief_execution_kappa=self.belief_execution_kappa,
            executed_hypothesis=selected, executed_hypothesis_previous=previous,
            execution_selection_probabilities=distribution.tolist(), execution_forced_switch=bool(forced),
            execution_voluntary_switch=bool(switched and not forced),
            execution_switch_probability=self.current_execution_switch_probability,
            execution_switch_event=bool(switched), execution_dwell_trials=self.execution_dwell_trials,
            choice_weights=self.execution_probabilities.tolist(), effective_participation=1.,
            persistent_execution_enabled=mode == 'STICKY', continuous_choice_enabled=False)
        # Engine readout uses the existing continuous-weight interface with a
        # one-hot vector; its belief prior remains untouched for all-rule learning.
        event.update(record)
        self.transition_log[-1].update(record)
        self.transition_rate_log[-1].update(record)
        audit['trials'] += 1
        audit['switches'] += int(switched)
        audit['forced_switches'] += int(forced)
        audit['voluntary_switches'] += int(switched and not forced)
        return event

    def restore(self, state):
        probability = np.asarray(state.get('execution_selection_probabilities'), dtype=float)
        active = np.asarray(state.get('active'), dtype=int)
        weights = np.asarray(state.get('execution_probabilities'), dtype=float)
        selected = state.get('executed_hypothesis')
        if (probability.shape != (self.total_hypo,) or not np.isfinite(probability).all()
                or np.any(probability < 0) or not np.isclose(probability.sum(), 1.)
                or np.any(probability[np.setdiff1d(np.arange(self.total_hypo), active)] != 0)
                or not isinstance(state.get('execution_forced_switch'), bool)):
            raise ValueError('Invalid saved execution selection distribution')
        expected = np.zeros(self.total_hypo)
        if (not isinstance(selected, (int, np.integer)) or isinstance(selected, bool)
                or not 0 <= selected < self.total_hypo):
            raise ValueError('Invalid saved execution identity')
        expected[selected] = 1.
        if not np.array_equal(weights, expected):
            raise ValueError('Single-rule execution requires one-hot choice weights')
        original['load_state_dict'](self, state)

    try:
        for name, method in zip(names, (initialize, finish, restore)):
            setattr(cls, name, method)
        yield audit
    finally:
        for name, method in original.items():
            setattr(cls, name, method)

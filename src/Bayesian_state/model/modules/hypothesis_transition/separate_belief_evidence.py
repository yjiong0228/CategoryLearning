"""Experimental online belief memory with a separate candidate-management decay.

The existing M gamma discounts the feedback evidence used for rule belief.
The existing H delta continues to discount admission and exit evidence. Both
accumulators consume the same completed feedback likelihood, evaluated before
that feedback changes precision. New entrants recall completed episodes at
beta_init and gamma; removal clears their belief evidence. Precision storage
and search/admission algorithms remain those of the enclosing experiment.

Install after POOL admission (and optional SEARCH/RETAIN). This replaces the
ONLINE adapter. No parameter is added. M still supplies the posterior used by
search and exit, and preserves the ordinary condition-3 pairing update.
"""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256

import numpy as np

from ..base_module import ModuleRole
from ..pairing_memory import HierarchicalPairingMemoryModule
from .episodic_admission import recalled_episodes, recalled_log_evidence
from .episodic_proposal import recalled_prior


@contextmanager
def separate_belief_evidence(cls, *, enabled: bool = False):
    """Use gamma for active-rule belief and retain delta for H management."""
    if type(enabled) is not bool:
        raise ValueError('Separate belief evidence must be explicitly enabled or disabled')
    events = []
    if not enabled:
        yield events
        return
    names = ('__init__', '_prepare_hypothesis_transition', '_assign_v3_prior',
             '_finish_hypothesis_transition', 'state_dict', 'load_state_dict')
    original = {name: getattr(cls, name) for name in names}

    def gamma(self):
        value = float(self.engine.get_module(ModuleRole.MEMORY).gamma)
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Belief memory requires gamma in [0,1]')
        return value

    def initialize(self, *args, **kwargs):
        original['__init__'](self, *args, **kwargs)
        if (not hasattr(self, 'admission_history') or not self.continuous_choice_enabled
                or self.intervention_kind != 'E' or hasattr(self, 'belief_evidence')):
            raise ValueError('Separate belief evidence requires POOL/full-v3 E exactly once')
        self._STATE_FIELDS += ('belief_evidence', 'belief_exposure')
        self.belief_evidence = np.zeros(self.total_hypo)
        self.belief_exposure = np.zeros(self.total_hypo)
        self.configuration_signature = sha256(
            f'{self.configuration_signature}:separate-belief-evidence-v1'.encode()).hexdigest()

    def prepare(self, **kwargs):
        if self.outcome_pending:
            decay = gamma(self)
            self.belief_evidence[self.active] = (
                decay * self.belief_evidence[self.active] + self._previous_log_evidence())
            self.belief_exposure[self.active] = decay * self.belief_exposure[self.active] + 1.
        return original['_prepare_hypothesis_transition'](self, **kwargs)

    def assign(self, context, selection):
        if self.admission_history is None:
            return original['_assign_v3_prior'](self, context, selection)
        decay = gamma(self)
        recalled_count, roundoff = 0, 0.
        if selection.newcomers.size:
            if (len(selection.newcomers) != 1 or self._recall is None
                    or self._recall['rules'][-1] != selection.newcomers[0]):
                raise ValueError('Belief initialization needs the admitted rule and completed history')
            if decay == self.retention_decay:
                # Reuse identical arithmetic at the nested ONLINE boundary.
                total = self._recall['total'][-1:]
                weights = self._recall['weights']
            else:
                # Delta's recalled history can be truncated by underflow. Gamma
                # therefore reads the original episode chain independently.
                episodes, weights = recalled_episodes(self.admission_history, decay)
                evidence, maximum = recalled_log_evidence(self.engine.observation_likelihood,
                    episodes, selection.newcomers,
                    self.engine.get_module(ModuleRole.BETA).beta_init)
                total = weights @ evidence
                roundoff = max(0., maximum - 1.)
            self.belief_evidence[selection.newcomers] = total
            self.belief_exposure[selection.newcomers] = weights.sum()
            recalled_count = len(weights)
        prior = recalled_prior(self.base_prior, selection.active_after,
                               np.arange(self.total_hypo), self.belief_evidence)
        memory = self.engine.get_module(ModuleRole.MEMORY)
        if isinstance(memory, HierarchicalPairingMemoryModule):
            memory.assign_relation_prior(prior, selection)
        selection.diagnostics.update(prior_initialization='separate_gamma_belief_evidence',
            belief_gamma=decay, belief_scores=self.belief_evidence[selection.active_after].tolist(),
            belief_exposure=self.belief_exposure[selection.active_after].tolist(),
            belief_newcomer_episodes=recalled_count, belief_feedback_roundoff_max=roundoff)
        return prior

    def finish(self, context, selection, prior, **kwargs):
        result = original['_finish_hypothesis_transition'](self, context, selection, prior, **kwargs)
        self.belief_evidence[selection.dropped] = 0.
        self.belief_exposure[selection.dropped] = 0.
        if 'belief_scores' in selection.diagnostics:
            if not np.array_equal(selection.diagnostics['belief_scores'],
                                  self.belief_evidence[selection.active_after]):
                raise RuntimeError('Choice belief and its persistent evidence disagree')
            equal_decay = gamma(self) == self.retention_decay
            if equal_decay and (not np.array_equal(self.belief_evidence, self.retention_evidence)
                    or not np.array_equal(self.belief_exposure, self.retention_exposure)):
                raise RuntimeError('Equal-decay belief differs from ONLINE evidence')
            events.append(dict(trial=int(context.trial_index),
                newcomer_count=len(selection.newcomers), gamma=gamma(self), delta=self.retention_decay,
                belief_matches_saved_state=True, equal_decay=equal_decay,
                newcomer_episodes=selection.diagnostics['belief_newcomer_episodes'],
                feedback_roundoff_max=selection.diagnostics['belief_feedback_roundoff_max']))
        return result

    def snapshot(self):
        return dict(original['state_dict'](self), belief_gamma=gamma(self))

    def restore(self, state):
        if state.get('belief_gamma') != gamma(self):
            raise ValueError('Belief snapshot gamma differs from the configured memory')
        active = np.asarray(state.get('active'), dtype=int)
        inactive = np.setdiff1d(np.arange(self.total_hypo), active)
        for name in ('belief_evidence', 'belief_exposure'):
            values = np.asarray(state.get(name))
            if (values.shape != (self.total_hypo,) or not np.issubdtype(values.dtype, np.number)
                    or not np.isfinite(values).all() or np.any(values[inactive] != 0)
                    or name == 'belief_exposure' and np.any(values < 0)):
                raise ValueError('Invalid separate belief evidence snapshot')
        original['load_state_dict'](self, {k: v for k, v in state.items() if k != 'belief_gamma'})

    try:
        for name, method in zip(names, (initialize, prepare, assign, finish, snapshot, restore)):
            setattr(cls, name, method)
        yield events
    finally:
        for name, method in original.items():
            setattr(cls, name, method)

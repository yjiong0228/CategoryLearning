"""Opt-in finite-workspace rule beliefs with the Model 1022 modules.

The memory module updates normalized rule beliefs from completed feedback.
Their retention rate is the existing delta coordinate; condition 3 keeps its
separate pairing retention. Search parents, replacement, choice and precision
inheritance all use these beliefs. A newcomer gets 1 / new workspace size;
survivors divide the remaining mass in their posterior proportions.

Install in an isolated numerical worker. The shared search, likelihood,
precision learning, pairing transport and snapshot lifecycle are reused.
"""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256

import numpy as np

from ..base_module import ModuleRole
from ..memory import DualMemoryModule
from ..pairing_memory import HierarchicalPairingMemoryModule, _normalize_joint


@contextmanager
def compact_belief(cls):
    """Use one normalized belief distribution, with no retention-score state."""
    names = ('__init__', '_prepare_hypothesis_transition', '_assign_v3_prior',
             '_finish_hypothesis_transition')
    original = {name: getattr(cls, name) for name in names}
    memory_process = DualMemoryModule.process
    pairing_prepare = HierarchicalPairingMemoryModule.prepare_feedback
    audit = dict(transitions=0, replacements=0, transferred_entries=0, feedback_updates=0)

    def initialize(self, *args, **kwargs):
        original['__init__'](self, *args, **kwargs)
        if (not self.continuous_choice_enabled or self.intervention_kind != 'E'
                or self.intervention.get('search') != 'accumulate'
                or self.exit_rate != 0 or self.execution_inertia != 0
                or self.local_structure_weight != 1 or self.global_search_failure_gain != 0
                or self.intervention['decay'] == 'memory'):
            raise ValueError('Belief comparison requires the compact search and precision configuration')
        # S1 selects the existing failure accumulator and posterior-ranked
        # replacement branches. No continuous retention evidence is maintained.
        self.intervention_kind = 'S1'
        self._STATE_FIELDS = tuple(k for k in self._STATE_FIELDS
                                   if k not in ('retention_evidence', 'retention_exposure'))
        del self.retention_evidence, self.retention_exposure
        self.configuration_signature = sha256(
            f'{self.configuration_signature}:compact-belief-uniform-entry-v1'.encode()).hexdigest()
        self.belief_framework = True

    def transition_for(memory):
        transition = memory.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        return transition if getattr(transition, 'belief_framework', False) else None

    def process(memory, **kwargs):
        transition = transition_for(memory)
        if transition is None:
            return memory_process(memory, **kwargs)
        if memory.forgetting_mode != 'exact_power':
            raise ValueError('Rule belief requires exact_power memory')
        saved = memory.gamma
        try:
            memory.gamma = transition.retention_decay
            result = memory_process(memory, **kwargs)
        finally:
            memory.gamma = saved
        audit['feedback_updates'] += 1
        return result

    def prepare_pairing(memory, kernel):
        evidence = pairing_prepare(memory, kernel)
        transition = transition_for(memory)
        if transition is None:
            return evidence
        if memory.forgetting_mode != 'factorized':
            raise ValueError('Rule belief requires factorized pairing memory')
        # The shared module has already discounted conditional pairings and
        # cached the feedback likelihood used by precision learning. Only its
        # rule-marginal retention changes to delta.
        prior = np.asarray(memory.engine.prior)
        active = np.asarray(memory.engine.hypotheses_mask) > 0
        weights = np.zeros_like(prior)
        weights[active] = (1. if transition.retention_decay == 0
                           else prior[active] ** transition.retention_decay)
        weights /= weights.sum()
        memory._pending_joint = _normalize_joint(
            weights[:, None] * memory.feedback_pairing_weights * np.asarray(kernel))
        audit['feedback_updates'] += 1
        return evidence

    def prepare(self, **kwargs):
        # The shared S1 controller reads the last outcome without maintaining E.
        original['_prepare_hypothesis_transition'](self, **kwargs)

    def assign(self, context, selection):
        retained, incoming = selection.survivors, selection.newcomers
        prior = np.zeros(self.total_hypo)
        newcomer_mass = len(incoming) / len(selection.active_after)
        if len(incoming):
            prior[incoming] = 1. / len(selection.active_after)
        if len(retained):
            mass = context.posterior[retained]
            if mass.sum() <= 0:
                raise ValueError('Surviving rules have no posterior belief')
            prior[retained] = (1. - newcomer_mass) * mass / mass.sum()
        memory = self.engine.get_module(ModuleRole.MEMORY)
        if isinstance(memory, HierarchicalPairingMemoryModule):
            memory.assign_relation_prior(prior, selection)
        selection.diagnostics.update(prior_initialization='belief_uniform_newcomer',
                                     belief_newcomer_mass=float(newcomer_mass))
        return prior

    def finish(self, context, selection, prior, **kwargs):
        beta = self.engine.get_module(ModuleRole.BETA)
        transferred = None
        if context.trial_index > 0 and len(selection.newcomers):
            active = context.active_before
            values = beta.beta[active]
            transferred = float(np.clip(np.average(values, weights=context.posterior[active]),
                                        values.min(), values.max()))
        result = original['_finish_hypothesis_transition'](self, context, selection, prior, **kwargs)
        if transferred is not None:
            beta.beta[selection.newcomers] = transferred
            self.engine.beta = beta.beta.copy()
            audit['transferred_entries'] += len(selection.newcomers)
        result.update(belief_framework=True, entry_precision_transferred=transferred)
        self.transition_log[-1].update(result)
        self.transition_rate_log[-1].update(result)
        audit['transitions'] += 1
        audit['replacements'] += int(bool(selection.dropped.size))
        return result

    try:
        for name, method in zip(names, (initialize, prepare, assign, finish)):
            setattr(cls, name, method)
        DualMemoryModule.process = process
        HierarchicalPairingMemoryModule.prepare_feedback = prepare_pairing
        yield audit
    finally:
        for name, method in original.items():
            setattr(cls, name, method)
        DualMemoryModule.process = memory_process
        HierarchicalPairingMemoryModule.prepare_feedback = pairing_prepare

"""Opt-in compact H: one active-rule evidence score, search and replacement.

Completed feedback updates the existing discounted evidence at the precision
used for that feedback. The same evidence determines local search parents,
capacity replacement and choice weights. A newly proposed rule starts at zero
evidence and learns from subsequent feedback. Search uses the existing failure
accumulator; global exploration has a fixed participant-level probability.
Structural local proposals, pairing updates and precision learning reuse the
shared implementation. There is no episodic history or inactive precision cache.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256

import numpy as np
from scipy.special import expit

from ..base_module import ModuleRole
from ..pairing_memory import HierarchicalPairingMemoryModule
from .mechanistic_rule_search import _softmax


def evidence_prior(base: np.ndarray, active: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Normalize positive rule priors times exp(online evidence) on the workspace."""
    base, scores, active = np.asarray(base), np.asarray(scores), np.asarray(active, dtype=int)
    if (base.ndim != 1 or scores.shape != base.shape or not len(active)
            or len(set(active.tolist())) != len(active) or np.any(active < 0)
            or np.any(active >= len(base)) or not np.isfinite(scores).all()
            or not np.isfinite(base).all() or np.any(base <= 0)):
        raise ValueError('Invalid active evidence or base rule weights')
    result = np.zeros(len(base))
    result[active] = _softmax(np.log(base[active]) + scores[active])
    return result


@contextmanager
def compact_evidence(cls):
    """Install the compact mechanism in an isolated worker; restore on exit.

The explicit zero/one configuration coordinates close old pathways and are
validated rather than silently overwritten. Existing snapshot and reseeding
contracts continue to carry active evidence and the failure accumulator.
"""
    names = ('__init__', '_prepare_hypothesis_transition', '_select_v3',
             '_assign_v3_prior', '_finish_hypothesis_transition')
    original = {name: getattr(cls, name) for name in names}
    audit = dict(transitions=0, replacements=0, max_active=0, evidence_checks=0)

    def initialize(self, *args, **kwargs):
        original['__init__'](self, *args, **kwargs)
        if (not self.continuous_choice_enabled or self.intervention_kind != 'E'
                or self.intervention.get('search') != 'accumulate'
                or self.intervention['capacity'] != 'evidence'
                or self.exit_rate != 0 or self.execution_inertia != 0
                or self.local_structure_weight != 1 or self.global_search_failure_gain != 0
                or hasattr(self, 'admission_history') or hasattr(self, 'reentry_precision_seen')):
            raise ValueError('Compact H requires evidence capacity, structural local search and closed legacy channels')
        self.configuration_signature = sha256(
            f'{self.configuration_signature}:compact-online-evidence-v1'.encode()).hexdigest()

    def prepare(self, **kwargs):
        if not self.outcome_pending:
            return
        self.retention_evidence[self.active] = (
            self.retention_decay * self.retention_evidence[self.active] + self._previous_log_evidence())
        # Exposure remains an audit counter; it does not divide the evidence.
        self.retention_exposure[self.active] = self.retention_decay * self.retention_exposure[self.active] + 1
        if self.previous_feedback == 1:
            self.accumulated_failure *= self.failure_decay
        else:
            self.accumulated_failure += 1 - self.previous_feedback
        logit = np.log(self.baseline_probability) - np.log1p(-self.baseline_probability)
        self.current_event_probability = float(expit(logit + self.error_gain * self.accumulated_failure))
        self.current_g = self.global_search
        self.outcome_pending = False

    def select(self, context):
        # The shared selector consumes this one score for parent sampling and
        # capacity decisions. Its independent exit channel is fixed at zero.
        support = evidence_prior(self.base_prior, context.active_before, self.retention_evidence)
        result = original['_select_v3'](self, replace(context, posterior=support))
        result.diagnostics.update(compact_evidence=True, support_source='active_online_evidence')
        if result.diagnostics['autonomous_exits']:
            raise RuntimeError('Compact H produced an autonomous exit')
        return result

    def assign(self, context, selection):
        scores = self.retention_evidence.copy()
        scores[selection.newcomers] = 0.
        prior = evidence_prior(self.base_prior, selection.active_after, scores)
        memory = self.engine.get_module(ModuleRole.MEMORY)
        if isinstance(memory, HierarchicalPairingMemoryModule):
            memory.assign_relation_prior(prior, selection)
        selection.diagnostics.update(prior_initialization='compact_online_evidence',
                                     online_belief_scores=scores[selection.active_after].tolist())
        return prior

    def finish(self, context, selection, prior, **kwargs):
        result = original['_finish_hypothesis_transition'](self, context, selection, prior, **kwargs)
        expected = evidence_prior(self.base_prior, selection.active_after, self.retention_evidence)
        if not np.array_equal(prior, expected):
            raise RuntimeError('Compact choice support differs from retained evidence')
        audit['transitions'] += 1
        audit['replacements'] += int(bool(selection.dropped.size))
        audit['max_active'] = max(audit['max_active'], len(selection.active_after))
        audit['evidence_checks'] += 1
        return result

    try:
        for name, method in zip(names, (initialize, prepare, select, assign, finish)):
            setattr(cls, name, method)
        yield audit
    finally:
        for name, method in original.items():
            setattr(cls, name, method)

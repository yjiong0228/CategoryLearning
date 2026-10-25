"""Opt-in oracle probes applied only after a captured training prefix.

The shared transition still proposes, assigns beliefs, inherits precision and
learns from generated feedback. Oracle availability replaces its selected set;
protection cancels a proposal that would evict the known target. Precision is
clamped at the existing upper bound, and execution can be directed to the
target. These diagnostic interventions use a task oracle, not participant data.
Install in an isolated worker, outside the fitted execution contexts.
"""
from __future__ import annotations

from contextlib import contextmanager
import numpy as np

ARMS = ('BASE', 'RETAIN', 'DISCOVER', 'AVAILABLE', 'PRECISION',
        'AVAILABLE_PRECISION', 'AVAILABLE_EXECUTE', 'AVAILABLE_EXECUTE_PRECISION')


def probe_selection(selection, context, *, target: int, capacity: int,
                    ensure: bool, protect: bool):
    """Override availability while preserving the shared selection contract."""
    before, after = context.active_before, selection.active_after
    diagnostics = dict(selection.diagnostics, probe_inserted=False, probe_protected=False)
    if ensure and target in before and target not in after:
        # An available target needs protection on this first step. Keeping
        # the workspace also resolves tied least-belief slots without a
        # duplicate target or a redundant remove/reinsert operation.
        after = before.copy()
        diagnostics.update(probe_protected=True, capacity_dropped=None)
    elif ensure and target not in after:
        # Use the original workspace for a single valid replacement. Lowest
        # belief and earliest slot resolve eviction without another RNG draw.
        after = before.copy()
        dropped = None
        if len(after) == capacity:
            index = int(np.argmin(context.posterior[after]))
            dropped = int(after[index])
            after = np.delete(after, index)
        after = np.r_[after, target]
        diagnostics.update(probe_inserted=True, capacity_dropped=dropped,
            proposed_hypothesis=target, proposal_source='task_oracle',
            remaining=before.copy(), gamma=context.posterior[before].copy())
    elif protect and target in before and target not in after:
        # Cancel the replacement; the original random draws remain consumed.
        after = before.copy()
        diagnostics.update(probe_protected=True, capacity_dropped=None)
    return type(selection).from_active_sets(before, after, diagnostics=diagnostics)


@contextmanager
def continuation_probe(cls, *, arm: str, target: int, cut: int):
    """Apply one declared probe to all descendants of an intact prefix cloud."""
    from ..model.modules.base_module import ModuleRole
    if arm not in ARMS or cut < 1:
        raise ValueError('Unknown continuation arm or invalid cut')
    if arm == 'BASE':
        yield
        return
    select, finish = cls.select_hypotheses, cls._finish_hypothesis_transition
    ensure = arm in ('DISCOVER', 'AVAILABLE', 'AVAILABLE_PRECISION',
                     'AVAILABLE_EXECUTE', 'AVAILABLE_EXECUTE_PRECISION')
    protect = arm in ('RETAIN', 'AVAILABLE', 'AVAILABLE_PRECISION',
                      'AVAILABLE_EXECUTE', 'AVAILABLE_EXECUTE_PRECISION')
    precision = arm in ('PRECISION', 'AVAILABLE_PRECISION', 'AVAILABLE_EXECUTE_PRECISION')
    execute = arm in ('AVAILABLE_EXECUTE', 'AVAILABLE_EXECUTE_PRECISION')

    def selection(self, context, **kwargs):
        value = select(self, context, **kwargs)
        return probe_selection(value, context, target=target, capacity=self.capacity,
            ensure=ensure and context.trial_index == cut, protect=protect)

    def ending(self, context, selected, prior, **kwargs):
        previous = int(self.executed_hypothesis)
        old_count, old_dwell = self.execution_switch_count, self.execution_dwell_trials
        value = finish(self, context, selected, prior, **kwargs)
        if precision and target in self.active:
            beta = self.engine.get_module(ModuleRole.BETA)
            beta.beta[target] = beta.beta_max
            self.engine.beta = beta.beta.copy()
        if execute:
            if target not in self.active:
                raise ValueError('Availability oracle lost its target')
            natural = int(self.executed_hypothesis)
            self.executed_hypothesis = int(target)
            self.execution_probabilities = np.zeros(self.total_hypo)
            self.execution_probabilities[target] = 1.
            if hasattr(self, 'execution_selection_probabilities'):
                self.execution_selection_probabilities = self.execution_probabilities.copy()
            # Switch counters describe the intervened path. They do not
            # determine candidate search or feedback learning.
            switched = previous != target
            self.execution_switch_count = old_count + int(switched)
            self.execution_dwell_trials = 1 if switched else old_dwell + 1
            self.current_execution_switch_event = switched
            self.current_execution_switch_probability = float(switched)
            value.update(probe_natural_execution=natural, choice_weights=self.execution_probabilities.tolist(),
                executed_hypothesis=int(target), executed_hypothesis_previous=previous,
                execution_switch_event=switched, execution_switch_probability=float(switched),
                execution_dwell_trials=self.execution_dwell_trials, effective_participation=1.)
        value.update(probe_arm=arm, probe_precision_clamped=precision and target in self.active)
        self.transition_log[-1].update(value)
        self.transition_rate_log[-1].update(value)
        return value

    cls.select_hypotheses, cls._finish_hypothesis_transition = selection, ending
    try:
        yield
    finally:
        cls.select_hypotheses, cls._finish_hypothesis_transition = select, finish

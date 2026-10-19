"""Execute the existing dominant rule while retaining the candidate beliefs.

Install inside the existing continuous-choice belief intervention. The original
transition chooses its dominant rule using the current belief, previous dominant
rule and execution inertia. This adapter gives that selected rule all response
weight. Active rules, belief evidence, admission, precision learning and the
dominant-rule selection algorithm retain their existing definitions. The changed
responses can change subsequent feedback and learning in autonomous simulations.
No task category or oral report enters this policy; no individual parameter is
added. Disabled mode preserves the class exactly.
"""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256

import numpy as np

POLICY = 'existing_dominant_rule_only'


@contextmanager
def dominant_rule_execution(cls, *, enabled: bool = False):
    """Give the already selected dominant rule unit response weight."""
    if type(enabled) is not bool:
        raise ValueError('Dominant-rule execution must be explicitly enabled or disabled')
    events = []
    if not enabled:
        yield events
        return
    names = ('__init__', '_finish_hypothesis_transition', 'state_dict', 'load_state_dict')
    original = {name: getattr(cls, name) for name in names}

    def select(self):
        rule = int(self.dominant_hypothesis)
        weights = np.asarray(self.execution_probabilities)
        if rule not in self.active or weights[rule] != weights.max():
            raise ValueError('The existing dominant rule must maximize response weights')
        before = weights.copy()
        self.execution_probabilities = np.zeros(self.total_hypo)
        self.execution_probabilities[rule] = 1.
        return rule, before

    def initialize(self, *args, **kwargs):
        original['__init__'](self, *args, **kwargs)
        if not self.continuous_choice_enabled or getattr(self, 'dominant_execution_enabled', False):
            raise ValueError('Dominant execution requires continuous-choice selection exactly once')
        self.dominant_execution_enabled = True
        self.configuration_signature = sha256(f'{self.configuration_signature}:{POLICY}:v1'.encode()).hexdigest()
        select(self)

    def finish(self, context, selection, prior, **kwargs):
        event = original['_finish_hypothesis_transition'](self, context, selection, prior, **kwargs)
        rule, before = select(self)
        diagnostics = dict(rule_execution_policy=POLICY,
            selection_weights_before_execution=before.tolist(),
            choice_weights=self.execution_probabilities.tolist(), effective_participation=1.)
        event.update(diagnostics)
        self.transition_log[-1].update(diagnostics)
        self.transition_rate_log[-1].update(diagnostics)
        events.append(dict(trial=int(context.trial_index), selected_rule=rule,
                           selected_prior_weight=float(before[rule]), active_count=len(self.active)))
        return event

    def snapshot(self):
        return dict(original['state_dict'](self), rule_execution_policy=POLICY)

    def restore(self, state):
        if state.get('rule_execution_policy') != POLICY:
            raise ValueError('Snapshot uses a different rule execution policy')
        weights = np.asarray(state.get('execution_probabilities'))
        rule = state.get('executed_hypothesis')
        expected = np.zeros(self.total_hypo)
        if (not isinstance(rule, (int, np.integer)) or isinstance(rule, bool)
                or not 0 <= rule < self.total_hypo):
            raise ValueError('Invalid saved dominant rule')
        expected[rule] = 1.
        if not np.array_equal(weights, expected):
            raise ValueError('Snapshot response weights must select only its dominant rule')
        original['load_state_dict'](self, {k: v for k, v in state.items() if k != 'rule_execution_policy'})

    try:
        for name, method in zip(names, (initialize, finish, snapshot, restore)):
            setattr(cls, name, method)
        yield events
    finally:
        for name, method in original.items():
            setattr(cls, name, method)

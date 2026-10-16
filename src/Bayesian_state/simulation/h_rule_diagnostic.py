"""Process-local H interventions for causal diagnosis at frozen parameters.

The oracle identity is supplied by a task-geometry audit. It is privileged
information, never an estimated cognitive parameter. All ordinary transition,
feedback, precision and pairing operations delegate to the shared model.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy

import numpy as np

from ..model.modules.base_module import ModuleRole
from ..model.modules.hypothesis_transition.contracts import HypothesisSelection
from ..model.modules.pairing_memory import HierarchicalPairingMemoryModule


ARMS = ('BASE', 'ONCE', 'ONCE_EQUAL', 'KEEP', 'KEEP_EQUAL', 'ONLY', 'REJECT')


def include_target(before, after, target, capacity, scores):
    """Protect a target, displacing the lowest-score other rule when necessary."""
    final = list(map(int, after))
    displaced = None
    if target not in final:
        if len(final) == capacity:
            index = int(np.argmin(scores))
            displaced = final.pop(index)
        final.append(int(target))
    # Retain the shared core's old-then-new ordering.
    return np.asarray([h for h in before if h in final] +
                      [h for h in final if h not in before], dtype=int), displaced


@contextmanager
def intervention(cls, *, arm, target, onset, tape=None):
    """Install one diagnostic arm; yield its transition RNG / proposal tape.

REPLAY reproduces BASE exactly. REJECT replays the same proposal opportunities
after onset and adds the zero-evidence newcomer to the existing capacity
comparison. An already-active replay candidate is recorded as already available.
Oracle arms intervene at onset; KEEP additionally prevents subsequent target
removal, EQUAL assigns uniform rule priors once, and ONLY clamps the active set
and rule prior to the target. Pairing beliefs still update through the core.
"""
    if arm not in (*ARMS, 'REPLAY') or onset < 1:
        raise ValueError('Unsupported diagnostic arm or onset')
    if arm in ('REJECT', 'REPLAY') and tape is None:
        raise ValueError('Replay requires a recorded baseline tape')
    select, draw, scores, assign = (cls.select_hypotheses, cls._draw_v3_proposal,
                                  cls._retention_scores, cls.assign_prior)
    recorded = []
    state = dict(forced=None, compare_new=False)

    def choose(self, context, **kwargs):
        t = context.trial_index
        row = dict(rng=deepcopy(self.trial_rng.bit_generator.state), proposed=-1)
        recorded.append(row)
        old_probability = self.current_event_probability
        state.update(forced=None, compare_new=False)
        available = False
        if arm in ('REJECT', 'REPLAY') and t >= onset:
            self.trial_rng.bit_generator.state = deepcopy(tape[t]['rng'])
            candidate = tape[t]['proposed']
            available = candidate >= 0 and candidate in context.active_before
            self.current_event_probability = float(candidate >= 0 and not available)
            state['forced'] = candidate if candidate >= 0 and not available else None
            state['compare_new'] = arm == 'REJECT'
        elif arm not in ('BASE', 'REJECT', 'REPLAY') and t == onset:
            if target not in context.active_before:
                self.current_event_probability = 1.
                state['forced'] = target
        try:
            selection = select(self, context, **kwargs)
        finally:
            self.current_event_probability = old_probability
            state['compare_new'] = False
        diagnostic = dict(selection.diagnostics)
        row['proposed'] = diagnostic.get('proposed_hypothesis')
        row['proposed'] = -1 if row['proposed'] is None else int(row['proposed'])
        diagnostic['tape_already_available'] = available
        after = selection.active_after
        protect = t >= onset and (arm in ('KEEP', 'KEEP_EQUAL', 'ONLY') or
                                  (t == onset and arm in ('ONCE', 'ONCE_EQUAL')))
        if protect:
            if arm == 'ONLY':
                after = np.array([target], dtype=int)
                diagnostic['oracle_clamp'] = True
            else:
                after, displaced = include_target(context.active_before, after, target,
                    self.capacity, scores(self, after, True))
                if target in diagnostic.get('autonomous_exits', []):
                    diagnostic['oracle_prevented_exit'] = True
                    diagnostic['autonomous_exits'] = [h for h in diagnostic['autonomous_exits'] if h != target]
                if diagnostic.get('capacity_dropped') == target:
                    diagnostic['oracle_prevented_capacity'] = True
                    diagnostic['capacity_dropped'] = None
                if displaced is not None and displaced in context.active_before:
                    diagnostic['capacity_dropped'] = displaced
            diagnostic['oracle_intervention'] = True
        return HypothesisSelection.from_active_sets(context.active_before, after, diagnostics=diagnostic)

    def propose(self, before, remaining, gamma, diagnostic):
        original = draw(self, before, remaining, gamma, diagnostic)
        if state['forced'] is None:
            return original
        candidate = int(state['forced'])
        if candidate in before:
            raise ValueError('Forced candidate must be inactive')
        diagnostic['proposed_hypothesis'] = candidate
        diagnostic['diagnostic_proposal'] = True
        return candidate

    def retention(self, rules, mean=False):
        values = scores(self, rules, mean)
        # _select_v3 already handles a minimum at len(remaining) as rejection.
        return np.r_[values, 0.] if state['compare_new'] else values

    def prior(self, context, selection, **kwargs):
        clamp = arm == 'ONLY' and context.trial_index >= onset
        equal = arm in ('ONCE_EQUAL', 'KEEP_EQUAL') and context.trial_index == onset
        if not (clamp or equal):
            return assign(self, context, selection, **kwargs)
        result = np.zeros(self.total_hypo)
        result[selection.active_after] = 1. / len(selection.active_after)
        memory = self.engine.get_module(ModuleRole.MEMORY)
        if isinstance(memory, HierarchicalPairingMemoryModule):
            memory.assign_relation_prior(result, selection)
        return result

    cls.select_hypotheses, cls._draw_v3_proposal = choose, propose
    cls._retention_scores, cls.assign_prior = retention, prior
    try:
        yield recorded
    finally:
        cls.select_hypotheses, cls._draw_v3_proposal = select, draw
        cls._retention_scores, cls.assign_prior = scores, assign

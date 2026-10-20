"""Initialize unseen candidates from discounted completed feedback episodes.

Install before POOL/RETAIN/SEPARATE and the finite-region association adapter.
At proposal, an unseen candidate starts uniformly in an independent scratch
bank. One chronological replay uses eta * delta**age per episode, at beta_init.
POOL and SEPARATE then evaluate its learned rows against completed history.
Admission retains the scratch bank; rejection restores the original bank.
Previously active rules recover their saved associations without rehearsal.

The retrospective admission score evaluates the learned state on the same
remembered episodes. Parameter fitting uses the prospective online choice
likelihood, which predicts each new response before its feedback.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from ...active_rule_associations import ActiveRuleAssociations
from ..base_module import ModuleRole
from .episodic_admission import recalled_episodes


@dataclass(frozen=True)
class CandidateRehearsal:
    candidate: int
    original: ActiveRuleAssociations
    scratch: ActiveRuleAssociations
    count: int
    exposure: float


def rehearse_candidate(transition, candidate: int) -> CandidateRehearsal:
    """Build a temporary candidate while leaving the live association bank intact."""
    original = transition.region_associations
    state = original.state_dict()
    if (transition.reentry_precision_seen[candidate]
            or state['feedback_counts'][candidate] != 0
            or np.any(state['logits'][candidate] != 0)):
        raise ValueError('Only an unseen, untrained candidate can be rehearsed')
    episodes, weights = recalled_episodes(transition.admission_history, transition.retention_decay)
    if any(e.feedback not in (0., 1.) or e.pairing is not None for e in episodes):
        raise ValueError('Candidate rehearsal requires completed binary feedback')
    scratch = ActiveRuleAssociations(*state['logits'].shape, learning_rate=original.learning_rate)
    scratch.load_state_dict(state)
    evaluator = transition.engine.observation_likelihood
    beta = transition.engine.get_module(ModuleRole.BETA).beta_init
    data = (np.asarray([e.stimulus for e in episodes]),
            np.asarray([e.choice for e in episodes]), np.asarray([e.feedback for e in episodes]))
    q = evaluator.partition.get_category_probabilities(candidate, data, beta,
                                                       distance_mode=evaluator.distance_mode).T
    for row in range(len(episodes)-1, -1, -1):
        episode = episodes[row]
        evidence = scratch.evaluate_feedback([candidate], q[row:row+1], episode.choice, episode.feedback)
        scratch.commit_feedback(evidence, learning_weight=float(weights[row]))
    return CandidateRehearsal(int(candidate), original, scratch, len(episodes), float(weights.sum()))


@contextmanager
def rehearse_new_candidates(cls, *, enabled: bool = False):
    """Add candidate preparation without changing proposal draws or capacity rules."""
    if type(enabled) is not bool:
        raise ValueError('Candidate rehearsal must be explicitly enabled or disabled')
    if not enabled:
        yield
        return
    names = ('__init__', '_draw_v3_proposal', '_finish_hypothesis_transition', 'state_dict', 'load_state_dict')
    original = {name: getattr(cls, name) for name in names}

    def initialize(self, *args, **kwargs):
        original['__init__'](self, *args, **kwargs)
        if not self.continuous_choice_enabled or hasattr(self, '_candidate_rehearsal'):
            raise ValueError('Candidate rehearsal requires the full-v3 finite-region profile exactly once')
        self._candidate_rehearsal = None
        self._STATE_FIELDS += ('region_rehearsal_count', 'region_rehearsal_exposure')
        self.region_rehearsal_count = np.zeros(self.total_hypo, dtype=np.int64)
        self.region_rehearsal_exposure = np.zeros(self.total_hypo)
        self.configuration_signature = sha256(
            f'{self.configuration_signature}:chronological-discounted-candidate-rehearsal-v1'.encode()).hexdigest()

    def draw(self, before, remaining, gamma, diagnostics):
        if self._candidate_rehearsal is not None:
            raise RuntimeError('Previous candidate rehearsal was not resolved')
        candidate = original['_draw_v3_proposal'](self, before, remaining, gamma, diagnostics)
        seen = bool(self.reentry_precision_seen[candidate])
        diagnostics.update(candidate_rehearsal=False, candidate_previously_active=seen,
                           candidate_rehearsal_count=0, candidate_rehearsal_exposure=0.)
        if not seen:
            pending = rehearse_candidate(self, candidate)
            self._candidate_rehearsal = pending
            self.region_associations = pending.scratch
            diagnostics.update(candidate_rehearsal=True, candidate_rehearsal_count=pending.count,
                               candidate_rehearsal_exposure=pending.exposure)
        return candidate

    def finish(self, context, selection, prior, **kwargs):
        pending = self._candidate_rehearsal
        admitted = pending is not None and pending.candidate in selection.newcomers
        if pending is not None:
            if self.region_associations is not pending.scratch:
                raise RuntimeError('Candidate scratch associations were replaced during comparison')
            self.region_associations = pending.scratch if admitted else pending.original
            if admitted:
                self.region_rehearsal_count[pending.candidate] = pending.count
                self.region_rehearsal_exposure[pending.candidate] = pending.exposure
        selection.diagnostics['candidate_rehearsal_committed'] = bool(admitted)
        self._candidate_rehearsal = None
        return original['_finish_hypothesis_transition'](self, context, selection, prior, **kwargs)

    def snapshot(self):
        if self._candidate_rehearsal is not None:
            raise RuntimeError('Snapshot requires a completed candidate transition')
        return original['state_dict'](self)

    def restore(self, state):
        counts = np.asarray(state.get('region_rehearsal_count'))
        exposure = np.asarray(state.get('region_rehearsal_exposure'))
        if (counts.shape != (self.total_hypo,) or not np.issubdtype(counts.dtype, np.integer)
                or np.any(counts < 0) or exposure.shape != counts.shape
                or not np.issubdtype(exposure.dtype, np.floating) or not np.isfinite(exposure).all()
                or np.any(exposure < 0) or np.any(exposure > counts)
                or np.any((counts == 0) != (exposure == 0))):
            raise ValueError('Invalid candidate rehearsal snapshot')
        original['load_state_dict'](self, state)
        self._candidate_rehearsal = None

    try:
        for name, method in zip(names, (initialize, draw, finish, snapshot, restore)):
            setattr(cls, name, method)
        yield
    finally:
        for name, method in original.items(): setattr(cls, name, method)

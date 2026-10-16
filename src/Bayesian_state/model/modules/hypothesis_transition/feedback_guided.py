"""Opt-in, process-local feedback-guided proposal/admission experiments.

The proposal adapter delegates route, parent, kernel and RNG selection to the
existing H implementation. It tilts only the final inactive-rule draw. State
needed by particle cloning is included in the ordinary transition snapshot.
Completed feedback is evaluated at newcomer beta_init; condition 3 uses the
pairing marginal available BEFORE that feedback. No task category is an input.
"""
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256

import numpy as np

from ..base_module import ModuleRole
from ..pairing_memory import HierarchicalPairingMemoryModule


def probability_roundoff(values):
    """Clip at most eight machine epsilons above one; reject substantive errors."""
    values = np.asarray(values, float)
    if (not np.isfinite(values).all() or np.any(values < 0)
            or np.any(values > 1 + 8 * np.finfo(float).eps)):
        raise ValueError('Feedback probability outside rounding tolerance')
    return np.minimum(values, 1.)


@contextmanager
def retention_roundoff_guard(cls):
    """Apply the same narrow bound check to every arm; record every correction."""
    original = cls._previous_log_evidence
    prepare = HierarchicalPairingMemoryModule.prepare_feedback
    corrections = []
    def pairing_feedback(self, kernel):
        bounded = probability_roundoff(kernel)
        if not np.array_equal(bounded, kernel):
            transition = self.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
            corrections.append(dict(phase='pairing_kernel', trial=int(transition.trial_index)-1,
                                    maximum=float(np.max(kernel))))
        return prepare(self, bounded)
    def checked(self):
        values = self.engine.likelihood
        active = np.asarray(values)[self.active]
        bounded = probability_roundoff(active)
        if np.array_equal(active, bounded):
            return original(self)
        corrections.append(dict(phase='retention', trial=int(self.trial_index), maximum=float(active.max())))
        self.engine.likelihood = np.asarray(values).copy()
        self.engine.likelihood[self.active] = bounded
        try:
            return original(self)
        finally:
            self.engine.likelihood = values
    try:
        cls._previous_log_evidence = checked
        HierarchicalPairingMemoryModule.prepare_feedback = pairing_feedback
        yield corrections
    finally:
        cls._previous_log_evidence = original
        HierarchicalPairingMemoryModule.prepare_feedback = prepare


def tilt_probability(probability, evidence):
    """Normalize q(h) L(feedback|h), preserving the base proposal's support."""
    q, likelihood = np.asarray(probability, float), probability_roundoff(evidence)
    if (q.shape != likelihood.shape or q.ndim != 1 or not np.isfinite(q).all()
            or not np.isfinite(likelihood).all() or np.any(q < 0)
            or np.any((likelihood < 0) | (likelihood > 1)) or not np.isclose(q.sum(), 1)):
        raise ValueError('Invalid proposal or feedback probabilities')
    weighted = q * likelihood
    if weighted.sum() <= 0:
        raise ValueError('Feedback eliminates the entire proposal support')
    return weighted / weighted.sum()


class _ProposalDraw:
    """Intercept exactly the final catalogue draw; leave other RNG calls intact."""
    def __init__(self, rng, inactive, likelihood):
        self.rng, self.inactive, self.likelihood = rng, inactive, likelihood
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.rng, name)

    def choice(self, a, *args, **kwargs):
        if np.ndim(a) == 1 and np.array_equal(a, self.inactive):
            kwargs['p'] = tilt_probability(kwargs['p'], self.likelihood)
            self.calls += 1
        return self.rng.choice(a, *args, **kwargs)


@contextmanager
def feedback_intervention(cls, *, proposal=False, admission=False):
    """Install an experimental arm and restore every method on exit.

Admission compares newcomer log feedback/chance evidence with incumbent mean
evidence and permits rejection. An accepted candidate starts with that evidence
and exposure one. Proposal-only leaves retention initialization unchanged.
"""
    if not proposal and not admission:
        yield
        return
    names = ('__init__', 'record_outcome', '_draw_v3_proposal', '_retention_scores',
             '_finish_hypothesis_transition')
    original = {name: getattr(cls, name) for name in names}

    def initialize(self, *args, **kwargs):
        original['__init__'](self, *args, **kwargs)
        if self.intervention_kind != 'E' or self.intervention['capacity'] != 'evidence_mean':
            raise ValueError('Feedback experiment requires E/evidence_mean retention')
        self._STATE_FIELDS += ('proposal_observation', 'proposal_pairing')
        self.proposal_observation = None
        self.proposal_pairing = None
        self._candidate_score = None
        self.configuration_signature = sha256(
            f'{self.configuration_signature}:feedback-v1:{proposal}:{admission}'.encode()).hexdigest()

    def record(self, observation):
        original['record_outcome'](self, observation)
        self.proposal_observation = deepcopy(observation)
        memory = self.engine.get_module(ModuleRole.MEMORY)
        self.proposal_pairing = (memory.pairing_marginal()
                                if hasattr(memory, 'pairing_marginal') else None)

    def evidence(self, rules):
        if self.proposal_observation is None:
            raise RuntimeError('Candidate evaluation requires completed feedback')
        likelihood = self.engine.observation_likelihood
        beta = self.engine.get_module(ModuleRole.BETA).beta_init
        if likelihood.feedback_likelihood_mode == 'hierarchical_pairing':
            return probability_roundoff(likelihood.pairing_kernel(self.proposal_observation, rules, beta) @ self.proposal_pairing)
        return probability_roundoff(likelihood.binary_feedback_likelihood(self.proposal_observation, rules, beta))

    def draw(self, before, remaining, gamma, diagnostics):
        self._candidate_score = None
        inactive = np.setdiff1d(np.arange(self.total_hypo), before)
        values = evidence(self, inactive) if proposal else None
        rng = self.trial_rng
        adapter = _ProposalDraw(rng, inactive, values) if proposal else None
        try:
            if adapter is not None:
                self.trial_rng = adapter
            candidate = original['_draw_v3_proposal'](self, before, remaining, gamma, diagnostics)
        finally:
            self.trial_rng = rng
        if adapter is not None and adapter.calls != 1:
            raise RuntimeError('Expected exactly one final candidate draw')
        value = (float(values[np.flatnonzero(inactive == candidate)[0]]) if proposal
                 else float(evidence(self, [candidate])[0]))
        feedback = self.proposal_observation[2]
        categories = self.engine.partition.n_cats
        chance = ((.5 if feedback == 0 else .25) if self.proposal_pairing is not None
                  else (1 / categories if feedback == 1 else 1 - 1 / categories))
        self._candidate_score = float(np.log(max(value, np.finfo(float).tiny) / chance))
        diagnostics.update(candidate_feedback_probability=value,
                           candidate_initial_evidence=self._candidate_score)
        return candidate

    def scores(self, rules, mean=False):
        values = original['_retention_scores'](self, rules, mean)
        if admission and self._candidate_score is not None and mean:
            return np.r_[values, self._candidate_score]
        return values

    def finish(self, context, selection, prior, **kwargs):
        result = original['_finish_hypothesis_transition'](self, context, selection, prior, **kwargs)
        if admission and selection.newcomers.size:
            self.retention_evidence[selection.newcomers] = self._candidate_score
            self.retention_exposure[selection.newcomers] = 1.
        if admission:
            # The base class assumes mandatory admission. Its analytic marginal
            # is unavailable under this adapter; recorded events remain valid.
            selection.diagnostics['admission_probability_given_search'] = np.nan
            result.update(admission_probability_given_search=np.nan, swap_probability=np.nan,
                          analytic_admission_available=False,
                          retention_evidence=self.retention_evidence[self.active].tolist(),
                          retention_exposure=self.retention_exposure[self.active].tolist())
            self.transition_rate_log[-1].update(result)
        self._candidate_score = None
        return result

    replacements = (initialize, record, draw, scores, finish)
    try:
        for name, method in zip(names, replacements):
            setattr(cls, name, method)
        yield
    finally:
        for name, method in original.items():
            setattr(cls, name, method)

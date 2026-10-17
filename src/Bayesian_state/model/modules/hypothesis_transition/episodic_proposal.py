"""Experimental use of recalled evidence in candidate generation and belief.

The POOL admission adapter supplies completed episodes, their shared decay and
newcomer-precision likelihoods. Two independent switches ask whether that same
information should guide the candidate draw and reconstruct belief when a
candidate is admitted. No oral content, correct-rule identity, new window or
additional free coefficient enters either operation. The unchanged route and
parent kernels define proposal support; cumulative recalled evidence tilts it.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from ..base_module import ModuleRole
from ..pairing_memory import HierarchicalPairingMemoryModule
from .episodic_admission import Episode, recalled_log_evidence


@dataclass(frozen=True, slots=True)
class RecallCache:
    """Immutable computational memo of evidence derivable from saved episodes."""
    head: Episode
    total: tuple[float, ...]
    exposure: float
    count: int
    decay: float
    beta: float

    def __post_init__(self):
        if (not isinstance(self.head, Episode) or type(self.total) is not tuple or not self.total
                or not np.isfinite(self.total).all() or not np.isfinite(self.exposure) or self.exposure <= 0
                or type(self.count) is not int or self.count < 1 or not 0 <= self.decay <= 1
                or not np.isfinite(self.beta) or self.beta <= 0):
            raise ValueError('Invalid recalled-evidence cache')

    def __deepcopy__(self, memo):
        return self


def advance_recall(head: Episode, cache: RecallCache | None, *, decay: float,
                   likelihood, rules: np.ndarray, beta: float) -> tuple[RecallCache, float]:
    """Evaluate newly completed episodes and compose the same geometric sum.

    The cache is read only at search. Including it in particle snapshots avoids
    repeating identical past geometry after resampling. Its values are fully
    determined by the episode chain and introduce no additional learning state.
    """
    if not isinstance(head, Episode) or not np.isfinite(decay) or not 0 <= decay <= 1:
        raise ValueError('Need completed episodes and a valid evidence decay')
    if not np.array_equal(rules, np.arange(len(rules))) or (cache is not None and not isinstance(cache, RecallCache)):
        raise ValueError('Cache requires the ordered complete catalogue and a valid snapshot')
    if cache is not None and (cache.decay != decay or cache.beta != beta or len(cache.total) != len(rules)):
        raise ValueError('Recall cache differs from the current numerical model')
    stop = cache.head if cache is not None else None
    episodes, node = [], head
    while node is not stop and node is not None:
        episodes.append(node)
        node = node.previous
    if node is not stop:
        raise ValueError('Recall cache is not an ancestor of the current history')
    if not episodes:
        return cache, 0.
    values, maximum = recalled_log_evidence(likelihood, episodes, rules, beta)
    weights = decay ** np.arange(len(episodes))
    total = weights @ values
    exposure = float(weights.sum())
    count = len(episodes)
    if cache is not None:
        multiplier = decay ** len(episodes)
        total += multiplier * np.asarray(cache.total)
        exposure += multiplier * cache.exposure
        count += cache.count
    return RecallCache(head, tuple(map(float, total)), exposure, count, decay, beta), maximum


def tilt_log_evidence(probability: np.ndarray, evidence: np.ndarray) -> np.ndarray:
    """Normalize q times exp(evidence), retaining exact zeros in q."""
    q, score = np.asarray(probability, float), np.asarray(evidence, float)
    if (q.ndim != 1 or not len(q) or q.shape != score.shape
            or not np.isfinite(q).all() or not np.isfinite(score).all()
            or np.any(q < 0) or not np.isclose(q.sum(), 1., atol=1e-12, rtol=0)):
        raise ValueError('Need normalized proposal and finite recalled log evidence')
    positive = q > 0
    values = np.log(q[positive]) + score[positive]
    values = np.exp(values - values.max())
    result = np.zeros_like(q)
    result[positive] = values / values.sum()
    return result


def recalled_prior(base_prior: np.ndarray, selected: np.ndarray,
                   recalled_rules: np.ndarray, evidence: np.ndarray) -> np.ndarray:
    """Reconstruct one prior on the final active set using common recall."""
    base = np.asarray(base_prior, float)
    selected, rules = np.asarray(selected, int), np.asarray(recalled_rules, int)
    score = np.asarray(evidence, float)
    if (base.ndim != 1 or selected.ndim != 1 or rules.ndim != 1 or not len(selected)
            or len(np.unique(selected)) != len(selected) or len(np.unique(rules)) != len(rules)
            or np.any(selected < 0) or np.any(selected >= len(base)) or score.shape != rules.shape
            or not set(selected) <= set(rules) or not np.isfinite(base).all() or np.any(base < 0)
            or base[selected].sum() <= 0):
        raise ValueError('Every retained rule needs common recalled evidence and base support')
    indices = np.array([np.flatnonzero(rules == rule).item() for rule in selected])
    result = np.zeros_like(base)
    result[selected] = tilt_log_evidence(base[selected] / base[selected].sum(), score[indices])
    return result


class _RecalledDraw:
    def __init__(self, rng, inactive: np.ndarray, evidence: np.ndarray):
        self.rng, self.inactive, self.evidence = rng, inactive, evidence
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.rng, name)

    def choice(self, a, *args, **kwargs):
        if np.ndim(a) == 1 and np.array_equal(a, self.inactive):
            kwargs['p'] = tilt_log_evidence(kwargs['p'], self.evidence)
            self.calls += 1
        return self.rng.choice(a, *args, **kwargs)


@contextmanager
def episodic_proposal(cls, *, proposal: bool = False, prior: bool = False):
    """Install after POOL admission; restore all methods on leaving the scope.

    Prior reconstruction uses base rule weights and common retrospective
    evidence once, when an admitted candidate changes the active set. Ordinary
    between-search belief updates retain their original memory operation.
    This is an explicit retrospective belief hypothesis; it is not an exact
    posterior for the full model with evolving precision and pairing states.
    """
    if not proposal and not prior:
        yield
        return
    names = ('__init__', '_draw_v3_proposal', '_assign_v3_prior')
    original = {name: getattr(cls, name) for name in names}

    def initialize(self, *args, **kwargs):
        original['__init__'](self, *args, **kwargs)
        if not hasattr(self, 'admission_history'):
            raise ValueError('Install POOL episodic admission before recalled proposal/prior')
        if proposal:
            self._STATE_FIELDS += ('proposal_recall_cache',)
            self.proposal_recall_cache = None
        self.configuration_signature = sha256(
            f'{self.configuration_signature}:episodic-proposal-v1:{proposal}:{prior}'.encode()).hexdigest()

    def draw(self, before, remaining, gamma, diagnostics):
        if not proposal:
            return original['_draw_v3_proposal'](self, before, remaining, gamma, diagnostics)
        inactive = np.setdiff1d(np.arange(self.total_hypo), before)
        cache, maximum = advance_recall(self.admission_history, self.proposal_recall_cache,
            decay=self.retention_decay, likelihood=self.engine.observation_likelihood,
            rules=np.arange(self.total_hypo), beta=self.engine.get_module(ModuleRole.BETA).beta_init)
        self.proposal_recall_cache = cache
        scores = np.asarray(cache.total)[inactive]
        rng = self.trial_rng
        adapter = _RecalledDraw(rng, inactive, scores)
        try:
            self.trial_rng = adapter
            candidate = original['_draw_v3_proposal'](self, before, remaining, gamma, diagnostics)
        finally:
            self.trial_rng = rng
        if adapter.calls != 1:
            raise RuntimeError('Expected exactly one final recalled candidate draw')
        diagnostics.update(proposal_recall='discounted_cumulative_evidence',
                           proposal_recalled_episodes=cache.count, proposal_recalled_exposure=cache.exposure,
                           proposal_evidence_range=float(np.ptp(scores)),
                           proposal_feedback_roundoff_max=max(0., maximum - 1.))
        return candidate

    def assign(self, context, selection):
        if not prior or not selection.newcomers.size:
            return original['_assign_v3_prior'](self, context, selection)
        recalled = self._recall
        if recalled is None or not set(selection.active_after) <= set(recalled['rules']):
            raise ValueError('Recalled prior requires POOL evidence for the final active set')
        result = recalled_prior(self.base_prior, selection.active_after, recalled['rules'], recalled['total'])
        memory = self.engine.get_module(ModuleRole.MEMORY)
        if isinstance(memory, HierarchicalPairingMemoryModule):
            memory.assign_relation_prior(result, selection)
        selection.diagnostics.update(prior_initialization='common_recalled_evidence',
                                     recalled_newcomer_mass=float(result[selection.newcomers].sum()))
        return result

    try:
        for name, replacement in zip(names, (initialize, draw, assign)):
            setattr(cls, name, replacement)
        yield
    finally:
        for name, method in original.items():
            setattr(cls, name, method)

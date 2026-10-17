"""Experimental admission from discounted, completed feedback episodes.

Only a proposed rule (NEW), or that rule and its surviving competitors (POOL
and DRAW), are evaluated retrospectively. Every recalled rule uses beta_init.
DRAW samples common episode weights from Dirichlet(discounted counts), making
the evidence comparison uncertain while preserving its across-rule dependence.
The fitted H evidence decay supplies the discount; no fixed window is imposed.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from ..base_module import ModuleRole
from .feedback_guided import probability_roundoff
from ....hypothesis_space.observation_model.pairing import pairing_feedback_kernel


@dataclass(frozen=True, slots=True)
class Episode:
    """Immutable persistent history, shared safely by resampled particles."""
    stimulus: tuple[float, ...]
    choice: int
    feedback: float
    pairing: tuple[float, ...] | None
    previous: Episode | None = None

    def __post_init__(self) -> None:
        if (type(self.stimulus) is not tuple or not self.stimulus
                or not np.isfinite(self.stimulus).all()
                or isinstance(self.choice, bool) or int(self.choice) != self.choice or self.choice < 1
                or self.feedback not in (0., .5, 1.)):
            raise ValueError('Invalid recalled feedback episode')
        if self.pairing is not None:
            if (type(self.pairing) is not tuple or len(self.pairing) != 3
                    or not np.isfinite(self.pairing).all() or min(self.pairing) < 0
                    or not np.isclose(sum(self.pairing), 1., atol=1e-12, rtol=0)):
                raise ValueError('Invalid recalled pairing weights')
        elif self.feedback == .5:
            raise ValueError('Half feedback requires pairing weights')
        if self.previous is not None and not isinstance(self.previous, Episode):
            raise ValueError('Invalid previous episode')

    def __deepcopy__(self, memo):
        # All payloads and links are immutable; history size does not multiply
        # at each particle resampling. New feedback creates a new head node.
        return self


def recalled_episodes(head: Episode, decay: float) -> tuple[list[Episode], np.ndarray]:
    """Newest first, with exact geometric weights until floating underflow."""
    if not isinstance(head, Episode) or not np.isfinite(decay) or not 0 <= decay <= 1:
        raise ValueError('Recall requires a history and decay in [0,1]')
    episodes, weights, weight = [], [], 1.
    node = head
    while node is not None and weight > 0:
        episodes.append(node); weights.append(weight)
        node = node.previous
        weight *= decay
        if weight < np.finfo(float).tiny:
            # Subnormal multiplication can otherwise stick at the smallest
            # float. Its omitted total is below ordinary probability precision.
            break
    return episodes, np.asarray(weights)


def recalled_log_evidence(likelihood, episodes: list[Episode], rules: np.ndarray,
                          beta: float) -> tuple[np.ndarray, float]:
    """Batch the shared category emissions; reuse the shared pairing kernel."""
    rules = np.asarray(rules, dtype=int)
    if not episodes or rules.ndim != 1 or not len(rules):
        raise ValueError('Recall needs episodes and rules')
    data = (np.asarray([e.stimulus for e in episodes]),
            np.asarray([e.choice for e in episodes]), np.asarray([e.feedback for e in episodes]))
    beta = likelihood.default_beta if likelihood.beta_source == 'fixed' else beta
    # The public partition operation accepts multiple stimuli, using exactly
    # the same geometry and category probabilities as one-trial likelihoods.
    p = np.stack([likelihood.partition.get_category_probabilities(int(h), data, beta,
                  distance_mode=likelihood.distance_mode).T for h in rules], axis=1)
    n, k, categories = p.shape
    if (n != len(episodes) or k != len(rules) or not np.isfinite(p).all()
            or np.any(p < 0) or not np.allclose(p.sum(axis=-1), 1., rtol=1e-10, atol=1e-12)):
        raise ValueError('Invalid recalled category probabilities')
    values = np.empty((n, k)); chance = np.empty(n)
    for choice, feedback in sorted(set(zip(data[1], data[2]))):
        mask = (data[1] == choice) & (data[2] == feedback)
        selected = p[mask]
        if likelihood.feedback_likelihood_mode == 'hierarchical_pairing':
            kernel = pairing_feedback_kernel(selected.reshape(-1, categories), int(choice), float(feedback))
            kernel = kernel.reshape(mask.sum(), k, 3)
            pairing = np.asarray([e.pairing for e, keep in zip(episodes, mask) if keep])
            values[mask] = np.sum(kernel * pairing[:, None, :], axis=-1)
            chance[mask] = .5 if feedback == 0 else .25
        else:
            if likelihood.feedback_likelihood_mode != 'category_feedback' or likelihood.feedback_lapse != 0:
                raise ValueError('Recall requires the 0923 exact feedback profile')
            values[mask] = (selected[:, :, choice-1] if feedback == 1 else
                            selected[:, :, np.arange(categories) != choice-1].sum(axis=-1))
            chance[mask] = 1/categories if feedback == 1 else 1-1/categories
    maximum = float(values.max())
    values = probability_roundoff(values)
    return np.log(np.maximum(values, np.finfo(float).tiny))-np.log(chance[:, None]), maximum


def comparison_weights(weights: np.ndarray, mode: str, rng) -> np.ndarray:
    """Return expected or sampled weights over the same remembered episodes."""
    weights = np.asarray(weights, dtype=float)
    if (weights.ndim != 1 or not len(weights) or not np.isfinite(weights).all()
            or np.any(weights <= 0) or mode not in ('NEW', 'POOL', 'DRAW')):
        raise ValueError('Invalid episode comparison weights')
    if mode == 'DRAW' and len(weights) > 1:
        return rng.dirichlet(weights)
    return weights/weights.sum()


@contextmanager
def episodic_admission(cls, *, mode: str):
    """Install NEW, POOL or DRAW without altering the default H implementation."""
    if mode not in ('NEW', 'POOL', 'DRAW'):
        raise ValueError('Unsupported episodic admission mode')
    names = ('__init__', 'record_outcome', '_draw_v3_proposal', '_retention_scores',
             '_finish_hypothesis_transition', 'load_state_dict')
    original = {name: getattr(cls, name) for name in names}
    corrections = []

    def initialize(self, *args, **kwargs):
        original['__init__'](self, *args, **kwargs)
        if self.intervention_kind != 'E' or self.intervention['capacity'] != 'evidence_mean':
            raise ValueError('Episodic admission requires E/evidence_mean')
        self._STATE_FIELDS += ('admission_history',)
        self.admission_history = None
        self._recall = None
        self.configuration_signature = sha256(
            f'{self.configuration_signature}:episodic-admission-v1:{mode}'.encode()).hexdigest()

    def record(self, observation):
        original['record_outcome'](self, observation)
        memory = self.engine.get_module(ModuleRole.MEMORY)
        pairing = tuple(map(float, memory.pairing_marginal())) if hasattr(memory, 'pairing_marginal') else None
        self.admission_history = Episode(tuple(map(float, observation[0])), int(observation[1]),
                                         float(observation[2]), pairing, self.admission_history)

    def draw(self, before, remaining, gamma, diagnostics):
        candidate = original['_draw_v3_proposal'](self, before, remaining, gamma, diagnostics)
        episodes, weights = recalled_episodes(self.admission_history, self.retention_decay)
        rules = np.array([candidate]) if mode == 'NEW' else np.r_[remaining, candidate]
        evidence, maximum = recalled_log_evidence(self.engine.observation_likelihood, episodes, rules,
                                                  self.engine.get_module(ModuleRole.BETA).beta_init)
        if maximum > 1.:
            corrections.append(dict(phase='recalled_feedback', trial=int(self.trial_index), maximum=maximum))
        total = weights @ evidence
        self._recall = dict(rules=rules, evidence=evidence, weights=weights, total=total,
                            diagnostics=diagnostics)
        diagnostics.update(recalled_episodes=len(episodes), recalled_exposure=float(weights.sum()),
            candidate_initial_evidence=float(total[-1]),
            candidate_recalled_mean=float(total[-1]/weights.sum()),
            recall_effective_count=float(weights.sum()**2/(weights @ weights)),
            recall_decay=self.retention_decay, recall_mode=mode)
        return candidate

    def scores(self, rules, mean=False):
        if self._recall is None or not mean:
            return original['_retention_scores'](self, rules, mean)
        r = self._recall
        if mode == 'NEW':
            result = np.r_[original['_retention_scores'](self, rules, True), r['total'][-1]/r['weights'].sum()]
        else:
            np.testing.assert_array_equal(r['rules'][:-1], rules)
            result = comparison_weights(r['weights'], mode, self.trial_rng) @ r['evidence']
        r['diagnostics'].update(admission_scores=result.tolist(),
            candidate_admission_score=float(result[-1]), best_incumbent_score=float(result[:-1].max()),
            weakest_incumbent_score=float(result[:-1].min()))
        return result

    def finish(self, context, selection, prior, **kwargs):
        # The base analytic marginal assumes mandatory admission. Record its
        # unavailability before both base log copies are created.
        selection.diagnostics['admission_probability_given_search'] = np.nan
        selection.diagnostics['analytic_admission_available'] = False
        result = original['_finish_hypothesis_transition'](self, context, selection, prior, **kwargs)
        if selection.newcomers.size:
            if self._recall is None:
                raise RuntimeError('New rule lacks recalled evidence')
            self.retention_evidence[selection.newcomers] = self._recall['total'][-1]
            self.retention_exposure[selection.newcomers] = self._recall['weights'].sum()
        result.update(retention_evidence=self.retention_evidence[self.active].tolist(),
                      retention_exposure=self.retention_exposure[self.active].tolist())
        self.transition_rate_log[-1].update(result)
        self._recall = None
        return result

    def restore(self, state):
        history = state.get('admission_history')
        if history is not None and not isinstance(history, Episode):
            raise ValueError('Invalid episodic admission snapshot')
        original['load_state_dict'](self, state)
        self._recall = None

    replacements = (initialize, record, draw, scores, finish, restore)
    try:
        for name, method in zip(names, replacements):
            setattr(cls, name, method)
        yield corrections
    finally:
        for name, method in original.items():
            setattr(cls, name, method)

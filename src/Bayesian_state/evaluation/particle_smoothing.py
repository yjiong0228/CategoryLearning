"""Passive genealogy summaries and a finite-state reference for recovery.

Latent values refer to the state before the current feedback. Terminal weights
must be taken after the terminal choice, before its feedback and resampling.
Ancestral indices are supplied by the existing particle-filter tracer.
"""
from __future__ import annotations

import numpy as np


def probabilities(value: np.ndarray, axis: int = -1) -> np.ndarray:
    """Validate normalized, finite, nonnegative probabilities without repair."""
    value = np.asarray(value, float)
    if not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError('Probabilities must be finite and nonnegative')
    if not np.allclose(value.sum(axis=axis), 1., rtol=0, atol=1e-10):
        raise ValueError('Probabilities must sum to one')
    return value


def resampling_parents(log: list[dict], trials: int, particles: int) -> np.ndarray:
    """Build identity rows except at registered post-feedback resampling steps.

    Row t maps the particle slot at t+1 back to the slot at t. The last row is
    retained for audit; it is unused when tracing pre-resampling terminal slots.
    """
    if trials < 1 or particles < 1:
        raise ValueError('Require positive trial and particle counts')
    parents = np.tile(np.arange(particles, dtype=np.int32), (trials, 1))
    seen = set()
    for row in log:
        t = row['trial_index']; raw = np.asarray(row['ancestors'])
        if (not isinstance(t, (int, np.integer)) or not 0 <= t < trials or t in seen
                or raw.shape != (particles,) or not np.issubdtype(raw.dtype, np.integer)
                or np.any((raw < 0) | (raw >= particles))):
            raise ValueError('Invalid or repeated resampling record')
        parents[t] = raw; seen.add(t)
    return parents


def ancestral_weights(indices: np.ndarray, terminal_weights: np.ndarray) -> dict:
    """Aggregate terminal weights at each ancestor; report numerical diversity."""
    indices = np.asarray(indices); w = probabilities(terminal_weights)
    if (indices.ndim != 2 or w.ndim != 1 or indices.shape[0] != len(w)
            or indices.shape[1] < 1 or not np.issubdtype(indices.dtype, np.integer)
            or np.any((indices < 0) | (indices >= len(w)))):
        raise ValueError('Invalid ancestral paths or terminal weights')
    mass = np.stack([np.bincount(row, weights=w, minlength=len(w)) for row in indices.T])
    probabilities(mass)
    return dict(weights=mass, unique_ancestors=np.count_nonzero(mass > 0, axis=1),
                ancestral_ess=1./np.square(mass).sum(axis=1))


def trace_values(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Select coherent values, returning trial by terminal-particle arrays."""
    values, indices = np.asarray(values), np.asarray(indices)
    if (values.ndim != 2 or indices.shape != values.T.shape
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any((indices < 0) | (indices >= values.shape[1]))):
        raise ValueError('Require aligned trial/particle values and ancestral indices')
    return values[np.arange(len(values))[:, None], indices.T]


def exact_hmm(initial: np.ndarray, transitions: np.ndarray, likelihood: np.ndarray) -> dict:
    """Scaled forward/backward reference, with observation t emitted by state t.

    transitions[t,i,j] is P(state[t+1]=j | state[t]=i). likelihood[t,i]
    is the probability of the observed choice at t given state i. All these
    arrays are fixed diagnostic inputs, independent of the inferred path.
    """
    initial = probabilities(initial); transitions = probabilities(transitions)
    likelihood = np.asarray(likelihood, float)
    if (initial.ndim != 1 or likelihood.ndim != 2 or likelihood.shape[1] != len(initial)
            or len(likelihood) < 1 or transitions.shape != (len(likelihood)-1, len(initial), len(initial))
            or not np.isfinite(likelihood).all() or np.any((likelihood < 0) | (likelihood > 1))):
        raise ValueError('Invalid finite-state reference dimensions or likelihood')
    n, states = likelihood.shape
    predictive = np.empty_like(likelihood); filtered = np.empty_like(likelihood); scale = np.empty(n)
    for t in range(n):
        predictive[t] = initial if t == 0 else filtered[t-1] @ transitions[t-1]
        mass = predictive[t]*likelihood[t]; scale[t] = mass.sum()
        if scale[t] <= 0:
            raise ValueError('Observed sequence has zero probability')
        filtered[t] = mass/scale[t]
    backward = np.ones_like(likelihood)
    for t in range(n-2, -1, -1):
        backward[t] = transitions[t] @ (likelihood[t+1]*backward[t+1])/scale[t+1]
    smoothed = filtered*backward; smoothed /= smoothed.sum(axis=1, keepdims=True)
    pairs = np.empty((n-1, states, states))
    for t in range(n-1):
        pairs[t] = filtered[t, :, None]*transitions[t]*(likelihood[t+1]*backward[t+1])[None, :]
        pairs[t] /= pairs[t].sum()
    return dict(predictive=predictive, filtered=filtered, smoothed=smoothed, pairs=pairs,
                log_evidence=float(np.log(scale).sum()))


def exact_hmm_paths(filtered: np.ndarray, transitions: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Draw finite-state posterior paths by forward filtering/backward sampling."""
    filtered = probabilities(filtered); transitions = probabilities(transitions)
    if (filtered.ndim != 2 or len(filtered) < 1 or count < 1
            or transitions.shape != (len(filtered)-1, filtered.shape[1], filtered.shape[1])):
        raise ValueError('Invalid backward-sampling dimensions')
    rng = np.random.default_rng(seed); n, states = filtered.shape
    result = np.empty((count, n), dtype=np.int16)
    result[:, -1] = rng.choice(states, size=count, p=filtered[-1])
    # Sample by destination groups to keep the diagnostic inexpensive.
    for t in range(n-2, -1, -1):
        for j in range(states):
            chosen = np.flatnonzero(result[:, t+1] == j)
            if len(chosen):
                mass = filtered[t]*transitions[t, :, j]
                if mass.sum() <= 0:
                    raise ValueError('Backward sample has an impossible destination')
                result[chosen, t] = rng.choice(states, size=len(chosen), p=mass/mass.sum())
    return result


class TargetGenealogyCollector:
    """Retain target-state particle values and post-choice weights, without RNG.

    Values are sufficient for the registered target events, not the complete
    cognitive state. The execution readout must select exactly one rule.
    """
    def __init__(self, targets: list[int]):
        self.targets = np.asarray(targets, dtype=int)
        if (self.targets.ndim != 1 or not len(self.targets) or np.any(self.targets < 0)
                or len(np.unique(self.targets)) != len(self.targets)):
            raise ValueError('Require unique nonnegative target rule IDs')
        self.values = []; self.weights = []

    def __call__(self, trial: int, phase: str, snapshot) -> None:
        if phase == 'predictive':
            if trial != len(self.values) or len(self.weights) != trial:
                raise ValueError('Snapshots must alternate predictive and filtered')
            active = np.asarray(snapshot['active']); belief = np.asarray(snapshot['belief'])
            contribution = probabilities(snapshot['choice_weights'])
            if (active.ndim != 2 or belief.shape != active.shape or contribution.shape != active.shape
                    or self.targets.max() >= active.shape[1] or not np.isin(active, [0, 1]).all()
                    or not np.isin(contribution, [0., 1.]).all()
                    or not np.isfinite(belief).all() or np.any(belief < 0)):
                raise ValueError('Require a valid single-rule execution snapshot')
            probabilities(belief)
            executed = contribution.argmax(axis=1)
            if np.any(active[np.arange(len(executed)), executed] == 0):
                raise ValueError('Executed rule is inactive')
            present = active[:, self.targets].any(axis=1)
            target_belief = belief[:, self.targets].sum(axis=1)
            target_execution = np.isin(executed, self.targets)
            self.values.append(dict(present=present.copy(), target_belief=target_belief.copy(),
                high_belief=(present & (target_belief >= .5)), executed=executed.astype(np.int16),
                target_execution=target_execution,
                source=np.where(~present, 0, np.where(target_execution, 2, 1)).astype(np.int8)))
        elif phase == 'filtered':
            if trial != len(self.weights) or len(self.values) != trial+1:
                raise ValueError('Filtered snapshot is out of order')
            self.weights.append(probabilities(snapshot['weights']).copy())
        else:
            raise ValueError('Unknown snapshot phase')

    def arrays(self) -> dict:
        if not self.values or len(self.values) != len(self.weights):
            raise ValueError('Incomplete genealogy capture')
        return dict({k: np.stack([v[k] for v in self.values]) for k in self.values[0]},
                    filtered_weights=np.stack(self.weights))

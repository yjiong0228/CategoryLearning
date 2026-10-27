"""Passive sparse state capture and common-endpoint retrospective summaries.

The arrays preserve each particle's complete active set, belief, precision and
choice-conditioned execution. Oral content uses the shared nearest-component
convention. Reports never enter this collector.
"""
from __future__ import annotations

import numpy as np

from .marginal_execution import execution_posterior
from .particle_smoothing import probabilities, ancestral_weights
from .oral.decision_basis import nearest_component_weights
from ..hypothesis_space.geometry.boundary import BoundaryGeometry


class IndividualStateCollector:
    """Record pre-feedback states and actual post-choice particle weights."""

    def __init__(self, space, choices: np.ndarray, capacity: int):
        self.space, self.choices, self.capacity = space, np.asarray(choices), capacity
        self.geometry = BoundaryGeometry(space)
        self.counts = np.array([[len(c.components) for c in h.categories] for h in space])
        self.components = int(self.counts.max())
        self.rows, self.weights = [], []
        self.minimum_choice_mass = 1.

    def __call__(self, trial: int, phase: str, snapshot) -> None:
        if phase == 'filtered':
            if trial != len(self.weights) or len(self.rows) != trial + 1:
                raise ValueError('Filtered snapshot is out of order')
            self.weights.append(probabilities(snapshot['weights']).copy())
            return
        if phase != 'predictive' or trial != len(self.rows) or len(self.weights) != trial:
            raise ValueError('Predictive snapshot is out of order')
        active = np.asarray(snapshot['active'], bool)
        belief = probabilities(snapshot['belief'])
        beta = np.asarray(snapshot['beta'])
        _, q = execution_posterior(snapshot['choice_weights'], snapshot['rule_choice_probabilities'],
                                   int(self.choices[trial]))
        self.minimum_choice_mass = min(self.minimum_choice_mass,
            float(np.min(snapshot['prediction'][:, int(self.choices[trial])-1])))
        if (active.shape != belief.shape or beta.shape != active.shape
                or np.any(active.sum(axis=1) > self.capacity)
                or np.any(belief[~active] != 0) or np.any(q[~active] != 0)
                or not np.isfinite(beta).all() or np.any(beta < 0)):
            raise ValueError('Invalid sparse workspace')
        p, h = active.shape
        rule = np.full((p, self.capacity), -1, np.int16)
        particle, identity = np.nonzero(active)
        slot = np.cumsum(active, axis=1)[particle, identity] - 1
        rule[particle, slot] = identity
        row = dict(rule=rule)
        for key, value in [('belief', belief), ('precision', beta), ('execution', q)]:
            sparse = np.zeros_like(rule, dtype=float)
            sparse[particle, slot] = value[particle, identity]
            row[key] = sparse
        component = np.zeros((p, self.capacity, self.components))
        perceived = np.asarray(snapshot['perceived_stimulus'])
        for identity in np.flatnonzero(q.sum(axis=0) > 0):
            use = np.flatnonzero(q[:, identity] > 0)
            slots = np.argmax(rule[use] == identity, axis=1)
            allocation = nearest_component_weights(self.space, self.geometry, int(identity),
                int(self.choices[trial])-1, perceived[use])
            component[use, slots, :allocation.shape[1]] = q[use, identity, None] * allocation
        np.testing.assert_allclose(component.sum(axis=(1, 2)), 1., atol=2e-12, rtol=0)
        row['component'] = component
        self.rows.append(row)

    def arrays(self) -> dict:
        if len(self.rows) != len(self.choices) or len(self.weights) != len(self.rows):
            raise ValueError('Incomplete state capture')
        return dict({k: np.stack([r[k] for r in self.rows]) for k in self.rows[0]},
                    filtered_weights=np.stack(self.weights), component_counts=self.counts)


def state_means(values: dict, weights: np.ndarray, rules: int, targets: list[int]) -> dict:
    """Aggregate without treating overlapping rule-presence events as a simplex."""
    weights = probabilities(weights)
    ids = values['rule']; n, p, k = ids.shape
    if weights.shape != (n, p):
        raise ValueError('Unaligned weights and states')
    valid = ids >= 0
    target = np.isin(ids, targets)
    present = target.any(axis=2)
    belief = (target * values['belief']).sum(axis=2)
    execution = (target * values['execution']).sum(axis=2)
    out = {name: np.sum(weights * v, axis=1) for name, v in [
        ('target_present', present), ('target_belief', belief),
        ('target_high_belief', present & (belief >= .5)), ('target_execution', execution)]}
    out['source'] = np.stack([np.sum(weights * ~present, axis=1),
        np.sum(weights * (present - execution), axis=1), out['target_execution']], axis=1)
    probabilities(out['source'])
    for key, value in [('active', valid), ('belief', values['belief']),
                       ('belief_second', values['belief']**2), ('execution', values['execution']),
                       ('active_precision', values['precision']),
                       ('active_precision_second', values['precision']**2)]:
        out[key] = np.stack([np.bincount(ids[t][valid[t]],
            weights=(weights[t, :, None] * value[t])[valid[t]], minlength=rules) for t in range(n)])
    d = values['component'].shape[-1]
    out['component'] = np.stack([np.stack([np.bincount(ids[t][valid[t]],
        weights=(weights[t, :, None]*values['component'][t, :, :, c])[valid[t]],
        minlength=rules) for c in range(d)], axis=-1) for t in range(n)])
    probabilities(out['belief']); probabilities(out['execution'])
    np.testing.assert_allclose(out['component'].sum(axis=(1, 2)), 1., atol=2e-12, rtol=0)
    return out


def summarize(values: dict, parents: np.ndarray, tracer, targets: list[int]) -> dict:
    """Filtering and complete-record reconstruction keep distinct conditioning."""
    w = values['filtered_weights']; h = len(values['component_counts'])
    indices = tracer(parents)
    ancestry = ancestral_weights(indices, w[-1])
    result = {}
    for label, weights in [('filtered', w), ('full', ancestry['weights'])]:
        result.update({label+'_'+k: v for k, v in state_means(values, weights, h, targets).items()})
    result.update(full_indices=indices.astype(np.int32), full_weights=ancestry['weights'],
                  full_unique_ancestors=ancestry['unique_ancestors'],
                  full_ancestral_ess=ancestry['ancestral_ess'])
    return result

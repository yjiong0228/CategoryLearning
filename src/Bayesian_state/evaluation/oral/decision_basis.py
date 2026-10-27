"""Record the complete local-basis distribution without reading oral reports.

Joint particle/rule/component masses are accumulated before marginalization.
Disconnected categories use the same nearest-component convention as OralReadout,
with uniform ties. Packed active sets retain joint availability for later audits.
"""
from __future__ import annotations

import numpy as np

from ...hypothesis_space.geometry.boundary import BoundaryGeometry


def nearest_component_weights(space, geometry, rule: int, choice: int,
                              perceived: np.ndarray) -> np.ndarray:
    """Allocate perceived stimuli to category components, uniformly on ties.

    ``choice`` is zero based. This is the shared local-content convention;
    it does not inspect an oral report or change the behavioral readout.
    """
    components = space[int(rule)].categories[choice].components
    if len(components) == 1:
        return np.ones((len(perceived), 1))
    unique, inverse = np.unique(perceived, axis=0, return_inverse=True)
    distances = np.column_stack([
        geometry.distances_to_polytope(unique, component) for component in components])
    nearest = np.isclose(distances, distances.min(axis=1, keepdims=True),
                         atol=geometry.tolerance, rtol=0)
    return (nearest / nearest.sum(axis=1, keepdims=True))[inverse]


class DecisionBasisCollector:
    """Consume predictive observer snapshots, all before current feedback."""

    def __init__(self, space, choices: np.ndarray, *, geometry=None):
        self.space = space
        self.choices = np.asarray(choices)
        if (self.choices.ndim != 1 or not np.isin(self.choices, range(1, space.n_cats + 1)).all()):
            raise ValueError('Expected one-based choices in the catalogue coordinate')
        self.geometry = geometry if geometry is not None else BoundaryGeometry(space)
        if self.geometry.space.signature != space.signature:
            raise ValueError('Geometry and catalogue differ')
        self.counts = np.array([[len(c.components) for c in h.categories] for h in space])
        self.max_components = int(self.counts.max())
        self.rows: list[dict[str, np.ndarray]] = []

    def __call__(self, trial: int, phase: str, snapshot) -> None:
        if phase == 'filtered':
            return
        if phase != 'predictive' or trial != len(self.rows) or trial >= len(self.choices):
            raise ValueError('Expected one predictive snapshot per ordered trial')
        p = np.asarray(snapshot['weights'])
        contribution = np.asarray(snapshot['choice_weights'])
        likelihood = np.asarray(snapshot['rule_choice_probabilities'])
        perceived = np.asarray(snapshot['perceived_stimulus'])
        active = np.asarray(snapshot['active'])
        prediction = np.asarray(snapshot['prediction'])
        n, h, c = len(p), len(self.space), self.space.n_cats
        if (p.shape != (n,) or contribution.shape != (n, h) or likelihood.shape != (n, h, c)
                or perceived.shape != (n, self.space.n_dims) or active.shape != (n, h)
                or prediction.shape != (n, c)
                or any(not np.isfinite(x).all() for x in (p, contribution, likelihood, perceived, active, prediction))
                or any(np.any(x < 0) for x in (p, contribution, likelihood, prediction))
                or not np.isin(active, [0., 1.]).all() or np.any(contribution[active == 0] != 0)):
            raise ValueError('Invalid decision-basis snapshot')
        np.testing.assert_allclose(p.sum(), 1., atol=1e-12, rtol=0)
        np.testing.assert_allclose(contribution.sum(axis=1), 1., atol=1e-12, rtol=0)
        np.testing.assert_allclose(np.einsum('ph,phc->pc', contribution, likelihood),
                                   prediction, atol=1e-12, rtol=0)
        choice = int(self.choices[trial]) - 1
        predictive = p[:, None] * contribution
        joint = predictive * likelihood[:, :, choice]
        choice_probability = float(p @ prediction[:, choice])
        if choice_probability <= 0:
            raise ValueError('Cannot infer a basis for an impossible choice')
        conditional_particles = p * prediction[:, choice] / choice_probability
        prior_components = np.zeros((h, self.max_components))
        joint_components = np.zeros_like(prior_components)
        for rule in np.flatnonzero(predictive.sum(axis=0) > 0):
            components = self.space[int(rule)].categories[choice].components
            if len(components) == 1:
                prior_components[rule, 0] = predictive[:, rule].sum()
                joint_components[rule, 0] = joint[:, rule].sum()
                continue
            # Many particles share an internal stimulus. Deduplication changes
            # only geometry work, retaining their distinct joint weights.
            use = np.flatnonzero(predictive[:, rule] > 0)
            component_weights = nearest_component_weights(
                self.space, self.geometry, int(rule), choice, perceived[use])
            prior_components[rule, :len(components)] = predictive[use, rule] @ component_weights
            joint_components[rule, :len(components)] = joint[use, rule] @ component_weights
        np.testing.assert_allclose(prior_components.sum(), 1., atol=1e-12, rtol=0)
        np.testing.assert_allclose(joint_components.sum(), choice_probability, atol=1e-12, rtol=0)
        self.rows.append(dict(
            prior_component_mass=prior_components,
            choice_component_joint_mass=joint_components,
            choice_probability=np.asarray(choice_probability),
            prediction=p @ prediction,
            filtered_active=conditional_particles @ active,
            filtered_belief=conditional_particles @ snapshot['belief'],
            filtered_active_beta=conditional_particles @ (active * snapshot['beta']),
            active_packed=np.packbits(active.astype(np.uint8), axis=1, bitorder='little'),
            conditional_particle_weights=conditional_particles,
            predictive_search=np.asarray(p @ snapshot['search']),
            predictive_admission=np.asarray(p @ snapshot['admission_event']),
        ))

    def arrays(self) -> dict[str, np.ndarray]:
        if len(self.rows) != len(self.choices) or not self.rows:
            raise ValueError('Incomplete decision-basis trajectory')
        return dict({key: np.stack([row[key] for row in self.rows]) for key in self.rows[0]},
                    choices=self.choices, component_counts=self.counts)


def joint_availability(active_packed: np.ndarray, particle_weights: np.ndarray,
                       rule_mask: np.ndarray) -> np.ndarray:
    """Probability that any matching rule is active; preserve set correlations."""
    mask = np.asarray(rule_mask, dtype=bool)
    weights = np.asarray(particle_weights, dtype=float)
    active = np.unpackbits(np.asarray(active_packed), axis=-1, count=mask.shape[-1], bitorder='little').astype(bool)
    if (mask.ndim != 2 or active.shape[:2] != weights.shape or active.shape[::2] != mask.shape
            or not np.isfinite(weights).all() or np.any(weights < 0)):
        raise ValueError('Availability requires trial/particle/rule identities')
    np.testing.assert_allclose(weights.sum(axis=1), 1., atol=1e-12, rtol=0)
    return np.sum(weights * np.any(active & mask[:, None, :], axis=2), axis=1)

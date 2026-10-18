"""Behavior-only local decision bases with learned response associations.

The existing DecisionBasisCollector supplies the nearest-component convention
for each geometrical region. Association and rule masses remain joint until the
observed response is conditioned on. Oral content is absent from this module.
"""
from __future__ import annotations

import numpy as np

from .decision_basis import DecisionBasisCollector
from ...hypothesis_space.geometry.boundary import BoundaryGeometry
from ...hypothesis_space.observation_model.response_association import (
    INVERSE_ASSOCIATIONS, association_choice_probabilities,
)


class AssociatedDecisionBasis:
    """Project one pre-feedback association state onto rule/region/components."""

    def __init__(self, space, *, geometry=None):
        if space.n_cats != 4: raise ValueError('Associated bases require four geometrical regions.')
        self.space = space
        self.geometry = geometry if geometry is not None else BoundaryGeometry(space)
        if self.geometry.space.signature != space.signature: raise ValueError('Geometry and catalogue differ.')
        self.counts = np.array([[len(c.components) for c in h.categories] for h in space])
        self.max_components = int(self.counts.max())

    def __call__(
        self, *, probabilities: np.ndarray, rule_weights: np.ndarray, associations: np.ndarray,
        stimulus: np.ndarray, choice: int,
    ) -> dict[str, np.ndarray]:
        """Return unconditional and choice-joint masses; do not consume feedback.

        For each region g, sum all association states mapping g to the actual
        response. Multiplying by the rule's region probability gives joint
        choice mass. Components within that region follow the same geometrical
        convention as the existing fixed-label collector, including ties.
        """
        q = np.asarray(probabilities, dtype=float); w = np.asarray(rule_weights, dtype=float)
        x = np.asarray(stimulus, dtype=float); a = np.asarray(associations, dtype=float)
        h = len(self.space)
        if (w.shape != (h,) or not np.isfinite(w).all() or np.any(w < 0)
                or not np.isclose(w.sum(), 1., atol=1e-12, rtol=0)
                or x.shape != (self.space.n_dims,) or not np.isfinite(x).all()
                or not np.isfinite(choice) or choice != int(choice) or not 1 <= int(choice) <= 4):
            raise ValueError('Need normalized rule weights, one finite stimulus and a valid response.')
        # Shared validator also checks both geometry and association distributions.
        per_rule = association_choice_probabilities(q, a)
        if len(q) != h: raise ValueError('Probability arrays differ from the catalogue.')
        prediction = w @ per_rule; probability = float(prediction[int(choice)-1])
        if probability <= 0.: raise ValueError('Cannot infer a basis for an impossible response.')
        region_selector = INVERSE_ASSOCIATIONS[:, int(choice)-1, None] == np.arange(4)[None,:]
        prior_region = w[:,None] * (a.sum(axis=2) @ region_selector.astype(float))
        prior = np.zeros((h,4,self.max_components)); joint = np.zeros_like(prior)
        for region in range(4):
            mass = float(prior_region[:,region].sum())
            if mass == 0.: continue
            contribution = prior_region[:,region] / mass
            possible = float(contribution @ q[:,region]) > 0.
            # A zero-probability region still carries unconditional basis mass.
            # Uniform dummy likelihood obtains only that geometry-dependent
            # prior from the existing collector; its actual joint mass is zero.
            local_q = q if possible else np.full_like(q, .25)
            snapshot = dict(weights=np.ones(1),choice_weights=contribution[None,:],
                rule_choice_probabilities=local_q[None,:,:],perceived_stimulus=x[None,:],
                active=np.ones((1,h)),prediction=(contribution @ local_q)[None,:],
                belief=w[None,:],beta=np.zeros((1,h)),search=np.zeros(1),admission_event=np.zeros(1))
            collector = DecisionBasisCollector(self.space,np.array([region+1]),geometry=self.geometry)
            collector(0,'predictive',snapshot)
            row = collector.rows[0]
            prior[:,region] = mass*row['prior_component_mass']
            if possible: joint[:,region] = mass*row['choice_component_joint_mass']
        np.testing.assert_allclose(prior.sum(),1.,atol=1e-12,rtol=0)
        np.testing.assert_allclose(joint.sum(),probability,atol=1e-12,rtol=0)
        np.testing.assert_allclose(prior.sum(axis=2),prior_region,atol=1e-12,rtol=0)
        np.testing.assert_allclose(joint.sum(axis=2),prior_region*q,atol=1e-12,rtol=0)
        return dict(prior_component_mass=prior,choice_component_joint_mass=joint,
                    choice_probability=np.asarray(probability),prediction=prediction)

"""Integrate local region reasons while retaining associations within particles.

Each particle contributes prior_weight * rule_weight * q(region) * rho(component).
For the current-choice joint mass, its own A(region, choice) multiplies this
quantity before particles are summed. The saved per-particle state remains the
source for analyses of jointly used rules. These masses measure compatibility
with geometric reasons; they do not define probabilities of spoken reports.
"""
from __future__ import annotations

import numpy as np

from .region_basis import region_basis


def particle_region_basis(state: dict[str, np.ndarray], components: np.ndarray) -> dict[str, np.ndarray]:
    """Project sparse trial/particle/slot states through matching component kernels.

    components has shape (trial, particle, slot, region, component). Each region
    row sums to one, including padded slots whose rule contribution is zero.
    Components must be evaluated on that particle's recorded perceived stimulus.
    """
    ids = np.asarray(state['rule_ids'])
    weights = np.asarray(state['prior_weights'])
    choices = np.asarray(state['choices'])
    rho = np.asarray(components, float)
    count = int(state['total_rules'])
    if (ids.ndim != 3 or not len(ids) or not np.issubdtype(ids.dtype, np.integer)
            or count < 1 or np.any((ids < -1) | (ids >= count))
            or weights.shape != ids.shape[:2] or choices.shape != (len(ids),)
            or rho.ndim != 5 or rho.shape[:3] != ids.shape or rho.shape[3] != 4
            or not rho.shape[4] or not np.isfinite(weights).all() or np.any(weights < 0)):
        raise ValueError('Require matching sparse states and particle-specific region components')
    np.testing.assert_allclose(weights.sum(1), 1., atol=1e-12, rtol=0)
    shape = (len(ids), count, 4, rho.shape[-1])
    prechoice, joint = np.zeros(shape), np.zeros(shape)
    prediction = np.empty((len(ids), 4))
    for trial, rule_ids in enumerate(ids):
        valid = rule_ids >= 0
        if any(len(np.unique(row[row >= 0])) != np.count_nonzero(row >= 0) for row in rule_ids):
            raise ValueError('Each particle needs distinct active rule IDs')
        w = np.array(state['choice_weights'][trial], copy=True)
        q = np.array(state['geometry'][trial], copy=True)
        a = np.array(state['associations'][trial], copy=True)
        if np.any(w[~valid] != 0.) or np.any(q[~valid] != 0.) or np.any(a[~valid] != 0.):
            raise ValueError('Sparse padding must contain zero numerical state')
        # The shared kernel checks normalized rows. Padding contributes no mass.
        q[~valid] = .25
        a[~valid] = .25
        basis = region_basis(geometry=q, rule_weights=w, associations=a, components=rho[trial],
                             choices=np.full(len(rule_ids), choices[trial]))
        np.testing.assert_allclose(basis['prediction'], state['particle_prediction'][trial], atol=1e-12, rtol=0)
        prediction[trial] = weights[trial] @ basis['prediction']
        for target, name in ((prechoice, 'prechoice_component_mass'), (joint, 'choice_component_joint_mass')):
            mass = basis[name] * weights[trial, :, None, None, None]
            np.add.at(target[trial], rule_ids[valid], mass[valid])
    probability = prediction[np.arange(len(choices)), choices.astype(int)-1]
    if np.any(probability <= 0.):
        raise ValueError('Cannot condition a reason basis on an impossible choice')
    np.testing.assert_allclose(prechoice.sum((1, 2, 3)), 1., atol=1e-12, rtol=0)
    np.testing.assert_allclose(joint.sum((1, 2, 3)), probability, atol=1e-12, rtol=0)
    np.testing.assert_allclose(prediction, state['prediction'], atol=1e-12, rtol=0)
    np.testing.assert_allclose(probability, state['choice_probability'], atol=1e-12, rtol=0)
    return dict(prechoice_component_mass=prechoice, choice_component_joint_mass=joint,
                prediction=prediction, choice_probability=probability, choices=choices.copy())

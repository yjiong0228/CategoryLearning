"""Hold rule-mixture weights fixed while retaining current-stimulus readout.

These are supplementary marginal-rule controls. Every condition uses the same
independent perception kernel. The original particle readout is retained to
measure the change caused by this factorization. Current choice determines the
queried category; it never reweights rules by their current choice likelihood.
"""
from __future__ import annotations

import numpy as np

from .decision_basis_scoring import label_masses

CONTROL_NAMES = ('original_prior', 'common_kernel_dynamic', 'boundary_frozen', 'prefix_mean_frozen',
                 'dynamic_minus_boundary', 'dynamic_minus_prefix_mean', 'factorization_difference')


def component_kernel(space, perceived: np.ndarray, choices: np.ndarray, geometry) -> np.ndarray:
    """Integrate the existing nearest-component convention over independent perceptions."""
    perceived, choices = np.asarray(perceived,float),np.asarray(choices)
    if (perceived.ndim != 3 or perceived.shape[2] != space.n_dims or perceived.shape[1] < 1
            or choices.shape != (len(perceived),) or not np.isfinite(perceived).all()
            or not np.isin(choices,range(1,space.n_cats+1)).all()
            or geometry.space.signature != space.signature):
        raise ValueError('Need matching trial/draw/stimulus coordinates and catalogue choices')
    maximum = max(len(c.components) for h in space for c in h.categories)
    result = np.zeros((len(choices),len(space),maximum))
    for choice in np.unique(choices):
        trials = np.flatnonzero(choices==choice)
        x = perceived[trials].reshape(-1,space.n_dims)
        for rule,hypothesis in enumerate(space):
            components = hypothesis.categories[int(choice)-1].components
            if len(components)==1:
                result[trials,rule,0]=1.
                continue
            distance=np.column_stack([geometry.distances_to_polytope(x,c) for c in components])
            nearest=np.isclose(distance,distance.min(1,keepdims=True),atol=geometry.tolerance,rtol=0)
            weights=nearest/nearest.sum(1,keepdims=True)
            result[trials,rule,:len(components)]=weights.reshape(len(trials),perceived.shape[1],-1).mean(1)
    np.testing.assert_allclose(result.sum(-1),1.,atol=1e-12,rtol=0)
    return result


def rule_mixture_controls(prior: np.ndarray, kernel: np.ndarray, labels: np.ndarray, cut: int) -> np.ndarray:
    """Return filter/control/trial/domain/verdict scores with an explicit prefix boundary.

    The boundary baseline freezes the predictive rule weights at the first
    later trial, after prefix feedback and before the first later choice.
    The second baseline averages predictive rule weights over prefix trials.
    Neither baseline accesses later states when establishing its rule mixture.
    """
    prior,kernel=np.asarray(prior,float),np.asarray(kernel,float)
    if (prior.ndim!=4 or kernel.shape!=prior.shape[1:] or len(prior)<2
            or labels.shape!=(*kernel.shape,2) or not 0<cut<prior.shape[1]
            or not np.isfinite(prior).all() or not np.isfinite(kernel).all()
            or np.any(prior<0) or np.any(kernel<0)):
        raise ValueError('Need full matching seed/trial/rule/component arrays and a nonempty prefix')
    np.testing.assert_allclose(prior.sum((2,3)),1.,rtol=0,atol=1e-12)
    np.testing.assert_allclose(kernel.sum(-1),1.,rtol=0,atol=1e-12)
    results=[]
    for mass in prior:
        rules=mass.sum(-1)
        dynamic=label_masses(rules[...,None]*kernel,labels)
        boundary=label_masses(rules[cut][None,:,None]*kernel,labels)
        mean=label_masses(rules[:cut].mean(0)[None,:,None]*kernel,labels)
        boundary[:cut]=np.nan;mean[:cut]=np.nan
        original=label_masses(mass,labels)
        results.append(np.stack([original,dynamic,boundary,mean,dynamic-boundary,dynamic-mean,dynamic-original]))
    return np.stack(results)

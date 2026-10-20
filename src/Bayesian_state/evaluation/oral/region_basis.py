"""Project independent region-response states onto local geometric reasons.

The pre-choice basis is w[h]*q[h,g]*rho[h,g,m]. Conditioning on response y
multiplies that joint mass by A[h,g,y] and normalizes once. Association rows
may share a preferred response; columns have no normalization constraint.
rho is the existing nearest-component kernel, computed without oral content.
"""
from __future__ import annotations

import numpy as np


def region_basis(*, geometry: np.ndarray, rule_weights: np.ndarray,
                 associations: np.ndarray, components: np.ndarray,
                 choices: np.ndarray) -> dict[str,np.ndarray]:
    """Return complete trial/rule/region/component masses before feedback."""
    q,w,a,rho,y=map(np.asarray,(geometry,rule_weights,associations,components,choices))
    if (q.ndim!=3 or q.shape[-1]!=4 or not len(q) or w.shape!=q.shape[:2]
            or a.shape!=(*q.shape,4) or rho.ndim!=4 or rho.shape[:3]!=q.shape
            or not rho.shape[-1] or y.shape!=(len(q),) or not np.isin(y,[1,2,3,4]).all()
            or any(not np.isfinite(v).all() or np.any(v<0) for v in (q,w,a,rho))):
        raise ValueError('Need aligned finite condition-2 probabilities and local component kernels')
    for v in (q,w,a,rho):
        if not np.allclose(v.sum(-1),1.,atol=1e-12,rtol=0):
            raise ValueError('Every probability row must be normalized')
    prechoice=w[:,:,None,None]*q[:,:,:,None]*rho
    selected=a[np.arange(len(y)),:,:,y.astype(int)-1]
    joint=prechoice*selected[:,:,:,None]
    prediction=np.einsum('th,thg,thgc->tc',w,q,a)
    probability=prediction[np.arange(len(y)),y.astype(int)-1]
    if np.any(probability<=0):raise ValueError('Cannot condition on an impossible response')
    np.testing.assert_allclose(prechoice.sum((1,2,3)),1.,atol=1e-12,rtol=0)
    np.testing.assert_allclose(joint.sum((1,2,3)),probability,atol=1e-12,rtol=0)
    return dict(prechoice_component_mass=prechoice,choice_component_joint_mass=joint,
                choice_probability=probability,prediction=prediction)

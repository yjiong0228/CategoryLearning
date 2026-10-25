"""Independent mathematical references for whole-system inference."""
from itertools import product

import numpy as np
import pytest

from src.Bayesian_state.evaluation.independent_particle_systems import (
    conditional_execution, evidence_weights, execution_kernel, pimh_indices, pool_curves,
)


def test_evidence_uses_common_endpoint_and_is_offset_invariant():
    z=np.log([[.1,.01],[.2,.001]])
    curves=np.array([[.1,.3],[.7,.9]])
    np.testing.assert_allclose(evidence_weights(z),evidence_weights(z+1000))
    np.testing.assert_allclose(pool_curves(curves,z,np.array([1,2])),[.5,.3*10/11+.9/11])
    np.testing.assert_allclose(pool_curves(curves,z,np.array([2,2])),[.1*10/11+.7/11,.3*10/11+.9/11])
    with pytest.raises(ValueError):pool_curves(curves,z,np.array([0,2]))


def test_pimh_retains_current_evidence_after_rejection():
    # Proposal 1 is rejected, so proposal 2 is compared with proposal 0.
    np.testing.assert_array_equal(pimh_indices(np.log([1.,.01,.5]),np.array([.5,.4])),[0,0,2])


def test_single_particle_extended_space_detailed_balance():
    paths=np.array(list(product(range(2),repeat=4)))
    initial=np.array([.4,.6]);kernel=np.array([[.8,.2],[.3,.7]])
    likelihood=np.array([[.2,.8],[.7,.3],[.4,.6],[.1,.9]])
    q=initial[paths[:,0]]
    for t in range(1,4):q*=kernel[paths[:,t-1],paths[:,t]]
    ell=np.prod(likelihood[np.arange(4)[None,:],paths],axis=1)
    posterior=q*ell;posterior/=posterior.sum()
    importance=evidence_weights(np.log(q)+np.log(ell))
    np.testing.assert_allclose(importance,posterior,atol=1e-14)
    transition=q[None,:]*np.minimum(1.,ell[None,:]/ell[:,None])
    transition[np.diag_indices(len(q))]+=1-transition.sum(axis=1)
    flow=posterior[:,None]*transition
    np.testing.assert_allclose(flow,flow.T,atol=1e-14)
    np.testing.assert_allclose(posterior@transition,posterior,atol=1e-14)


@pytest.mark.parametrize('kappa',[0.,2.])
def test_variable_workspace_conditional_execution_matches_enumeration(kappa):
    ids=[np.array([3,4]),np.array([4,5,6]),np.array([5,6])]
    beliefs=[np.array([.4,.6]),np.array([.2,.5,.3]),np.array([.3,.7])]
    ell=[np.array([.7,.2]),np.array([.1,.6,.9]),np.array([.8,.3])]
    result=conditional_execution(ids,beliefs,ell,kappa)
    paths=list(product(range(2),range(3),range(2)));mass=[]
    for a,b,c in paths:
        mass.append(beliefs[0][a]*ell[0][a]*result['kernels'][0][a,b]*ell[1][b]*result['kernels'][1][b,c]*ell[2][c])
    mass=np.asarray(mass);logz=np.log(mass.sum());mass/=mass.sum()
    for t in range(3):
        exact=np.bincount([p[t] for p in paths],weights=mass,minlength=len(ids[t]))
        np.testing.assert_allclose(exact,result['smoothed'][t],atol=1e-14)
        if kappa==0:
            local=beliefs[t]*ell[t];local/=local.sum()
            np.testing.assert_allclose(local,result['smoothed'][t],atol=1e-14)
    assert abs(result['log_evidence']-logz)<1e-14
    # Previous rule 3 has left the workspace: its stay bonus disappears.
    np.testing.assert_allclose(execution_kernel(ids[0],ids[1],beliefs[1],kappa)[0],beliefs[1])

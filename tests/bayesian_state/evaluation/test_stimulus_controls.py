"""Check fixed-weight controls with changing stimulus-dependent components."""
from types import SimpleNamespace

import numpy as np
import pytest

from src.Bayesian_state.evaluation.oral.stimulus_controls import component_kernel,rule_mixture_controls


def example():
    q=np.array([[[.7,.3],[.6,.4],[.9,.1],[.2,.8]],[[.8,.2],[.4,.6],[.6,.4],[.3,.7]]])
    kernel=np.array([[[1,0],[1,0]],[[0,1],[1,0]],[[1,0],[1,0]],[[0,1],[1,0]]],float)
    prior=q[...,None]*kernel[None]
    labels=np.broadcast_to(np.array([[[0,0],[1,1]],[[2,2],[-1,-1]]]),(4,2,2,2)).copy()
    return prior,kernel,labels,2


def test_frozen_rules_keep_current_stimulus_and_first_later_state_is_equal():
    values=rule_mixture_controls(*example())
    assert values.shape==(2,7,4,2,3)
    np.testing.assert_allclose(values[:,4,2],0.)
    np.testing.assert_allclose(values[:,6],0.)
    # At the next stimulus the same frozen rule now contributes conflict.
    np.testing.assert_allclose(values[:,2,3,0,1],[.9,.6])
    np.testing.assert_allclose(values[:,1,3,0,1],[.2,.3])
    assert np.isnan(values[:,2:6,:2]).all()
    np.testing.assert_allclose(values[:,:4,2:].sum(-1),1.)


def test_later_weights_do_not_change_either_frozen_baseline():
    args=example();baseline=rule_mixture_controls(*args)
    changed=args[0].copy();changed[:,3]=changed[:,3,::-1]
    # Preserve legal component padding while altering later rule weights.
    changed[:,3]=np.array([[[0,.8],[.2,0]],[[0,.9],[.1,0]]])
    result=rule_mixture_controls(changed,*args[1:])
    np.testing.assert_array_equal(baseline[:,2:4],result[:,2:4])
    assert not np.array_equal(baseline[:,1,3],result[:,1,3])


def test_kernel_preserves_nearest_component_ties_and_category_query():
    class Space(list):
        n_dims=1;n_cats=2;signature='test'
    space=Space([SimpleNamespace(categories=[SimpleNamespace(components=[0.,1.]),SimpleNamespace(components=[.2])])])
    geometry=SimpleNamespace(space=space,tolerance=1e-12,distances_to_polytope=lambda x,c:abs(x[:,0]-c))
    x=np.array([[[0.],[1.],[.5]],[[.9],[.1],[.2]]])
    kernel=component_kernel(space,x,np.array([1,2]),geometry)
    np.testing.assert_allclose(kernel,[[[.5,.5]],[[1.,0.]]])


@pytest.mark.parametrize('fault',['prefix','mass','kernel','shape'])
def test_invalid_comparison_is_rejected(fault):
    prior,kernel,labels,cut=example()
    if fault=='prefix':cut=4
    elif fault=='mass':prior[0,0]*=2
    elif fault=='kernel':kernel[0,0]*=2
    elif fault=='shape':labels=labels[:2]
    with pytest.raises((ValueError,AssertionError)):rule_mixture_controls(prior,kernel,labels,cut)

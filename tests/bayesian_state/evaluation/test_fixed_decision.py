"""Fixed-state readout invariants, uncertainty cancellation and redundancy."""
import numpy as np
import pytest

from src.Bayesian_state.evaluation.oral.fixed_decision import (
    feature_grid, fixed_rule_readout, summarize_feature_sweep,
)
from src.Bayesian_state.utils.numeric import softmax


def distances(rule, x):
    d = np.column_stack((np.maximum(.5-x[:, 0], 0), np.maximum(x[:, 0]-.5, 0)))
    return d if rule == 0 else d[:, ::-1]


def test_original_model_readout_with_particle_specific_betas():
    ids=np.array([[0,1],[1,-1]]);u=np.array([[.25,.75],[1.,0.]])
    b=np.array([[2.,8.],[4.,0.]])
    x=np.array([[[.2,.3],[.8,.3]],[[.9,.8],[.4,.8]]])
    result=fixed_rule_readout(ids,u,b,x,category_distances=distances,softmax=softmax)
    expected=np.zeros_like(result)
    for p in range(2):
        for g in range(2):
            for k in range(2):
                if u[p,k]>0:expected[p,g]+=u[p,k]*softmax(distances(ids[p,k],x[p,g:g+1]).T,-b[p,k],axis=0)[:,0]
    np.testing.assert_allclose(result,expected,atol=1e-15,rtol=0)
    np.testing.assert_allclose(result.sum(-1),1.,atol=1e-15,rtol=0)


def test_unused_feature_is_exactly_inert_and_inputs_unchanged():
    x=np.array([[.2,.3],[.8,.4]]);grid=np.linspace(0,1,5)
    probes=feature_grid(x,grid);before=probes.copy()
    result=fixed_rule_readout(np.zeros((2,1),int),np.ones((2,1)),np.full((2,1),3.),probes,
                              category_distances=distances,softmax=softmax)
    curves=result[:,1:,0].reshape(2,2,5)
    np.testing.assert_array_equal(curves[:,1],np.repeat(result[:,0,0,None],5,axis=1))
    np.testing.assert_array_equal(probes,before)
    np.testing.assert_array_equal(probes[:,0],x)


def test_opposing_particle_effects_survive_before_pooling():
    curves=np.array([[[.1,.5,.9]],[[.9,.5,.1]]])
    summary=summarize_feature_sweep(curves,np.array([.5,.5]))
    np.testing.assert_allclose(summary['mean_particle_range'],.8)
    np.testing.assert_allclose(summary['pooled_range'],0.)
    np.testing.assert_allclose(summary['endpoint_effect'],0.)
    np.testing.assert_allclose(summary['positive_endpoint_effect'],.4)
    np.testing.assert_allclose(summary['negative_endpoint_effect'],.4)


def test_fixed_original_choice_weights_are_not_reconditioned_per_probe():
    curves=np.array([[[.1,.9]],[[.7,.3]]])
    weights=np.array([.9,.1]);summary=summarize_feature_sweep(curves,weights)
    np.testing.assert_allclose(summary['curve'],[[.16,.84]])
    np.testing.assert_array_equal(weights,[.9,.1])


def test_nonmonotonic_sensitivity_is_kept_with_zero_endpoint_difference():
    summary=summarize_feature_sweep(np.array([[[.1,.9,.1]]]),np.ones(1))
    assert summary['endpoint_effect'][0]==0.
    assert summary['mean_particle_range'][0]==pytest.approx(.8)


def test_redundant_reasons_need_joint_contrasts():
    def either_distance(rule,x):
        active=(x>.5).any(axis=1)
        return np.column_stack((~active,active)).astype(float)
    probes=np.array([[[1.,1.],[0.,1.],[1.,0.],[0.,0.]]])
    q=fixed_rule_readout(np.array([[0]]),np.ones((1,1)),np.full((1,1),3.),probes,
                        category_distances=either_distance,softmax=softmax)[0,:,0]
    assert q[0]==q[1]==q[2] and q[0]>q[3]


@pytest.mark.parametrize('bad', ['weights','padding','duplicates','outside'])
def test_invalid_state_rejected(bad):
    ids=np.array([[0,-1]]);u=np.array([[1.,0.]]);b=np.ones((1,2));x=np.zeros((1,2,2))
    if bad=='weights':u[0,0]=.5
    if bad=='padding':u=np.array([[.5,.5]])
    if bad=='duplicates':ids=np.array([[0,0]])
    if bad=='outside':x[0,0,0]=-1.
    with pytest.raises((ValueError,AssertionError)):
        fixed_rule_readout(ids,u,b,x,category_distances=distances,softmax=softmax)

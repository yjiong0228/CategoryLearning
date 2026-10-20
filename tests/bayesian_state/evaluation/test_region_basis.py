import numpy as np
import pytest

from src.Bayesian_state.evaluation.oral.region_basis import region_basis
from src.Bayesian_state.evaluation.oral.decision_basis import DecisionBasisCollector
from src.Bayesian_state.evaluation.oral.stimulus_controls import component_kernel
from src.Bayesian_state.hypothesis_space.spaces.structural_0923 import build_relational_0923_space
from src.Bayesian_state.hypothesis_space.geometry.boundary import BoundaryGeometry


def test_repeated_response_rows_preserve_all_local_region_mass():
    q=np.array([[[.1,.2,.3,.4]]]);w=np.ones((1,1));a=np.full((1,1,4,4),.1);a[:,:,:,2]=.7
    rho=np.ones((1,1,4,1));before=a.copy()
    result=region_basis(geometry=q,rule_weights=w,associations=a,components=rho,choices=np.array([3]))
    np.testing.assert_allclose(result['choice_probability'],.7)
    np.testing.assert_allclose(result['prechoice_component_mass'].reshape(4),q.reshape(4))
    np.testing.assert_allclose(result['choice_component_joint_mass'].reshape(4)/.7,q.reshape(4))
    np.testing.assert_array_equal(a,before)


def test_prechoice_basis_independent_of_response_and_association_strength():
    q=np.array([[[.1,.2,.3,.4]]]);w=np.ones((1,1));rho=np.ones((1,1,4,1));a=np.full((1,1,4,4),.25)
    one=region_basis(geometry=q,rule_weights=w,associations=a,components=rho,choices=np.array([1]))
    a[0,0,0]=[.7,.1,.1,.1]
    two=region_basis(geometry=q,rule_weights=w,associations=a,components=rho,choices=np.array([4]))
    np.testing.assert_array_equal(one['prechoice_component_mass'],two['prechoice_component_mass'])
    assert not np.allclose(one['choice_component_joint_mass'],two['choice_component_joint_mass'])


def test_fixed_identity_matches_existing_collector_joint_mass():
    space=build_relational_0923_space();geometry=BoundaryGeometry(space);h=len(space)
    x=np.array([[.2,.3,.6,.7],[.5,.5,.5,.5]])
    rho=np.stack([component_kernel(space,x[:,None,:],np.full(2,g+1),geometry) for g in range(4)],axis=2)
    q=np.random.default_rng(9).dirichlet(np.ones(4),size=(2,h));w=np.full((2,h),1/h)
    a=np.broadcast_to(np.eye(4),(2,h,4,4));choices=np.array([1,4])
    result=region_basis(geometry=q,rule_weights=w,associations=a,components=rho,choices=choices)
    old=DecisionBasisCollector(space,choices,geometry=geometry)
    for t,y in enumerate(choices):
        old(t,'predictive',dict(weights=np.ones(1),choice_weights=w[t,None],rule_choice_probabilities=q[t,None],
            perceived_stimulus=x[t,None],active=np.ones((1,h)),prediction=(w[t]@q[t])[None],
            belief=w[t,None],beta=np.zeros((1,h)),search=np.zeros(1),admission_event=np.zeros(1)))
        np.testing.assert_allclose(result['choice_component_joint_mass'][t,:,y-1],old.rows[t]['choice_component_joint_mass'],atol=1e-14,rtol=0)
        assert np.count_nonzero(np.delete(result['choice_component_joint_mass'][t],y-1,axis=1))==0


@pytest.mark.parametrize('bad',['unnormalized','impossible'])
def test_invalid_probability_states_rejected(bad):
    q=np.zeros((1,1,4));q[:,:,0]=1;w=np.ones((1,1));a=np.eye(4)[None,None];rho=np.ones((1,1,4,1))
    if bad=='unnormalized':a=a*2
    with pytest.raises(ValueError):region_basis(geometry=q,rule_weights=w,associations=a,components=rho,choices=np.array([4]))

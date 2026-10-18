from itertools import permutations

import numpy as np
import pytest

from src.Bayesian_state.evaluation.oral.association_basis import AssociatedDecisionBasis
from src.Bayesian_state.evaluation.oral.decision_basis import DecisionBasisCollector
from src.Bayesian_state.hypothesis_space.spaces.structural_0923 import build_relational_0923_space


@pytest.fixture(scope='module')
def space(): return build_relational_0923_space()


def test_marginals_match_independent_association_enumeration(space):
    h=len(space); rng=np.random.default_rng(77); q=rng.dirichlet(np.ones(4),size=h)
    w=np.zeros(h);w[:2]=[.3,.7]
    a=rng.uniform(size=(h,24,3));a/=a.sum(axis=(1,2),keepdims=True);before=a.copy()
    result=AssociatedDecisionBasis(space)(probabilities=q,rule_weights=w,associations=a,
                                        stimulus=np.array([.2,.3,.6,.7]),choice=3)
    expected=np.zeros((h,4))
    for rule in range(h):
        for i,mapping in enumerate(permutations(range(4))):
            expected[rule,mapping.index(2)] += w[rule]*sum(a[rule,i])
    np.testing.assert_allclose(result['prior_component_mass'].sum(axis=2),expected,atol=1e-14)
    np.testing.assert_allclose(result['choice_component_joint_mass'].sum(axis=2),expected*q,atol=1e-14)
    np.testing.assert_allclose(result['choice_probability'],(expected*q).sum())
    np.testing.assert_array_equal(a,before)


def test_fixed_identity_reduces_to_existing_collector_with_disconnected_regions(space):
    rule,region=next((h.index,g) for h in space for g,c in enumerate(h.categories) if len(c.components)>1)
    h=len(space);w=np.zeros(h);w[rule]=1.;q=np.full((h,4),.25)
    a=np.zeros((h,24,1));a[:,0]=1.;x=np.full(4,.5)
    result=AssociatedDecisionBasis(space)(probabilities=q,rule_weights=w,associations=a,stimulus=x,choice=region+1)
    snapshot=dict(weights=np.ones(1),choice_weights=w[None,:],rule_choice_probabilities=q[None,:,:],
        perceived_stimulus=x[None,:],active=np.ones((1,h)),prediction=(w@q)[None,:],
        belief=w[None,:],beta=np.zeros((1,h)),search=np.zeros(1),admission_event=np.zeros(1))
    old=DecisionBasisCollector(space,np.array([region+1]));old(0,'predictive',snapshot)
    for key in ('prior_component_mass','choice_component_joint_mass'):
        np.testing.assert_array_equal(result[key][:,region],old.rows[0][key])
        assert np.count_nonzero(np.delete(result[key],region,axis=1))==0


def test_zero_probability_regions_keep_only_unconditional_mass(space):
    h=len(space);w=np.zeros(h);w[0]=1.;q=np.zeros((h,4));q[:,0]=1.
    a=np.full((h,24,1),1/24)
    result=AssociatedDecisionBasis(space)(probabilities=q,rule_weights=w,associations=a,
                                        stimulus=np.full(4,.3),choice=4)
    np.testing.assert_allclose(result['prior_component_mass'].sum(axis=(0,2)),.25)
    np.testing.assert_allclose(result['choice_component_joint_mass'].sum(axis=(0,2)),[.25,0,0,0])
    assert result['choice_probability']==pytest.approx(.25)


def test_invalid_weights_or_impossible_response_rejected(space):
    h=len(space);w=np.zeros(h);w[0]=1.;q=np.zeros((h,4));q[:,0]=1.
    a=np.zeros((h,24,1));a[:,0]=1.;basis=AssociatedDecisionBasis(space)
    for weights,choice in [(w*2,1),(w,2),(w,5)]:
        with pytest.raises(ValueError):basis(probabilities=q,rule_weights=weights,associations=a,
                                            stimulus=np.full(4,.3),choice=choice)

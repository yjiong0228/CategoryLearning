"""Active alternatives must retain their state-specific identity and weight."""
import numpy as np
import pytest

from src.Bayesian_state.evaluation.oral.candidate_cues import candidate_feature_profiles


def fixture():
    return dict(rule_ids=np.array([[0,1,-1],[0,2,-1]]),
        rule_features=np.array([[1,0],[0,1],[1,1]],bool),
        probabilities=np.array([[[.9,.1],[.1,.9],[0.,0.]],[[.8,.2],[.5,.5],[0.,0.]]]),
        belief=np.array([[.99,.01,0],[.8,.2,0]]),
        contribution=np.array([[1.,0.,0.],[1.,0.,0.]]),weights=np.array([.25,.75]),choice=1)


def test_downweighted_opposing_rule_remains_available_without_gaining_choice_weight():
    result=candidate_feature_profiles(**fixture())
    values=result['means']
    np.testing.assert_allclose(values['availability'],[1.,1.])
    np.testing.assert_allclose(values['choice_weight'],[1.,0.])
    assert values['belief_mass'][1]==pytest.approx(.1525)
    assert values['max_opposing_margin'][1]==pytest.approx(.2)
    assert values['belief_opposing_margin'][1]==pytest.approx(.002)
    assert values['contribution_opposing_margin'][1]==0.


def test_feature_membership_can_overlap_without_normalizing_across_features():
    result=candidate_feature_profiles(**fixture())
    assert result['means']['rule_count'].sum()>2
    assert result['means']['belief_mass'].sum()>1


def test_univariate_scope_changes_coverage_without_reweighting_existing_states():
    data=fixture();data['rule_features'][2]=False
    before={k:v.copy() for k,v in data.items() if isinstance(v,np.ndarray)}
    result=candidate_feature_profiles(**data)
    assert result['means']['availability'][1]==.25
    assert result['means']['belief_mass'][1]==pytest.approx(.0025)
    for key,value in before.items():np.testing.assert_array_equal(data[key],value)


def test_joint_coverage_retains_particle_associations():
    data=fixture();data['rule_ids']=np.array([[0,-1,-1],[1,-1,-1]])
    data['probabilities']=np.array([[[.9,.1],[0,0],[0,0]],[[.1,.9],[0,0],[0,0]]])
    data['belief']=data['contribution'].copy()
    result=candidate_feature_profiles(**data)
    np.testing.assert_allclose(result['means']['availability'],[.25,.75])
    assert np.all(result['particle']['availability'],axis=1).sum()==0


@pytest.mark.parametrize('fault',['padding','duplicate','empty','noninteger','unknown_rule','feature_dtype',
    'probability_sum','belief_sum','contribution_sum','weight_sum','nan','choice'])
def test_invalid_sparse_state_rejected(fault):
    data=fixture()
    if fault=='padding':data['probabilities'][0,2]=[.5,.5]
    if fault=='duplicate':data['rule_ids'][0,1]=0
    if fault=='empty':data['rule_ids'][0]=-1
    if fault=='noninteger':data['rule_ids']=data['rule_ids'].astype(float)
    if fault=='unknown_rule':data['rule_ids'][0,0]=3
    if fault=='feature_dtype':data['rule_features']=data['rule_features'].astype(int)
    if fault=='probability_sum':data['probabilities'][0,0]=[.8,.1]
    if fault=='belief_sum':data['belief'][0]=0
    if fault=='contribution_sum':data['contribution'][0]=0
    if fault=='weight_sum':data['weights']=np.array([.5,.75])
    if fault=='nan':data['belief'][0,0]=np.nan
    if fault=='choice':data['choice']=3
    with pytest.raises((ValueError,AssertionError)):
        candidate_feature_profiles(**data)

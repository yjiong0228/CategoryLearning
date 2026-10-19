"""Joint effects must preserve state uncertainty, redundancy and contrast scope."""
import numpy as np
import pytest

from src.Bayesian_state.evaluation.oral.joint_relevance import context_effects,weighted_profile


def masks(n=2):return ((np.arange(2**n)[:,None]>>np.arange(n))&1).astype(bool)


def test_joint_use_differs_from_uncertain_single_rules_with_identical_marginals():
    separate=np.array([[.9,.1,.9,.1],[.9,.9,.1,.1]])
    simultaneous=np.repeat(separate.mean(0)[None,:],2,axis=0)
    a,b=[context_effects(q,masks(),np.array([True,True])) for q in (separate,simultaneous)]
    np.testing.assert_allclose(a['positive_context_effect'].mean(0),b['positive_context_effect'].mean(0))
    np.testing.assert_array_equal(a['minimum_positive_effect'],[0.,0.])
    np.testing.assert_allclose(b['minimum_positive_effect'],[.4,.4])


def test_redundant_reasons_retain_context_effects_and_zero_current_removal():
    e=context_effects(np.array([[.9,.9,.9,.1]]),masks(),np.array([True,True]))
    np.testing.assert_array_equal(e['actual_effect'],[[0.,0.]])
    np.testing.assert_allclose(e['positive_context_effect'],[[.8,.8]])
    np.testing.assert_allclose(e['minimum_positive_effect'],[.8])


def test_unused_extra_reason_cannot_inherit_effect_from_used_feature():
    e=context_effects(np.array([[.9,.1,.9,.1]]),masks(),np.array([True,True]))
    np.testing.assert_allclose(e['mean_positive_effect'],[.4])
    np.testing.assert_array_equal(e['minimum_positive_effect'],[0.])


def test_report_restricted_contexts_hold_unreported_features_fixed():
    q=np.array([[.9,.9,.9,.1]])
    e=context_effects(q,masks(),np.array([True,False]))
    np.testing.assert_array_equal(e['minimum_positive_effect'],[0.])
    q=np.array([[.9,.1,.9,.1]])
    e=context_effects(q,masks(),np.array([True,False]))
    np.testing.assert_allclose(e['minimum_positive_effect'],[.8])


def test_same_global_endpoints_do_not_erase_conditional_effects():
    e=context_effects(np.array([[.9,.1,.1,.9]]),masks(),np.array([True,True]))
    np.testing.assert_allclose(e['minimum_positive_effect'],[.8])
    np.testing.assert_allclose(e['negative_context_effect'],[[.8,.8]])


def test_direction_and_absolute_relevance_are_distinct():
    e=context_effects(np.array([[.1,.9]]),masks(1),np.array([True]))
    np.testing.assert_array_equal(e['minimum_positive_effect'],[0.])
    np.testing.assert_allclose(e['minimum_absolute_effect'],[.8])
    np.testing.assert_allclose(e['negative_context_effect'],[[.8]])


def test_probe_column_order_irrelevant():
    q=np.array([[.9,.7,.5,.1]]);order=[3,1,0,2]
    a=context_effects(q,masks(),np.array([True,True]))
    b=context_effects(q[:,order],masks()[order],np.array([True,True]))
    for key in a:np.testing.assert_array_equal(a[key],b[key])


def test_threshold_integral_matches_weighted_minimum_and_preserves_zero_mass():
    x=np.array([0.,.2,.8]);w=np.array([.3,.2,.5]);t=np.array([0.,.2,.8,1.])
    a=weighted_profile(x,w,t)
    assert a['mean']==pytest.approx(.44)
    assert np.dot(a['mass_above'][:-1],np.diff(t))==pytest.approx(a['mean'])
    np.testing.assert_allclose(a['mass_above'],[.7,.5,0.,0.])
    b=weighted_profile(x,w,t,np.array([True,True,False]))
    assert b['mean']==pytest.approx(.04) and b['eligible_mass']==pytest.approx(.5)


def test_reflection_at_midpoint_can_be_uninformative_even_for_relevant_feature():
    # A smooth f(x)=x has full unit-range sensitivity; reflection of x=.5
    # leaves the same input. This diagnostic must preserve that limitation.
    e=context_effects(np.array([[.5,.5]]),masks(1),np.array([True]))
    assert e['minimum_absolute_effect'][0]==0.


@pytest.mark.parametrize('fault',['duplicate_mask','empty_report','bad_probability'])
def test_invalid_probes_rejected(fault):
    q=np.array([[.9,.7,.5,.1]]);m=masks();r=np.array([True,True])
    if fault=='duplicate_mask':m[1]=m[0]
    if fault=='empty_report':r[:]=False
    if fault=='bad_probability':q[0,0]=np.nan
    with pytest.raises(ValueError):context_effects(q,m,r)


def test_numeric_semantics_and_relevance_feature_set_remain_separate():
    from src.Bayesian_state.workflows.analysis.model_0923_joint_reason_profile import report_content
    tokens=['facet:[[0.0,0.0,-1.0,1.0,0.0]]']
    mentioned,truth=report_content(tokens,np.array([[.2,.3,.8,.2],[.2,.3,.2,.8]]))
    np.testing.assert_array_equal(mentioned,[False,False,True,True])
    np.testing.assert_array_equal(truth,[True,False])


def test_symbolic_reference_retains_feature_but_full_truth_is_unresolved():
    from src.Bayesian_state.workflows.analysis.model_0923_joint_reason_profile import report_content
    mentioned,truth=report_content(['body:0:lt'],np.array([[.2,.3,.8,.2]]))
    np.testing.assert_array_equal(mentioned,[True,False,False,False])
    assert truth is None


def test_common_contexts_recover_redundancy_with_an_unreported_reason():
    from src.Bayesian_state.evaluation.model_0923_joint_reason_contexts import full_context_effects
    q=np.array([[.9,.9,.9,.1]]);reported=np.array([True,False])
    restricted=context_effects(q,masks(),reported)
    common=full_context_effects(q,masks(),reported)
    assert restricted['minimum_positive_effect'][0]==0.
    assert common['minimum_positive_effect'][0]==pytest.approx(.8)
    assert common['positive_context_effect'].shape==(1,1)


def test_common_contexts_still_reject_an_unused_reported_feature():
    from src.Bayesian_state.evaluation.model_0923_joint_reason_contexts import full_context_effects
    q=np.array([[.9,.1,.9,.1]])
    e=full_context_effects(q,masks(),np.array([False,True]))
    assert e['minimum_positive_effect'][0]==0.

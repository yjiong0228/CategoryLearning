"""Exact conditional execution information and truth-centred numerical windows."""
import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis.model_1024_behavior_recovery import execution_reference, event_prefix, monitor_class


def test_indistinguishable_rules_preserve_the_execution_prior():
    result = execution_reference(np.array([.2,.8]), np.array([[.3,.7],[.3,.7]]), 2)
    np.testing.assert_allclose(result['posterior'], [.2,.8])
    assert result['information'] == pytest.approx(0., abs=1e-14)
    assert result['optimal_accuracy'] == pytest.approx(.8)


def test_disjoint_choices_identify_execution_and_reveal_prior_entropy():
    result = execution_reference(np.array([.2,.8]), np.eye(2), 1)
    np.testing.assert_array_equal(result['posterior'], [1.,0.])
    assert result['information'] == pytest.approx(-.2*np.log(.2)-.8*np.log(.8))
    assert result['optimal_accuracy'] == 1.


def test_posterior_uses_unobserved_execution_distribution_and_one_based_choice():
    result = execution_reference(np.array([.25,.75]), np.array([[.9,.1],[.1,.9]]), 1)
    np.testing.assert_allclose(result['posterior'], [.75,.25])
    np.testing.assert_allclose(result['marginal'], [.3,.7])
    assert result['optimal_accuracy'] >= result['prior_optimal_accuracy']


def test_zero_prior_rule_and_impossible_category_do_not_create_nan_information():
    result = execution_reference(np.array([1.,0.]), np.array([[1.,0.],[0.,1.]]), 1)
    assert np.isfinite(result['information']) and result['information'] == 0
    with pytest.raises(ValueError):
        execution_reference(np.array([1.,0.]), np.array([[1.,0.],[0.,1.]]), 2)


@pytest.mark.parametrize('prior,likelihood,choice', [([.1,.2],[[.5,.5],[.5,.5]],1),
    ([.5,.5],[[1.,1.],[.5,.5]],1),([.5,-.5],[[.5,.5],[.5,.5]],1),
    ([.5,.5],[[.5,.5],[.5,.5]],0)])
def test_invalid_probability_coordinates_fail(prior,likelihood,choice):
    with pytest.raises((ValueError,AssertionError)):
        execution_reference(np.array(prior),np.array(likelihood),choice)


def test_window_preserves_history_and_the_prespecified_post_entry_length():
    present=np.zeros(1000,bool);present[155:169]=True
    assert event_prefix(present,32,64,32)==(123,219,224)
    present=np.zeros(130,bool);present[100:]=True
    assert event_prefix(present,32,64,32)==(68,130,130)
    with pytest.raises(ValueError):event_prefix(np.zeros(100,bool),32,64,32)


def test_passive_monitor_records_weight_and_target_support_without_mutation():
    class Base:
        def __init__(self):self.targets=np.array([0]);self.choices=np.array([1]);self.calls=[]
        def __call__(self,t,phase,snapshot):self.calls.append(phase)
        def arrays(self):return dict(choices=self.choices)
    cls=monitor_class(Base);observer=cls()
    weights=np.array([.25,.75]);active=np.array([[1,0],[0,1]])
    snapshot=dict(weights=weights,active=active)
    observer(0,'predictive',snapshot);observer(0,'filtered',snapshot)
    z=observer.arrays()
    np.testing.assert_array_equal(weights,[.25,.75]);np.testing.assert_array_equal(active,[[1,0],[0,1]])
    assert z['monitor_filtered_ess'][0]==pytest.approx(1/.625)
    assert z['monitor_filtered_target_particles'][0]==1
    assert z['monitor_filtered_target_ess'][0]==1


def test_presence_spells_and_censoring_preserve_trial_definitions():
    from src.Bayesian_state.workflows.analysis.model_1024_behavior_recovery_summary import runs,episode_scores
    assert runs(np.array([False,True,True,False,True]))==[(1,3),(4,5)]
    raw=dict(rule=np.array([[-1],[0],[0],[-1],[0]]))
    state=dict(target_rules=np.array([0]),choices=np.ones(5),filtered_target_present=np.array([.1,.3,.7,.1,.8]))
    rows=episode_scores(state,raw,5)
    assert rows[0]['entry_trial']==2 and rows[0]['last_present_trial']==3 and rows[0]['exit_trial']==4
    assert rows[0]['true_length']==2 and rows[0]['detection_delay']==1 and not rows[0]['right_censored']
    assert rows[1]['right_censored'] and rows[1]['exit_trial'] is None

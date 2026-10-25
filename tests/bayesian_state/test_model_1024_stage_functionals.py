"""Functional uncertainty must retain execution randomness and censoring."""
import numpy as np

from src.Bayesian_state.workflows.analysis.model_1024_stage_summary import functional_rows


def test_integrated_execution_count_keeps_conditional_bernoulli_variance():
    part=dict(full_indices=np.tile(np.arange(2)[:,None],(1,3)),filtered_weights=np.full((3,2),.5),
        particle_present=np.ones((3,2),bool),particle_target_execution=np.full((3,2),.5),
        particle_target_belief=np.full((3,2),.5))
    truth=dict(present=np.ones(3,bool),target_execution=np.array([True,False,True]),target_belief=np.full(3,.5))
    rows=functional_rows([part],np.zeros((1,3)),truth,'full',True)
    count=next(r for r in rows if r['function']=='target_execution_trials')
    assert count['mean']==1.5
    assert count['variance']==.75
    assert count['within_execution_variance']==.75
    assert np.isnan(count['q50'])


def test_no_entry_is_retained_as_probability_and_endpoint_censor_code():
    part=dict(full_indices=np.tile(np.arange(2)[:,None],(1,3)),filtered_weights=np.tile([.25,.75],(3,1)),
        particle_present=np.array([[False,False],[True,False],[True,False]]),
        particle_target_execution=np.array([[0,0],[1,0],[1,0]]),particle_target_belief=np.array([[0.,0.],[1.,0.],[1.,0.]]))
    truth=dict(present=np.zeros(3,bool),target_execution=np.zeros(3,bool),target_belief=np.zeros(3))
    rows=functional_rows([part],np.zeros((1,3)),truth,'full',True)
    entry=next(r for r in rows if r['function']=='first_entry_trial')
    assert entry['no_entry_mass']==.75
    assert entry['q50']==4
    assert entry['true_value']==4

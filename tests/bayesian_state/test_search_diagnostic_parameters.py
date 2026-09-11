"""Search reports must distinguish Model0826 parameter profiles."""
from copy import deepcopy
import numpy as np
from src.Bayesian_state.optimization.diagnostics.search import flatten_hyperparams


def test_model0826_packed_parameters_are_not_missing_or_identical():
    transition='engine.modules.hypo_transitions_mod.kwargs'
    memory='engine.modules.memory_mod.kwargs'
    point={memory+'.gamma':.97,'__profile_candidate__:workspace_execution':{
        transition+'.capacity':5,transition+'.persistent_execution.enabled':False},
        '__profile_candidate__:global_search':{
            transition+'.nested_feedback_accumulator_controller.global_search':.85,
            transition+'.nested_feedback_accumulator_controller.global_search_failure_gain':.5},
        transition+'.nested_feedback_accumulator_controller.accumulator_logit_gain':.75}
    original=deepcopy(point)
    row=flatten_hyperparams(point)
    assert row['gamma']==.97
    assert row['capacity']==5 and row['persistent_execution']==0
    assert row['global_search']==.85 and row['global_search_failure_gain']==.5
    assert row['accumulator_logit_gain']==.75
    alternative=deepcopy(point)
    alternative['__profile_candidate__:global_search'][transition+'.nested_feedback_accumulator_controller.global_search']=.15
    assert row['strategy_id']!=flatten_hyperparams(alternative)['strategy_id']
    assert point==original


def test_legacy_mapping_and_flat_memory_parameters_agree():
    prefix='engine.modules.memory_mod.kwargs'
    nested=flatten_hyperparams({prefix:{'gamma':.8,'w0':0}})
    flat=flatten_hyperparams({prefix+'.gamma':.8,prefix+'.w0':0})
    assert flat['gamma']==nested['gamma']==.8
    assert flat['w0']==nested['w0']==0
    assert np.isnan(flat['capacity'])

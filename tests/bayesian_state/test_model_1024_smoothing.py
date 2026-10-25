"""Independent enumeration, coherent ancestry and pre-resampling timing."""
import numpy as np
import pytest

from src.Bayesian_state.evaluation.particle_smoothing import (
    ancestral_weights, exact_hmm, exact_hmm_paths, probabilities,
    resampling_parents, TargetGenealogyCollector, trace_values)
from src.Bayesian_state.inference.backends.particle_filter import _trace_ancestral_indices
from src.Bayesian_state.workflows.analysis.model_1024_smoothing import brute_hmm, genealogy_summary


def diagnostic():
    initial=np.array([.6,.4])
    transitions=np.array([[[.9,.1],[.2,.8]],[[.7,.3],[.05,.95]],[[.8,.2],[.3,.7]]])
    likelihood=np.array([[.9,.1],[.5,.5],[.2,.8],[.05,.95]])
    return initial,transitions,likelihood


def test_forward_backward_matches_independent_nonstationary_path_enumeration():
    initial,transitions,likelihood=diagnostic();exact=exact_hmm(initial,transitions,likelihood)
    brute=brute_hmm(initial,transitions,likelihood)
    for key in ('smoothed','pairs','log_evidence'):
        np.testing.assert_allclose(exact[key],brute[key],atol=1e-14,rtol=0)
    for end in range(1,len(likelihood)+1):
        prefix=brute_hmm(initial,transitions[:end-1],likelihood[:end])
        np.testing.assert_allclose(exact['filtered'][end-1],prefix['smoothed'][-1],atol=1e-14,rtol=0)
    np.testing.assert_allclose(exact['pairs'].sum(2),exact['smoothed'][:-1],atol=1e-14)
    np.testing.assert_allclose(exact['pairs'].sum(1),exact['smoothed'][1:],atol=1e-14)
    assert exact['smoothed'][0,1]>exact['filtered'][0,1]
    np.testing.assert_allclose(exact['smoothed'][-1],exact['filtered'][-1])


def test_indistinguishable_emissions_add_no_state_information():
    initial,transitions,_=diagnostic();likelihood=np.tile([.3,.3],(4,1))
    exact=exact_hmm(initial,transitions,likelihood)
    np.testing.assert_allclose(exact['filtered'],exact['predictive'],atol=1e-14)
    np.testing.assert_allclose(exact['smoothed'],exact['filtered'],atol=1e-14)


def test_backward_sampling_respects_joint_path_distribution_and_seed():
    initial,transitions,likelihood=diagnostic();exact=exact_hmm(initial,transitions,likelihood)
    samples=exact_hmm_paths(exact['filtered'],transitions,40000,41)
    np.testing.assert_array_equal(samples,exact_hmm_paths(exact['filtered'],transitions,40000,41))
    for t in range(3):
        joint=np.bincount(samples[:,t]*2+samples[:,t+1],minlength=4).reshape(2,2)/len(samples)
        np.testing.assert_allclose(joint,exact['pairs'][t],atol=.009,rtol=0)


def test_single_trial_and_impossible_paths():
    result=exact_hmm(np.array([.4,.6]),np.empty((0,2,2)),np.array([[1.,0.]]))
    np.testing.assert_array_equal(result['smoothed'],[[1.,0.]])
    assert result['pairs'].shape==(0,2,2)
    with pytest.raises(ValueError,match='zero probability'):
        exact_hmm(np.array([.4,.6]),np.empty((0,2,2)),np.zeros((1,2)))


def test_final_resampling_row_is_unused_for_pre_resampling_terminal_weights():
    # At t=0, child slots 0/1 share ancestor 1; the final row maps all
    # particles to slot 2. Applying that last row would be a timing error.
    log=[dict(trial_index=0,ancestors=[1,1,2]),dict(trial_index=2,ancestors=[2,2,2])]
    parents=resampling_parents(log,3,3);indices=_trace_ancestral_indices(parents)
    w=np.array([.6,.3,.1]);summary=ancestral_weights(indices,w)
    np.testing.assert_allclose(summary['weights'],[[0.,.9,.1],[.6,.3,.1],[.6,.3,.1]])
    np.testing.assert_array_equal(summary['unique_ancestors'],[2,3,3])
    assert summary['ancestral_ess'][0]==pytest.approx(1/.82)
    parents[-1]=np.arange(3)
    np.testing.assert_array_equal(indices,_trace_ancestral_indices(parents))
    values=np.array([[10,20,30],[40,50,60],[70,80,90]])
    np.testing.assert_array_equal(trace_values(values,indices),[[20,20,30],[40,50,60],[70,80,90]])


def test_zero_lag_equals_filter_and_prefix_endpoint_does_not_use_future_rows():
    parents=np.array([[1,1,2],[0,2,2],[2,2,2]])
    weights=np.array([[.2,.7,.1],[.6,.3,.1],[.05,.05,.9]])
    present=np.array([[False,True,True],[False,True,False],[True,False,False]])
    values=dict(present=present,high_belief=present,target_execution=present,
                target_belief=present.astype(float),source=np.where(present,2,0))
    result=genealogy_summary(values,parents,weights,_trace_ancestral_indices,2,0)
    for name in ('present','high_belief','target_execution','target_belief','source'):
        np.testing.assert_allclose(result['filtered_'+name],result['lag_'+name],atol=1e-14)
    assert result['prefix_present'][0]==pytest.approx(1.)
    assert result['prefix_present'][1]==pytest.approx(.3)
    assert result['full_present'][-1]==pytest.approx(.05)
    np.testing.assert_array_equal(result['lag_endpoints'],[1,2,3])


@pytest.mark.parametrize('log',[
    [dict(trial_index=0,ancestors=[0,1])],
    [dict(trial_index=3,ancestors=[0,1,2])],
    [dict(trial_index=0,ancestors=[0.,1.,2.])],
    [dict(trial_index=0,ancestors=[0,1,3])],
    [dict(trial_index=0,ancestors=[0,1,2])]*2])
def test_malformed_resampling_records_fail(log):
    with pytest.raises(ValueError):resampling_parents(log,3,3)


@pytest.mark.parametrize('value',[[.2,.2],[-.1,1.1],[np.nan,1.],[np.inf,0.]])
def test_invalid_weights_are_not_silently_repaired(value):
    with pytest.raises(ValueError):probabilities(np.array(value))


def test_sparse_target_recorder_preserves_inputs_and_event_definitions():
    collector=TargetGenealogyCollector([1]);active=np.array([[1,1],[1,0],[1,1]])
    belief=np.array([[.4,.6],[1.,0.],[.7,.3]])
    contribution=np.array([[1.,0.],[1.,0.],[0.,1.]])
    weights=np.array([.2,.3,.5]);snapshot=dict(active=active,belief=belief,
        choice_weights=contribution,weights=weights)
    collector(0,'predictive',snapshot);collector(0,'filtered',snapshot)
    result=collector.arrays();np.testing.assert_array_equal(result['source'],[[1,0,2]])
    np.testing.assert_array_equal(result['high_belief'],[[True,False,False]])
    weights[:]=1/3;belief[:]=.5
    np.testing.assert_array_equal(result['filtered_weights'],[[.2,.3,.5]])
    np.testing.assert_allclose(result['target_belief'],[[.6,0.,.3]])


def test_recorder_rejects_mixture_and_out_of_order_callbacks():
    collector=TargetGenealogyCollector([0])
    with pytest.raises(ValueError):collector(0,'filtered',dict(weights=np.array([1.])))
    snapshot=dict(active=np.ones((1,2)),belief=np.array([[.5,.5]]),choice_weights=np.array([[.5,.5]]))
    with pytest.raises(ValueError,match='single-rule'):collector(0,'predictive',snapshot)


def test_common_endpoint_path_features_preserve_spell_and_censor_definitions():
    from src.Bayesian_state.workflows.analysis.model_1024_smoothing_summary import coherent_features
    present=np.array([[False,False,True],[True,False,True],[True,False,False],[False,False,True]])
    result=coherent_features(present,present,present.astype(float))
    np.testing.assert_array_equal(result['presence_trials'],[2,0,3])
    np.testing.assert_array_equal(result['longest_presence'],[2,0,2])
    np.testing.assert_array_equal(result['entry_count'],[1,0,2])
    np.testing.assert_array_equal(result['first_entry_trial'],[2,5,1])
    np.testing.assert_array_equal(result['no_entry'],[False,True,False])


def test_error_metrics_keep_rare_class_denominator_and_false_positives():
    from src.Bayesian_state.workflows.analysis.model_1024_smoothing_summary import metric_rows
    truth=dict(present=np.array([1,1,0,1],bool),high_belief=np.array([1,0,0,1],bool),
        target_execution=np.array([0,1,0,0],bool),target_belief=np.array([.6,.3,0.,.8]),
        source=np.array([1,2,0,1]),wrong=np.array([1,1,1,0],bool))
    state={k:truth[k].astype(float) for k in ['present','high_belief','target_execution','target_belief']}
    state['source']=np.array([[.1,.7,.2],[.1,.6,.3],[.8,.1,.1],[.1,.8,.1]])
    rows=metric_rows(state,truth);rare=next(r for r in rows if r['state']=='error_available_other')
    assert rare['trials']==3 and rare['positives']==1
    assert rare['hard_true_positive']==1 and rare['hard_false_positive']==1
    assert rare['hard_precision']==pytest.approx(.5)

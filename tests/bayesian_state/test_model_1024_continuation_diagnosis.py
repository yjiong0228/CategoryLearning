"""Oracle probes preserve valid active sets and remain explicitly opt-in."""
from types import SimpleNamespace
import numpy as np
import pytest
from src.Bayesian_state.model.modules.hypothesis_transition.contracts import HypothesisSelection, TransitionContext
from src.Bayesian_state.simulation.continuation_probes import continuation_probe, probe_selection
from src.Bayesian_state.workflows.analysis.model_1024_continuation_diagnosis import cutoffs


def context():
    return TransitionContext(trial_index=32,posterior=np.array([.7,.2,.1,0.]),active_before=np.array([0,1,2]))


def select(after):
    return HypothesisSelection.from_active_sets([0,1,2],after,diagnostics={'capacity_dropped':2})


def test_protection_cancels_eviction_without_reassigning_beliefs():
    c=context(); s=probe_selection(select([0,1,3]),c,target=2,capacity=3,ensure=False,protect=True)
    np.testing.assert_array_equal(s.active_after,[0,1,2])
    assert not len(s.newcomers) and not len(s.dropped)
    assert s.diagnostics['probe_protected'] and s.diagnostics['capacity_dropped'] is None
    np.testing.assert_array_equal(c.posterior,[.7,.2,.1,0.])


def test_discovery_is_one_valid_least_belief_replacement():
    c=context(); s=probe_selection(select([0,1,2]),c,target=3,capacity=3,ensure=True,protect=False)
    np.testing.assert_array_equal(s.active_after,[0,1,3])
    np.testing.assert_array_equal(s.newcomers,[3]); np.testing.assert_array_equal(s.dropped,[2])
    assert s.diagnostics['capacity_dropped']==2 and s.diagnostics['probe_inserted']


def test_discovery_open_slot_and_tie_order():
    c=TransitionContext(trial_index=4,posterior=np.array([.5,.5,0.,0.]),active_before=np.array([1,0]))
    s=HypothesisSelection.from_active_sets([1,0],[1,0])
    a=probe_selection(s,c,target=3,capacity=3,ensure=True,protect=True)
    np.testing.assert_array_equal(a.active_after,[1,0,3]); assert not len(a.dropped)
    a=probe_selection(s,c,target=3,capacity=2,ensure=True,protect=True)
    np.testing.assert_array_equal(a.active_after,[0,3]); assert a.diagnostics['capacity_dropped']==1


def test_retention_keeps_natural_discovery_and_other_proposals():
    c=context()
    for after in ([0,1,3],[0,1,2]):
        s=select(after)
        a=probe_selection(s,c,target=0,capacity=3,ensure=False,protect=True)
        np.testing.assert_array_equal(a.active_after,s.active_after)
        assert not a.diagnostics['probe_protected']


def test_context_restores_methods_after_failure():
    class H:
        def select_hypotheses(self,*a,**k): pass
        def _finish_hypothesis_transition(self,*a,**k): pass
    before=(H.select_hypotheses,H._finish_hypothesis_transition)
    with pytest.raises(RuntimeError):
        with continuation_probe(H,arm='AVAILABLE',target=0,cut=32):
            assert H.select_hypotheses is not before[0]
            raise RuntimeError('abort')
    assert (H.select_hypotheses,H._finish_hypothesis_transition)==before
    with continuation_probe(H,arm='BASE',target=0,cut=32):
        assert (H.select_hypotheses,H._finish_hypothesis_transition)==before


def test_cutoffs_keep_actual_training_boundary():
    assert cutoffs(160,32)==dict(half_training_block=64,training_end=160)
    assert cutoffs(704,32)==dict(half_training_block=352,training_end=704)
    with pytest.raises(ValueError): cutoffs(32,32)


def test_clustered_score_uses_only_independent_filter_pairs():
    from src.Bayesian_state.workflows.analysis.model_1024_continuation_summary import cross_filter_energy, clustered_se
    values=np.array([[0.],[0.],[2.],[2.]])
    labels=np.array([0,0,1,1])
    # Mean distance to 1 is 1. Across-filter distance is exactly 2.
    assert cross_filter_energy(values,np.array([1.]),labels)==0.
    assert clustered_se(np.array([[0.,0.],[2.,2.]]))==pytest.approx(1.)
    with pytest.raises(ValueError): cross_filter_energy(values,np.array([1.]),np.zeros(4))


def test_future_events_use_identical_complete_followup_for_real_and_generated():
    from src.Bayesian_state.workflows.analysis.model_1024_continuation_summary import events
    labels=np.ones(128,dtype=int)
    choices=np.r_[np.repeat(2,32),np.ones(96,dtype=int)]
    records,truth=events(np.stack([choices,np.ones(128,dtype=int)]),labels,choices,2)
    np.testing.assert_array_equal(records['jump_joint'],[True,False])
    assert truth['gain']==1. and truth['maintenance']==1.
    assert truth['minimum128']==.75


def test_initial_availability_preserves_existing_target_under_tied_eviction():
    c=TransitionContext(trial_index=32,posterior=np.array([.5,.5,0.]),active_before=np.array([0,1]))
    natural=HypothesisSelection.from_active_sets([0,1],[0,2],diagnostics={'capacity_dropped':1})
    result=probe_selection(natural,c,target=1,capacity=2,ensure=True,protect=False)
    np.testing.assert_array_equal(result.active_after,[0,1])
    assert len(np.unique(result.active_after))==2 and not len(result.newcomers)
    assert result.diagnostics['probe_protected'] and not result.diagnostics['probe_inserted']

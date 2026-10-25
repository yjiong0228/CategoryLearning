"""Check clustered denominators and probability-bin boundary behavior."""
import numpy as np
import pytest

from src.Bayesian_state.evaluation.trajectory_calibration import bootstrap_counts, calibration, ratio_interval


def test_stratified_resampling_keeps_each_context_size():
    w=bootstrap_counts(['a','a','b','b','b'],100,123)
    np.testing.assert_array_equal(w[:,:2].sum(axis=1),np.full(100,2))
    np.testing.assert_array_equal(w[:,2:].sum(axis=1),np.full(100,3))
    np.testing.assert_array_equal(w,bootstrap_counts(['a','a','b','b','b'],100,123))


def test_whole_path_ratio_keeps_unequal_trial_denominators():
    # Equal numbers of paths do not imply equal numbers of observed trials.
    w=np.array([[2,0],[1,1],[0,2]])
    interval=ratio_interval([1,0],[1,9],w)
    assert interval['value']==.1
    assert interval['lower']<.1<interval['upper']
    # A constant per-trial rate is exact under every cluster resample.
    constant=ratio_interval([.2,1.8],[1,9],w)
    np.testing.assert_allclose([constant['lower'],constant['upper']],[.2,.2])


def test_calibration_assigns_zero_one_and_shared_edges_once():
    predictions=[np.array([0.,.2,.4,.6,.8,1.])]*8
    truths=[np.array([0,0,0,1,1,1],bool)]*8
    counts=bootstrap_counts(['a']*8,20,1)
    rows=calibration(predictions,truths,counts,[0,.2,.4,.6,.8,1.],8,1,.1)
    assert sum(row['trials'] for row in rows)==48
    assert [row['trials'] for row in rows]==[8,8,8,8,16]
    assert rows[0]['practical_equivalence']
    assert not rows[1]['practical_equivalence']


def test_only_probability_boundary_roundoff_is_tolerated():
    counts=np.ones((4,1),int)
    rows=calibration([np.array([-1e-15,1+1e-15])],[np.array([0,1])],counts,[0,.5,1],1,1,.1)
    assert [r['mean_prediction'] for r in rows]==[0.,1.]
    with pytest.raises(ValueError):
        calibration([np.array([1.01])],[np.array([1])],counts,[0,1],1,1,.1)

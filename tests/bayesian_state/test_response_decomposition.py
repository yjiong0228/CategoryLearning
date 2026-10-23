"""Limiting cases distinguish readout changes from state-distribution changes."""
import numpy as np
import pytest

from src.Bayesian_state.evaluation.model_1023_response_decomposition import accuracy_components


def test_same_states_assign_all_gain_to_readout():
    p=np.array([[[0.,.8,.2],[.1,.2,.7]],[[.2,.5,.3],[.4,.4,.2]]])
    q=p**2;q/=q.sum(-1,keepdims=True)
    value=accuracy_components(p,q,np.array([2,3]))
    np.testing.assert_allclose(value['state_prediction_change'],0,atol=1e-15)
    np.testing.assert_allclose(value['total_change'],value['readout_change'],atol=1e-15)
    assert value['readout_change'][0,0]==pytest.approx(.64/.68-.8)


def test_uniform_new_states_assign_all_change_to_state_predictions():
    original=np.array([[[.9,.1],[.2,.8]]])
    squared=np.full_like(original,.5)
    value=accuracy_components(original,squared,np.array([1,2]))
    np.testing.assert_array_equal(value['readout_change'],0)
    np.testing.assert_allclose(value['state_prediction_change'],[[-.4,-.3]])
    np.testing.assert_array_equal(value['total_change'],value['state_prediction_change'])


def test_rejects_invalid_task_labels_and_probabilities():
    p=np.array([[[.4,.6]]])
    with pytest.raises(ValueError,match='one-based'):accuracy_components(p,p,np.array([0]))
    with pytest.raises(ValueError,match='one-based'):accuracy_components(p,p,np.array([1.5]))
    with pytest.raises(AssertionError):accuracy_components(p,p*.5,np.array([1]))

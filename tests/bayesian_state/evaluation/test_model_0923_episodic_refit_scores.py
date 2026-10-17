import numpy as np
import pytest

from src.Bayesian_state.evaluation.model_0923_episodic_refit import loss_parts, active_precision_mean
from src.Bayesian_state.evaluation.model_0923_h_validation import nll


def test_error_and_correct_losses_add_to_sequence_score():
    p = np.array([[.8,.25,.6],[.6,.5,.5]])
    values = loss_parts(p,[True,False,True])
    assert values['correct_loss']+values['error_loss'] == pytest.approx(nll(p))
    assert values['error_loss'] == pytest.approx(-np.log(.375)/3)


def test_single_correctness_class_has_zero_other_contribution():
    p = np.array([[.1,.8],[.4,.6]])
    assert loss_parts(p,[True,True])['error_loss'] == 0.
    assert loss_parts(p,[False,False])['correct_loss'] == 0.
    with pytest.raises(ValueError): loss_parts(p,[True])


def test_absent_rule_is_not_zero_precision_observation():
    assert active_precision_mean([5.,0.,10.],[1,0,1]) == 7.5
    assert np.isnan(active_precision_mean([0.,0.],[0,0]))
    with pytest.raises(ValueError): active_precision_mean([5.,0.],[1])

"""Scientific channel identities for graded feedback at the retention boundary."""
import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis.model_0923_pairing_retention_audit import (
    PRECISIONS, feedback_state, probabilities,
)


@pytest.mark.parametrize('choice', [1, 2, 3, 4])
def test_zero_retention_removes_family_information_from_rule_and_precision_updates(choice):
    beta=PRECISIONS[1]
    partial=feedback_state(0.,True,beta,choice,.5)
    wrong=feedback_state(0.,True,beta,choice,0.)
    other=probabilities(beta)[:,np.arange(4)!=choice-1].sum(axis=1)
    np.testing.assert_allclose(partial['h_increment'],np.log(other/.75),atol=1e-14)
    for key in ('h_increment','h_evidence','rule_posterior','gradient'):
        np.testing.assert_allclose(partial[key],wrong[key],atol=1e-14)
    np.testing.assert_allclose(partial['rule_posterior'],other/other.sum(),atol=1e-14)
    assert np.max(abs(partial['pairing_posterior']-wrong['pairing_posterior']))>.1
    assert partial['failure_pressure']==pytest.approx(.32)
    assert wrong['failure_pressure']==pytest.approx(.52)
    assert partial['accumulated_failure']==pytest.approx(.9)
    assert wrong['accumulated_failure']==pytest.approx(1.4)
    assert wrong['search_probability']>partial['search_probability']


@pytest.mark.parametrize('gamma', [.8, 1.])
def test_retained_informed_pairings_restore_feedback_distinctions(gamma):
    a=feedback_state(gamma,True,PRECISIONS[1],2,.5)
    b=feedback_state(gamma,True,PRECISIONS[1],2,0.)
    for key in ('h_increment','rule_posterior','gradient'):
        assert np.max(abs(a[key]-b[key]))>1e-3


def test_uniform_pairing_still_collapses_relative_feedback_with_positive_retention():
    a=feedback_state(.8,False,PRECISIONS[1],2,.5)
    b=feedback_state(.8,False,PRECISIONS[1],2,0.)
    for key in ('h_increment','rule_posterior','gradient'):
        np.testing.assert_allclose(a[key],b[key],atol=1e-14)

"""Target-emission diagnostics preserve fixed mixture weights and error mass."""
import numpy as np
import pytest

from src.Bayesian_state.evaluation.model_0923_target_emission import emission_contrasts


def example():
    return [np.array([row],float) for row in ([1,0],[.4,0],[.6,.7],[.5,np.nan],
        [.7,np.nan],[.8,.2],[.9,.9])]


def test_counterfactual_changes_only_existing_target_contribution():
    result=emission_contrasts(*example())
    np.testing.assert_allclose(result['accuracy_clean'],[[.68,.7]])
    np.testing.assert_allclose(result['accuracy_upper'],[[.72,.7]])
    np.testing.assert_allclose(result['accuracy_clean_upper'],[[.76,.7]])
    np.testing.assert_allclose(result['perception_share'],[[.06,0]])
    np.testing.assert_allclose(result['precision_share'],[[.10,0]])
    np.testing.assert_allclose(result['target_error_remaining'],[[.04,0]])


def test_negative_perception_effect_is_retained_instead_of_forced_into_positive_error():
    args=example();args[3][0,0]=.8;args[4][0,0]=.5;args[5][0,0]=.9;args[6][0,0]=.7
    result=emission_contrasts(*args)
    assert result['perception_share'][0,0]<0
    np.testing.assert_allclose(result['perception_share']+result['precision_share']+result['target_error_remaining'],
                               result['target_error_original'])


@pytest.mark.parametrize('problem',['absent_weight','shape','nan','negative','above_one'])
def test_invalid_saved_mixture_or_target_probabilities_fail(problem):
    args=example()
    if problem=='absent_weight':args[1][0,1]=.1
    elif problem=='shape':args[4]=args[4][:,:1]
    elif problem=='nan':args[4][0,0]=np.nan
    elif problem=='negative':args[4][0,0]=-.1
    elif problem=='above_one':args[6][0,0]=1.1
    with pytest.raises(ValueError):emission_contrasts(*args)

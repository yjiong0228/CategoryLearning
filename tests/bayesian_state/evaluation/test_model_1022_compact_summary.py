"""Hand-computable summary contrasts retain shared simulation errors."""
import math
import numpy as np

from src.Bayesian_state.evaluation.model_1022_compact_refit import paired_contrast


def test_paired_difference_and_jackknife_have_the_declared_sign_and_scale():
    points = {'COMPACT':dict(choice_nll=math.log(2),energy=.4),
              'TRANSFER':dict(choice_nll=-math.log(.75),energy=.2)}
    jacks = {'COMPACT':dict(choice_nll=np.array([math.log(2)]*2),energy=np.array([.3,.5])),
             'TRANSFER':dict(choice_nll=np.array([-math.log(.75)]*2),energy=np.array([.2,.2]))}
    result = paired_contrast(points,jacks,('choice_nll','energy'))
    np.testing.assert_allclose(result['delta_choice_nll'],math.log(2/3),atol=1e-14)
    assert result['delta_choice_nll_mc_se']==0
    np.testing.assert_allclose(result['delta_energy'],-.2,atol=1e-14)
    np.testing.assert_allclose(result['delta_energy_mc_se'],.1,atol=1e-14)

"""Profile bank preserves nuisance parameters and explicit grid membership."""
from src.Bayesian_state.workflows.analysis.model_0923_h_profiles import profile_points


def test_profile_cross_grid_shares_points_and_preserves_fixed_coordinate():
    anchor={'a':1,'b':2,'fixed':7};space={'a':[0,1,2],'b':[1,2],'fixed':[6,7,8]}
    bank,members=profile_points(anchor,space,['a','b'],[['a','b']])
    assert len(bank)==6 and len(members)==11
    assert all(c['hyperparams']['fixed']==7 for c in bank.values())
    assert {(c['hyperparams']['a'],c['hyperparams']['b']) for c in bank.values()}=={(a,b) for a in [0,1,2] for b in [1,2]}
    assert all(m['candidate'] in bank for m in members)


def test_profile_numerical_error_uses_complete_seeds(tmp_path):
    import numpy as np
    from src.Bayesian_state.evaluation.model_0923_h_validation import jackknife_scores
    from src.Bayesian_state.workflows.analysis.model_0923_h_profiles import numerical_scores
    from src.Bayesian_state.workflows.analysis.model_0923_h_validation import digest,write
    p=np.array([[.1,.9],[.5,.5],[.8,.2]])
    np.savez(tmp_path/'point.npz',observed_probability=p,seeds=[11,12,13])
    write(tmp_path/'point.json',dict(predictions_sha256=digest(tmp_path/'point.npz')))
    seeds,values=numerical_scores(tmp_path/'point.json')
    np.testing.assert_array_equal(seeds,[11,12,13])
    np.testing.assert_allclose(values,jackknife_scores(p))

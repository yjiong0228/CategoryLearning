"""Independent small distributions expose spread, dependence and scoring timing."""
import numpy as np
import pytest

from src.Bayesian_state.evaluation.model_0923_h_validation import block_paths,distribution_scores,jackknife_scores,jackknife_se


def test_point_mass_scores_equal_absolute_error_and_penalize_missing_intervals():
    score=distribution_scores(np.array([[.2],[.2]]),np.array([.8]))
    assert score['energy']==pytest.approx(.6)
    assert score['crps']==pytest.approx(.6)
    assert score['interval_coverage']==0 and score['interval_width']==0
    assert score['interval_score']==pytest.approx(6.)


def test_narrow_wrong_forecast_can_score_worse_than_wide_forecast():
    narrow=distribution_scores(np.array([[0.],[0.]]),np.array([.5]))
    wide=distribution_scores(np.array([[0.],[1.]]),np.array([.5]))
    assert wide['crps']==pytest.approx(.25)
    assert wide['crps']<narrow['crps']
    assert wide['interval_score']<narrow['interval_score']


def test_energy_distinguishes_joint_paths_with_identical_block_marginals():
    a=distribution_scores(np.array([[0.,0.],[1.,1.]]),np.array([0.,0.]))
    b=distribution_scores(np.array([[0.,1.],[1.,0.]]),np.array([0.,0.]))
    assert a['crps']==pytest.approx(b['crps'])
    assert a['energy']<b['energy']


def test_blocking_keeps_suffix_boundary_and_final_partial_block():
    paths=np.arange(10.)[None,:]
    np.testing.assert_array_equal(block_paths(paths,3,10,4),[[4.5,8.]])


def test_seed_jackknife_keeps_all_trials_and_pools_before_log():
    p=np.array([[.1,.9],[.5,.5],[.9,.1]])
    values=jackknife_scores(p)
    np.testing.assert_allclose(values,[-np.log([.7,.3]).mean(),np.log(2),-np.log([.3,.7]).mean()])
    assert jackknife_se(values)>0


@pytest.mark.parametrize('samples,observed',[(np.array([1.]),np.array([1.])),(np.ones((2,2)),np.ones(3)),
                                            (np.array([[np.nan]]),np.array([0.]))])
def test_invalid_distribution_rejected(samples,observed):
    with pytest.raises(ValueError):distribution_scores(samples,observed)


def test_participant_and_trial_weighted_scores_can_have_opposite_directions():
    import pandas as pd
    from src.Bayesian_state.evaluation.model_0923_h_validation import aggregate_choices
    test=pd.DataFrame([dict(subject=s,model=m,trials=n,choice_nll=p)
        for s,n,b,e in [(101,100,.4,.3),(201,1000,.8,.82)] for m,p in [('B',b),('E',e)]])
    contrasts=pd.DataFrame([dict(subject=101,model='E',choice_nll_mc_se=.01),
                            dict(subject=201,model='E',choice_nll_mc_se=.02)])
    row=aggregate_choices(test,contrasts,['B','E']).set_index('model').loc['E']
    assert row.delta_participant_mean_nll==pytest.approx(-.04)
    assert row.delta_total_negative_log_score==pytest.approx(10.)
    assert row.delta_total_mc_se==pytest.approx(np.sqrt(401))
    assert row.participants_lower_nll==1


def test_reused_seed_halves_keep_declared_development_labels(tmp_path):
    import json
    import pandas as pd
    from src.Bayesian_state.evaluation.model_0923_h_validation import score_screen
    source=tmp_path/'source';case=source/'case';(case/'B').mkdir(parents=True)
    pd.DataFrame(dict(choice=[1]*32,category=[1]*32)).to_csv(case/'observed_trials.csv',index=False)
    np.savez(case/'B/correct_paths.npz',realized=np.ones((32,64)),correct_probability=np.ones((32,64)))
    (source/'simulation_manifest.json').write_text(json.dumps(dict(status='complete',smoke=False,
        records=[dict(id='case',subject=101)],settings=dict(variants=dict(B={}),
        seed_group_labels=['development_first32','development_second32']))))
    score_screen(source,tmp_path/'scores')
    scores=pd.read_csv(tmp_path/'scores/scores.csv')
    assert set(scores.seed_group)=={'all','development_first32','development_second32'}

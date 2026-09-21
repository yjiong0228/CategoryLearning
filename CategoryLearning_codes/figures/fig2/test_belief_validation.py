"""Scientific guards for report ambiguity, timing and interval comparisons."""
import numpy as np
import pandas as pd
import pytest

from CategoryLearning_codes.figures.fig2.build_belief_validation import compatible, matched_change, project, report_pairs


def test_partial_report_does_not_require_a_unique_full_rule():
    oral=np.array([[.5,.5,0],[.5,.5,0],[.5,.5,0]])
    belief=np.array([[1,0,0],[0,1,0],[0,0,1]])
    np.testing.assert_allclose(compatible(belief,oral),[1,1,0])
    np.testing.assert_allclose(compatible(np.ones((1,3))/3,oral[:1]),[2/3])


def test_compatibility_retains_soft_report_likelihood_and_pf_linearity():
    oral=np.array([[.6,.3,.1]])
    a=np.array([[.7,.1,.2]]);b=np.array([[.1,.7,.2]])
    np.testing.assert_allclose(compatible(a,oral),[.7+.1*.5+.2/6])
    np.testing.assert_allclose(compatible((a+b)/2,oral),(compatible(a,oral)+compatible(b,oral))/2)


@pytest.mark.parametrize('bad',[np.array([[np.nan,.5]]),np.array([[.3,.3]]),np.array([[-.1,1.1]])])
def test_compatibility_rejects_invalid_distributions(bad):
    with pytest.raises((ValueError,AssertionError)):compatible(bad,np.array([[.5,.5]]))


def test_equivalence_projection_ignores_unobservable_rule_swaps():
    q=np.array([[1,0,0],[0,1,0],[0,0,1]])
    np.testing.assert_array_equal(project(q,np.array([0,0,1])),[[1,0],[1,0],[0,1]])


def test_report_pairs_respect_category_session_and_missing_reports():
    frame=pd.DataFrame({'iSub':[1]*8,'task':[1]*8,'iSession':[1]*6+[2]*2,
                        'choice':[1,2,1,1,2,1,1,1]})
    q=np.array([[.9,.1]]*8);oral=q.copy();valid=np.ones(8,bool);valid[3]=False
    result=report_pairs(frame,q,oral,valid,{1:np.arange(2),2:np.arange(2)})
    # t4's invalid category-1 report breaks its chain; t7 starts a new session.
    assert list(zip(result.previous_trial,result.trial))==[(1,3),(2,5),(7,8)]
    assert result.gap.tolist()==[2,3,1]


def test_report_pairs_reject_long_gaps_and_project_by_current_category():
    frame=pd.DataFrame({'iSub':[1]*35,'task':[1]*35,'iSession':[1]*35,'choice':[1]+[2]*33+[1]})
    q=np.tile([1.,0.,0.],(35,1));q[1]=[0,1,0]
    pairs=report_pairs(frame,q,q,np.ones(35,bool),{1:np.arange(3),2:np.array([0,0,1])})
    assert 35 not in pairs.trial.values
    assert pairs.oral_tv.eq(0).all()
    assert pairs.belief_tv_per_trial.eq(0).all()


def test_exact_gap_matching_avoids_comparing_long_changes_with_short_stability():
    pairs=pd.DataFrame({'gap':[1,1,1,2,2,10], 'oral_tv':[0,0,1,0,1,1],
                        'belief_tv_per_trial':[.1,.3,.5,.4,.5,99.]})
    result=matched_change(pairs)
    assert result['matched_weight']==2
    assert result['stable_change']==pytest.approx(.3)
    assert result['report_change']==pytest.approx(.5)
    assert result['change_difference']==pytest.approx(.2)
    assert np.isnan(matched_change(pairs.iloc[[-1]])['change_difference'])

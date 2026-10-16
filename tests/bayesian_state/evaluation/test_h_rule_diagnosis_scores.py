"""Episode censoring and absent-rule contribution in diagnostic summaries."""
import numpy as np

from src.Bayesian_state.evaluation.model_0923_h_rule_diagnosis import episodes, target_values


def test_episodes_separate_exit_capacity_and_censoring():
    rule=np.array([[[2,-1],[2,1],[1,-1],[2,1],[1,-1],[2,1]]])
    exits=np.full_like(rule,-1); exits[0,2,0]=2
    drops=np.full((1,6),-1); drops[0,4]=2
    a=dict(rule=rule,oracle=np.array(2),seeds=np.array([17]),exits=exits,capacity_dropped=drops)
    rows=episodes(a,207,'FREE','BASE')
    assert [r['duration'] for r in rows]==[2,1,1]
    assert [r['removal'] for r in rows]==['autonomous_exit','capacity','right_censored']
    assert [r['start_trial'] for r in rows]==[1,4,6]
    assert [r['right_censored'] for r in rows]==[False,False,True]


def test_absence_is_zero_contribution_and_slots_do_not_define_identity():
    a=dict(rule=np.array([[[2,1],[1,2],[1,-1]]]),oracle=np.array(2),
        contribution=np.array([[[.7,.3],[.2,.8],[1.,np.nan]]]))
    np.testing.assert_array_equal(target_values(a,'contribution'),[[.7,.8,0.]])

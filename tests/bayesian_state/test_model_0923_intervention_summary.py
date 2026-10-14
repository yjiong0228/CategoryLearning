"""Counterevidence clearance needs complete follow-up and rule identity."""
import numpy as np
from src.Bayesian_state.evaluation.model_0923_interventions import wrong_rule_clearance


def test_clearance_includes_eighth_transition_and_tracks_censoring():
    ids = np.full((12,1,2),-1)
    ids[:10,0,0] = 1
    ids[10:,0,0] = 0
    ids[5:,0,1] = 2
    evidence = np.ones(ids.shape)
    evidence[2,0,0] = -1
    evidence[5,0,1] = -1
    rows = wrong_rule_clearance({'rule':ids,'evidence_score':evidence},[0],8)
    assert len(rows)==2
    first,second = rows
    assert first['first_negative_trial']==3 and first['duration']==8
    assert first['clearance_eligible'] and first['cleared_within_horizon']
    assert not first['right_censored']
    assert second['right_censored'] and not second['clearance_eligible']
    assert not second['cleared_within_horizon']


def test_reentry_opens_a_new_counterevidence_episode():
    ids = np.array([1,1,0,1,1,0])[:,None,None]
    evidence = -np.ones(ids.shape)
    rows = wrong_rule_clearance({'rule':ids,'evidence_score':evidence},[0],1)
    assert [r['first_negative_trial'] for r in rows]==[1,4]
    assert [r['duration'] for r in rows]==[2,2]
    assert not any(r['cleared_within_horizon'] for r in rows)


def test_residence_sensitivity_uses_one_based_event_timing_and_records_censoring():
    from src.Bayesian_state.evaluation.model_0923_retention_diagnostics import retention_diagnostics
    ids=np.array([1,1,1,0,1,1])[:,None,None]
    a=dict(rule=ids,prediction=np.ones((6,1,2))*.5,feedback=np.ones((6,1)),
           evidence=np.ones(ids.shape)*.25,contribution=np.ones(ids.shape),
           exits=np.full(ids.shape,-1),capacity_dropped=np.full((6,1),-1))
    a['exits'][3,0,0]=1
    a['capacity_dropped'][4,0]=0
    paths,episodes,summary=retention_diagnostics(a,[0],1,6,[1,4,8],[1,3,4])
    assert len(episodes)==2
    assert episodes[0]['duration']==3 and episodes[0]['cause']=='autonomous_exit'
    assert episodes[1]['right_censored'] and episodes[1]['start_trial']==5
    assert paths[0]['total_autonomous_exits']==1
    assert paths[0]['total_capacity_replacements']==1
    assert paths[0]['incompatible_residences_eligible_3']==1
    assert paths[0]['incompatible_residences_retained_3']==1
    for row in summary:
        assert row['eligible']==(2 if row['horizon']==1 else 1)
        assert row['cleared']==(0 if row['horizon']==1 else 1)

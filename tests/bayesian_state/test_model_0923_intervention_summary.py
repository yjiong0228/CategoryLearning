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

"""Admission reconstruction uses evidence available before the transition."""
import numpy as np
import pytest

from src.Bayesian_state.evaluation.model_0923_h_capacity_competition import proposal_events


def test_supported_incumbent_can_block_a_compatible_new_proposal():
    a = dict(rule=np.array([[[1]],[[1]]]), proposed=np.array([[-1],[2]]),
        retention_score_post=np.array([[[np.log(9.)]],[[0.]]]),
        retention_mean_post=np.array([[[np.log(9.)]],[[0.]]]),
        admitted=np.array([[False],[False]]),rejected=np.array([[False],[True]]),seeds=np.array([7]))
    row = proposal_events(a,dict(stay_bias=0.),[2])[0]
    assert row['admission_probability']==pytest.approx(.1)
    assert row['compatible_proposal'] and not row['admitted']
    assert row['trial']==2 and row['capacity_full']


def test_spare_capacity_admits_candidate_despite_strong_incumbent():
    a = dict(rule=np.array([[[1,-1]],[[1,2]]]),proposed=np.array([[-1],[2]]),
        retention_score_post=np.array([[[100.,np.nan]],[[0.,0.]]]),
        retention_mean_post=np.array([[[10.,np.nan]],[[0.,0.]]]),
        admitted=np.array([[False],[True]]),rejected=np.array([[False],[False]]),seeds=np.array([7]))
    row = proposal_events(a,dict(stay_bias=3.,score='mean'),[2])[0]
    assert row['admission_probability']==1 and row['admitted']
    assert not row['capacity_full'] and row['old_score_min']==13

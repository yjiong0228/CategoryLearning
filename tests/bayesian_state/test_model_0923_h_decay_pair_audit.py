"""Two-value recovery separates likelihood direction, numerical error and ties."""
import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis.model_0923_h_decay_pair_audit import paired_contrast


def test_identical_predictions_do_not_count_as_recovery():
    common=dict(subject=101,true_retention=.6,trials=3,mean_nll=np.log(2))
    arrays=[(np.array([1,2]),np.full((2,3),.5))]*2
    row=paired_contrast('case',dict(common,candidate_retention=.6),dict(common,candidate_retention=.95),arrays)
    assert row['exact_tie'] and not row['correct']
    assert row['mc_se']==0 and not row['difference_exceeds_two_mc_se']


def test_lower_high_retention_loss_has_negative_contrast():
    common=dict(subject=101,true_retention=.95,trials=3)
    arrays=[(np.array([1,2]),np.full((2,3),p)) for p in [.5,.8]]
    row=paired_contrast('case',dict(common,candidate_retention=.6,mean_nll=-np.log(.5)),
        dict(common,candidate_retention=.95,mean_nll=-np.log(.8)),arrays)
    assert row['difference_high_minus_low']==pytest.approx(np.log(.5/.8))
    assert row['selected_retention']==.95 and row['correct'] and not row['exact_tie']
    assert row['difference_exceeds_two_mc_se']


def test_misaligned_filter_seeds_are_rejected():
    common=dict(subject=101,true_retention=.6,trials=3,mean_nll=np.log(2))
    with pytest.raises(AssertionError):
        paired_contrast('case',dict(common,candidate_retention=.6),dict(common,candidate_retention=.95),
            [(np.array([1,2]),np.full((2,3),.5)),(np.array([2,1]),np.full((2,3),.5))])

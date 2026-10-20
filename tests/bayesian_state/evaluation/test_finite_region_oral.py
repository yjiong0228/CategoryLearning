import numpy as np
import pandas as pd
import pytest

from src.Bayesian_state.workflows.analysis.model_0923_finite_region_oral import (
    paired_contrasts, score_bank,
)


def example(shift=0.):
    frame = pd.DataFrame(dict(state_trial=[0, 1, 2, 3], feedback=[1., 0., 0., 1.],
                              scope=['complete', 'complete', 'parsed_fragment', 'needs_review']))
    probability = np.array([[.1, .2, .4, .5], [.9, .8, .6, .5]])
    support = np.array([[.1, .2, .3, np.nan], [.9, .6, .7, np.nan]])+shift
    conditional = np.stack([support, np.zeros_like(support), 1-support], axis=-1)
    conditional[:, -1] = np.nan
    joint = np.repeat(conditional[:, :, None, :], 2, axis=2)*probability[:, :, None, None]
    before = np.array([[.2, .2, .2, np.nan], [.4, .4, .4, np.nan]])
    prior = np.stack([before, np.zeros_like(before), 1-before], axis=-1)
    prior[:, -1] = np.nan
    prior = np.repeat(prior[:, :, None, :], 2, axis=2)
    return frame, prior, joint, probability, np.array([11, 29])


def test_joint_pooling_and_whole_seed_uncertainty_preserve_report_boundaries():
    frame, prior, joint, probability, seeds = example()
    trial, rows, _ = score_bank(frame, prior, joint, probability, seeds, 2, 'relative', 208, 'FROZEN')
    # The two seeds assign unequal probabilities to the observed response.
    # Conditioning each seed first would incorrectly yield .5 for trial zero.
    assert trial.conditional_functional_support.iloc[0] == pytest.approx(.82)
    assert trial.prechoice_functional_support.iloc[0] == pytest.approx(.3)
    assert np.isnan(trial.conditional_functional_support.iloc[3])
    summary = pd.DataFrame(rows)
    selected = summary.query('window == "training" and scope == "complete" and domain == "functional"')
    assert selected.set_index('feedback_stratum').reports.to_dict() == dict(all=2, correct=1, incorrect=1)
    all_row = selected[selected.feedback_stratum.eq('all')].iloc[0]
    assert all_row.conditional_support == pytest.approx(.67)
    assert all_row.conditional_support_mcse == pytest.approx(.3)
    later = summary.query('window == "test" and feedback_stratum == "all" and domain == "functional"')
    assert len(later) == 1 and later.iloc[0].scope == 'parsed_fragment' and later.iloc[0].reports == 1


def test_paired_contrast_uses_matching_seed_deletions():
    rows, deleted = [], {}
    for arm, shift in [('FROZEN', 0.), ('REHEARSE', .05)]:
        _, summary, jackknife = score_bank(*example(shift), 2, 'midpoint', 207, arm)
        rows.extend(summary); deleted.update({(arm, *k): v for k, v in jackknife.items()})
    contrast = paired_contrasts(pd.DataFrame(rows), deleted)
    np.testing.assert_allclose(contrast.delta_conditional_support, .05, atol=1e-15, rtol=0)
    np.testing.assert_allclose(contrast.delta_conditional_support_mcse, 0., atol=1e-15, rtol=0)
    np.testing.assert_allclose(contrast.delta_prechoice_support, 0., atol=1e-15, rtol=0)


@pytest.mark.parametrize('change', ['seed', 'identity', 'coverage'])
def test_incomplete_or_misaligned_banks_fail(change):
    frame, prior, joint, probability, seeds = example()
    if change == 'seed': seeds[1] = seeds[0]
    elif change == 'identity': frame.loc[1, 'state_trial'] = 9
    else: frame.loc[3, 'scope'] = 'complete'
    with pytest.raises((ValueError, AssertionError)):
        score_bank(frame, prior, joint, probability, seeds, 2, 'relative', 208, 'FROZEN')

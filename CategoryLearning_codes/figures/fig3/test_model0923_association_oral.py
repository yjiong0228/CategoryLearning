"""Protect report denominators, missing blocks and cross-arm identities."""
import numpy as np
import pandas as pd
import pytest

from CategoryLearning_codes.figures.fig3.model0923_association_oral import ARMS, time_blocks


def example():
    rows = []
    for arm in ARMS:
        for t in range(1, 7):
            rows.append(dict(iSub=207, arm_name=arm, ordinal_trial=t, choice=1,
                scope='complete' if t in (1, 5, 6) else 'needs_review', tokens='[]', training_trials=4,
                structural_support=.3 if t==1 else .6, structural_conflict=.1,
                structural_undetermined=.6 if t==1 else .3,
                functional_support=.4, functional_conflict=.1, functional_undetermined=.5))
    return pd.DataFrame(rows)


def test_empty_blocks_and_complete_denominators_remain_visible():
    result = time_blocks(example(), width=2)
    for _, group in result.groupby('arm_name'):
        np.testing.assert_array_equal(group.reports, [1, 0, 2])
        np.testing.assert_array_equal(group.unresolved, [1, 2, 0])
        np.testing.assert_allclose(group.structural_support, [.3, np.nan, .6], equal_nan=True)
        assert group.trials.sum() == 6


def test_masks_must_agree_across_arms():
    frame = example()
    frame.loc[frame.arm_name.eq('LEARN_SHARED') & frame.ordinal_trial.eq(1), 'scope'] = 'needs_review'
    with pytest.raises(AssertionError):
        time_blocks(frame, width=2)


def test_blocking_must_preserve_training_boundary():
    with pytest.raises(ValueError, match='training boundary'):
        time_blocks(example(), width=3)

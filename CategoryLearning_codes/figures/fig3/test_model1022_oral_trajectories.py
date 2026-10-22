"""Protect temporal alignment and complete-report denominators."""
import numpy as np
import pandas as pd
import pytest

from CategoryLearning_codes.figures.fig3.model1022_oral_trajectories import (
    check_summary, time_blocks,
)


def example():
    rows = []
    for trial in range(1, 7):
        complete = trial in (1, 5, 6)
        support = {1: .2, 5: .7, 6: .9}.get(trial, .99)
        row = dict(iSub=205, arm='TRANSFER', ordinal_trial=trial, training_trials=4,
                   scope='complete' if complete else 'parsed_fragment',
                   phase='training' if trial <= 4 else 'test')
        for domain in ('functional', 'structural'):
            row.update({f'{domain}_support': support,
                        f'{domain}_conflict': 0., f'{domain}_undetermined': 1 - support,
                        f'{domain}_support_any_component': 1})
        rows.append(row)
    return pd.DataFrame(rows)


def test_missing_blocks_stay_undefined_and_fragments_do_not_enter_means():
    blocks = time_blocks(example(), width=2)
    np.testing.assert_array_equal(blocks.trial, [2, 4, 6])
    np.testing.assert_array_equal(blocks.reports, [1, 0, 2])
    np.testing.assert_array_equal(blocks.unresolved, [1, 2, 0])
    np.testing.assert_allclose(blocks.functional_support, [.2, np.nan, .8], equal_nan=True)
    np.testing.assert_allclose(blocks.functional_coverage, [1., np.nan, 1.], equal_nan=True)
    assert blocks.trials.sum() == 6


def test_summary_reconstruction_weights_unequal_blocks_by_report_count():
    frame = example()
    rows = []
    for domain in ('functional', 'structural'):
        for window, trials, reports, mean in [('whole', 6, 3, .6), ('training', 4, 1, .2), ('test', 2, 2, .8)]:
            rows.append(dict(subject=205, arm='TRANSFER', subset='complete', domain=domain,
                             window=window, window_trials=trials, reports=reports,
                             support=mean, conflict=0., undetermined=1 - mean))
    assert check_summary(frame, time_blocks(frame, 2), pd.DataFrame(rows)) == 18


def test_blocks_cannot_cross_the_parameter_training_cut():
    with pytest.raises(ValueError, match='training boundary'):
        time_blocks(example(), width=3)


def test_duplicate_trial_cannot_shift_the_time_axis():
    frame = example()
    frame.loc[1, 'ordinal_trial'] = 1
    with pytest.raises(ValueError, match='unique and consecutive'):
        time_blocks(frame, width=2)


def test_complete_report_masses_must_sum_to_one():
    frame = example()
    frame.loc[0, 'functional_conflict'] = .3
    with pytest.raises(AssertionError):
        time_blocks(frame, width=2)

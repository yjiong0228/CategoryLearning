"""Tests of event timing and the feedback comparison's temporal boundaries."""
import numpy as np
import pandas as pd

from CategoryLearning_codes.figures.fig3.nine_subject_analysis import (
    first_sustained, feedback_table, block_table,
)


def test_sustained_event_is_one_based_strict_and_not_first_discovery():
    values = np.array([.8, .1, .5, .6, .7, .8, .1])
    assert first_sustained(values, .5, 3) == 4
    assert np.isnan(first_sustained(values, .5, 4))
    assert np.isnan(first_sustained(np.array([np.nan, .8, np.nan]), .5, 2))


def test_feedback_uses_previous_trial_and_does_not_bridge_sessions():
    frame = pd.DataFrame({'iSession': [1,1,1,2,2], 'feedback': [1,0,1,0,1],
        'belief': [.8,.7,.4,.9,.6], 'search': [.99,.1,.6,.99,.4],
        'global_range': [.9,.2,.3,.9,.5], 'replacement': [.9,.01,.02,.9,.03]})
    table = feedback_table(frame).set_index('feedback')
    assert table.loc[1,'n'] == 1
    assert table.loc[0,'n'] == 2
    np.testing.assert_allclose(table.loc[1,'search'], .1)
    np.testing.assert_allclose(table.loc[0,'search'], .5)
    np.testing.assert_allclose(table.loc[0,'strong_belief_change'], -.3)
    np.testing.assert_allclose(table.loc[0,'strong_retention'], .5)


def test_all_trials_and_missing_execution_survive_block_summary():
    frame = pd.DataFrame({'trial': np.arange(1,71), 'correct': np.ones(70),
        **{k: np.ones(70)*.4 for k in ['predicted','available','belief','search','replacement','global_range']},
        'executed': np.full(70,np.nan), 'pairing': np.full(70,np.nan)})
    blocks = block_table(frame)
    assert blocks.n.tolist() == [64,6]
    assert blocks.n.sum() == 70
    assert blocks.executed.isna().all()

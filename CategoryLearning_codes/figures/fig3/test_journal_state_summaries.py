"""Protect state episode and feedback timing boundaries in the journal figures."""
import numpy as np
import pandas as pd

from CategoryLearning_codes.figures.fig3.build_journal_figure import support_episodes
from CategoryLearning_codes.figures.fig4.build_journal_figure import case_window_accuracy, feedback_search


def test_sustained_support_does_not_join_sessions():
    trials = pd.DataFrame({'iSub': [1]*32, 'iSession': [1]*15+[2]*17,
                           'trial': np.arange(1, 33), 'belief': [.7]*32})
    episodes = support_episodes(trials)
    assert episodes[['session', 'start', 'stop', 'length']].to_dict('records') == [
        {'session': 2, 'start': 16, 'stop': 32, 'length': 17}]


def test_sustained_support_breaks_at_threshold_and_missing_trials():
    trials = pd.DataFrame({'iSub': [1]*8, 'iSession': [1]*8,
                           'trial': [1, 2, 3, 4, 5, 7, 8, 9],
                           'belief': [.7, .7, .5, .7, .7, .7, .7, .7]})
    episodes = support_episodes(trials, minimum=3)
    assert episodes[['start', 'stop']].values.tolist() == [[7, 9]]


def test_feedback_pairs_use_next_search_and_never_cross_session_or_gap():
    trials = pd.DataFrame({'iSub': [1]*6, 'iSession': [1, 1, 2, 2, 2, 2],
                           'trial': [1, 2, 3, 4, 6, 7], 'task': [2]*6,
                           'feedback': [0., 1., .5, 0., 1., 0.],
                           'search': [.9, .2, .8, .4, .7, .6]})
    result = feedback_search(trials).set_index('feedback')
    assert result.n_pairs.to_dict() == {1.: 1, .5: 1, 0.: 1}
    assert result.mean_next_search.to_dict() == {1.: .6, .5: .4, 0.: .2}


def test_missing_feedback_level_has_zero_pairs_and_no_fabricated_value():
    trials = pd.DataFrame({'iSub': [1, 1], 'iSession': [1, 1],
                           'trial': [1, 2], 'task': [1, 1],
                           'feedback': [1., 0.], 'search': [.1, .3]})
    result = feedback_search(trials).set_index('feedback')
    assert result.loc[.5, 'n_pairs'] == 0
    assert np.isnan(result.loc[.5, 'mean_next_search'])


def test_behavioral_window_split_is_last_before_trial():
    trials = pd.DataFrame({'iSub': [1]*8, 'trial': np.arange(1, 9),
                           'correct': [1, 1, 0, 0, 1, 1, 0, 0]})
    assert case_window_accuracy(trials, subject=1, split=4, window=2) == (0., 1.)

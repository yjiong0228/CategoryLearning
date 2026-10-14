"""Training-profile selection preserves frozen parameters in exact ties."""
import pandas as pd
import pytest

from src.Bayesian_state.workflows.analysis.model_0923_h_profile_audit import select_alternative


def test_profile_membership_duplicates_do_not_change_selection():
    frame = pd.DataFrame(dict(candidate=['a', 'b', 'b', 'c'], mean_nll=[2., 1., 1., 3.]))
    assert select_alternative(frame, 'a') == 'b'
    assert select_alternative(frame.iloc[::-1], 'a') == 'b'


def test_frozen_anchor_wins_exact_training_tie():
    assert select_alternative(pd.DataFrame(dict(candidate=['a', 'z'], mean_nll=[1., 1.])), 'z') == 'z'


def test_missing_anchor_and_inconsistent_scores_are_rejected():
    with pytest.raises(ValueError):
        select_alternative(pd.DataFrame(dict(candidate=['a'], mean_nll=[1.])), 'z')
    with pytest.raises(ValueError):
        select_alternative(pd.DataFrame(dict(candidate=['a', 'a'], mean_nll=[1., 2.])), 'a')

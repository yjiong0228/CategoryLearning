"""Scientific invariants for the descriptive figure analysis."""
import numpy as np
import pandas as pd

from CategoryLearning_codes.figures.fig1 import behavior


def test_trial_order_survives_session_reset_and_partial_credit_is_not_correct():
    frame = pd.DataFrame({"condition": [3]*3, "iSub": [301]*3,
                          "iSession": [2, 1, 1], "iBlock": [1]*3,
                          "iTrial": [1, 2, 1], "feedback": [1, .5, 0]})
    actual = behavior.order_trials(frame)
    assert actual.iSession.tolist() == [1, 1, 2]
    assert actual.correct.tolist() == [0, 0, 1]
    assert actual.trial.tolist() == [1, 2, 3]


def test_adjacent_gain_uses_disjoint_windows_without_endpoint_padding():
    gain, boundary = behavior.adjacent_gain(np.r_[np.zeros(32), np.ones(32)], 32)
    assert gain == 1
    assert boundary == 32
    assert np.isnan(behavior.adjacent_gain(np.ones(63), 32)[0])
    assert behavior.adjacent_gain(np.ones(64), 32)[0] == 0


def test_explicit_mentions_do_not_turn_unspecified_other_parts_into_observations():
    assert behavior.explicit_features("头和脖子长，尾巴短") == frozenset({"head", "neck", "tail"})
    assert behavior.explicit_features("其他部分一样") == frozenset()
    assert behavior.explicit_features(np.nan) == frozenset()


def test_missing_report_breaks_comparison_and_new_category_does_not_fake_a_switch():
    frame = pd.DataFrame({"iSession": [1]*5, "choice": [1, 2, 1, 1, 1],
                          "trial": [1, 2, 3, 4, 5],
                          "text": ["头长", "尾巴短", "头短", None, "尾巴长"]})
    out = behavior.report_features(frame, 32)
    assert np.isnan(out.report_set_change.iloc[1])
    assert out.report_set_change.iloc[2] == 0
    assert np.isnan(out.report_set_change.iloc[4])


def test_gain_does_not_cross_subjects_and_correctness_does_not_use_labels():
    frame = pd.DataFrame({"condition": [1, 1], "iSub": [101, 102],
                          "iSession": [1, 1], "iBlock": [1, 1], "iTrial": [1, 1],
                          "feedback": [1, 0], "choice": [1, 1], "category": [2, 1]})
    out = behavior.order_trials(frame)
    assert out.trial.tolist() == [1, 1]
    assert out.correct.tolist() == [1, 0]


def test_two_category_task_coarsens_four_leaf_stimulus_labels():
    assert behavior.task_categories(np.array([1, 2, 3, 4]), 1).tolist() == [1, 1, 2, 2]
    assert behavior.task_categories(np.array([1, 2, 3, 4]), 3).tolist() == [1, 2, 3, 4]


def test_required_features_follow_stimulus_branch_and_individual_anatomy():
    frame = pd.DataFrame({"condition": [3, 3], "feature1": [.2, .8],
                          "feature1_name": ["tail", "tail"], "feature2_name": ["head", "head"],
                          "feature3_name": ["neck", "neck"], "feature4_name": ["leg", "leg"],
                          "feature_set": ["head|tail", "head|tail"], "report_recognized": [True, True]})
    result = behavior.task_relative_reports(frame)
    assert result.path_coverage.tolist() == [1, .5]
    assert result.irrelevant_fraction.tolist() == [0, 0]
    assert result.mention_head.tolist() == [1, 1]


def test_unrecognized_text_is_missing_and_irrelevance_is_task_relative():
    frame = pd.DataFrame({"condition": [1, 1], "feature1": [.2, .2],
                          "feature1_name": ["tail"]*2, "feature2_name": ["head"]*2,
                          "feature3_name": ["neck"]*2, "feature4_name": ["leg"]*2,
                          "feature_set": ["head|tail", ""], "report_recognized": [True, False]})
    result = behavior.task_relative_reports(frame)
    assert result.irrelevant_fraction.iloc[0] == 1/3
    assert np.isnan(result.path_coverage.iloc[1])
    assert np.isnan(result.mention_head.iloc[1])


def test_hierarchical_rule_matches_the_correct_branch_not_other_branch():
    frame = pd.DataFrame({"condition": [1, 3, 3], "feature1": [.2, .2, .8],
                          "feature2": [.9, .9, .9], "feature3": [.9, .9, .2],
                          "category": [1, 2, 3]})
    behavior.validate_task_rule(frame)
    frame.loc[2, "category"] = 4
    import pytest
    with pytest.raises(ValueError, match="task rule"):
        behavior.validate_task_rule(frame)

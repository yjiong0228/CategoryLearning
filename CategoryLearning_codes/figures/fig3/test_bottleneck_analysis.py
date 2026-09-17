"""Tests of scientific summaries and source-order safeguards, not plot styling."""
import numpy as np
import pandas as pd
import pytest

from CategoryLearning_codes.figures.fig3.bottleneck_analysis import (
    conditional_support, contiguous_runs, stage_profiles, validate_frame,
)


def test_conditional_support_uses_ratio_of_aggregated_marginals():
    active = np.array([.1, .9])
    mass = np.array([.09, .09])
    pooled = conditional_support(np.array([mass.mean()]), np.array([active.mean()]))
    assert pooled[0] == pytest.approx(.18)
    assert np.mean(conditional_support(mass, active)) == pytest.approx(.5)
    assert np.isnan(conditional_support(np.array([0.]), np.array([0.]))[0])


def test_belief_mass_cannot_exceed_active_probability():
    with pytest.raises(ValueError, match="exceeds"):
        conditional_support(np.array([.4]), np.array([.2]))


def test_stage_profile_pools_support_and_preserves_execution_inapplicability():
    table = pd.DataFrame({
        "trial": [1, 2], "target_active": [.1, .9], "target_mass": [.09, .09],
        "target_execution": [np.nan, np.nan], "observed_accuracy": [0., 1.],
        "model_correct_probability": [.3, .8], "oral_report_valid": [True, False],
        "oral_target_state": [.2, .2], "oral_target_current_report": [.2, np.nan],
        "oral_model_overlap": [.7, .9],
    })
    case = {"table": table, "metadata": {"subject": 129, "task": 1, "persistent_execution": False}}
    row = stage_profiles(case, 1).iloc[0]
    assert row.target_support_pooled == pytest.approx(.18)
    assert row.current_report_model_overlap_mean == pytest.approx(.7)
    assert row.oral_reports == 1
    assert np.isnan(row.target_execution_mean)
    assert row.n_trials == 2


def test_screens_preserve_gaps_and_last_trial():
    assert contiguous_runs(np.array([True, True, False, True, True, True]), 2) == [(0, 2), (3, 6)]
    assert contiguous_runs(np.array([True, False, True]), 2) == []


def test_source_order_and_integer_choices_are_not_silently_repaired():
    frame = pd.DataFrame({"iSub": [129, 129], "condition": [1, 1], "iSession": [1, 1],
                          "iBlock": [1, 1], "iTrial": [1, 2], "choice": [1, 2],
                          "category": [1, 2], "feedback": [1., 1.]})
    validate_frame(frame, 129, 1)
    with pytest.raises(ValueError, match="ordered"):
        validate_frame(frame.iloc[::-1], 129, 1)
    bad = frame.copy()
    bad["choice"] = [1.25, 2.]
    with pytest.raises(ValueError, match="non-integer"):
        validate_frame(bad, 129, 1)

"""Protect censored residences and conditional-state metric denominators."""
import numpy as np
import pandas as pd
import pytest

from src.Bayesian_state.evaluation.model_0923_path_diagnosis import residence_retention, trace_tables


def test_retention_excludes_insufficient_followup_without_counting_censoring_as_failure():
    episodes = pd.DataFrame({"start_trial": [1, 4, 9], "duration": [2, 7, 2],
                             "right_censored": [False, True, True]})
    result = residence_retention(episodes, 10, (1, 4, 8))
    assert result.eligible.tolist() == [3, 2, 1]
    assert result.retained.tolist() == [3, 1, 0]
    np.testing.assert_allclose(result.fraction, [1., .5, 0.])


def test_retention_empty_denominator_is_unknown():
    result = residence_retention(pd.DataFrame({"start_trial": [10], "duration": [1]}), 10, (8,))
    assert result.eligible.iloc[0] == 0
    assert np.isnan(result.fraction.iloc[0])
    with pytest.raises(ValueError, match="positive"):
        residence_retention(pd.DataFrame(), 10, (0,))


def test_trace_tables_preserve_absence_and_exit_timing(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    rule = np.array([[[42], [1]], [[42], [42]], [[1], [42]]])
    values = np.full(rule.shape, .8)
    np.savez(traces / "B_00.npz", rule=rule, prior=values, posterior=values,
        forgotten=values-.1, contribution=values, rule_correct=values,
        beta_pre=values, beta_post=values+.05,
        pairing_pre_true=values, pairing_post_true=values,
        proposed=np.array([[-1, -1], [-1, 42], [1, -1]]),
        source=np.array([[-1, -1], [-1, 1], [1, -1]]),
        capacity_dropped=np.full((3, 2), -1), feedback=np.ones((3, 2)))
    curves, batches, feedback, exits = trace_tables(tmp_path, 42, 1)
    np.testing.assert_allclose(curves.active, [.5, 1., .5])
    np.testing.assert_allclose(curves.ever, [.5, 1., 1.])
    assert batches.ever_paths.iloc[0] == 2
    assert batches.late_active_trials.iloc[0] == 1
    assert batches.late_prior_sum.iloc[0] == .8
    assert exits.exit_trial.tolist() == [3]
    assert exits.posterior.tolist() == [.8]
    assert feedback.path_trials.sum() == 4

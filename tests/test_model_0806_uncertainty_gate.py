from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_model_0806_uncertainty_gate import (
    choice_designs,
    evaluate_support_gate,
)


def test_choice_designs_adds_standardized_uncertainty_last() -> None:
    train = pd.DataFrame({
        "subject_id": [1, 1, 2, 2],
        "log_trial": [1.0, 2.0, 1.5, 2.5],
        "ambiguous": [0.0, 1.0, 0.0, 1.0],
        "previous_error_within": [0.0, 1.0, 1.0, 0.0],
        "lag_uncertainty_within": [0.2, 0.4, 0.6, 0.8],
    })
    evaluation = pd.DataFrame({
        "subject_id": [1, 2],
        "log_trial": [3.0, 3.5],
        "ambiguous": [0.0, 1.0],
        "previous_error_within": [1.0, 0.0],
        "lag_uncertainty_within": [0.3, 0.7],
    })
    baseline_train, _, candidate_train, _, scaling = choice_designs(
        train,
        evaluation,
        lag_column="lag_uncertainty_within",
        previous_error_column="previous_error_within",
    )
    assert candidate_train.shape[1] == baseline_train.shape[1] + 1
    assert np.isclose(candidate_train[:, -1].mean(), 0.0)
    assert np.isclose(candidate_train[:, -1].std(ddof=0), 1.0)
    assert np.isclose(scaling["uncertainty_mean"], 0.5)


def test_support_gate_requires_predictive_gain_and_phase_stability() -> None:
    summary = {
        "bootstrap_mean_95_interval": [0.001, 0.010],
        "coefficient": {"median": 0.2},
    }
    stable = {
        "early": {"mean": 0.1},
        "middle": {"mean": 0.2},
        "late": {"mean": 0.05},
    }
    assert evaluate_support_gate(summary, stable)["passed"]
    unstable = {**stable, "middle": {"mean": -0.1}}
    assert not evaluate_support_gate(summary, unstable)["passed"]
    no_gain = {**summary, "bootstrap_mean_95_interval": [-0.001, 0.010]}
    assert not evaluate_support_gate(no_gain, stable)["passed"]

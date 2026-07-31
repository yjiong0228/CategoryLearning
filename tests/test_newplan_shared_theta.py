from __future__ import annotations

import numpy as np

from src.Bayesian_state.utils.newplan_shared_theta import (
    binary_recovery_metrics,
    choose_membership_penalty,
    select_shared_theta,
)


def test_shared_theta_selects_one_strength_and_subject_boundaries() -> None:
    result = select_shared_theta(
        b0_losses=[0.30, 0.20, 0.40],
        dynamic_losses=[
            [0.29, 0.18],
            [0.21, 0.22],
            [0.39, 0.25],
        ],
        positive_theta_grid=[0.25, 0.75],
        membership_penalty=0.01,
    )

    assert result.theta_plus == 0.75
    assert result.membership.tolist() == [True, False, True]
    assert np.allclose(
        result.selected_unpenalized_losses,
        [0.18, 0.20, 0.25],
    )


def test_shared_theta_exact_ties_prefer_all_b0() -> None:
    result = select_shared_theta(
        b0_losses=[0.2, 0.3],
        dynamic_losses=[[0.2, 0.2], [0.3, 0.3]],
        positive_theta_grid=[0.25, 0.75],
        membership_penalty=0.0,
    )

    assert result.theta_plus == 0.0
    assert not np.any(result.membership)


def test_membership_penalty_removes_only_small_training_gains() -> None:
    result = select_shared_theta(
        b0_losses=[0.30, 0.30],
        dynamic_losses=[[0.295], [0.20]],
        positive_theta_grid=[0.5],
        membership_penalty=0.01,
    )

    assert result.theta_plus == 0.5
    assert result.membership.tolist() == [False, True]


def test_binary_metrics_and_penalty_calibration() -> None:
    metrics = binary_recovery_metrics(
        [False, False, True, True],
        [False, True, True, False],
    )
    assert metrics["accuracy_count"] == 2
    assert metrics["sensitivity"] == 0.5
    assert metrics["specificity"] == 0.5

    selected = choose_membership_penalty(
        penalties=[0.0, 0.01, 0.02],
        specificities=[0.7, 0.9, 1.0],
        sensitivities=[1.0, 0.8, 0.4],
        target_specificity=0.9,
    )
    assert selected == 1

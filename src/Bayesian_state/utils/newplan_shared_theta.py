"""Selection helpers for the zero-boundary shared-theta new-plan model.

The model writes each subject's transition strength as

    theta_s = z_s * theta_plus,

where ``z_s`` is an exact B0/D0 boundary and ``theta_plus`` is shared by all
subjects assigned to D0.  These helpers contain no simulation code and are
kept deterministic so the recovery decision rule can be unit tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SharedThetaSelection:
    theta_plus: float
    membership: np.ndarray
    objective: float
    selected_unpenalized_losses: np.ndarray
    selected_penalized_losses: np.ndarray
    penalty: float


def _finite_vector(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite vector.")
    return array


def select_shared_theta(
    *,
    b0_losses: Sequence[float] | np.ndarray,
    dynamic_losses: Sequence[Sequence[float]] | np.ndarray,
    positive_theta_grid: Sequence[float] | np.ndarray,
    membership_penalty: float,
) -> SharedThetaSelection:
    """Select one shared non-zero theta and exact subject-level zero boundaries.

    ``membership_penalty`` is added once to a subject's mean training loss
    when that subject is assigned to D0.  The all-B0 solution is an explicit
    candidate.  Exact ties prefer fewer dynamic memberships and then the
    smaller shared theta.
    """

    b0 = _finite_vector(b0_losses, "b0_losses")
    dynamic = np.asarray(dynamic_losses, dtype=float)
    theta_grid = _finite_vector(positive_theta_grid, "positive_theta_grid")
    penalty = float(membership_penalty)
    if (
        dynamic.ndim != 2
        or dynamic.shape[0] != b0.size
        or dynamic.shape[1] != theta_grid.size
    ):
        raise ValueError(
            "dynamic_losses must have shape "
            "(len(b0_losses), len(positive_theta_grid))."
        )
    if not np.all(np.isfinite(dynamic)):
        raise ValueError("dynamic_losses must be finite.")
    if np.any(theta_grid <= 0.0) or np.any(theta_grid > 1.0):
        raise ValueError("positive_theta_grid must lie in (0, 1].")
    if np.any(np.diff(theta_grid) <= 0.0):
        raise ValueError("positive_theta_grid must be strictly increasing.")
    if not np.isfinite(penalty) or penalty < 0.0:
        raise ValueError("membership_penalty must be finite and non-negative.")

    candidates: list[
        tuple[
            tuple[float, int, float],
            float,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]
    ] = []
    all_b0 = np.zeros(b0.size, dtype=bool)
    candidates.append(
        (
            (float(np.mean(b0)), 0, 0.0),
            0.0,
            all_b0,
            b0.copy(),
            b0.copy(),
        )
    )
    for theta_index, theta_plus in enumerate(theta_grid):
        dynamic_unpenalized = dynamic[:, theta_index]
        dynamic_penalized = dynamic_unpenalized + penalty
        membership = dynamic_penalized < b0
        selected_unpenalized = np.where(
            membership, dynamic_unpenalized, b0
        )
        selected_penalized = np.where(membership, dynamic_penalized, b0)
        objective = float(np.mean(selected_penalized))
        candidates.append(
            (
                (
                    objective,
                    int(np.sum(membership)),
                    float(theta_plus),
                ),
                float(theta_plus),
                membership,
                selected_unpenalized,
                selected_penalized,
            )
        )

    key, theta_plus, membership, selected, selected_penalized = min(
        candidates, key=lambda item: item[0]
    )
    return SharedThetaSelection(
        theta_plus=float(theta_plus),
        membership=membership.copy(),
        objective=float(key[0]),
        selected_unpenalized_losses=selected.copy(),
        selected_penalized_losses=selected_penalized.copy(),
        penalty=penalty,
    )


def binary_recovery_metrics(
    true_membership: Sequence[bool] | np.ndarray,
    estimated_membership: Sequence[bool] | np.ndarray,
) -> dict[str, float | int]:
    truth = np.asarray(true_membership, dtype=bool).reshape(-1)
    estimate = np.asarray(estimated_membership, dtype=bool).reshape(-1)
    if truth.size == 0 or estimate.shape != truth.shape:
        raise ValueError(
            "true_membership and estimated_membership must be aligned "
            "non-empty vectors."
        )
    true_positive = int(np.sum(truth & estimate))
    true_negative = int(np.sum((~truth) & (~estimate)))
    false_positive = int(np.sum((~truth) & estimate))
    false_negative = int(np.sum(truth & (~estimate)))
    positive_n = int(np.sum(truth))
    negative_n = int(np.sum(~truth))
    return {
        "n": int(truth.size),
        "accuracy_count": int(np.sum(truth == estimate)),
        "sensitivity": (
            float(true_positive / positive_n)
            if positive_n
            else float("nan")
        ),
        "specificity": (
            float(true_negative / negative_n)
            if negative_n
            else float("nan")
        ),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def choose_membership_penalty(
    *,
    penalties: Sequence[float] | np.ndarray,
    specificities: Sequence[float] | np.ndarray,
    sensitivities: Sequence[float] | np.ndarray,
    target_specificity: float,
) -> int:
    """Return the smallest calibrated penalty meeting the specificity target.

    If no candidate meets the target, the deterministic fallback maximizes
    specificity, then sensitivity, and then prefers the smaller penalty.
    """

    penalty_values = _finite_vector(penalties, "penalties")
    specificity_values = _finite_vector(specificities, "specificities")
    sensitivity_values = _finite_vector(sensitivities, "sensitivities")
    target = float(target_specificity)
    if (
        penalty_values.shape != specificity_values.shape
        or penalty_values.shape != sensitivity_values.shape
    ):
        raise ValueError("penalty and metric vectors must have equal lengths.")
    if np.any(penalty_values < 0.0) or np.any(np.diff(penalty_values) <= 0.0):
        raise ValueError("penalties must be strictly increasing and non-negative.")
    if (
        np.any((specificity_values < 0.0) | (specificity_values > 1.0))
        or np.any((sensitivity_values < 0.0) | (sensitivity_values > 1.0))
    ):
        raise ValueError("specificities and sensitivities must lie in [0, 1].")
    if not np.isfinite(target) or not 0.0 <= target <= 1.0:
        raise ValueError("target_specificity must lie in [0, 1].")

    passing = np.flatnonzero(specificity_values >= target)
    if passing.size:
        return int(passing[0])
    return int(
        min(
            range(penalty_values.size),
            key=lambda index: (
                -float(specificity_values[index]),
                -float(sensitivity_values[index]),
                float(penalty_values[index]),
            ),
        )
    )


__all__ = [
    "SharedThetaSelection",
    "binary_recovery_metrics",
    "choose_membership_penalty",
    "select_shared_theta",
]

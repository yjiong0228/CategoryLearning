"""Observed feedback in exp123 (Task2_learning.m, compare function)."""
from __future__ import annotations

import numpy as np


def category_count(condition: int) -> int:
    if condition not in (1, 2, 3):
        raise ValueError("condition must be 1, 2 or 3")
    return 2 if condition == 1 else 4


def category_indices(labels, n_categories: int, name: str = "labels") -> np.ndarray:
    labels = np.asarray(labels, dtype=float)
    if (not np.all(np.isfinite(labels)) or
            not np.all(np.isin(labels, np.arange(1, n_categories + 1)))):
        raise ValueError(f"{name} must contain category IDs 1..{n_categories}")
    return labels.astype(int) - 1


def task_feedback(choice: int, category: int, condition: int) -> float:
    """Score an observed or simulated choice using the experimental protocol."""
    k = category_count(condition)
    category_indices([choice, category], k)
    if choice == category:
        return 1.0
    if condition == 3 and (choice - 1) // 2 == (category - 1) // 2:
        return 0.5
    return 0.0


def feedback_mask(choice: int, feedback: float, condition: int) -> np.ndarray:
    """Categories compatible with the displayed feedback, given the chosen label.

    Condition 3 assumes the disclosed pairing {1, 2}, {3, 4} is known.
    A negative response in condition 2 leaves three possible target categories.
    """
    k = category_count(condition)
    category_indices([choice], k, "choice")
    allowed = (0.0, 0.5, 1.0) if condition == 3 else (0.0, 1.0)
    if feedback not in allowed:
        raise ValueError(f"feedback must be one of {allowed} for condition {condition}")
    return np.asarray([task_feedback(choice, c, condition) == feedback
                       for c in range(1, k + 1)], dtype=bool)

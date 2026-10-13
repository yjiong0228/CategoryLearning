"""Evaluate the experiment's per-category stopping criterion on saved blocks."""
from __future__ import annotations

import numpy as np


def block_criterion(feedback: np.ndarray, categories: np.ndarray,
                    ambiguous: np.ndarray, n_categories: int,
                    accuracy_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Return per-category exact accuracy and whether all categories reach threshold.

    Ambiguous stimuli are excluded. Every task trial in the block, including
    the first task trial, enters this behavioral criterion when unambiguous.
    """
    outcomes = np.asarray(feedback, dtype=float)
    category = np.asarray(categories, dtype=int)
    ambiguity = np.asarray(ambiguous)
    if outcomes.ndim != 2 or category.shape != (outcomes.shape[1],) or ambiguity.shape != category.shape:
        raise ValueError('Need aligned trajectory × trial feedback and trial labels')
    if not np.isin(outcomes, [0, .5, 1]).all() or not np.isin(ambiguity, [0, 1]).all():
        raise ValueError('Invalid task feedback or ambiguity flag')
    if n_categories < 2 or np.any((category < 1) | (category > n_categories)) or not 0 <= accuracy_threshold <= 1:
        raise ValueError('Invalid category labels or accuracy threshold')
    accuracy = np.full((outcomes.shape[0], n_categories), np.nan)
    for k in range(1, n_categories + 1):
        mask = (category == k) & (ambiguity == 0)
        if np.any(mask):
            accuracy[:, k - 1] = (outcomes[:, mask] == 1).mean(axis=1)
    reached = np.all(np.isfinite(accuracy) & (accuracy >= accuracy_threshold), axis=1)
    return accuracy, reached

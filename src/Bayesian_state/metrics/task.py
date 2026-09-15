"""Experimenter-side category-learning feedback and distinct task metrics.

The canonical family groups are task truth, not a learner's pairing prior.
These functions belong in environment scoring and post hoc evaluation only.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .trial import family_correct


def category_learning_feedback(choice: int, category: int, *, condition: int) -> float:
    """Score a sampled choice after responding, in the task's choice encoding."""
    if int(choice) == int(category):
        return 1.0
    if int(condition) == 3 and (int(choice) - 1) // 2 == (int(category) - 1) // 2:
        return 0.5
    return 0.0


def category_learning_metrics(
    *,
    choices: Sequence[int] | np.ndarray,
    categories: Sequence[int] | np.ndarray,
    feedback: Sequence[float] | np.ndarray,
    n_categories: int,
) -> dict[str, float]:
    """Summarize one aligned trajectory without treating reward as accuracy.

    Binary tasks retain the established family convention: for two choices,
    family accuracy equals exact-category accuracy.
    """
    responses = np.asarray(choices, dtype=int).reshape(-1)
    targets = np.asarray(categories, dtype=int).reshape(-1)
    rewards = np.asarray(feedback, dtype=float).reshape(-1)
    if responses.size == 0 or targets.shape != responses.shape or rewards.shape != responses.shape:
        raise ValueError("choices, categories, and feedback must align and be non-empty")
    return {
        "species_accuracy": float(np.mean(responses == targets)),
        "family_accuracy": float(np.mean(family_correct(targets, responses, int(n_categories)))),
        "mean_reward": float(np.mean(rewards)),
    }


def condition3_task_metric_summaries(
    *,
    probabilities: np.ndarray,
    choices: Sequence[int] | np.ndarray,
    feedback: Sequence[float] | np.ndarray,
    categories: Sequence[int] | np.ndarray | None,
    valid_trial_mask: Sequence[bool] | np.ndarray,
) -> dict[str, Any]:
    """Score species, family and reward using the caller's exact scoring mask.

    True categories are experimenter-side targets used only for evaluation.
    Without them, ternary feedback still identifies observed task scores, but
    the predicted task scores are unavailable: chosen-response probability is
    not species accuracy. In condition 3, expected reward is half the sum of
    species and family correctness probabilities.
    """
    probs = np.asarray(probabilities, dtype=float)
    responses = np.asarray(choices, dtype=int).reshape(-1)
    rewards = np.asarray(feedback, dtype=float).reshape(-1)
    valid = np.asarray(valid_trial_mask, dtype=bool).reshape(-1)
    if probs.shape != (responses.size, 4) or rewards.shape != responses.shape or valid.shape != responses.shape:
        raise ValueError("condition 3 task summaries require aligned four-response probabilities and trials")
    targets = None if categories is None else np.asarray(categories, dtype=int).reshape(-1)
    if targets is not None and targets.shape != responses.shape:
        raise ValueError("categories must align with condition 3 task summary trials")
    observed = {key: float("nan") for key in ("species_accuracy", "family_accuracy", "mean_reward")}
    predicted = observed.copy()
    if np.any(valid):
        if targets is None:
            observed = {
                "species_accuracy": float(np.mean(rewards[valid] == 1.0)),
                "family_accuracy": float(np.mean(rewards[valid] > 0.0)),
                "mean_reward": float(np.mean(rewards[valid])),
            }
        else:
            observed = category_learning_metrics(
                choices=responses[valid], categories=targets[valid], feedback=rewards[valid], n_categories=4,
            )
            target_indices = targets[valid] - 1
            if np.all((target_indices >= 0) & (target_indices < 4)):
                scored_probs = probs[valid]
                rows = np.arange(target_indices.size)
                species = float(np.mean(scored_probs[rows, target_indices]))
                family_probs = scored_probs.reshape(-1, 2, 2).sum(axis=2)
                family = float(np.mean(family_probs[rows, target_indices // 2]))
                predicted = {
                    "species_accuracy": species,
                    "family_accuracy": family,
                    "mean_reward": 0.5 * (species + family),
                }
    return {
        "observed_task_metrics": observed,
        "predicted_task_metrics": predicted,
        "task_metrics_n_trials": int(valid.sum()),
        "task_metrics_trial_mask": "valid_trial_mask",
    }


__all__ = ["category_learning_feedback", "category_learning_metrics", "condition3_task_metric_summaries"]

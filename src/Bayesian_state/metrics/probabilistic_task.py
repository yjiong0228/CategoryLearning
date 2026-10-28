"""Task-side scores for binary probabilistic category feedback."""
from __future__ import annotations

import numpy as np


def validate_task_probabilities(values: np.ndarray) -> np.ndarray:
    """Validate exp4's two complementary reward probabilities per trial."""
    probabilities = np.asarray(values, dtype=float)
    if (probabilities.ndim != 2 or probabilities.shape[1] != 2
            or len(probabilities) == 0 or not np.isfinite(probabilities).all()
            or np.any((probabilities < 0) | (probabilities > 1))
            or not np.allclose(probabilities.sum(axis=1), 1., atol=1e-12, rtol=0)):
        raise ValueError("Task probabilities must be nonempty normalized binary rows")
    return probabilities


def probabilistic_task_metrics(*, choices: np.ndarray, feedback: np.ndarray,
                               target_probabilities: np.ndarray) -> dict[str, float]:
    """Separate realized reward from choice quality on the given schedule.

    Expected reward averages the task probability of each actual choice.
    Expected regret is the reward-probability gap to the best available choice.
    Both choices count as optimal on an exactly tied trial.
    """
    q = validate_task_probabilities(target_probabilities)
    choices = np.asarray(choices)
    feedback = np.asarray(feedback)
    if (choices.shape != (len(q),) or feedback.shape != choices.shape
            or not np.isin(choices, [1, 2]).all()
            or not np.isin(feedback, [0., 1.]).all()):
        raise ValueError("Binary choices and feedback must align with task probabilities")
    selected = q[np.arange(len(q)), choices.astype(int) - 1]
    optimal = q.max(axis=1)
    return dict(optimal_choice_rate=float(np.mean(selected == optimal)),
                mean_reward=float(feedback.mean()),
                expected_reward=float(selected.mean()),
                optimal_expected_reward=float(optimal.mean()),
                expected_regret=float(np.mean(optimal - selected)))

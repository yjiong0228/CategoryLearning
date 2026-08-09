"""Autonomous StateModel execution in a category-learning task."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..problems.model import GeneratedBehaviorTrajectory, StateModel
from ..utils.seeding import inject_module_seed_from_trajectory, stable_seed


@dataclass
class AutonomousModelResult:
    """A generated behavior trajectory together with its task provenance."""

    trajectory: GeneratedBehaviorTrajectory
    categories: np.ndarray
    condition: int
    subject_id: int
    trajectory_seed: int


def run_autonomous_category_learning(
    *,
    engine_config: Mapping[str, Any],
    subject_id: int,
    condition: int,
    stimulus: Sequence[Sequence[float]] | np.ndarray,
    categories: Sequence[int] | np.ndarray,
    trajectory_seed: int,
    choice_readout_config: Mapping[str, Any] | None = None,
    output_noise_config: Mapping[str, Any] | None = None,
    processed_data_dir: Path | str | None = None,
    dataset_paths: Mapping[str, Path | str] | None = None,
) -> AutonomousModelResult:
    """Generate choices, task feedback, and cognitive states in trial order.

    The model receives the physical stimulus but never the correct category
    before choosing.  The category schedule is owned by the task environment
    and is used only after the sampled choice to produce deterministic
    correctness feedback.
    """

    physical = np.asarray(stimulus, dtype=float)
    task_categories = np.asarray(categories, dtype=int).reshape(-1)
    if physical.ndim != 2 or physical.shape[0] != task_categories.size:
        raise ValueError("stimulus must be 2-D and aligned with categories.")
    if physical.shape[0] == 0:
        raise ValueError("cannot generate an empty autonomous trajectory.")

    resolved_config = deepcopy(dict(engine_config))
    inject_module_seed_from_trajectory(resolved_config, int(trajectory_seed))
    model = StateModel(
        resolved_config,
        condition=int(condition),
        subject_id=int(subject_id),
        processed_data_dir=processed_data_dir,
        dataset_paths=dataset_paths,
    )
    if not np.all(
        (task_categories >= 1) & (task_categories <= int(model.n_cats))
    ):
        raise ValueError(
            f"categories must be 1-indexed in [1, {model.n_cats}]."
        )

    choice_seed = stable_seed(
        {
            "seed_role": "state_model_autonomous_choice",
            "trajectory_seed": int(trajectory_seed),
        }
    )

    def task_feedback(trial_index: int, choice: int) -> float:
        return float(int(choice) == int(task_categories[trial_index]))

    legacy_random_state = np.random.get_state()
    np.random.seed(int(trajectory_seed))
    try:
        trajectory = model.generate_step_by_step(
            physical,
            task_feedback,
            choice_seed=int(choice_seed),
            choice_readout_config=choice_readout_config,
            output_noise_config=output_noise_config,
        )
    finally:
        np.random.set_state(legacy_random_state)

    return AutonomousModelResult(
        trajectory=trajectory,
        categories=task_categories.copy(),
        condition=int(condition),
        subject_id=int(subject_id),
        trajectory_seed=int(trajectory_seed),
    )


__all__ = ["AutonomousModelResult", "run_autonomous_category_learning"]

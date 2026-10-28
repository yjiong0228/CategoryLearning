"""Choice-contingent stochastic environment, separate from learner beliefs."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..metrics.probabilistic_task import probabilistic_task_metrics, validate_task_probabilities
from ..model.config import ModelContext
from ..model.state_model import GeneratedBehaviorTrajectory, StateModel
from ..utils.seeding import inject_module_seed_from_trajectory, stable_seed


class BinaryProbabilisticEnvironment:
    """Draw r_t = 1[u_t < q_t(choice_t)] after a 1-indexed choice.

    The environment owns a private feedback RNG and the full-precision task
    probabilities. The callback returns only the realized binary feedback.
    Sequential access prevents silently drawing a second outcome for a trial.
    """

    def __init__(self, target_probabilities: np.ndarray, *, seed: int) -> None:
        self.probabilities = validate_task_probabilities(target_probabilities).copy()
        self.rng = np.random.default_rng(seed)
        self.selected_probabilities: list[float] = []
        self.uniform_draws: list[float] = []

    def __call__(self, trial_index: int, choice: int) -> float:
        if trial_index != len(self.uniform_draws) or trial_index >= len(self.probabilities):
            raise ValueError("Feedback must be requested once per trial in order")
        if choice not in (1, 2):
            raise ValueError("Feedback requires a binary 1-indexed choice")
        probability = float(self.probabilities[trial_index, int(choice) - 1])
        uniform = float(self.rng.random())
        self.selected_probabilities.append(probability)
        self.uniform_draws.append(uniform)
        return float(uniform < probability)


@dataclass
class ProbabilisticModelResult:
    trajectory: GeneratedBehaviorTrajectory
    target_probabilities: np.ndarray
    selected_probabilities: np.ndarray
    feedback_uniforms: np.ndarray
    trajectory_seed: int
    feedback_seed: int
    metrics: dict[str, float]


def seeded_cognitive_config(engine_config: Mapping[str, Any], trajectory_seed: int) -> dict:
    """Give perception and rule transitions reproducible, separate RNG streams."""
    config = deepcopy(dict(engine_config))
    for name in ("hypo_transitions_mod", "perception_mod"):
        inject_module_seed_from_trajectory(config, trajectory_seed, module_name=name)
    return config


def run_autonomous_probabilistic_learning(
    *, engine_config: Mapping[str, Any], subject_id: int, stimulus: np.ndarray,
    target_probabilities: np.ndarray, trajectory_seed: int,
    feedback_seed: int | None = None,
    processed_data_dir: Path | str | None = None,
    dataset_paths: Mapping[str, Path | str] | None = None,
) -> ProbabilisticModelResult:
    """Generate one binary-task history using the shared StateModel lifecycle.

    Call inside the runtime for the intended model variant. Each new call starts
    a fresh history; session boundaries within the stimulus schedule keep state.
    """
    physical = np.asarray(stimulus, dtype=float)
    q = validate_task_probabilities(target_probabilities)
    if physical.ndim != 2 or len(physical) != len(q) or not np.isfinite(physical).all():
        raise ValueError("Finite 2-D stimuli must align with task probabilities")
    if feedback_seed is None:
        feedback_seed = stable_seed(dict(seed_role="state_model_probabilistic_feedback",
                                         trajectory_seed=int(trajectory_seed)))
    environment = BinaryProbabilisticEnvironment(q, seed=feedback_seed)
    model = StateModel(seeded_cognitive_config(engine_config, trajectory_seed),
                       context=ModelContext(condition=1, subject_id=int(subject_id),
                                            processed_data_dir=processed_data_dir,
                                            dataset_paths=dataset_paths))
    if model.n_cats != 2:
        raise ValueError("Probabilistic binary environment requires two model categories")
    choice_seed = stable_seed(dict(seed_role="state_model_autonomous_choice",
                                   trajectory_seed=int(trajectory_seed)))
    trajectory = model.generate_step_by_step(physical, environment, choice_seed=choice_seed)
    return ProbabilisticModelResult(
        trajectory=trajectory, target_probabilities=q.copy(),
        selected_probabilities=np.asarray(environment.selected_probabilities),
        feedback_uniforms=np.asarray(environment.uniform_draws),
        trajectory_seed=int(trajectory_seed), feedback_seed=int(feedback_seed),
        metrics=probabilistic_task_metrics(choices=trajectory.choices, feedback=trajectory.feedback,
                                          target_probabilities=q))

"""Autonomous trajectory generation for the minimal active-set model."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..utils.seeding import inject_module_seed_from_trajectory, stable_seed


@dataclass
class GeneratedCondition1Trajectory:
    stimulus: np.ndarray
    perceived_stimulus: np.ndarray
    categories: np.ndarray
    choices: np.ndarray
    feedback: np.ndarray
    cognitive_probabilities: np.ndarray
    observed_probabilities: np.ndarray
    prior: np.ndarray
    posterior: np.ndarray
    beta: np.ndarray
    transition_log: list[dict[str, Any]]
    trajectory_seed: int


def _normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError("Probability weights must be finite and non-negative.")
    total = float(np.sum(array))
    if total <= 0.0:
        raise ValueError("Probability weights sum to zero.")
    return array / total


def generate_condition1_trajectory(
    *,
    engine_config: Mapping[str, Any],
    subject_id: int,
    stimulus: Sequence[Sequence[float]] | np.ndarray,
    categories: Sequence[int] | np.ndarray,
    epsilon: float,
    rho: float,
    trajectory_seed: int,
    processed_data_dir: Path | str | None = None,
    dataset_paths: Mapping[str, Path | str] | None = None,
) -> GeneratedCondition1Trajectory:
    """Generate choices and feedback before updating the same state model.

    The physical stimulus/category schedule is fixed.  At each trial the
    function prepares perception and the active-set transition, samples a
    choice from the lapse-mixed pre-feedback prediction, lets the deterministic
    condition-1 task generate feedback, and only then updates likelihood,
    memory, and beta.
    """

    from ..problems import StateModel

    x = np.asarray(stimulus, dtype=float)
    y = np.asarray(categories, dtype=int).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("stimulus must be 2-D and aligned with categories.")
    if x.shape[0] == 0:
        raise ValueError("Cannot generate an empty trajectory.")
    if not np.all(np.isin(y, [1, 2])):
        raise ValueError("Condition-1 categories must be encoded as 1 or 2.")
    epsilon_value = float(epsilon)
    rho_value = float(rho)
    if not np.isfinite(epsilon_value) or not 0.0 <= epsilon_value <= 1.0:
        raise ValueError("epsilon must be in [0, 1].")
    if not np.isfinite(rho_value) or rho_value <= 0.0:
        raise ValueError("rho must be finite and positive.")

    resolved_config = deepcopy(dict(engine_config))
    inject_module_seed_from_trajectory(resolved_config, int(trajectory_seed))
    model = StateModel(
        resolved_config,
        condition=1,
        subject_id=int(subject_id),
        processed_data_dir=processed_data_dir,
        dataset_paths=dataset_paths,
    )
    engine = model.engine
    required_modules = (
        "perception_mod",
        "hypo_transitions_mod",
        "likelihood_mod",
        "memory_mod",
        "beta_mod",
    )
    missing = [name for name in required_modules if name not in engine.modules]
    if missing:
        raise ValueError(f"Autonomous generation is missing required modules: {missing}.")
    transition = engine.modules["hypo_transitions_mod"]
    if not hasattr(transition, "record_outcome_feedback"):
        raise ValueError(
            "Autonomous active-set generation requires FeedbackSwapHypothesisModule."
        )

    n_trials = x.shape[0]
    n_hypotheses = int(engine.set_size)
    perceived_log = np.zeros_like(x, dtype=float)
    choices = np.zeros(n_trials, dtype=int)
    feedback = np.zeros(n_trials, dtype=float)
    cognitive = np.zeros((n_trials, 2), dtype=float)
    observed = np.zeros((n_trials, 2), dtype=float)
    prior_log = np.zeros((n_trials, n_hypotheses), dtype=float)
    posterior_log = np.zeros((n_trials, n_hypotheses), dtype=float)
    beta_log = np.zeros((n_trials, n_hypotheses), dtype=float)

    choice_seed = stable_seed(
        {
            "seed_role": "active_set_autonomous_choice",
            "trajectory_seed": int(trajectory_seed),
        }
    )
    perception_seed = stable_seed(
        {
            "seed_role": "active_set_autonomous_perception",
            "trajectory_seed": int(trajectory_seed),
        }
    )
    choice_rng = np.random.default_rng(choice_seed)
    legacy_random_state = np.random.get_state()
    np.random.seed(int(perception_seed))

    try:
        for trial_idx in range(n_trials):
            if engine.posterior is not None:
                engine.prior = np.asarray(engine.posterior, dtype=float).copy()

            # Provisional outcome fields let the existing perception/transition
            # interfaces prepare a pre-choice state.  The transition only reads
            # its internally stored previous feedback; the current provisional
            # feedback is replaced below before the state update.
            engine.observation = (x[trial_idx].copy(), 1, 1.0)
            engine.modules["perception_mod"].process()
            perceived = np.asarray(engine.observation[0], dtype=float).copy()
            engine.modules["hypo_transitions_mod"].process()

            prior = np.asarray(engine.prior, dtype=float).copy()
            active = np.flatnonzero(np.asarray(engine.hypotheses_mask, dtype=float) > 0.0)
            readout_weights = np.zeros(n_hypotheses, dtype=float)
            readout_weights[active] = _normalize(np.power(prior[active], rho_value))

            beta_now = np.asarray(engine.beta, dtype=float).copy()
            hypothesis_category = np.zeros((n_hypotheses, 2), dtype=float)
            for hypothesis in active:
                category_probability = model.partition_model.get_category_probabilities(
                    hypo=int(hypothesis),
                    data=([perceived], [1], [1.0]),
                    beta=float(beta_now[hypothesis]),
                    distance_mode=getattr(engine, "distance_mode", "prototype"),
                )
                probability_vector = np.asarray(category_probability[:, 0], dtype=float)
                hypothesis_category[hypothesis] = _normalize(probability_vector)

            cognitive_probability = _normalize(
                np.sum(readout_weights[:, None] * hypothesis_category, axis=0)
            )
            observed_probability = _normalize(
                (1.0 - epsilon_value) * cognitive_probability
                + epsilon_value / 2.0
            )
            choice = int(choice_rng.choice(2, p=observed_probability)) + 1
            outcome = float(choice == int(y[trial_idx]))

            engine.observation = (perceived, choice, outcome)
            transition.record_outcome_feedback(outcome)
            engine.modules["likelihood_mod"].process()
            engine.modules["memory_mod"].process()
            engine.modules["beta_mod"].process()

            perceived_log[trial_idx] = perceived
            choices[trial_idx] = choice
            feedback[trial_idx] = outcome
            cognitive[trial_idx] = cognitive_probability
            observed[trial_idx] = observed_probability
            prior_log[trial_idx] = prior
            posterior_log[trial_idx] = np.asarray(engine.posterior, dtype=float)
            beta_log[trial_idx] = beta_now
    finally:
        np.random.set_state(legacy_random_state)

    return GeneratedCondition1Trajectory(
        stimulus=x.copy(),
        perceived_stimulus=perceived_log,
        categories=y.copy(),
        choices=choices,
        feedback=feedback,
        cognitive_probabilities=cognitive,
        observed_probabilities=observed,
        prior=prior_log,
        posterior=posterior_log,
        beta=beta_log,
        transition_log=[dict(item) for item in transition.transition_log],
        trajectory_seed=int(trajectory_seed),
    )


__all__ = [
    "GeneratedCondition1Trajectory",
    "generate_condition1_trajectory",
]

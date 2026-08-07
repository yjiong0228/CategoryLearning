"""Simulation helpers for FA2/FA2R mechanism-recovery experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model_0803 import TransitionKernels
from .model_0804 import (
    EPS,
    Model0804Parameters,
    _choice_probability,
    _feedback_update,
    _initial_state,
    _sample_transition,
    _transition_uniform_dimension,
    _validate_inputs,
    _weighted_wor_from_uniforms,
)


@dataclass
class Model0804Simulation:
    choices: np.ndarray
    feedback: np.ndarray
    correct_choices: np.ndarray
    choice_probabilities: np.ndarray
    active: np.ndarray
    replacement_count: np.ndarray
    regenerated: np.ndarray
    simulation_seed: int
    model_id: str


def infer_correct_choices(
    choices: np.ndarray,
    feedback: np.ndarray,
) -> np.ndarray:
    """Recover the binary task-correct response from observed choice/feedback."""

    y = np.asarray(choices, dtype=int).reshape(-1)
    r = np.asarray(feedback, dtype=float).reshape(-1)
    if y.size != r.size or np.any((y < 0) | (y > 1)):
        raise ValueError("choices and feedback must be matching binary arrays")
    if np.any(~np.isclose(r, 0.0) & ~np.isclose(r, 1.0)):
        raise ValueError("feedback must contain only 0 or 1")
    return np.where(r >= 0.5, y, 1 - y).astype(int)


def simulate_model0804_choices(
    q_values: np.ndarray,
    correct_choices: np.ndarray,
    prior: np.ndarray,
    kernels: TransitionKernels,
    *,
    model_id: str,
    parameters: Model0804Parameters,
    capacity: int,
    simulation_seed: int,
    epsilon: float = EPS,
) -> Model0804Simulation:
    """Generate choices and deterministic correctness feedback in trial order."""

    correct = np.asarray(correct_choices, dtype=int).reshape(-1)
    dummy_feedback = np.ones(correct.size, dtype=float)
    q, _, _, p0, decoded = _validate_inputs(
        q_values,
        correct,
        dummy_feedback,
        prior,
        kernels,
        capacity,
        model_id,
        parameters,
    )
    n_trials, n_hypotheses, n_categories = q.shape
    if n_categories != 2:
        raise ValueError("model_0804 recovery simulation requires binary choices")
    rng = np.random.default_rng(int(simulation_seed))
    initial = _weighted_wor_from_uniforms(
        np.arange(n_hypotheses, dtype=int),
        p0,
        int(capacity),
        rng.random(int(capacity)),
    )
    state = _initial_state(initial, p0)
    choices = np.zeros(n_trials, dtype=int)
    feedback = np.zeros(n_trials, dtype=float)
    probabilities = np.zeros((n_trials, n_categories), dtype=float)
    active = np.zeros((n_trials, n_hypotheses), dtype=bool)
    replacement_count = np.zeros(n_trials, dtype=int)
    regenerated = np.zeros(n_trials, dtype=bool)
    transition_dimension = _transition_uniform_dimension(decoded, int(capacity))

    for trial_index in range(n_trials):
        if trial_index > 0:
            state, summary, _ = _sample_transition(
                state,
                p0,
                kernels,
                decoded,
                int(capacity),
                rng.random(transition_dimension),
            )
            replacement_count[trial_index] = summary.replacement_count
            regenerated[trial_index] = summary.regenerated
        active[trial_index, state.active] = True
        probabilities[trial_index] = _choice_probability(
            state,
            q[trial_index],
            decoded.kappa,
            decoded.lapse,
        )
        cumulative = np.cumsum(probabilities[trial_index])
        cumulative[-1] = 1.0
        choices[trial_index] = int(
            np.searchsorted(cumulative, float(rng.random()), side="right")
        )
        feedback[trial_index] = float(
            choices[trial_index] == correct[trial_index]
        )
        state, _, _ = _feedback_update(
            state,
            q[trial_index],
            int(choices[trial_index]),
            float(feedback[trial_index]),
            decoded,
            float(epsilon),
        )

    return Model0804Simulation(
        choices=choices,
        feedback=feedback,
        correct_choices=correct.copy(),
        choice_probabilities=probabilities,
        active=active,
        replacement_count=replacement_count,
        regenerated=regenerated,
        simulation_seed=int(simulation_seed),
        model_id=str(model_id),
    )

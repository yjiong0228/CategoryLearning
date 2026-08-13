"""Minimal surprise-driven FA3-M utilities for manuscript/model_0806.tex."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .model_0803 import TransitionKernels
from .model_0804.core import (
    EPS,
    Model0804Parameters,
    Model0804RTParameters,
    _choice_probability,
    _feedback_update,
    _sample_initial_state,
    _sample_transition,
    _transition_uniform_dimension,
    _validate_inputs,
    _validate_rt_parameters,
)


@dataclass
class Model0806Simulation:
    """One autonomous latent-path simulation from static FA2 or dynamic FA3-M."""

    choices: np.ndarray
    feedback: np.ndarray
    probabilities: np.ndarray
    predictive_m: np.ndarray
    feedback_surprise: np.ndarray
    feedback_uncertainty: np.ndarray
    replacement_count: np.ndarray
    replacement_fraction: np.ndarray
    newcomer_distance: np.ndarray
    active_path: np.ndarray


def simulate_model0806_log_rt(
    simulation: Model0806Simulation,
    parameters: Model0804RTParameters,
    *,
    seed: int,
) -> np.ndarray:
    """Generate log RT from the same autonomous latent search path."""

    decoded = _validate_rt_parameters(parameters)
    probabilities = np.asarray(simulation.probabilities, dtype=float)
    entropy = -np.sum(
        probabilities * np.log(np.clip(probabilities, EPS, 1.0)), axis=1
    )
    location = (
        float(decoded.intercept)
        + float(decoded.choice_entropy) * entropy
        + float(decoded.replacement_fraction)
        * np.asarray(simulation.replacement_fraction, dtype=float)
        + float(decoded.newcomer_distance)
        * np.asarray(simulation.newcomer_distance, dtype=float)
    )
    rng = np.random.default_rng(int(seed))
    noise = rng.standard_t(
        float(decoded.degrees_of_freedom), size=location.size
    )
    return location + float(decoded.sigma) * noise


def _logit(probability: float) -> float:
    value = float(probability)
    if not 0.0 < value < 1.0:
        raise ValueError("dynamic baseline m must lie strictly between 0 and 1")
    return math.log(value / (1.0 - value))


def _expit(value: float) -> float:
    clipped = min(max(float(value), -30.0), 30.0)
    return 1.0 / (1.0 + math.exp(-clipped))


def simulate_model0806_choices(
    q_values: np.ndarray,
    categories: np.ndarray,
    prior: np.ndarray,
    kernels: TransitionKernels,
    *,
    parameters: Model0804Parameters,
    capacity: int,
    seed: int,
    epsilon: float = EPS,
) -> Model0806Simulation:
    """Generate choices and feedback without reading an observed choice path."""

    q = np.asarray(q_values, dtype=float)
    category = np.asarray(categories, dtype=int).reshape(-1)
    if q.ndim != 3 or q.shape[0] != category.size or q.shape[2] != 2:
        raise ValueError("q_values and categories have incompatible shapes")
    if np.any((category < 0) | (category > 1)):
        raise ValueError("condition-1 categories must be zero-based 0 or 1")
    placeholder_choice = np.zeros(category.size, dtype=int)
    placeholder_feedback = np.ones(category.size, dtype=float)
    q, _, _, p0, decoded = _validate_inputs(
        q,
        placeholder_choice,
        placeholder_feedback,
        prior,
        kernels,
        int(capacity),
        "FA2",
        parameters,
    )
    rng = np.random.default_rng(int(seed))
    state = _sample_initial_state(
        p0,
        int(capacity),
        int(rng.integers(0, 2**32 - 1)),
    )
    n_trials, n_hypotheses, n_categories = q.shape
    choices = np.zeros(n_trials, dtype=int)
    feedback = np.zeros(n_trials, dtype=float)
    probabilities = np.zeros((n_trials, n_categories), dtype=float)
    predictive_m = np.full(n_trials, float(decoded.m), dtype=float)
    surprise = np.zeros(n_trials, dtype=float)
    uncertainty = np.zeros(n_trials, dtype=float)
    replacement_count = np.zeros(n_trials, dtype=int)
    newcomer_distance = np.zeros(n_trials, dtype=float)
    active_path = np.zeros((n_trials, n_hypotheses), dtype=bool)
    baseline_logit = _logit(float(decoded.m))
    control_logit = baseline_logit

    for trial_index in range(n_trials):
        current_m = (
            _expit(control_logit)
            if decoded.dynamic_m
            and (
                decoded.m_beta_surprise > 0.0
                or decoded.m_beta_uncertainty > 0.0
            )
            else float(decoded.m)
        )
        predictive_m[trial_index] = current_m
        if trial_index > 0:
            transition_parameters = replace(
                decoded,
                m=current_m,
                dynamic_m=False,
                m_phi=0.0,
                m_beta_surprise=0.0,
                surprise_center=0.0,
                surprise_scale=1.0,
                m_beta_uncertainty=0.0,
                uncertainty_center=0.0,
                uncertainty_scale=1.0,
            )
            unit = rng.random(
                _transition_uniform_dimension(
                    transition_parameters,
                    int(capacity),
                )
            )
            state, summary, _ = _sample_transition(
                state,
                p0,
                kernels,
                transition_parameters,
                int(capacity),
                unit,
            )
            replacement_count[trial_index] = int(summary.replacement_count)
            newcomer_distance[trial_index] = float(summary.newcomer_distance)

        active_path[trial_index, state.active] = True
        probabilities[trial_index] = _choice_probability(
            state,
            q[trial_index],
            float(decoded.kappa),
            float(decoded.lapse),
        )
        choices[trial_index] = int(
            rng.choice(n_categories, p=probabilities[trial_index])
        )
        feedback[trial_index] = float(
            choices[trial_index] == category[trial_index]
        )
        state, surprise[trial_index], uncertainty[trial_index] = _feedback_update(
            state,
            q[trial_index],
            int(choices[trial_index]),
            float(feedback[trial_index]),
            decoded,
            float(epsilon),
        )
        if decoded.dynamic_m and (
            decoded.m_beta_surprise > 0.0
            or decoded.m_beta_uncertainty > 0.0
        ):
            standardized_surprise = (
                surprise[trial_index] - float(decoded.surprise_center)
            ) / float(decoded.surprise_scale)
            standardized_uncertainty = (
                uncertainty[trial_index] - float(decoded.uncertainty_center)
            ) / float(decoded.uncertainty_scale)
            control_logit = (
                baseline_logit
                + float(decoded.m_phi) * (control_logit - baseline_logit)
                + float(decoded.m_beta_surprise) * standardized_surprise
                + float(decoded.m_beta_uncertainty) * standardized_uncertainty
            )

    return Model0806Simulation(
        choices=choices,
        feedback=feedback,
        probabilities=probabilities,
        predictive_m=predictive_m,
        feedback_surprise=surprise,
        feedback_uncertainty=uncertainty,
        replacement_count=replacement_count,
        replacement_fraction=replacement_count.astype(float) / float(capacity),
        newcomer_distance=newcomer_distance,
        active_path=active_path,
    )

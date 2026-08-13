"""Coupled-history forgetting diagnostics for the model_0804 HFW state.

The audit starts two filtered HFW states from different histories and then
forces them to receive identical transition innovations, observed choices,
and feedback.  Any remaining divergence is therefore inherited from the
anchor histories rather than future Monte Carlo noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..model_0803 import TransitionKernels
from .core import (
    EPS,
    HFWState,
    Model0804Parameters,
    _choice_probability,
    _feedback_update,
    _initial_state,
    _sample_transition,
    _transition_uniform_dimension,
    _validate_inputs,
    _weighted_wor_from_uniforms,
    effective_sample_size,
    systematic_resample,
)


@dataclass
class FilteredAnchorPanel:
    """Posterior state draws at requested post-feedback anchor trials."""

    states_by_anchor: dict[int, list[HFWState]]
    pre_resampling_ess: np.ndarray
    resampled: np.ndarray
    particle_count: int
    sample_count: int
    filter_seed: int


@dataclass
class CoupledForgettingTrace:
    """Pairwise distances after common-random-number future propagation."""

    anchor_trial: int
    trial_indices: np.ndarray
    active_distance: np.ndarray
    active_equal: np.ndarray
    state_exact_equal: np.ndarray
    regenerated: np.ndarray
    ever_regenerated: np.ndarray
    omega_total_variation: np.ndarray
    choice_probability_difference: np.ndarray
    signed_choice_probability_difference: np.ndarray
    common_memory_delta_difference: np.ndarray
    pair_count: int
    horizon: int
    coupling_seed: int


def sample_model0804_filtered_anchor_states(
    q_values: np.ndarray,
    choices: np.ndarray,
    feedback: np.ndarray,
    prior: np.ndarray,
    kernels: TransitionKernels,
    *,
    model_id: str,
    parameters: Model0804Parameters,
    capacity: int,
    anchors: list[int] | tuple[int, ...] | np.ndarray,
    particle_count: int,
    sample_count: int,
    filter_seed: int,
    resample_threshold_fraction: float = 0.5,
    epsilon: float = EPS,
) -> FilteredAnchorPanel:
    """Run a bootstrap filter and draw post-feedback states at each anchor."""

    q, y, r, p0, decoded = _validate_inputs(
        q_values, choices, feedback, prior, kernels, capacity, model_id, parameters
    )
    n_trials, n_hypotheses, _ = q.shape
    anchor_values = sorted({int(value) for value in anchors})
    if not anchor_values or anchor_values[0] < 0 or anchor_values[-1] >= n_trials:
        raise ValueError("anchors must be non-empty valid zero-based trial indices")
    n_particles = int(particle_count)
    n_samples = int(sample_count)
    threshold = float(resample_threshold_fraction)
    if n_particles < 2 or n_samples < 1:
        raise ValueError("particle_count must be >= 2 and sample_count positive")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("resample_threshold_fraction must lie in [0, 1]")

    rng = np.random.default_rng(int(filter_seed))
    candidates = np.arange(n_hypotheses, dtype=int)
    states = [
        _initial_state(
            _weighted_wor_from_uniforms(
                candidates,
                p0,
                int(capacity),
                rng.random(int(capacity)),
            ),
            p0,
        )
        for _ in range(n_particles)
    ]
    weights = np.full(n_particles, 1.0 / float(n_particles), dtype=float)
    ess = np.full(n_trials, np.nan, dtype=float)
    resampled = np.zeros(n_trials, dtype=bool)
    states_by_anchor: dict[int, list[HFWState]] = {}
    transition_dimension = _transition_uniform_dimension(decoded, int(capacity))

    for trial_index in range(anchor_values[-1] + 1):
        if trial_index > 0:
            transition_unit = rng.random((n_particles, transition_dimension))
            states = [
                _sample_transition(
                    state,
                    p0,
                    kernels,
                    decoded,
                    int(capacity),
                    transition_unit[particle_index],
                )[0]
                for particle_index, state in enumerate(states)
            ]
        observed_probability = np.asarray(
            [
                _choice_probability(
                    state,
                    q[trial_index],
                    decoded.kappa,
                    decoded.lapse,
                )[y[trial_index]]
                for state in states
            ],
            dtype=float,
        )
        weights *= np.clip(observed_probability, float(epsilon), None)
        weights /= float(weights.sum())
        ess[trial_index] = effective_sample_size(weights)
        if ess[trial_index] < threshold * n_particles:
            indices = systematic_resample(weights, float(rng.random()))
            states = [states[int(index)].copy() for index in indices]
            weights.fill(1.0 / float(n_particles))
            resampled[trial_index] = True
        states = [
            _feedback_update(
                state,
                q[trial_index],
                int(y[trial_index]),
                float(r[trial_index]),
                decoded,
                float(epsilon),
            )[0]
            for state in states
        ]
        if trial_index in anchor_values:
            selected = rng.choice(
                n_particles,
                size=n_samples,
                replace=True,
                p=weights,
            )
            states_by_anchor[trial_index] = [
                states[int(index)].copy() for index in selected
            ]

    return FilteredAnchorPanel(
        states_by_anchor=states_by_anchor,
        pre_resampling_ess=ess,
        resampled=resampled,
        particle_count=n_particles,
        sample_count=n_samples,
        filter_seed=int(filter_seed),
    )


def _state_distances(
    left: HFWState,
    right: HFWState,
    q_trial: np.ndarray,
    parameters: Model0804Parameters,
    capacity: int,
) -> tuple[float, bool, float, float, float, float]:
    overlap = np.intersect1d(left.active, right.active, assume_unique=True)
    active_distance = 1.0 - overlap.size / float(capacity)
    active_equal = bool(
        np.array_equal(np.sort(left.active), np.sort(right.active))
    )
    omega_tv = 0.5 * float(np.sum(np.abs(left.omega - right.omega)))
    left_choice = _choice_probability(
        left, q_trial, parameters.kappa, parameters.lapse
    )
    right_choice = _choice_probability(
        right, q_trial, parameters.kappa, parameters.lapse
    )
    choice_difference = float(np.max(np.abs(left_choice - right_choice)))
    signed_choice_difference = float(left_choice[1] - right_choice[1])
    if overlap.size:
        left_delta = left.static[overlap] - left.fade[overlap]
        right_delta = right.static[overlap] - right.fade[overlap]
        memory_difference = float(np.mean(np.abs(left_delta - right_delta)))
    else:
        memory_difference = float("nan")
    return (
        active_distance,
        active_equal,
        omega_tv,
        choice_difference,
        signed_choice_difference,
        memory_difference,
    )


def couple_model0804_histories(
    left_states: list[HFWState] | tuple[HFWState, ...],
    right_states: list[HFWState] | tuple[HFWState, ...],
    q_values: np.ndarray,
    choices: np.ndarray,
    feedback: np.ndarray,
    prior: np.ndarray,
    kernels: TransitionKernels,
    *,
    model_id: str,
    parameters: Model0804Parameters,
    capacity: int,
    anchor_trial: int,
    horizon: int,
    coupling_seed: int,
    epsilon: float = EPS,
) -> CoupledForgettingTrace:
    """Propagate paired anchor states with identical future random inputs."""

    q, y, r, p0, decoded = _validate_inputs(
        q_values, choices, feedback, prior, kernels, capacity, model_id, parameters
    )
    left_input = list(left_states)
    right_input = list(right_states)
    if len(left_input) != len(right_input) or len(left_input) < 1:
        raise ValueError("left_states and right_states must have equal positive size")
    anchor = int(anchor_trial)
    lag_count = int(horizon)
    if anchor < 0 or lag_count < 1 or anchor + lag_count >= q.shape[0]:
        raise ValueError("anchor_trial plus horizon must stay inside the data")
    pair_count = len(left_input)
    left = [state.copy() for state in left_input]
    right = [state.copy() for state in right_input]
    active_distance = np.zeros((pair_count, lag_count), dtype=float)
    active_equal = np.zeros((pair_count, lag_count), dtype=bool)
    state_exact_equal = np.zeros((pair_count, lag_count), dtype=bool)
    regenerated = np.zeros((pair_count, lag_count), dtype=bool)
    omega_tv = np.zeros((pair_count, lag_count), dtype=float)
    choice_difference = np.zeros((pair_count, lag_count), dtype=float)
    signed_choice_difference = np.zeros((pair_count, lag_count), dtype=float)
    memory_difference = np.full((pair_count, lag_count), np.nan, dtype=float)
    rng = np.random.default_rng(int(coupling_seed))
    transition_dimension = _transition_uniform_dimension(decoded, int(capacity))
    innovations = rng.random((pair_count, lag_count, transition_dimension))

    for lag_index in range(lag_count):
        trial_index = anchor + lag_index + 1
        for pair_index in range(pair_count):
            left_next, left_summary, _ = _sample_transition(
                left[pair_index],
                p0,
                kernels,
                decoded,
                int(capacity),
                innovations[pair_index, lag_index],
            )
            right_next, right_summary, _ = _sample_transition(
                right[pair_index],
                p0,
                kernels,
                decoded,
                int(capacity),
                innovations[pair_index, lag_index],
            )
            if left_summary.regenerated != right_summary.regenerated:
                raise AssertionError("common reset innovation produced unequal events")
            regenerated[pair_index, lag_index] = left_summary.regenerated
            left[pair_index] = left_next
            right[pair_index] = right_next
            state_exact_equal[pair_index, lag_index] = bool(
                np.array_equal(left_next.active, right_next.active)
                and np.array_equal(left_next.omega, right_next.omega)
                and np.array_equal(left_next.fade, right_next.fade)
                and np.array_equal(left_next.static, right_next.static)
            )
            (
                active_distance[pair_index, lag_index],
                active_equal[pair_index, lag_index],
                omega_tv[pair_index, lag_index],
                choice_difference[pair_index, lag_index],
                signed_choice_difference[pair_index, lag_index],
                memory_difference[pair_index, lag_index],
            ) = _state_distances(
                left[pair_index],
                right[pair_index],
                q[trial_index],
                decoded,
                int(capacity),
            )
            left[pair_index] = _feedback_update(
                left[pair_index],
                q[trial_index],
                int(y[trial_index]),
                float(r[trial_index]),
                decoded,
                float(epsilon),
            )[0]
            right[pair_index] = _feedback_update(
                right[pair_index],
                q[trial_index],
                int(y[trial_index]),
                float(r[trial_index]),
                decoded,
                float(epsilon),
            )[0]

    ever_regenerated = np.maximum.accumulate(regenerated, axis=1)
    if np.any(ever_regenerated & ~state_exact_equal):
        raise AssertionError("states diverged after a common regeneration")
    return CoupledForgettingTrace(
        anchor_trial=anchor,
        trial_indices=np.arange(anchor + 1, anchor + lag_count + 1),
        active_distance=active_distance,
        active_equal=active_equal,
        state_exact_equal=state_exact_equal,
        regenerated=regenerated,
        ever_regenerated=ever_regenerated,
        omega_total_variation=omega_tv,
        choice_probability_difference=choice_difference,
        signed_choice_probability_difference=signed_choice_difference,
        common_memory_delta_difference=memory_difference,
        pair_count=pair_count,
        horizon=lag_count,
        coupling_seed=int(coupling_seed),
    )

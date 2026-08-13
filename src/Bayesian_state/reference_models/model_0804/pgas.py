"""Innovation-space PGAS diagnostics for the finite-workspace model_0804.

PGAS samples latent transition histories conditional on observed choices.  It
does not estimate the marginal likelihood; the normalizing-constant audit must
remain a separate particle-filter calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

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
    _stable_seed,
    _trial_masks,
    _validate_inputs,
    _weighted_wor_from_uniforms,
    enumerate_initial_states,
    enumerate_transition_outcomes,
)


@dataclass
class Model0804InnovationPath:
    """Uniform innovations that deterministically generate one HFW path."""

    initial_unit: np.ndarray
    transition_unit: np.ndarray

    def copy(self) -> "Model0804InnovationPath":
        return Model0804InnovationPath(
            initial_unit=self.initial_unit.copy(),
            transition_unit=self.transition_unit.copy(),
        )


@dataclass
class Model0804InnovationReplay:
    """Deterministic cognitive trajectory induced by one innovation path."""

    active: np.ndarray
    replacement_count: np.ndarray
    observed_choice_probability: np.ndarray
    log_choice_likelihood: float


@dataclass
class ExactModel0804Smoothing:
    """Exact tiny-space posterior path marginals used as a PGAS oracle."""

    nll: float
    active_probability: np.ndarray
    expected_replacement_count: np.ndarray
    path_count: int


@dataclass
class Model0804PGASTrace:
    """Posterior history summaries and PGAS mixing diagnostics."""

    active_probability: np.ndarray
    expected_replacement_count: np.ndarray
    retained_active_samples: np.ndarray
    retained_replacement_samples: np.ndarray
    iteration_log_choice_likelihood: np.ndarray
    iteration_path_change_fraction: np.ndarray
    iteration_ancestor_switch_fraction: np.ndarray
    iteration_minimum_ancestor_ess: np.ndarray
    iteration_trial_active_change_fraction: np.ndarray
    iteration_trial_ancestor_switched: np.ndarray
    iteration_trial_ancestor_ess: np.ndarray
    particle_count: int
    iterations: int
    burn_in: int
    thin: int
    ancestor_lookahead: int | None
    retained_samples: int
    normalizing_constant_estimated: bool = False


def _normalize_log_weights(log_weights: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(log_weights, dtype=float).reshape(-1)
    if values.size == 0 or np.all(np.isneginf(values)) or np.any(np.isnan(values)):
        raise ValueError(f"{name} has no finite mass")
    maximum = float(np.max(values))
    weights = np.exp(values - maximum)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"{name} cannot be normalized")
    return weights / total


def _sample_index(probabilities: np.ndarray, rng: np.random.Generator) -> int:
    cumulative = np.cumsum(np.asarray(probabilities, dtype=float))
    cumulative[-1] = 1.0
    return int(np.searchsorted(cumulative, float(rng.random()), side="right"))


def draw_model0804_innovation_path(
    n_trials: int,
    capacity: int,
    rng: np.random.Generator,
) -> Model0804InnovationPath:
    """Draw one complete path from the innovation prior."""

    trials = int(n_trials)
    workspace = int(capacity)
    if trials < 1 or workspace < 1:
        raise ValueError("n_trials and capacity must be positive")
    return Model0804InnovationPath(
        initial_unit=rng.random(workspace),
        transition_unit=rng.random(
            (trials - 1, 1 + 2 * workspace)
        ),
    )


def replay_model0804_innovation_path(
    path: Model0804InnovationPath,
    q_values: np.ndarray,
    choices: np.ndarray,
    feedback: np.ndarray,
    prior: np.ndarray,
    kernels: TransitionKernels,
    *,
    model_id: str,
    parameters: Model0804Parameters,
    capacity: int,
    condition_on_choice_mask: np.ndarray | None = None,
    epsilon: float = EPS,
) -> Model0804InnovationReplay:
    """Replay a complete innovation path in strict choice-before-feedback order."""

    q, y, r, p0, decoded = _validate_inputs(
        q_values, choices, feedback, prior, kernels, capacity, model_id, parameters
    )
    n_trials, n_hypotheses, _ = q.shape
    _, condition = _trial_masks(
        n_trials,
        np.zeros(n_trials, dtype=bool),
        condition_on_choice_mask,
    )
    initial_unit = np.asarray(path.initial_unit, dtype=float).reshape(-1)
    transition_unit = np.asarray(path.transition_unit, dtype=float)
    if initial_unit.shape != (int(capacity),):
        raise ValueError("innovation initial_unit has the wrong shape")
    if transition_unit.shape != (
        n_trials - 1,
        1 + 2 * int(capacity),
    ):
        raise ValueError("innovation transition_unit has the wrong shape")
    if np.any((initial_unit < 0.0) | (initial_unit >= 1.0)) or np.any(
        (transition_unit < 0.0) | (transition_unit >= 1.0)
    ):
        raise ValueError("innovation values must lie in [0, 1)")

    entered = _weighted_wor_from_uniforms(
        np.arange(n_hypotheses, dtype=int),
        p0,
        int(capacity),
        initial_unit,
    )
    state = _initial_state(entered, p0)
    active = np.zeros((n_trials, n_hypotheses), dtype=bool)
    replacement_count = np.zeros(n_trials, dtype=int)
    observed_probability = np.ones(n_trials, dtype=float)
    log_likelihood = 0.0
    for trial_index in range(n_trials):
        if trial_index > 0:
            state, summary, _ = _sample_transition(
                state,
                p0,
                kernels,
                decoded,
                int(capacity),
                transition_unit[trial_index - 1],
            )
            replacement_count[trial_index] = summary.replacement_count
        active[trial_index, state.active] = True
        probability = _choice_probability(
            state,
            q[trial_index],
            decoded.kappa,
            decoded.lapse,
        )
        observed_probability[trial_index] = float(
            probability[y[trial_index]]
        )
        if condition[trial_index]:
            log_likelihood += math.log(
                max(observed_probability[trial_index], float(epsilon))
            )
        state, _, _ = _feedback_update(
            state,
            q[trial_index],
            int(y[trial_index]),
            float(r[trial_index]),
            decoded,
            float(epsilon),
        )
    return Model0804InnovationReplay(
        active=active,
        replacement_count=replacement_count,
        observed_choice_probability=observed_probability,
        log_choice_likelihood=float(log_likelihood),
    )


def run_model0804_exact_smoothing(
    q_values: np.ndarray,
    choices: np.ndarray,
    feedback: np.ndarray,
    prior: np.ndarray,
    kernels: TransitionKernels,
    *,
    model_id: str,
    parameters: Model0804Parameters,
    capacity: int,
    condition_on_choice_mask: np.ndarray | None = None,
    epsilon: float = EPS,
    max_paths: int = 250_000,
) -> ExactModel0804Smoothing:
    """Enumerate the exact posterior over tiny HFW trajectories."""

    q, y, r, p0, decoded = _validate_inputs(
        q_values, choices, feedback, prior, kernels, capacity, model_id, parameters
    )
    n_trials, n_hypotheses, _ = q.shape
    _, condition = _trial_masks(
        n_trials,
        np.zeros(n_trials, dtype=bool),
        condition_on_choice_mask,
    )
    branches: list[tuple[HFWState, float, np.ndarray, np.ndarray]] = []
    for state, initial_probability in enumerate_initial_states(p0, int(capacity)):
        prediction = _choice_probability(
            state, q[0], decoded.kappa, decoded.lapse
        )
        weight = float(initial_probability)
        if condition[0]:
            weight *= float(prediction[y[0]])
        active_history = np.zeros((n_trials, n_hypotheses), dtype=bool)
        active_history[0, state.active] = True
        replacement_history = np.zeros(n_trials, dtype=int)
        updated, _, _ = _feedback_update(
            state, q[0], int(y[0]), float(r[0]), decoded, float(epsilon)
        )
        branches.append(
            (updated, weight, active_history, replacement_history)
        )

    for trial_index in range(1, n_trials):
        expanded: list[tuple[HFWState, float, np.ndarray, np.ndarray]] = []
        for state, branch_weight, active_history, replacement_history in branches:
            for new_state, summary, _, transition_probability in enumerate_transition_outcomes(
                state, p0, kernels, decoded, int(capacity)
            ):
                prediction = _choice_probability(
                    new_state,
                    q[trial_index],
                    decoded.kappa,
                    decoded.lapse,
                )
                weight = branch_weight * float(transition_probability)
                if condition[trial_index]:
                    weight *= float(prediction[y[trial_index]])
                new_active_history = active_history.copy()
                new_active_history[trial_index, new_state.active] = True
                new_replacement_history = replacement_history.copy()
                new_replacement_history[trial_index] = summary.replacement_count
                updated, _, _ = _feedback_update(
                    new_state,
                    q[trial_index],
                    int(y[trial_index]),
                    float(r[trial_index]),
                    decoded,
                    float(epsilon),
                )
                expanded.append(
                    (
                        updated,
                        weight,
                        new_active_history,
                        new_replacement_history,
                    )
                )
                if len(expanded) > int(max_paths):
                    raise RuntimeError("exact smoothing path limit exceeded")
        branches = expanded

    evidence = float(sum(item[1] for item in branches))
    if not np.isfinite(evidence) or evidence <= 0.0:
        raise RuntimeError("exact smoothing evidence is zero")
    active_probability = np.zeros((n_trials, n_hypotheses), dtype=float)
    expected_replacement = np.zeros(n_trials, dtype=float)
    for _, weight, active_history, replacement_history in branches:
        posterior_weight = float(weight) / evidence
        active_probability += posterior_weight * active_history
        expected_replacement += posterior_weight * replacement_history
    return ExactModel0804Smoothing(
        nll=float(-math.log(evidence)),
        active_probability=active_probability,
        expected_replacement_count=expected_replacement,
        path_count=len(branches),
    )


def _suffix_log_likelihood(
    state: HFWState,
    reference: Model0804InnovationPath,
    start_trial: int,
    stop_trial: int,
    q: np.ndarray,
    y: np.ndarray,
    r: np.ndarray,
    p0: np.ndarray,
    kernels: TransitionKernels,
    parameters: Model0804Parameters,
    capacity: int,
    condition: np.ndarray,
    epsilon: float,
) -> float:
    """Replay a reference innovation suffix from one candidate ancestor."""

    current = state.copy()
    total = 0.0
    for trial_index in range(int(start_trial), int(stop_trial) + 1):
        current, _, _ = _sample_transition(
            current,
            p0,
            kernels,
            parameters,
            int(capacity),
            reference.transition_unit[trial_index - 1],
        )
        prediction = _choice_probability(
            current,
            q[trial_index],
            parameters.kappa,
            parameters.lapse,
        )
        if condition[trial_index]:
            total += math.log(
                max(float(prediction[y[trial_index]]), float(epsilon))
            )
        current, _, _ = _feedback_update(
            current,
            q[trial_index],
            int(y[trial_index]),
            float(r[trial_index]),
            parameters,
            float(epsilon),
        )
    return float(total)


def _conditional_smc_pgas(
    reference: Model0804InnovationPath,
    q: np.ndarray,
    y: np.ndarray,
    r: np.ndarray,
    p0: np.ndarray,
    kernels: TransitionKernels,
    parameters: Model0804Parameters,
    *,
    capacity: int,
    particle_count: int,
    condition: np.ndarray,
    ancestor_lookahead: int | None,
    rng: np.random.Generator,
    epsilon: float,
) -> tuple[
    Model0804InnovationPath,
    float,
    float,
    np.ndarray,
    np.ndarray,
]:
    """Run one innovation-space conditional SMC sweep with ancestor sampling."""

    n_trials, n_hypotheses, _ = q.shape
    n_particles = int(particle_count)
    transition_dimension = 1 + 2 * int(capacity)
    initial_paths = np.empty((n_particles, int(capacity)), dtype=float)
    transition_paths = np.zeros(
        (n_particles, n_trials - 1, transition_dimension), dtype=float
    )
    initial_paths[:-1] = rng.random((n_particles - 1, int(capacity)))
    initial_paths[-1] = reference.initial_unit
    states: list[HFWState] = []
    log_weights = np.zeros(n_particles, dtype=float)
    for particle_index in range(n_particles):
        entered = _weighted_wor_from_uniforms(
            np.arange(n_hypotheses, dtype=int),
            p0,
            int(capacity),
            initial_paths[particle_index],
        )
        state = _initial_state(entered, p0)
        prediction = _choice_probability(
            state, q[0], parameters.kappa, parameters.lapse
        )
        if condition[0]:
            log_weights[particle_index] = math.log(
                max(float(prediction[y[0]]), float(epsilon))
            )
        state, _, _ = _feedback_update(
            state, q[0], int(y[0]), float(r[0]), parameters, float(epsilon)
        )
        states.append(state)

    ancestor_switches = 0
    minimum_ancestor_ess = float(n_particles)
    trial_ancestor_switched = np.zeros(n_trials - 1, dtype=bool)
    trial_ancestor_ess = np.full(n_trials - 1, float(n_particles), dtype=float)
    for trial_index in range(1, n_trials):
        weights = _normalize_log_weights(log_weights, "PGAS particle weights")
        ancestors = np.empty(n_particles, dtype=int)
        ancestors[:-1] = rng.choice(
            n_particles, size=n_particles - 1, replace=True, p=weights
        )
        if ancestor_lookahead is None:
            suffix_stop = n_trials - 1
        else:
            suffix_stop = min(
                n_trials - 1,
                trial_index + int(ancestor_lookahead) - 1,
            )
        log_ancestor = np.log(np.clip(weights, float(epsilon), 1.0))
        for particle_index, state in enumerate(states):
            log_ancestor[particle_index] += _suffix_log_likelihood(
                state,
                reference,
                trial_index,
                suffix_stop,
                q,
                y,
                r,
                p0,
                kernels,
                parameters,
                int(capacity),
                condition,
                float(epsilon),
            )
        ancestor_probabilities = _normalize_log_weights(
            log_ancestor, "PGAS ancestor weights"
        )
        current_ancestor_ess = 1.0 / float(
            np.sum(np.square(ancestor_probabilities))
        )
        trial_ancestor_ess[trial_index - 1] = current_ancestor_ess
        minimum_ancestor_ess = min(minimum_ancestor_ess, current_ancestor_ess)
        ancestors[-1] = _sample_index(ancestor_probabilities, rng)
        if ancestors[-1] != n_particles - 1:
            ancestor_switches += 1
            trial_ancestor_switched[trial_index - 1] = True

        new_initial_paths = initial_paths[ancestors].copy()
        new_transition_paths = transition_paths[ancestors].copy()
        current_unit = np.empty(
            (n_particles, transition_dimension), dtype=float
        )
        current_unit[:-1] = rng.random(
            (n_particles - 1, transition_dimension)
        )
        current_unit[-1] = reference.transition_unit[trial_index - 1]
        new_transition_paths[:, trial_index - 1] = current_unit
        new_states: list[HFWState] = []
        new_log_weights = np.zeros(n_particles, dtype=float)
        for particle_index in range(n_particles):
            state, _, _ = _sample_transition(
                states[int(ancestors[particle_index])],
                p0,
                kernels,
                parameters,
                int(capacity),
                current_unit[particle_index],
            )
            prediction = _choice_probability(
                state,
                q[trial_index],
                parameters.kappa,
                parameters.lapse,
            )
            if condition[trial_index]:
                new_log_weights[particle_index] = math.log(
                    max(
                        float(prediction[y[trial_index]]),
                        float(epsilon),
                    )
                )
            state, _, _ = _feedback_update(
                state,
                q[trial_index],
                int(y[trial_index]),
                float(r[trial_index]),
                parameters,
                float(epsilon),
            )
            new_states.append(state)
        initial_paths = new_initial_paths
        transition_paths = new_transition_paths
        states = new_states
        log_weights = new_log_weights

    final_weights = _normalize_log_weights(log_weights, "PGAS final weights")
    selected = _sample_index(final_weights, rng)
    path = Model0804InnovationPath(
        initial_unit=initial_paths[selected].copy(),
        transition_unit=transition_paths[selected].copy(),
    )
    return (
        path,
        ancestor_switches / float(max(n_trials - 1, 1)),
        float(minimum_ancestor_ess),
        trial_ancestor_switched,
        trial_ancestor_ess,
    )


def run_model0804_pgas(
    q_values: np.ndarray,
    choices: np.ndarray,
    feedback: np.ndarray,
    prior: np.ndarray,
    kernels: TransitionKernels,
    *,
    model_id: str,
    parameters: Model0804Parameters,
    capacity: int,
    particle_count: int = 16,
    iterations: int = 500,
    burn_in: int = 100,
    thin: int = 1,
    ancestor_lookahead: int | None = None,
    chain_seed: int = 20260804,
    condition_on_choice_mask: np.ndarray | None = None,
    epsilon: float = EPS,
) -> Model0804PGASTrace:
    """Sample the full innovation-path posterior with PGAS.

    ``ancestor_lookahead=None`` uses the exact full reference suffix.  A finite
    value is the explicitly approximate non-Markovian truncation described by
    Lindsten, Jordan, and Schoen (2014).
    """

    q, y, r, p0, decoded = _validate_inputs(
        q_values, choices, feedback, prior, kernels, capacity, model_id, parameters
    )
    if model_id not in {"FA1", "FA2"}:
        raise ValueError("PGAS is restricted to dynamic FA1/FA2")
    n_trials, _, _ = q.shape
    n_particles = int(particle_count)
    n_iterations = int(iterations)
    burn = int(burn_in)
    thinning = int(thin)
    if n_particles < 2:
        raise ValueError("PGAS particle_count must be at least 2")
    if n_iterations < 1 or not 0 <= burn < n_iterations:
        raise ValueError("PGAS burn_in must lie in [0, iterations)")
    if thinning < 1:
        raise ValueError("PGAS thin must be positive")
    if ancestor_lookahead is not None and int(ancestor_lookahead) < 1:
        raise ValueError("ancestor_lookahead must be positive or None")
    _, condition = _trial_masks(
        n_trials,
        np.zeros(n_trials, dtype=bool),
        condition_on_choice_mask,
    )

    reference = draw_model0804_innovation_path(
        n_trials,
        int(capacity),
        np.random.default_rng(_stable_seed(chain_seed, "pgas_initial")),
    )
    log_likelihood = np.zeros(n_iterations, dtype=float)
    path_change = np.zeros(n_iterations, dtype=float)
    ancestor_switch = np.zeros(n_iterations, dtype=float)
    minimum_ancestor_ess = np.zeros(n_iterations, dtype=float)
    trial_active_change = np.zeros((n_iterations, n_trials), dtype=float)
    trial_ancestor_switched = np.zeros(
        (n_iterations, n_trials - 1), dtype=bool
    )
    trial_ancestor_ess = np.zeros(
        (n_iterations, n_trials - 1), dtype=float
    )
    retained_active: list[np.ndarray] = []
    retained_replacement: list[np.ndarray] = []
    previous_replay = replay_model0804_innovation_path(
        reference,
        q,
        y,
        r,
        p0,
        kernels,
        model_id=model_id,
        parameters=decoded,
        capacity=int(capacity),
        condition_on_choice_mask=condition,
        epsilon=float(epsilon),
    )
    for iteration in range(n_iterations):
        rng = np.random.default_rng(
            _stable_seed(chain_seed, "pgas_iteration", iteration)
        )
        (
            reference,
            ancestor_switch[iteration],
            minimum_ancestor_ess[iteration],
            trial_ancestor_switched[iteration],
            trial_ancestor_ess[iteration],
        ) = _conditional_smc_pgas(
                reference,
                q,
                y,
                r,
                p0,
                kernels,
                decoded,
                capacity=int(capacity),
                particle_count=n_particles,
                condition=condition,
                ancestor_lookahead=(
                    None
                    if ancestor_lookahead is None
                    else int(ancestor_lookahead)
                ),
                rng=rng,
                epsilon=float(epsilon),
            )
        replay = replay_model0804_innovation_path(
            reference,
            q,
            y,
            r,
            p0,
            kernels,
            model_id=model_id,
            parameters=decoded,
            capacity=int(capacity),
            condition_on_choice_mask=condition,
            epsilon=float(epsilon),
        )
        log_likelihood[iteration] = replay.log_choice_likelihood
        path_change[iteration] = float(
            np.mean(replay.active != previous_replay.active)
        )
        trial_active_change[iteration] = np.mean(
            replay.active != previous_replay.active, axis=1
        )
        previous_replay = replay
        if iteration >= burn and (iteration - burn) % thinning == 0:
            retained_active.append(replay.active.copy())
            retained_replacement.append(replay.replacement_count.copy())

    active_samples = np.asarray(retained_active, dtype=bool)
    replacement_samples = np.asarray(retained_replacement, dtype=int)
    if active_samples.size == 0:
        raise AssertionError("PGAS retained no posterior samples")
    return Model0804PGASTrace(
        active_probability=np.mean(active_samples, axis=0),
        expected_replacement_count=np.mean(replacement_samples, axis=0),
        retained_active_samples=active_samples,
        retained_replacement_samples=replacement_samples,
        iteration_log_choice_likelihood=log_likelihood,
        iteration_path_change_fraction=path_change,
        iteration_ancestor_switch_fraction=ancestor_switch,
        iteration_minimum_ancestor_ess=minimum_ancestor_ess,
        iteration_trial_active_change_fraction=trial_active_change,
        iteration_trial_ancestor_switched=trial_ancestor_switched,
        iteration_trial_ancestor_ess=trial_ancestor_ess,
        particle_count=n_particles,
        iterations=n_iterations,
        burn_in=burn,
        thin=thinning,
        ancestor_lookahead=(
            None if ancestor_lookahead is None else int(ancestor_lookahead)
        ),
        retained_samples=int(active_samples.shape[0]),
    )

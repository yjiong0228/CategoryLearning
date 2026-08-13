"""Finite-active-set condition-1 models from ``manuscript/model_0804.tex``.

The implementation deliberately covers the first frozen gate only:

* HFW (hard finite workspace), not the reservoir-access variant;
* FA0, FA1, and FA2 with constant controls;
* the model_0806 single-signal FA3-M extensions, enabled explicitly through
  the Model0804Parameters.dynamic_m flag;
* fixed integrated rule predictions ``q[t, h, c]`` from the model_0803 cache;
* mass-conserving multi-slot replacement and synchronized dual memory;
* bootstrap-particle and alive-particle marginal choice likelihoods;
* exact small-space enumeration for implementation validation.

An optional frozen RT emission is available for joint recovery checks.  Oral
reports, autonomous generation, and formal group inference remain outside this
module's current scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import itertools
import math
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.stats import qmc

from ..model_0803 import ParameterDefinition, TransitionKernels


EPS = 1e-12
FA_MODEL_IDS = ("FA0", "FA1", "FA2")
HFW_MODEL_IDS = (*FA_MODEL_IDS, "FA2R")


@dataclass(frozen=True)
class Model0804Parameters:
    """Decoded HFW parameters shared by one subject's particles."""

    gamma: float
    w0: float
    kappa: float
    m: float
    g: float
    lapse: float = 0.0
    rho: float = 0.0
    dynamic_m: bool = False
    m_phi: float = 0.0
    m_beta_surprise: float = 0.0
    surprise_center: float = 0.0
    surprise_scale: float = 1.0
    m_beta_uncertainty: float = 0.0
    uncertainty_center: float = 0.0
    uncertainty_scale: float = 1.0


@dataclass(frozen=True)
class Model0804RTParameters:
    """Frozen pre-choice log-RT emission for recovery and external validation."""

    intercept: float
    choice_entropy: float
    replacement_fraction: float
    newcomer_distance: float = 0.0
    sigma: float = 0.15
    degrees_of_freedom: float = 5.0


@dataclass
class HFWState:
    """One latent hard-workspace state after the previous feedback update."""

    active: np.ndarray
    omega: np.ndarray
    fade: np.ndarray
    static: np.ndarray

    def copy(self) -> "HFWState":
        return HFWState(
            active=self.active.copy(),
            omega=self.omega.copy(),
            fade=self.fade.copy(),
            static=self.static.copy(),
        )


@dataclass(frozen=True)
class TransitionSummary:
    """Realized finite-set change before one choice."""

    replacement_count: int
    removed_mass: float
    newcomer_distance: float
    dropped: tuple[int, ...]
    newcomers: tuple[int, ...]
    regenerated: bool = False


@dataclass
class Model0804Trace:
    """Particle-marginal predictions and numerical diagnostics."""

    nll: float
    probabilities: np.ndarray
    marginal_hypothesis_prior: np.ndarray
    marginal_active_probability: np.ndarray
    predictive_replacement_count: np.ndarray
    predictive_replacement_fraction: np.ndarray
    predictive_removed_mass: np.ndarray
    predictive_newcomer_distance: np.ndarray
    pre_choice_ess: np.ndarray
    post_choice_ess: np.ndarray
    resampled: np.ndarray
    resampling_unique_ancestors: np.ndarray
    memory_sync_error: np.ndarray
    final_weights: np.ndarray
    particle_count: int
    capacity: int
    filter_seed: int
    transition_proposals_per_particle: int
    integration_mode: str
    replacement_count_stratified: bool
    inference_method: str = "bootstrap"
    alive_attempt_count: np.ndarray | None = None
    alive_incremental_likelihood: np.ndarray | None = None
    rejuvenation_window: int = 0
    rejuvenation_sweeps: int = 0
    rejuvenation_acceptance_rate: np.ndarray | None = None
    rejuvenation_unique_active_sets: np.ndarray | None = None
    predictive_m: np.ndarray | None = None
    feedback_surprise: np.ndarray | None = None
    feedback_uncertainty: np.ndarray | None = None
    joint_nll: float | None = None
    rt_conditional_nll: float | None = None
    rt_predictive_log_density: np.ndarray | None = None


@dataclass
class ExactModel0804Trace:
    """Exact small-space filtering output used as a particle-filter oracle."""

    nll: float
    probabilities: np.ndarray
    marginal_hypothesis_prior: np.ndarray
    marginal_active_probability: np.ndarray
    branch_counts: np.ndarray


@dataclass
class Model0804IslandEnsemble:
    """Coherent likelihood-weighted combination of independent alive filters."""

    nll: float
    probabilities: np.ndarray
    incremental_likelihood: np.ndarray
    pretrial_island_weights: np.ndarray
    effective_island_count: np.ndarray
    final_island_log_evidence: np.ndarray
    island_count: int


@dataclass
class Model0804Fit:
    """Best deterministic-CRN multi-start fit for one FA candidate."""

    model_id: str
    memory_id: str
    raw_vector: np.ndarray
    parameters: Model0804Parameters
    reported_parameters: dict[str, float]
    train_nll: float
    diagnostics: dict[str, Any]


def _normalize(values: np.ndarray, name: str = "probability vector") -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must be finite, non-negative, and non-empty")
    total = float(array.sum())
    if total <= 0.0:
        raise ValueError(f"{name} has zero mass")
    return array / total


def effective_sample_size(weights: np.ndarray) -> float:
    normalized = _normalize(weights, "particle weights")
    return 1.0 / float(np.sum(np.square(normalized)))


def systematic_resample(weights: np.ndarray, uniform: float) -> np.ndarray:
    normalized = _normalize(weights, "particle weights")
    value = float(uniform)
    if not np.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError("systematic-resampling uniform must lie in [0, 1)")
    n_particles = normalized.size
    cumulative = np.cumsum(normalized)
    cumulative[-1] = 1.0
    positions = (
        value / float(n_particles)
        + np.arange(n_particles, dtype=float) / float(n_particles)
    )
    return np.searchsorted(cumulative, positions, side="right").astype(int)


def _stable_seed(base_seed: int, *parts: object) -> int:
    joined = ":".join([str(int(base_seed)), *(str(value) for value in parts)])
    digest = hashlib.blake2b(joined.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


def _validate_model_and_parameters(
    model_id: str,
    parameters: Model0804Parameters,
) -> Model0804Parameters:
    if model_id not in HFW_MODEL_IDS:
        raise ValueError(
            f"unknown model_id {model_id!r}; expected one of {HFW_MODEL_IDS}"
        )
    values = {
        "gamma": float(parameters.gamma),
        "w0": float(parameters.w0),
        "kappa": float(parameters.kappa),
        "m": float(parameters.m),
        "g": float(parameters.g),
        "lapse": float(parameters.lapse),
        "rho": float(parameters.rho),
        "dynamic_m": bool(parameters.dynamic_m),
        "m_phi": float(parameters.m_phi),
        "m_beta_surprise": float(parameters.m_beta_surprise),
        "surprise_center": float(parameters.surprise_center),
        "surprise_scale": float(parameters.surprise_scale),
        "m_beta_uncertainty": float(parameters.m_beta_uncertainty),
        "uncertainty_center": float(parameters.uncertainty_center),
        "uncertainty_scale": float(parameters.uncertainty_scale),
    }
    numeric_values = {
        key: value for key, value in values.items() if key != "dynamic_m"
    }
    if not all(np.isfinite(value) for value in numeric_values.values()):
        raise ValueError("all model_0804 parameters must be finite")
    if not 0.0 <= values["gamma"] <= 1.0:
        raise ValueError("gamma must lie in [0, 1]")
    if not 0.0 <= values["w0"] <= 1.0:
        raise ValueError("w0 must lie in [0, 1]")
    if values["kappa"] <= 0.0:
        raise ValueError("kappa must be positive")
    if not 0.0 <= values["m"] <= 1.0:
        raise ValueError("m must lie in [0, 1]")
    if not 0.0 <= values["g"] <= 1.0:
        raise ValueError("g must lie in [0, 1]")
    if not 0.0 <= values["lapse"] < 1.0:
        raise ValueError("lapse must lie in [0, 1)")
    if not 0.0 <= values["rho"] <= 1.0:
        raise ValueError("rho must lie in [0, 1]")
    if not -1.0 < values["m_phi"] < 1.0:
        raise ValueError("m_phi must lie in (-1, 1)")
    if values["m_beta_surprise"] < 0.0:
        raise ValueError("m_beta_surprise must be non-negative")
    if values["m_beta_uncertainty"] < 0.0:
        raise ValueError("m_beta_uncertainty must be non-negative")
    if values["surprise_scale"] <= 0.0:
        raise ValueError("surprise_scale must be positive")
    if values["uncertainty_scale"] <= 0.0:
        raise ValueError("uncertainty_scale must be positive")
    if values["dynamic_m"] and values["rho"] > 0.0:
        raise ValueError("the first FA3-M implementation requires rho=0")
    if values["dynamic_m"] and not 0.0 < values["m"] < 1.0:
        raise ValueError("dynamic_m requires a baseline m strictly between 0 and 1")
    if not values["dynamic_m"]:
        values["m_phi"] = 0.0
        values["m_beta_surprise"] = 0.0
        values["surprise_center"] = 0.0
        values["surprise_scale"] = 1.0
        values["m_beta_uncertainty"] = 0.0
        values["uncertainty_center"] = 0.0
        values["uncertainty_scale"] = 1.0
    if model_id == "FA0":
        values["m"] = 0.0
        values["g"] = 0.0
        values["rho"] = 0.0
    elif model_id == "FA1":
        values["g"] = 0.0
        values["rho"] = 0.0
    elif model_id == "FA2":
        values["rho"] = 0.0
    return Model0804Parameters(**values)


def _validate_inputs(
    q_values: np.ndarray,
    choices: np.ndarray,
    feedback: np.ndarray,
    prior: np.ndarray,
    kernels: TransitionKernels,
    capacity: int,
    model_id: str,
    parameters: Model0804Parameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Model0804Parameters]:
    q = np.asarray(q_values, dtype=float)
    y = np.asarray(choices, dtype=int).reshape(-1)
    r = np.asarray(feedback, dtype=float).reshape(-1)
    p0 = _normalize(np.asarray(prior, dtype=float), "rule prior")
    if np.any(p0 <= 0.0):
        raise ValueError("rule prior must be strictly positive")
    if q.ndim != 3 or q.shape[2] != 2:
        raise ValueError("condition-1 q_values must have shape [trial, hypothesis, 2]")
    if q.shape[0] != y.size or y.size != r.size:
        raise ValueError("q_values, choices, and feedback must have matching trials")
    if q.shape[1] != p0.size:
        raise ValueError("q_values hypothesis count must match the rule prior")
    if not np.all(np.isfinite(q)) or np.any(q < 0.0):
        raise ValueError("q_values must be finite and non-negative")
    if not np.allclose(q.sum(axis=2), 1.0, atol=1e-7, rtol=0.0):
        raise ValueError("q_values must normalize across choice categories")
    if np.any((y < 0) | (y > 1)):
        raise ValueError("condition-1 choices must be zero-based values 0 or 1")
    if np.any(~np.isclose(r, 0.0) & ~np.isclose(r, 1.0)):
        raise ValueError("condition-1 feedback must contain only 0 or 1")
    if kernels.local.shape != (p0.size, p0.size):
        raise ValueError("local kernel does not match the hypothesis space")
    if kernels.distance.shape != (p0.size, p0.size):
        raise ValueError("distance matrix does not match the hypothesis space")
    capacity_value = int(capacity)
    if capacity_value != float(capacity) or not 1 <= capacity_value <= p0.size:
        raise ValueError("capacity must be an integer in [1, hypothesis_count]")
    decoded = _validate_model_and_parameters(model_id, parameters)
    if decoded.m > 0.0 and 2 * capacity_value > p0.size:
        raise ValueError(
            "dynamic no-same-trial-reentry HFW requires 2 * capacity <= hypothesis_count"
        )
    return q, y, r, p0, decoded


def _mask_from_active(active: np.ndarray, size: int) -> np.ndarray:
    mask = np.zeros(int(size), dtype=bool)
    mask[np.asarray(active, dtype=int)] = True
    return mask


def _weighted_wor_sample(
    candidates: np.ndarray,
    weights: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    return _weighted_wor_from_uniforms(
        candidates,
        weights,
        count,
        rng.random(int(count)),
    )


def _weighted_wor_from_uniforms(
    candidates: np.ndarray,
    weights: np.ndarray,
    count: int,
    uniforms: Sequence[float] | np.ndarray,
) -> np.ndarray:
    remaining = np.asarray(candidates, dtype=int).reshape(-1).copy()
    remaining_weights = np.asarray(weights, dtype=float).reshape(-1).copy()
    k = int(count)
    unit_values = np.asarray(uniforms, dtype=float).reshape(-1)
    if remaining.size != remaining_weights.size:
        raise ValueError("weighted-WOR candidates and weights must have equal lengths")
    if not 0 <= k <= remaining.size:
        raise ValueError("weighted-WOR count exceeds the candidate pool")
    if unit_values.size < k or np.any(~np.isfinite(unit_values[:k])) or np.any(
        (unit_values[:k] < 0.0) | (unit_values[:k] >= 1.0)
    ):
        raise ValueError("weighted-WOR requires count uniforms in [0, 1)")
    selected = np.empty(k, dtype=int)
    for index in range(k):
        probabilities = _normalize(remaining_weights, "weighted-WOR weights")
        cumulative = np.cumsum(probabilities)
        cumulative[-1] = 1.0
        chosen_position = int(
            np.searchsorted(cumulative, float(unit_values[index]), side="right")
        )
        selected[index] = int(remaining[chosen_position])
        remaining = np.delete(remaining, chosen_position)
        remaining_weights = np.delete(remaining_weights, chosen_position)
    return selected


def enumerate_weighted_wor(
    candidates: Sequence[int] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
    count: int,
) -> list[tuple[tuple[int, ...], float]]:
    """Enumerate ordered weighted-without-replacement outcomes.

    This helper is exponential and is intended only for tiny-space tests.
    """

    candidate_array = np.asarray(candidates, dtype=int).reshape(-1)
    weight_array = np.asarray(weights, dtype=float).reshape(-1)
    k = int(count)
    if candidate_array.size != weight_array.size:
        raise ValueError("enumeration candidates and weights must have equal lengths")
    if np.unique(candidate_array).size != candidate_array.size:
        raise ValueError("enumeration candidates must be unique")
    if not 0 <= k <= candidate_array.size:
        raise ValueError("enumeration count exceeds the candidate pool")
    if k == 0:
        return [(tuple(), 1.0)]

    outcomes: list[tuple[tuple[int, ...], float]] = []

    def visit(
        remaining_candidates: np.ndarray,
        remaining_weights: np.ndarray,
        prefix: tuple[int, ...],
        probability: float,
    ) -> None:
        if len(prefix) == k:
            outcomes.append((prefix, float(probability)))
            return
        probabilities = _normalize(remaining_weights, "enumeration weights")
        for position, item in enumerate(remaining_candidates):
            visit(
                np.delete(remaining_candidates, position),
                np.delete(remaining_weights, position),
                (*prefix, int(item)),
                probability * float(probabilities[position]),
            )

    visit(candidate_array.copy(), weight_array.copy(), tuple(), 1.0)
    if not np.isclose(sum(probability for _, probability in outcomes), 1.0, atol=1e-12):
        raise AssertionError("weighted-WOR enumeration does not normalize")
    return outcomes


def _initial_state(active: np.ndarray, prior: np.ndarray) -> HFWState:
    active_sorted = np.sort(np.asarray(active, dtype=int).reshape(-1))
    if np.unique(active_sorted).size != active_sorted.size:
        raise ValueError("initial active set contains duplicates")
    omega = np.zeros(prior.size, dtype=float)
    omega[active_sorted] = _normalize(prior[active_sorted], "active initial prior")
    fade = np.full(prior.size, -np.inf, dtype=float)
    static = np.full(prior.size, -np.inf, dtype=float)
    fade[active_sorted] = np.log(omega[active_sorted])
    static[active_sorted] = np.log(omega[active_sorted])
    return HFWState(active=active_sorted, omega=omega, fade=fade, static=static)


def _sample_initial_state(
    prior: np.ndarray,
    capacity: int,
    seed: int,
) -> HFWState:
    candidates = np.arange(prior.size, dtype=int)
    rng = np.random.default_rng(int(seed))
    active = _weighted_wor_sample(candidates, prior, int(capacity), rng)
    return _initial_state(active, prior)


def _sobol_unit_points(count: int, dimension: int, seed: int) -> np.ndarray:
    n = int(count)
    d = int(dimension)
    if n < 1 or d < 1:
        raise ValueError("Sobol count and dimension must be positive")
    exponent = int(math.ceil(math.log2(n))) if n > 1 else 0
    sampler = qmc.Sobol(d=d, scramble=True, seed=int(seed))
    return np.asarray(sampler.random_base2(exponent)[:n], dtype=float)


def _sample_initial_states_qmc(
    prior: np.ndarray,
    capacity: int,
    particle_count: int,
    filter_seed: int,
) -> list[HFWState]:
    unit = _sobol_unit_points(
        int(particle_count),
        int(capacity),
        _stable_seed(filter_seed, "initial_sobol"),
    )
    candidates = np.arange(prior.size, dtype=int)
    return [
        _initial_state(
            _weighted_wor_from_uniforms(
                candidates,
                prior,
                int(capacity),
                unit[particle_index],
            ),
            prior,
        )
        for particle_index in range(int(particle_count))
    ]


def _sample_initial_indices_qmc(
    prior: np.ndarray,
    capacity: int,
    particle_count: int,
    filter_seed: int,
) -> np.ndarray:
    """Vectorized sequential PPS-WOR initial sets for large FA0 runs."""

    n_particles = int(particle_count)
    n_hypotheses = int(prior.size)
    unit = _sobol_unit_points(
        n_particles,
        int(capacity),
        _stable_seed(filter_seed, "initial_sobol"),
    )
    available = np.ones((n_particles, n_hypotheses), dtype=bool)
    selected = np.empty((n_particles, int(capacity)), dtype=np.int32)
    rows = np.arange(n_particles)
    for slot in range(int(capacity)):
        weights = available * prior[None, :]
        weights /= weights.sum(axis=1, keepdims=True)
        cumulative = np.cumsum(weights, axis=1)
        cumulative[:, -1] = 1.0
        positions = np.sum(unit[:, slot, None] >= cumulative, axis=1)
        selected[:, slot] = positions.astype(np.int32)
        available[rows, positions] = False
    return selected


@lru_cache(maxsize=None)
def _initial_indices_exact(
    n_hypotheses: int,
    capacity: int,
) -> np.ndarray:
    """Enumerate all unordered initial sets."""

    total = math.comb(int(n_hypotheses), int(capacity))
    flattened = np.fromiter(
        (
            hypothesis
            for active_set in itertools.combinations(
                range(int(n_hypotheses)), int(capacity)
            )
            for hypothesis in active_set
        ),
        dtype=np.int32,
        count=total * int(capacity),
    )
    active = flattened.reshape(total, int(capacity))
    active.setflags(write=False)
    return active


@lru_cache(maxsize=8)
def _successive_wor_set_probabilities_exact(
    prior_values: tuple[float, ...],
    capacity: int,
) -> np.ndarray:
    """Exact unordered-set probabilities under sequential weighted WOR.

    A 2^M dynamic program sums all M! draw orders for each set without
    materializing the permutations.  Batching bounds temporary memory for the
    38-choose-5 condition-1 state space.
    """

    prior = _normalize(np.asarray(prior_values, dtype=float), "cached rule prior")
    active = _initial_indices_exact(prior.size, int(capacity))
    probabilities = np.empty(active.shape[0], dtype=float)
    final_mask = (1 << int(capacity)) - 1
    batch_size = 100_000
    for start in range(0, active.shape[0], batch_size):
        stop = min(start + batch_size, active.shape[0])
        selected_weights = prior[active[start:stop]]
        dynamic = np.zeros((stop - start, final_mask + 1), dtype=float)
        dynamic[:, 0] = 1.0
        for mask in range(final_mask):
            included = [
                slot for slot in range(int(capacity)) if mask & (1 << slot)
            ]
            if included:
                removed_mass = selected_weights[:, included].sum(axis=1)
            else:
                removed_mass = np.zeros(stop - start, dtype=float)
            denominator = 1.0 - removed_mass
            for slot in range(int(capacity)):
                if mask & (1 << slot):
                    continue
                dynamic[:, mask | (1 << slot)] += (
                    dynamic[:, mask]
                    * selected_weights[:, slot]
                    / denominator
                )
        probabilities[start:stop] = dynamic[:, final_mask]
    probabilities = _normalize(
        probabilities, "exact successive-WOR initial-set probabilities"
    )
    probabilities.setflags(write=False)
    return probabilities


def enumerate_initial_states(
    prior: Sequence[float] | np.ndarray,
    capacity: int,
) -> list[tuple[HFWState, float]]:
    p0 = _normalize(np.asarray(prior, dtype=float), "rule prior")
    ordered = enumerate_weighted_wor(
        np.arange(p0.size, dtype=int), p0, int(capacity)
    )
    probabilities: dict[tuple[int, ...], float] = {}
    for selection, probability in ordered:
        key = tuple(sorted(selection))
        probabilities[key] = probabilities.get(key, 0.0) + float(probability)
    states = [
        (_initial_state(np.asarray(active, dtype=int), p0), probability)
        for active, probability in sorted(probabilities.items())
    ]
    if not np.isclose(sum(probability for _, probability in states), 1.0, atol=1e-12):
        raise AssertionError("initial active-set enumeration does not normalize")
    return states


def _binomial_probabilities(capacity: int, m: float) -> np.ndarray:
    probabilities = np.asarray(
        [
            math.comb(int(capacity), count)
            * float(m) ** count
            * (1.0 - float(m)) ** (int(capacity) - count)
            for count in range(int(capacity) + 1)
        ],
        dtype=float,
    )
    return _normalize(probabilities, "binomial replacement probabilities")


def _newcomer_proposal(
    state: HFWState,
    prior: np.ndarray,
    kernels: TransitionKernels,
    g: float,
) -> tuple[np.ndarray, np.ndarray]:
    active_mask = _mask_from_active(state.active, prior.size)
    inactive = np.flatnonzero(~active_mask)
    local_full = np.asarray(state.omega @ kernels.local, dtype=float)
    local = _normalize(local_full[inactive], "inactive local proposal")
    global_ = _normalize(prior[inactive], "inactive global proposal")
    proposal = _normalize(
        (1.0 - float(g)) * local + float(g) * global_,
        "newcomer mixture",
    )
    return inactive, proposal


def _apply_replacement(
    state: HFWState,
    dropped: Sequence[int],
    newcomers: Sequence[int],
    w0: float,
    distance: np.ndarray,
) -> tuple[HFWState, TransitionSummary, float]:
    dropped_array = np.asarray(dropped, dtype=int).reshape(-1)
    newcomer_array = np.asarray(newcomers, dtype=int).reshape(-1)
    if dropped_array.size != newcomer_array.size:
        raise ValueError("dropped and newcomer sequences must have equal lengths")
    if dropped_array.size == 0:
        active = state.active.copy()
        delta = state.static[active] - state.fade[active]
        log_omega = np.log(np.clip(state.omega[active], EPS, None))
        fade = np.full_like(state.fade, -np.inf)
        static = np.full_like(state.static, -np.inf)
        fade[active] = log_omega - float(w0) * delta
        static[active] = log_omega + (1.0 - float(w0)) * delta
        combined = (
            float(w0) * static[active]
            + (1.0 - float(w0)) * fade[active]
        )
        sync_error = float(np.max(np.abs(combined - log_omega)))
        return (
            HFWState(
                active=active,
                omega=state.omega.copy(),
                fade=fade,
                static=static,
            ),
            TransitionSummary(0, 0.0, 0.0, tuple(), tuple()),
            sync_error,
        )
    active_before = state.active.copy()
    active_mask = _mask_from_active(active_before, state.omega.size)
    if np.any(~active_mask[dropped_array]):
        raise ValueError("all dropped hypotheses must be active")
    if np.any(active_mask[newcomer_array]):
        raise ValueError("newcomers must be inactive before the transition")
    if np.unique(dropped_array).size != dropped_array.size:
        raise ValueError("dropped hypotheses must be unique")
    if np.unique(newcomer_array).size != newcomer_array.size:
        raise ValueError("newcomers must be unique")

    old_omega = state.omega.copy()
    old_delta = np.zeros_like(state.fade)
    old_delta[active_before] = (
        state.static[active_before] - state.fade[active_before]
    )
    retained = active_before[~np.isin(active_before, dropped_array)]
    active_after = np.sort(np.concatenate([retained, newcomer_array]))
    omega = np.zeros_like(old_omega)
    omega[retained] = old_omega[retained]
    for dropped_hypothesis, newcomer in zip(dropped_array, newcomer_array):
        omega[int(newcomer)] = old_omega[int(dropped_hypothesis)]

    fade = np.full_like(state.fade, -np.inf)
    static = np.full_like(state.static, -np.inf)
    log_omega = np.log(np.clip(omega[active_after], EPS, None))
    delta_star = np.zeros(active_after.size, dtype=float)
    retained_mask = np.isin(active_after, retained)
    delta_star[retained_mask] = old_delta[active_after[retained_mask]]
    fade[active_after] = log_omega - float(w0) * delta_star
    static[active_after] = log_omega + (1.0 - float(w0)) * delta_star
    combined = (
        float(w0) * static[active_after]
        + (1.0 - float(w0)) * fade[active_after]
    )
    sync_error = float(np.max(np.abs(combined - log_omega)))

    removed_mass = float(np.sum(old_omega[dropped_array]))
    distance_total = 0.0
    for newcomer in newcomer_array:
        distance_total += float(
            np.sum(old_omega[active_before] * distance[active_before, int(newcomer)])
        )
    newcomer_distance = distance_total / float(newcomer_array.size)
    if active_after.size != active_before.size:
        raise AssertionError("HFW replacement changed active-set capacity")
    if not np.isclose(float(omega.sum()), 1.0, atol=1e-12, rtol=0.0):
        raise AssertionError("HFW replacement did not conserve rule mass")
    return (
        HFWState(active=active_after, omega=omega, fade=fade, static=static),
        TransitionSummary(
            replacement_count=int(dropped_array.size),
            removed_mass=removed_mass,
            newcomer_distance=float(newcomer_distance),
            dropped=tuple(int(value) for value in dropped_array),
            newcomers=tuple(int(value) for value in newcomer_array),
        ),
        sync_error,
    )


def _transition_uniform_dimension(
    parameters: Model0804Parameters,
    capacity: int,
) -> int:
    """Random-map dimension, preserving the FA2 prefix when rho is zero."""

    ordinary = 1 + 2 * int(capacity)
    return ordinary if float(parameters.rho) <= 0.0 else ordinary + 1 + int(capacity)


def _apply_regeneration(
    state: HFWState,
    prior: np.ndarray,
    kernels: TransitionKernels,
    capacity: int,
    uniforms: np.ndarray,
) -> tuple[HFWState, TransitionSummary, float]:
    """Draw a state-independent full workspace and reset rule memory."""

    active = _weighted_wor_from_uniforms(
        np.arange(prior.size, dtype=int),
        prior,
        int(capacity),
        uniforms,
    )
    regenerated = _initial_state(active, prior)
    distance_total = 0.0
    for newcomer in regenerated.active:
        distance_total += float(
            np.sum(
                state.omega[state.active]
                * kernels.distance[state.active, int(newcomer)]
            )
        )
    return (
        regenerated,
        TransitionSummary(
            replacement_count=int(capacity),
            removed_mass=1.0,
            newcomer_distance=distance_total / float(capacity),
            dropped=tuple(int(value) for value in state.active),
            newcomers=tuple(int(value) for value in regenerated.active),
            regenerated=True,
        ),
        0.0,
    )


def _sample_transition(
    state: HFWState,
    prior: np.ndarray,
    kernels: TransitionKernels,
    parameters: Model0804Parameters,
    capacity: int,
    uniforms: np.ndarray,
) -> tuple[HFWState, TransitionSummary, float]:
    unit = np.asarray(uniforms, dtype=float).reshape(-1)
    ordinary_dimension = 1 + 2 * int(capacity)
    required_dimension = _transition_uniform_dimension(parameters, int(capacity))
    if unit.size < required_dimension:
        raise ValueError("transition Sobol row is too short")
    if parameters.rho > 0.0:
        reset_uniform = float(unit[ordinary_dimension])
        if reset_uniform < float(parameters.rho):
            return _apply_regeneration(
                state,
                prior,
                kernels,
                int(capacity),
                unit[
                    ordinary_dimension
                    + 1 : ordinary_dimension
                    + 1
                    + int(capacity)
                ],
            )
    if parameters.m <= 0.0:
        return _apply_replacement(state, (), (), parameters.w0, kernels.distance)
    unit = unit[:ordinary_dimension]
    count_probabilities = _binomial_probabilities(int(capacity), parameters.m)
    count_cumulative = np.cumsum(count_probabilities)
    count_cumulative[-1] = 1.0
    replacement_count = int(
        np.searchsorted(count_cumulative, float(unit[0]), side="right")
    )
    if replacement_count == 0:
        return _apply_replacement(state, (), (), parameters.w0, kernels.distance)
    exit_weights = 1.0 - state.omega[state.active] + 1e-9
    dropped = _weighted_wor_from_uniforms(
        state.active,
        exit_weights,
        replacement_count,
        unit[1 : 1 + int(capacity)],
    )
    inactive, proposal = _newcomer_proposal(state, prior, kernels, parameters.g)
    newcomers = _weighted_wor_from_uniforms(
        inactive,
        proposal,
        replacement_count,
        unit[1 + int(capacity) : 1 + 2 * int(capacity)],
    )
    return _apply_replacement(
        state,
        dropped,
        newcomers,
        parameters.w0,
        kernels.distance,
    )


def enumerate_transition_outcomes(
    state: HFWState,
    prior: Sequence[float] | np.ndarray,
    kernels: TransitionKernels,
    parameters: Model0804Parameters,
    capacity: int,
) -> list[tuple[HFWState, TransitionSummary, float, float]]:
    """Enumerate one HFW transition as ``(state, summary, sync, prob)``."""

    p0 = _normalize(np.asarray(prior, dtype=float), "rule prior")
    decoded = _validate_model_and_parameters(
        "FA2R" if float(parameters.rho) > 0.0 else "FA2", parameters
    )
    if decoded.m > 0.0 and 2 * int(capacity) > p0.size:
        raise ValueError("exact dynamic HFW requires 2 * capacity <= hypothesis_count")
    ordinary_mass = 1.0 - float(decoded.rho)
    outcomes: list[tuple[HFWState, TransitionSummary, float, float]] = []
    if decoded.m <= 0.0:
        new_state, summary, sync_error = _apply_replacement(
            state, (), (), decoded.w0, kernels.distance
        )
        if ordinary_mass > 0.0:
            outcomes.append(
                (new_state, summary, sync_error, float(ordinary_mass))
            )
    else:
        count_probabilities = _binomial_probabilities(int(capacity), decoded.m)
        inactive, proposal = _newcomer_proposal(state, p0, kernels, decoded.g)
        exit_weights = 1.0 - state.omega[state.active] + 1e-9
        for replacement_count, count_probability in enumerate(count_probabilities):
            if count_probability <= 0.0:
                continue
            drop_outcomes = enumerate_weighted_wor(
                state.active, exit_weights, replacement_count
            )
            newcomer_outcomes = enumerate_weighted_wor(
                inactive, proposal, replacement_count
            )
            for dropped, drop_probability in drop_outcomes:
                for newcomers, newcomer_probability in newcomer_outcomes:
                    new_state, summary, sync_error = _apply_replacement(
                        state,
                        dropped,
                        newcomers,
                        decoded.w0,
                        kernels.distance,
                    )
                    probability = ordinary_mass * (
                        float(count_probability)
                        * float(drop_probability)
                        * float(newcomer_probability)
                    )
                    outcomes.append(
                        (new_state, summary, sync_error, probability)
                    )
    if decoded.rho > 0.0:
        for regenerated, reset_probability in enumerate_initial_states(
            p0, int(capacity)
        ):
            distance_total = 0.0
            for newcomer in regenerated.active:
                distance_total += float(
                    np.sum(
                        state.omega[state.active]
                        * kernels.distance[state.active, int(newcomer)]
                    )
                )
            summary = TransitionSummary(
                replacement_count=int(capacity),
                removed_mass=1.0,
                newcomer_distance=distance_total / float(capacity),
                dropped=tuple(int(value) for value in state.active),
                newcomers=tuple(int(value) for value in regenerated.active),
                regenerated=True,
            )
            outcomes.append(
                (
                    regenerated,
                    summary,
                    0.0,
                    float(decoded.rho) * float(reset_probability),
                )
            )
    total = sum(probability for _, _, _, probability in outcomes)
    if not np.isclose(total, 1.0, atol=1e-11, rtol=0.0):
        raise AssertionError(f"enumerated HFW transition has mass {total}")
    return outcomes


def _choice_probability(
    state: HFWState,
    q_trial: np.ndarray,
    kappa: float,
    lapse: float = 0.0,
) -> np.ndarray:
    core = np.asarray(state.omega @ q_trial, dtype=float)
    core = _normalize(core, "core choice probabilities")
    logits = float(kappa) * np.log(np.clip(core, EPS, None))
    logits -= float(np.max(logits))
    readout = _normalize(np.exp(logits), "choice readout")
    return (
        (1.0 - float(lapse)) * readout
        + float(lapse) / float(readout.size)
    )


def _feedback_update(
    state: HFWState,
    q_trial: np.ndarray,
    choice: int,
    feedback: float,
    parameters: Model0804Parameters,
    epsilon: float,
) -> tuple[HFWState, float, float]:
    compatible = int(choice) if float(feedback) >= 0.5 else 1 - int(choice)
    active = state.active
    likelihood = np.clip(q_trial[active, compatible], float(epsilon), 1.0)
    feedback_probability = float(np.sum(state.omega[active] * likelihood))
    surprise = -math.log(max(feedback_probability, float(epsilon)))

    fade = np.full_like(state.fade, -np.inf)
    static = np.full_like(state.static, -np.inf)
    fade[active] = parameters.gamma * state.fade[active] + np.log(likelihood)
    static[active] = state.static[active] + np.log(likelihood)
    ell = (
        parameters.w0 * static[active]
        + (1.0 - parameters.w0) * fade[active]
    )
    ell -= float(np.max(ell))
    posterior_active = _normalize(np.exp(ell), "feedback posterior")
    omega = np.zeros_like(state.omega)
    omega[active] = posterior_active
    if active.size <= 1:
        uncertainty = 0.0
    else:
        uncertainty = float(
            -np.sum(posterior_active * np.log(np.clip(posterior_active, EPS, None)))
            / math.log(active.size)
        )
    return HFWState(active.copy(), omega, fade, static), surprise, uncertainty


def _dense_initial_states_qmc(
    prior: np.ndarray,
    capacity: int,
    particle_count: int,
    filter_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a dense particle panel while keeping active membership explicit."""

    active_indices = _sample_initial_indices_qmc(
        prior, int(capacity), int(particle_count), int(filter_seed)
    )
    n_particles = int(particle_count)
    n_hypotheses = int(prior.size)
    rows = np.arange(n_particles)[:, None]
    active = np.zeros((n_particles, n_hypotheses), dtype=bool)
    active[rows, active_indices] = True
    omega = np.zeros((n_particles, n_hypotheses), dtype=float)
    selected_prior = prior[active_indices]
    selected_prior /= selected_prior.sum(axis=1, keepdims=True)
    omega[rows, active_indices] = selected_prior
    fade = np.full_like(omega, -np.inf)
    static = np.full_like(omega, -np.inf)
    log_omega = np.log(np.clip(omega[active], EPS, None))
    fade[active] = log_omega
    static[active] = log_omega
    return active, omega, fade, static


def _dense_initial_states_iid(
    prior: np.ndarray,
    capacity: int,
    particle_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample IID initial workspaces for the alive-filter stopping rule."""

    n_particles = int(particle_count)
    n_hypotheses = int(prior.size)
    if n_particles < 1:
        raise ValueError("particle_count must be positive")
    unit = rng.random((n_particles, int(capacity)))
    available = np.ones((n_particles, n_hypotheses), dtype=bool)
    active_indices = np.empty((n_particles, int(capacity)), dtype=np.int32)
    rows = np.arange(n_particles)
    for slot in range(int(capacity)):
        weights = available * prior[None, :]
        weights /= weights.sum(axis=1, keepdims=True)
        cumulative = np.cumsum(weights, axis=1)
        cumulative[:, -1] = 1.0
        positions = np.sum(unit[:, slot, None] >= cumulative, axis=1)
        active_indices[:, slot] = positions.astype(np.int32)
        available[rows, positions] = False

    active = np.zeros((n_particles, n_hypotheses), dtype=bool)
    active[rows[:, None], active_indices] = True
    omega = np.zeros((n_particles, n_hypotheses), dtype=float)
    selected_prior = prior[active_indices].astype(float)
    selected_prior /= selected_prior.sum(axis=1, keepdims=True)
    omega[rows[:, None], active_indices] = selected_prior
    fade = np.full_like(omega, -np.inf)
    static = np.full_like(omega, -np.inf)
    log_omega = np.log(np.clip(omega[active], EPS, None))
    fade[active] = log_omega
    static[active] = log_omega
    return active, omega, fade, static


def _rowwise_weighted_wor_indices(
    weights: np.ndarray,
    counts: np.ndarray,
    uniforms: np.ndarray,
    capacity: int,
) -> np.ndarray:
    """Vectorized row-wise weighted sampling without replacement."""

    working = np.asarray(weights, dtype=float).copy()
    count_values = np.asarray(counts, dtype=int).reshape(-1)
    unit = np.asarray(uniforms, dtype=float)
    n_rows, n_hypotheses = working.shape
    if count_values.size != n_rows or unit.shape != (n_rows, int(capacity)):
        raise ValueError("row-wise weighted-WOR arrays have incompatible shapes")
    selected = np.full((n_rows, int(capacity)), -1, dtype=np.int32)
    for slot in range(int(capacity)):
        rows = np.flatnonzero(count_values > slot)
        if rows.size == 0:
            continue
        row_weights = working[rows]
        totals = row_weights.sum(axis=1)
        if np.any(~np.isfinite(totals)) or np.any(totals <= 0.0):
            raise ValueError("row-wise weighted-WOR encountered zero mass")
        cumulative = np.cumsum(row_weights / totals[:, None], axis=1)
        cumulative[:, -1] = 1.0
        positions = np.sum(unit[rows, slot, None] >= cumulative, axis=1)
        if np.any(positions >= n_hypotheses):
            raise AssertionError("row-wise weighted-WOR selection overflow")
        selected[rows, slot] = positions.astype(np.int32)
        working[rows, positions] = 0.0
    return selected


def _choice_probability_dense(
    omega: np.ndarray,
    q_trial: np.ndarray,
    kappa: float,
    lapse: float = 0.0,
) -> np.ndarray:
    core = np.asarray(omega @ q_trial, dtype=float)
    core /= core.sum(axis=1, keepdims=True)
    logits = float(kappa) * np.log(np.clip(core, EPS, None))
    logits -= np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return (
        (1.0 - float(lapse)) * probabilities
        + float(lapse) / float(probabilities.shape[1])
    )


def _sample_categorical_dense(
    probabilities: np.ndarray,
    uniforms: np.ndarray,
) -> np.ndarray:
    """Sample one category per row without depending on NumPy's choice API."""

    values = np.asarray(probabilities, dtype=float)
    unit = np.asarray(uniforms, dtype=float).reshape(-1)
    if values.ndim != 2 or values.shape[0] != unit.size:
        raise ValueError("categorical probabilities and uniforms are incompatible")
    if np.any(values < 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("categorical probabilities must be finite and non-negative")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("categorical probabilities must sum to one")
    cumulative = np.cumsum(values, axis=1)
    cumulative[:, -1] = 1.0
    categories = np.sum(unit[:, None] >= cumulative, axis=1)
    if np.any(categories >= values.shape[1]):
        raise AssertionError("categorical sampling overflow")
    return categories.astype(np.int32)


def _sample_transition_candidates_dense(
    active: np.ndarray,
    omega: np.ndarray,
    fade: np.ndarray,
    static: np.ndarray,
    prior: np.ndarray,
    kernels: TransitionKernels,
    parameters: Model0804Parameters,
    capacity: int,
    proposal_count: int,
    transition_unit: np.ndarray,
    replacement_count_override: np.ndarray | None = None,
    m_by_parent: np.ndarray | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Propagate every parent through B transition proposals in one batch."""

    n_particles, n_hypotheses = omega.shape
    n_candidates = n_particles * int(proposal_count)
    unit = np.asarray(transition_unit, dtype=float)
    ordinary_dimension = 1 + 2 * int(capacity)
    expected_shape = (
        n_candidates,
        _transition_uniform_dimension(parameters, int(capacity)),
    )
    if unit.shape != expected_shape:
        raise ValueError(
            f"transition Sobol array has shape {unit.shape}, expected {expected_shape}"
        )

    ordinary_unit = unit[:, :ordinary_dimension]
    if replacement_count_override is not None and parameters.rho > 0.0:
        raise ValueError(
            "replacement-count stratification is not defined for FA2R resets"
        )
    parent_rows = np.arange(n_particles, dtype=int)
    active_indices = np.nonzero(active)[1].reshape(
        n_particles, int(capacity)
    )
    parent_local = np.zeros_like(omega)
    for slot in range(int(capacity)):
        hypothesis = active_indices[:, slot]
        parent_local += (
            omega[parent_rows, hypothesis, None] * kernels.local[hypothesis]
        )
    parent_local *= ~active
    parent_local /= parent_local.sum(axis=1, keepdims=True)
    parent_global = (~active) * prior[None, :]
    parent_global /= parent_global.sum(axis=1, keepdims=True)
    parent_proposal = (
        (1.0 - float(parameters.g)) * parent_local
        + float(parameters.g) * parent_global
    )
    parent_proposal /= parent_proposal.sum(axis=1, keepdims=True)

    candidate_active = np.repeat(active, int(proposal_count), axis=0)
    candidate_omega = np.repeat(omega, int(proposal_count), axis=0)
    candidate_fade = np.repeat(fade, int(proposal_count), axis=0)
    candidate_static = np.repeat(static, int(proposal_count), axis=0)
    newcomer_weights = np.repeat(parent_proposal, int(proposal_count), axis=0)

    if replacement_count_override is None:
        if m_by_parent is None:
            count_probabilities = _binomial_probabilities(
                int(capacity), parameters.m
            )
            count_cumulative = np.cumsum(count_probabilities)
            count_cumulative[-1] = 1.0
            replacement_count = np.searchsorted(
                count_cumulative, ordinary_unit[:, 0], side="right"
            ).astype(int)
        else:
            parent_m = np.asarray(m_by_parent, dtype=float).reshape(-1)
            if parent_m.size != n_particles or np.any(~np.isfinite(parent_m)) or np.any(
                (parent_m < 0.0) | (parent_m > 1.0)
            ):
                raise ValueError("m_by_parent must contain one probability per particle")
            candidate_m = np.repeat(parent_m, int(proposal_count))
            counts = np.arange(int(capacity) + 1, dtype=float)
            combinations = np.asarray(
                [math.comb(int(capacity), int(value)) for value in counts],
                dtype=float,
            )
            count_probabilities = (
                combinations[None, :]
                * candidate_m[:, None] ** counts[None, :]
                * (1.0 - candidate_m[:, None])
                ** (float(capacity) - counts[None, :])
            )
            count_probabilities /= count_probabilities.sum(axis=1, keepdims=True)
            count_cumulative = np.cumsum(count_probabilities, axis=1)
            count_cumulative[:, -1] = 1.0
            replacement_count = np.sum(
                ordinary_unit[:, 0, None] >= count_cumulative, axis=1
            ).astype(int)
    else:
        replacement_count = np.asarray(
            replacement_count_override, dtype=int
        ).reshape(-1)
        if replacement_count.size != n_candidates or np.any(
            (replacement_count < 0) | (replacement_count > int(capacity))
        ):
            raise ValueError("replacement-count override is invalid")

    exit_weights = (1.0 - candidate_omega + 1e-9) * candidate_active
    dropped = _rowwise_weighted_wor_indices(
        exit_weights,
        replacement_count,
        ordinary_unit[:, 1 : 1 + int(capacity)],
        int(capacity),
    )
    newcomers = _rowwise_weighted_wor_indices(
        newcomer_weights,
        replacement_count,
        ordinary_unit[:, 1 + int(capacity) : 1 + 2 * int(capacity)],
        int(capacity),
    )

    old_omega = candidate_omega.copy()
    old_active = candidate_active.copy()
    old_active_indices = np.nonzero(old_active)[1].reshape(
        n_candidates, int(capacity)
    )
    delta = np.zeros_like(candidate_omega)
    delta[old_active] = candidate_static[old_active] - candidate_fade[old_active]
    removed_mass = np.zeros(n_candidates, dtype=float)
    distance_total = np.zeros(n_candidates, dtype=float)
    for slot in range(int(capacity)):
        rows = np.flatnonzero(replacement_count > slot)
        if rows.size == 0:
            continue
        dropped_index = dropped[rows, slot]
        newcomer_index = newcomers[rows, slot]
        transferred_mass = old_omega[rows, dropped_index]
        removed_mass[rows] += transferred_mass
        for active_slot in range(int(capacity)):
            source_index = old_active_indices[rows, active_slot]
            distance_total[rows] += (
                old_omega[rows, source_index]
                * kernels.distance[source_index, newcomer_index]
            )
        candidate_omega[rows, dropped_index] = 0.0
        candidate_omega[rows, newcomer_index] = transferred_mass
        candidate_active[rows, dropped_index] = False
        candidate_active[rows, newcomer_index] = True
        delta[rows, dropped_index] = 0.0
        delta[rows, newcomer_index] = 0.0

    if parameters.rho > 0.0:
        reset = unit[:, ordinary_dimension] < float(parameters.rho)
        reset_rows = np.flatnonzero(reset)
        if reset_rows.size:
            reset_weights = np.broadcast_to(
                prior[None, :], (reset_rows.size, n_hypotheses)
            )
            reset_count = np.full(
                reset_rows.size, int(capacity), dtype=int
            )
            reset_active_indices = _rowwise_weighted_wor_indices(
                reset_weights,
                reset_count,
                unit[
                    reset_rows,
                    ordinary_dimension
                    + 1 : ordinary_dimension
                    + 1
                    + int(capacity),
                ],
                int(capacity),
            )
            candidate_active[reset_rows] = False
            candidate_omega[reset_rows] = 0.0
            delta[reset_rows] = 0.0
            selected_prior = prior[reset_active_indices]
            selected_prior /= selected_prior.sum(axis=1, keepdims=True)
            for slot in range(int(capacity)):
                newcomer_index = reset_active_indices[:, slot]
                candidate_active[reset_rows, newcomer_index] = True
                candidate_omega[reset_rows, newcomer_index] = selected_prior[
                    :, slot
                ]
            replacement_count[reset_rows] = int(capacity)
            removed_mass[reset_rows] = 1.0
            distance_total[reset_rows] = 0.0
            for slot in range(int(capacity)):
                newcomer_index = reset_active_indices[:, slot]
                for active_slot in range(int(capacity)):
                    source_index = old_active_indices[
                        reset_rows, active_slot
                    ]
                    distance_total[reset_rows] += (
                        old_omega[reset_rows, source_index]
                        * kernels.distance[source_index, newcomer_index]
                    )

    candidate_fade.fill(-np.inf)
    candidate_static.fill(-np.inf)
    log_omega = np.log(np.clip(candidate_omega[candidate_active], EPS, None))
    candidate_fade[candidate_active] = (
        log_omega - float(parameters.w0) * delta[candidate_active]
    )
    candidate_static[candidate_active] = (
        log_omega + (1.0 - float(parameters.w0)) * delta[candidate_active]
    )
    combined = (
        float(parameters.w0) * candidate_static[candidate_active]
        + (1.0 - float(parameters.w0)) * candidate_fade[candidate_active]
    )
    sync_error = np.zeros(n_candidates, dtype=float)
    active_rows, _ = np.nonzero(candidate_active)
    element_error = np.abs(combined - log_omega)
    np.maximum.at(sync_error, active_rows, element_error)
    newcomer_distance = np.divide(
        distance_total,
        replacement_count,
        out=np.zeros_like(distance_total),
        where=replacement_count > 0,
    )
    if np.any(candidate_active.sum(axis=1) != int(capacity)):
        raise AssertionError("dense HFW transition changed active-set capacity")
    if not np.allclose(candidate_omega.sum(axis=1), 1.0, atol=1e-12, rtol=0.0):
        raise AssertionError("dense HFW transition did not conserve rule mass")
    return (
        candidate_active,
        candidate_omega,
        candidate_fade,
        candidate_static,
        replacement_count.astype(float),
        removed_mass,
        newcomer_distance,
        sync_error,
    )


def _feedback_update_dense(
    active: np.ndarray,
    omega: np.ndarray,
    fade: np.ndarray,
    static: np.ndarray,
    q_trial: np.ndarray,
    choice: int,
    feedback: float,
    parameters: Model0804Parameters,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    compatible = int(choice) if float(feedback) >= 0.5 else 1 - int(choice)
    log_likelihood = np.log(
        np.clip(q_trial[:, compatible], float(epsilon), 1.0)
    )
    updated_fade = np.full_like(fade, -np.inf)
    updated_static = np.full_like(static, -np.inf)
    updated_fade[active] = (
        float(parameters.gamma) * fade[active]
        + np.broadcast_to(log_likelihood, fade.shape)[active]
    )
    updated_static[active] = (
        static[active] + np.broadcast_to(log_likelihood, static.shape)[active]
    )
    # Compute only on active entries.  Multiplying an inactive ``-inf`` by
    # the exact endpoint weight zero would otherwise create NaN (0 * -inf),
    # even though that channel is mathematically absent from the mixture.
    ell = np.full_like(fade, -np.inf)
    if float(parameters.w0) <= 0.0:
        ell[active] = updated_fade[active]
    elif float(parameters.w0) >= 1.0:
        ell[active] = updated_static[active]
    else:
        ell[active] = (
            float(parameters.w0) * updated_static[active]
            + (1.0 - float(parameters.w0)) * updated_fade[active]
        )
    ell -= np.max(ell, axis=1, keepdims=True)
    updated_omega = np.where(active, np.exp(ell), 0.0)
    updated_omega /= updated_omega.sum(axis=1, keepdims=True)
    return updated_omega, updated_fade, updated_static


def _feedback_surprise_dense(
    omega: np.ndarray,
    q_trial: np.ndarray,
    choice: int,
    feedback: float,
    epsilon: float,
) -> np.ndarray:
    """Return one pre-feedback surprise value per latent particle."""

    compatible = int(choice) if float(feedback) >= 0.5 else 1 - int(choice)
    likelihood = np.clip(q_trial[:, compatible], float(epsilon), 1.0)
    probability = np.asarray(omega @ likelihood, dtype=float)
    return -np.log(np.clip(probability, float(epsilon), 1.0))


def _rule_uncertainty_dense(
    omega: np.ndarray,
    capacity: int,
) -> np.ndarray:
    """Return normalized active-rule entropy after feedback."""

    if int(capacity) <= 1:
        return np.zeros(omega.shape[0], dtype=float)
    values = np.where(
        omega > 0.0,
        omega * np.log(np.clip(omega, EPS, None)),
        0.0,
    )
    return -np.sum(values, axis=1) / math.log(int(capacity))


def _validate_rt_parameters(
    parameters: Model0804RTParameters,
) -> Model0804RTParameters:
    values = {
        "intercept": float(parameters.intercept),
        "choice_entropy": float(parameters.choice_entropy),
        "replacement_fraction": float(parameters.replacement_fraction),
        "newcomer_distance": float(parameters.newcomer_distance),
        "sigma": float(parameters.sigma),
        "degrees_of_freedom": float(parameters.degrees_of_freedom),
    }
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("all RT parameters must be finite")
    if values["sigma"] <= 0.0:
        raise ValueError("RT sigma must be positive")
    if values["degrees_of_freedom"] <= 2.0:
        raise ValueError("RT degrees_of_freedom must exceed 2")
    return Model0804RTParameters(**values)


def _student_t_log_density(
    values: np.ndarray,
    location: np.ndarray,
    sigma: float,
    degrees_of_freedom: float,
) -> np.ndarray:
    observed = np.asarray(values, dtype=float)
    mean = np.asarray(location, dtype=float)
    nu = float(degrees_of_freedom)
    scale = float(sigma)
    standardized = (observed - mean) / scale
    normalizer = (
        math.lgamma((nu + 1.0) / 2.0)
        - math.lgamma(nu / 2.0)
        - 0.5 * math.log(nu * math.pi)
        - math.log(scale)
    )
    return normalizer - 0.5 * (nu + 1.0) * np.log1p(
        np.square(standardized) / nu
    )


def _weighted_log_mean_density(
    log_density: np.ndarray,
    weights: np.ndarray,
) -> float:
    values = np.log(np.clip(np.asarray(weights, dtype=float), EPS, None)) + np.asarray(
        log_density, dtype=float
    )
    maximum = float(np.max(values))
    return maximum + math.log(float(np.sum(np.exp(values - maximum))))


def _trial_masks(
    n_trials: int,
    score_mask: np.ndarray | None,
    condition_on_choice_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if score_mask is None:
        score = np.ones(n_trials, dtype=bool)
    else:
        score = np.asarray(score_mask, dtype=bool).reshape(-1)
    if condition_on_choice_mask is None:
        condition = np.ones(n_trials, dtype=bool)
    else:
        condition = np.asarray(condition_on_choice_mask, dtype=bool).reshape(-1)
    if score.size != n_trials or condition.size != n_trials:
        raise ValueError("trial masks must have one value per trial")
    if np.any(score & ~condition):
        raise ValueError("every scored choice must also condition the particle filter")
    return score, condition


def _run_fa0_vectorized(
    q: np.ndarray,
    y: np.ndarray,
    r: np.ndarray,
    p0: np.ndarray,
    decoded: Model0804Parameters,
    *,
    capacity: int,
    particle_count: int,
    filter_seed: int,
    exact_initial_sets: bool,
    maximum_exact_initial_sets: int,
    score: np.ndarray,
    condition: np.ndarray,
    epsilon: float,
) -> Model0804Trace:
    """Vectorized integration for the static-set FA0 endpoint.

    Static latent sets must not be bootstrap-resampled without a rejuvenation
    kernel.  It can enumerate every unordered initial set and its exact
    sequential weighted-WOR probability; otherwise it keeps the complete QMC
    panel and updates only its importance weights.
    """

    n_trials, n_hypotheses, n_categories = q.shape
    initial_weights = None
    if exact_initial_sets:
        exact_count = math.comb(n_hypotheses, int(capacity))
        if exact_count > int(maximum_exact_initial_sets):
            raise ValueError(
                f"exact FA0 integration requires {exact_count} sets, exceeding "
                f"the configured maximum {int(maximum_exact_initial_sets)}"
            )
        active = _initial_indices_exact(n_hypotheses, int(capacity))
        initial_weights = _successive_wor_set_probabilities_exact(
            tuple(float(value) for value in p0), int(capacity)
        )
        integration_mode = "exact_successive_wor_initial_sets"
    else:
        active = _sample_initial_indices_qmc(
            p0, int(capacity), int(particle_count), int(filter_seed)
        )
        integration_mode = "qmc_static_panel"
    n_particles = int(active.shape[0])
    omega = p0[active].astype(float)
    omega /= omega.sum(axis=1, keepdims=True)
    fade = np.log(np.clip(omega, float(epsilon), None))
    static = fade.copy()
    if initial_weights is None:
        weights = np.full(n_particles, 1.0 / float(n_particles), dtype=float)
    else:
        weights = initial_weights.copy()

    probabilities = np.zeros((n_trials, n_categories), dtype=float)
    marginal_prior = np.zeros((n_trials, n_hypotheses), dtype=float)
    marginal_active = np.zeros((n_trials, n_hypotheses), dtype=float)
    pre_ess = np.zeros(n_trials, dtype=float)
    post_ess = np.zeros(n_trials, dtype=float)
    sync_error = np.zeros(n_trials, dtype=float)
    nll = 0.0

    for trial_index in range(n_trials):
        if trial_index > 0:
            delta = static - fade
            log_omega = np.log(np.clip(omega, float(epsilon), None))
            fade = log_omega - decoded.w0 * delta
            static = log_omega + (1.0 - decoded.w0) * delta
            combined = decoded.w0 * static + (1.0 - decoded.w0) * fade
            sync_error[trial_index] = float(
                np.max(np.abs(combined - log_omega))
            )

        pre_ess[trial_index] = effective_sample_size(weights)
        q_active = q[trial_index][active]
        core = np.sum(omega[:, :, None] * q_active, axis=1)
        core /= core.sum(axis=1, keepdims=True)
        logits = decoded.kappa * np.log(np.clip(core, float(epsilon), None))
        logits -= np.max(logits, axis=1, keepdims=True)
        particle_probabilities = np.exp(logits)
        particle_probabilities /= particle_probabilities.sum(axis=1, keepdims=True)
        particle_probabilities = (
            (1.0 - decoded.lapse) * particle_probabilities
            + decoded.lapse / float(n_categories)
        )
        probabilities[trial_index] = _normalize(
            weights @ particle_probabilities,
            "FA0 QMC marginal choice probabilities",
        )
        for slot in range(int(capacity)):
            marginal_prior[trial_index] += np.bincount(
                active[:, slot],
                weights=weights * omega[:, slot],
                minlength=n_hypotheses,
            )
            marginal_active[trial_index] += np.bincount(
                active[:, slot],
                weights=weights,
                minlength=n_hypotheses,
            )
        marginal_prior[trial_index] = _normalize(
            marginal_prior[trial_index], "FA0 QMC marginal rule prior"
        )

        observed_probability = float(probabilities[trial_index, y[trial_index]])
        if score[trial_index]:
            nll -= math.log(max(observed_probability, float(epsilon)))
        if condition[trial_index]:
            weights *= np.clip(
                particle_probabilities[:, y[trial_index]], float(epsilon), 1.0
            )
            weights = _normalize(weights, "FA0 choice-filtered QMC weights")
        post_ess[trial_index] = effective_sample_size(weights)

        compatible = int(y[trial_index]) if r[trial_index] >= 0.5 else 1 - int(y[trial_index])
        likelihood = np.clip(
            q[trial_index, active, compatible], float(epsilon), 1.0
        )
        fade = decoded.gamma * fade + np.log(likelihood)
        static = static + np.log(likelihood)
        ell = decoded.w0 * static + (1.0 - decoded.w0) * fade
        ell -= np.max(ell, axis=1, keepdims=True)
        omega = np.exp(ell)
        omega /= omega.sum(axis=1, keepdims=True)

    zeros = np.zeros(n_trials, dtype=float)
    return Model0804Trace(
        nll=float(nll),
        probabilities=probabilities,
        marginal_hypothesis_prior=marginal_prior,
        marginal_active_probability=marginal_active,
        predictive_replacement_count=zeros.copy(),
        predictive_replacement_fraction=zeros.copy(),
        predictive_removed_mass=zeros.copy(),
        predictive_newcomer_distance=zeros.copy(),
        pre_choice_ess=pre_ess,
        post_choice_ess=post_ess,
        resampled=np.zeros(n_trials, dtype=bool),
        resampling_unique_ancestors=np.full(n_trials, n_particles, dtype=int),
        memory_sync_error=sync_error,
        final_weights=weights.copy(),
        particle_count=n_particles,
        capacity=int(capacity),
        filter_seed=int(filter_seed),
        transition_proposals_per_particle=1,
        integration_mode=integration_mode,
        replacement_count_stratified=False,
    )


def run_model0804_particle_filter(
    q_values: np.ndarray,
    choices: np.ndarray,
    feedback: np.ndarray,
    prior: np.ndarray,
    kernels: TransitionKernels,
    *,
    model_id: str,
    parameters: Model0804Parameters,
    capacity: int,
    particle_count: int,
    filter_seed: int = 20260804,
    resample_threshold_fraction: float = 0.5,
    transition_proposals_per_particle: int = 1,
    stratify_replacement_count: bool = False,
    fa0_exact_initial_sets: bool = False,
    fa0_maximum_exact_initial_sets: int = 1_000_000,
    score_mask: np.ndarray | None = None,
    condition_on_choice_mask: np.ndarray | None = None,
    log_rt_values: np.ndarray | None = None,
    rt_parameters: Model0804RTParameters | None = None,
    score_rt_mask: np.ndarray | None = None,
    condition_on_rt_mask: np.ndarray | None = None,
    epsilon: float = EPS,
) -> Model0804Trace:
    """Run FA0--FA3-M in strict order, optionally with a joint RT emission."""

    q, y, r, p0, decoded = _validate_inputs(
        q_values, choices, feedback, prior, kernels, capacity, model_id, parameters
    )
    n_trials, n_hypotheses, n_categories = q.shape
    n_particles = int(particle_count)
    if n_particles < 2:
        raise ValueError("particle_count must be at least 2")
    threshold = float(resample_threshold_fraction)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("resample_threshold_fraction must lie in [0, 1]")
    proposal_count = int(transition_proposals_per_particle)
    if proposal_count != float(transition_proposals_per_particle) or proposal_count < 1:
        raise ValueError("transition_proposals_per_particle must be a positive integer")
    count_stratified = bool(stratify_replacement_count)
    dynamic_signal_active = bool(
        decoded.dynamic_m
        and (
            decoded.m_beta_surprise > 0.0
            or decoded.m_beta_uncertainty > 0.0
        )
    )
    if count_stratified and dynamic_signal_active:
        raise ValueError(
            "replacement-count stratification is not available for dynamic m"
        )
    if count_stratified:
        proposal_count = int(capacity) + 1
        proposal_weights = _binomial_probabilities(int(capacity), decoded.m)
    else:
        proposal_weights = np.full(
            proposal_count, 1.0 / float(proposal_count), dtype=float
        )
    score, condition = _trial_masks(
        n_trials, score_mask, condition_on_choice_mask
    )
    rt_requested = log_rt_values is not None or rt_parameters is not None
    if rt_requested and (log_rt_values is None or rt_parameters is None):
        raise ValueError("log_rt_values and rt_parameters must be supplied together")
    if rt_requested:
        if model_id == "FA0":
            raise ValueError("joint RT emission is currently implemented for FA1--FA3-M")
        if proposal_count != 1:
            raise ValueError("joint RT emission requires one transition proposal per particle")
        log_rt = np.asarray(log_rt_values, dtype=float).reshape(-1)
        if log_rt.size != n_trials or not np.all(np.isfinite(log_rt)):
            raise ValueError("log_rt_values must be finite with one value per trial")
        decoded_rt = _validate_rt_parameters(rt_parameters)
        rt_score, rt_condition = _trial_masks(
            n_trials, score_rt_mask, condition_on_rt_mask
        )
    else:
        log_rt = np.zeros(n_trials, dtype=float)
        decoded_rt = None
        rt_score = np.zeros(n_trials, dtype=bool)
        rt_condition = np.zeros(n_trials, dtype=bool)

    if model_id == "FA0":
        return _run_fa0_vectorized(
            q,
            y,
            r,
            p0,
            decoded,
            capacity=int(capacity),
            particle_count=n_particles,
            filter_seed=int(filter_seed),
            exact_initial_sets=bool(fa0_exact_initial_sets),
            maximum_exact_initial_sets=int(fa0_maximum_exact_initial_sets),
            score=score,
            condition=condition,
            epsilon=float(epsilon),
        )

    active, omega, fade, static = _dense_initial_states_qmc(
        p0, int(capacity), n_particles, int(filter_seed)
    )
    weights = np.full(n_particles, 1.0 / float(n_particles), dtype=float)
    probabilities = np.zeros((n_trials, n_categories), dtype=float)
    marginal_prior = np.zeros((n_trials, n_hypotheses), dtype=float)
    marginal_active = np.zeros((n_trials, n_hypotheses), dtype=float)
    predictive_count = np.zeros(n_trials, dtype=float)
    predictive_removed = np.zeros(n_trials, dtype=float)
    predictive_distance = np.zeros(n_trials, dtype=float)
    predictive_m = np.full(n_trials, float(decoded.m), dtype=float)
    feedback_surprise = np.zeros(n_trials, dtype=float)
    feedback_uncertainty = np.zeros(n_trials, dtype=float)
    pre_ess = np.zeros(n_trials, dtype=float)
    post_ess = np.zeros(n_trials, dtype=float)
    resampled = np.zeros(n_trials, dtype=bool)
    unique_ancestors = np.full(n_trials, n_particles, dtype=int)
    sync_error = np.zeros(n_trials, dtype=float)
    rt_predictive_log_density = np.full(n_trials, np.nan, dtype=float)
    nll = 0.0
    rt_conditional_nll = 0.0
    baseline_m_logit = (
        math.log(float(decoded.m) / (1.0 - float(decoded.m)))
        if dynamic_signal_active
        else 0.0
    )
    control_m_logit = np.full(n_particles, baseline_m_logit, dtype=float)

    for trial_index in range(n_trials):
        if dynamic_signal_active:
            particle_m = 1.0 / (
                1.0 + np.exp(-np.clip(control_m_logit, -30.0, 30.0))
            )
        else:
            particle_m = np.full(n_particles, float(decoded.m), dtype=float)
        predictive_m[trial_index] = float(np.sum(weights * particle_m))
        particle_probabilities = np.zeros((n_particles, n_categories), dtype=float)
        particle_prior = np.zeros((n_particles, n_hypotheses), dtype=float)
        particle_active = np.zeros((n_particles, n_hypotheses), dtype=float)
        particle_count_values = np.zeros(n_particles, dtype=float)
        particle_removed = np.zeros(n_particles, dtype=float)
        particle_distance = np.zeros(n_particles, dtype=float)
        trial_sync_error = 0.0

        predictions_ready = False
        if trial_index > 0:
            transition_unit = _sobol_unit_points(
                n_particles * proposal_count,
                _transition_uniform_dimension(decoded, int(capacity)),
                _stable_seed(filter_seed, "transition_sobol", trial_index),
            )
            # A separately scrambled Sobol net is low-discrepancy marginally,
            # but pairing rows by their original index with the initial-state
            # net can induce a deterministic cross-net correlation.  An
            # independent seeded permutation preserves the transition net
            # while randomizing its assignment to current particle states.
            permutation = np.random.default_rng(
                _stable_seed(filter_seed, "transition_permutation", trial_index)
            ).permutation(n_particles * proposal_count)
            transition_unit = transition_unit[permutation]
            proposal_batch_size = min(proposal_count, 2)
            best_selection_score = np.full(n_particles, -np.inf, dtype=float)
            selected_active = np.empty_like(active)
            selected_omega = np.empty_like(omega)
            selected_fade = np.empty_like(fade)
            selected_static = np.empty_like(static)
            parent_rows = np.arange(n_particles, dtype=int)
            for proposal_start in range(0, proposal_count, proposal_batch_size):
                proposal_stop = min(
                    proposal_start + proposal_batch_size, proposal_count
                )
                batch_count = proposal_stop - proposal_start
                proposal_slots = np.arange(proposal_start, proposal_stop)
                batch_weights = proposal_weights[proposal_slots]
                unit_rows = (
                    parent_rows[:, None] * proposal_count
                    + proposal_slots[None, :]
                ).reshape(-1)
                (
                    candidate_active,
                    candidate_omega,
                    candidate_fade,
                    candidate_static,
                    candidate_count,
                    candidate_removed,
                    candidate_distance,
                    candidate_sync,
                ) = _sample_transition_candidates_dense(
                    active,
                    omega,
                    fade,
                    static,
                    p0,
                    kernels,
                    decoded,
                    int(capacity),
                    batch_count,
                    transition_unit[unit_rows],
                    replacement_count_override=(
                        np.tile(proposal_slots, n_particles)
                        if count_stratified
                        else None
                    ),
                    m_by_parent=(
                        particle_m if dynamic_signal_active else None
                    ),
                )
                candidate_probabilities = _choice_probability_dense(
                    candidate_omega,
                    q[trial_index],
                    decoded.kappa,
                    decoded.lapse,
                ).reshape(n_particles, batch_count, n_categories)
                candidate_active_panel = candidate_active.reshape(
                    n_particles, batch_count, n_hypotheses
                )
                candidate_omega_panel = candidate_omega.reshape(
                    n_particles, batch_count, n_hypotheses
                )
                particle_probabilities += np.sum(
                    candidate_probabilities * batch_weights[None, :, None],
                    axis=1,
                )
                particle_prior += np.sum(
                    candidate_omega_panel * batch_weights[None, :, None], axis=1
                )
                particle_active += np.sum(
                    candidate_active_panel * batch_weights[None, :, None], axis=1
                )
                particle_count_values += np.sum(
                    candidate_count.reshape(n_particles, batch_count)
                    * batch_weights[None, :],
                    axis=1,
                )
                particle_removed += np.sum(
                    candidate_removed.reshape(n_particles, batch_count)
                    * batch_weights[None, :],
                    axis=1,
                )
                particle_distance += np.sum(
                    candidate_distance.reshape(n_particles, batch_count)
                    * batch_weights[None, :],
                    axis=1,
                )
                trial_sync_error = max(
                    trial_sync_error, float(np.max(candidate_sync))
                )

                for local_index, proposal_index in enumerate(
                    range(proposal_start, proposal_stop)
                ):
                    if condition[trial_index]:
                        selection_mass = candidate_probabilities[
                            :, local_index, y[trial_index]
                        ]
                    else:
                        selection_mass = np.ones(n_particles, dtype=float)
                    selection_mass = (
                        selection_mass * batch_weights[local_index]
                    )
                    if proposal_count == 1:
                        selection_score = np.zeros(n_particles, dtype=float)
                    else:
                        selection_uniform = np.random.default_rng(
                            _stable_seed(
                                filter_seed,
                                "proposal_selection_gumbel",
                                trial_index,
                                proposal_index,
                            )
                        ).random(n_particles)
                        selection_uniform = np.clip(
                            selection_uniform, EPS, 1.0 - EPS
                        )
                        gumbel = -np.log(-np.log(selection_uniform))
                        selection_score = (
                            np.log(np.clip(selection_mass, EPS, None)) + gumbel
                        )
                        selection_score = np.where(
                            selection_mass > 0.0, selection_score, -np.inf
                        )
                    replace = selection_score > best_selection_score
                    if np.any(replace):
                        candidate_rows = parent_rows * batch_count + local_index
                        selected_active[replace] = candidate_active[
                            candidate_rows[replace]
                        ]
                        selected_omega[replace] = candidate_omega[
                            candidate_rows[replace]
                        ]
                        selected_fade[replace] = candidate_fade[
                            candidate_rows[replace]
                        ]
                        selected_static[replace] = candidate_static[
                            candidate_rows[replace]
                        ]
                        best_selection_score[replace] = selection_score[replace]
            active = selected_active
            omega = selected_omega
            fade = selected_fade
            static = selected_static
            predictions_ready = True

        pre_ess[trial_index] = effective_sample_size(weights)
        if not predictions_ready:
            particle_probabilities[:] = _choice_probability_dense(
                omega, q[trial_index], decoded.kappa, decoded.lapse
            )
            particle_prior[:] = omega
            particle_active[:] = active

        probabilities[trial_index] = _normalize(
            np.sum(weights[:, None] * particle_probabilities, axis=0),
            "particle-marginal choice probabilities",
        )
        marginal_prior[trial_index] = _normalize(
            np.sum(weights[:, None] * particle_prior, axis=0),
            "particle-marginal rule prior",
        )
        marginal_active[trial_index] = np.sum(
            weights[:, None] * particle_active, axis=0
        )
        predictive_count[trial_index] = float(np.sum(weights * particle_count_values))
        predictive_removed[trial_index] = float(np.sum(weights * particle_removed))
        predictive_distance[trial_index] = float(np.sum(weights * particle_distance))
        sync_error[trial_index] = trial_sync_error

        observed_probability = float(probabilities[trial_index, y[trial_index]])
        if score[trial_index]:
            nll -= math.log(max(observed_probability, float(epsilon)))
        particle_rt_log_density = None
        if rt_requested:
            particle_entropy = -np.sum(
                particle_probabilities
                * np.log(np.clip(particle_probabilities, float(epsilon), 1.0)),
                axis=1,
            )
            particle_rt_location = (
                float(decoded_rt.intercept)
                + float(decoded_rt.choice_entropy) * particle_entropy
                + float(decoded_rt.replacement_fraction)
                * particle_count_values
                / float(capacity)
                + float(decoded_rt.newcomer_distance) * particle_distance
            )
            particle_rt_log_density = _student_t_log_density(
                np.full(n_particles, log_rt[trial_index], dtype=float),
                particle_rt_location,
                float(decoded_rt.sigma),
                float(decoded_rt.degrees_of_freedom),
            )
            joint_log_density = _weighted_log_mean_density(
                np.log(
                    np.clip(
                        particle_probabilities[:, y[trial_index]],
                        float(epsilon),
                        1.0,
                    )
                )
                + particle_rt_log_density,
                weights,
            )
            conditional_rt_log_density = (
                joint_log_density
                - math.log(max(observed_probability, float(epsilon)))
            )
            rt_predictive_log_density[trial_index] = conditional_rt_log_density
            if rt_score[trial_index]:
                rt_conditional_nll -= conditional_rt_log_density
        if condition[trial_index]:
            weights *= np.clip(
                particle_probabilities[:, y[trial_index]], float(epsilon), 1.0
            )
            weights = _normalize(weights, "choice-filtered particle weights")
        if rt_condition[trial_index]:
            assert particle_rt_log_density is not None
            stabilized_rt_density = np.exp(
                particle_rt_log_density - float(np.max(particle_rt_log_density))
            )
            weights *= stabilized_rt_density
            weights = _normalize(weights, "RT-filtered particle weights")
        post_ess[trial_index] = effective_sample_size(weights)

        particle_surprise = _feedback_surprise_dense(
            omega,
            q[trial_index],
            int(y[trial_index]),
            float(r[trial_index]),
            float(epsilon),
        )
        omega, fade, static = _feedback_update_dense(
            active,
            omega,
            fade,
            static,
            q[trial_index],
            int(y[trial_index]),
            float(r[trial_index]),
            decoded,
            float(epsilon),
        )
        particle_uncertainty = _rule_uncertainty_dense(omega, int(capacity))
        feedback_surprise[trial_index] = float(
            np.sum(weights * particle_surprise)
        )
        feedback_uncertainty[trial_index] = float(
            np.sum(weights * particle_uncertainty)
        )
        if dynamic_signal_active:
            standardized_surprise = (
                particle_surprise - float(decoded.surprise_center)
            ) / float(decoded.surprise_scale)
            standardized_uncertainty = (
                particle_uncertainty - float(decoded.uncertainty_center)
            ) / float(decoded.uncertainty_scale)
            control_m_logit = (
                baseline_m_logit
                + float(decoded.m_phi)
                * (control_m_logit - baseline_m_logit)
                + float(decoded.m_beta_surprise) * standardized_surprise
                + float(decoded.m_beta_uncertainty) * standardized_uncertainty
            )

        if threshold > 0.0 and post_ess[trial_index] < threshold * float(n_particles):
            uniform = float(
                np.random.default_rng(
                    _stable_seed(filter_seed, "resample", trial_index)
                ).random()
            )
            ancestors = systematic_resample(weights, uniform)
            active = active[ancestors].copy()
            omega = omega[ancestors].copy()
            fade = fade[ancestors].copy()
            static = static[ancestors].copy()
            control_m_logit = control_m_logit[ancestors].copy()
            weights.fill(1.0 / float(n_particles))
            resampled[trial_index] = True
            unique_ancestors[trial_index] = int(np.unique(ancestors).size)

    return Model0804Trace(
        nll=float(nll),
        probabilities=probabilities,
        marginal_hypothesis_prior=marginal_prior,
        marginal_active_probability=marginal_active,
        predictive_replacement_count=predictive_count,
        predictive_replacement_fraction=predictive_count / float(capacity),
        predictive_removed_mass=predictive_removed,
        predictive_newcomer_distance=predictive_distance,
        pre_choice_ess=pre_ess,
        post_choice_ess=post_ess,
        resampled=resampled,
        resampling_unique_ancestors=unique_ancestors,
        memory_sync_error=sync_error,
        final_weights=weights.copy(),
        particle_count=n_particles,
        capacity=int(capacity),
        filter_seed=int(filter_seed),
        transition_proposals_per_particle=proposal_count,
        integration_mode=(
            "particle_qmc_stratified_replacement_count"
            if count_stratified
            else "particle_qmc_multiple_proposal"
        ),
        replacement_count_stratified=count_stratified,
        predictive_m=predictive_m,
        feedback_surprise=feedback_surprise,
        feedback_uncertainty=feedback_uncertainty,
        joint_nll=float(nll + rt_conditional_nll) if rt_requested else None,
        rt_conditional_nll=float(rt_conditional_nll) if rt_requested else None,
        rt_predictive_log_density=(
            rt_predictive_log_density if rt_requested else None
        ),
    )


def run_model0804_alive_particle_filter(
    q_values: np.ndarray,
    choices: np.ndarray,
    feedback: np.ndarray,
    prior: np.ndarray,
    kernels: TransitionKernels,
    *,
    model_id: str,
    parameters: Model0804Parameters,
    capacity: int,
    particle_count: int,
    filter_seed: int = 20260804,
    alive_batch_size: int = 8_192,
    maximum_attempts_per_trial: int = 100_000_000,
    score_mask: np.ndarray | None = None,
    condition_on_choice_mask: np.ndarray | None = None,
    epsilon: float = EPS,
) -> Model0804Trace:
    """Run an indicator-augmented alive filter for dynamic FA1/FA2.

    Each proposal augments a latent transition with a synthetic categorical
    choice drawn from that state's readout distribution.  A proposal is alive
    exactly when the synthetic choice equals the observed choice.  The filter
    retains ``N`` alive proposals, continues until the ``N + 1``-th success,
    and estimates the incremental normalizing constant by ``N / (T - 1)``.
    This is the standard alive-filter stopping correction applied to an exact
    categorical augmentation of the non-indicator choice likelihood.
    """

    q, y, r, p0, decoded = _validate_inputs(
        q_values, choices, feedback, prior, kernels, capacity, model_id, parameters
    )
    if model_id == "FA0":
        raise ValueError(
            "the alive filter is restricted to dynamic FA1/FA2; "
            "use exact initial-set integration for static FA0"
        )
    n_trials, n_hypotheses, n_categories = q.shape
    n_particles = int(particle_count)
    if n_particles < 2:
        raise ValueError("particle_count must be at least 2")
    batch_size = int(alive_batch_size)
    if batch_size < 1:
        raise ValueError("alive_batch_size must be positive")
    maximum_attempts = int(maximum_attempts_per_trial)
    if maximum_attempts <= n_particles:
        raise ValueError("maximum_attempts_per_trial must exceed particle_count")
    score, condition = _trial_masks(
        n_trials, score_mask, condition_on_choice_mask
    )

    probabilities = np.zeros((n_trials, n_categories), dtype=float)
    marginal_prior = np.zeros((n_trials, n_hypotheses), dtype=float)
    marginal_active = np.zeros((n_trials, n_hypotheses), dtype=float)
    predictive_count = np.zeros(n_trials, dtype=float)
    predictive_removed = np.zeros(n_trials, dtype=float)
    predictive_distance = np.zeros(n_trials, dtype=float)
    pre_ess = np.full(n_trials, float(n_particles), dtype=float)
    post_ess = np.full(n_trials, float(n_particles), dtype=float)
    resampled = np.asarray(condition, dtype=bool).copy()
    unique_ancestors = np.full(n_trials, n_particles, dtype=int)
    sync_error = np.zeros(n_trials, dtype=float)
    attempt_count = np.zeros(n_trials, dtype=np.int64)
    incremental_likelihood = np.zeros(n_trials, dtype=float)
    nll = 0.0

    active: np.ndarray | None = None
    omega: np.ndarray | None = None
    fade: np.ndarray | None = None
    static: np.ndarray | None = None

    for trial_index in range(n_trials):
        rng = np.random.default_rng(
            _stable_seed(filter_seed, "alive_trial", trial_index)
        )
        retained_active = np.empty(
            (n_particles, n_hypotheses), dtype=bool
        )
        retained_omega = np.empty(
            (n_particles, n_hypotheses), dtype=float
        )
        retained_fade = np.empty_like(retained_omega)
        retained_static = np.empty_like(retained_omega)
        retained_parent = np.full(n_particles, -1, dtype=np.int64)
        retained = 0
        attempts = 0

        prior_total = np.zeros(n_hypotheses, dtype=float)
        active_total = np.zeros(n_hypotheses, dtype=float)
        count_total = 0.0
        removed_total = 0.0
        distance_total = 0.0
        probability_total = np.zeros(n_categories, dtype=float)
        category_counts = np.zeros(n_categories, dtype=np.int64)
        predictive_denominator = 0
        trial_sync_error = 0.0

        def propose(batch_count: int):
            if trial_index == 0:
                (
                    candidate_active,
                    candidate_omega,
                    candidate_fade,
                    candidate_static,
                ) = _dense_initial_states_iid(
                    p0, int(capacity), int(batch_count), rng
                )
                candidate_parent = np.full(batch_count, -1, dtype=np.int64)
                candidate_count = np.zeros(batch_count, dtype=float)
                candidate_removed = np.zeros(batch_count, dtype=float)
                candidate_distance = np.zeros(batch_count, dtype=float)
                candidate_sync = np.zeros(batch_count, dtype=float)
            else:
                if active is None or omega is None or fade is None or static is None:
                    raise AssertionError("alive parent panel is uninitialized")
                candidate_parent = rng.integers(
                    0, n_particles, size=int(batch_count), dtype=np.int64
                )
                transition_unit = rng.random(
                    (
                        int(batch_count),
                        _transition_uniform_dimension(decoded, int(capacity)),
                    )
                )
                (
                    candidate_active,
                    candidate_omega,
                    candidate_fade,
                    candidate_static,
                    candidate_count,
                    candidate_removed,
                    candidate_distance,
                    candidate_sync,
                ) = _sample_transition_candidates_dense(
                    active[candidate_parent],
                    omega[candidate_parent],
                    fade[candidate_parent],
                    static[candidate_parent],
                    p0,
                    kernels,
                    decoded,
                    int(capacity),
                    1,
                    transition_unit,
                )
            candidate_probabilities = _choice_probability_dense(
                candidate_omega,
                q[trial_index],
                decoded.kappa,
                decoded.lapse,
            )
            return (
                candidate_active,
                candidate_omega,
                candidate_fade,
                candidate_static,
                candidate_parent,
                candidate_count,
                candidate_removed,
                candidate_distance,
                candidate_sync,
                candidate_probabilities,
            )

        if condition[trial_index]:
            success_count = 0
            target_successes = n_particles + 1
            while success_count < target_successes:
                remaining_attempts = maximum_attempts - attempts
                if remaining_attempts <= 0:
                    raise RuntimeError(
                        f"alive filter exceeded {maximum_attempts} attempts at "
                        f"trial {trial_index}; increase the cap or use a shared "
                        "non-zero lapse observation layer"
                    )
                current_batch = min(batch_size, remaining_attempts)
                candidates = propose(current_batch)
                candidate_probabilities = candidates[9]
                synthetic_choice = _sample_categorical_dense(
                    candidate_probabilities, rng.random(current_batch)
                )
                success_positions = np.flatnonzero(
                    synthetic_choice == int(y[trial_index])
                )
                successes_needed = target_successes - success_count
                finished = success_positions.size >= successes_needed
                if finished:
                    used = int(success_positions[successes_needed - 1]) + 1
                    prefix = used - 1
                else:
                    used = current_batch
                    prefix = used

                if prefix > 0:
                    candidate_active = candidates[0][:prefix]
                    candidate_omega = candidates[1][:prefix]
                    prior_total += candidate_omega.sum(axis=0)
                    active_total += candidate_active.sum(axis=0)
                    count_total += float(np.sum(candidates[5][:prefix]))
                    removed_total += float(np.sum(candidates[6][:prefix]))
                    distance_total += float(np.sum(candidates[7][:prefix]))
                    trial_sync_error = max(
                        trial_sync_error, float(np.max(candidates[8][:prefix]))
                    )
                    category_counts += np.bincount(
                        synthetic_choice[:prefix], minlength=n_categories
                    )
                    predictive_denominator += prefix

                    retained_positions = np.flatnonzero(
                        synthetic_choice[:prefix] == int(y[trial_index])
                    )
                    take = int(retained_positions.size)
                    if take > 0:
                        destination = slice(retained, retained + take)
                        retained_active[destination] = candidates[0][
                            retained_positions
                        ]
                        retained_omega[destination] = candidates[1][
                            retained_positions
                        ]
                        retained_fade[destination] = candidates[2][
                            retained_positions
                        ]
                        retained_static[destination] = candidates[3][
                            retained_positions
                        ]
                        retained_parent[destination] = candidates[4][
                            retained_positions
                        ]
                        retained += take

                attempts += used
                success_count += int(
                    np.sum(synthetic_choice[:used] == int(y[trial_index]))
                )
                if finished:
                    break

            if retained != n_particles or predictive_denominator != attempts - 1:
                raise AssertionError("alive stopping-rule accounting failed")
            if category_counts[int(y[trial_index])] != n_particles:
                raise AssertionError("alive observed-choice count is not N")
            probabilities[trial_index] = (
                category_counts.astype(float) / float(predictive_denominator)
            )
            incremental_likelihood[trial_index] = (
                float(n_particles) / float(attempts - 1)
            )
        else:
            while retained < n_particles:
                current_batch = min(batch_size, n_particles - retained)
                candidates = propose(current_batch)
                destination = slice(retained, retained + current_batch)
                retained_active[destination] = candidates[0]
                retained_omega[destination] = candidates[1]
                retained_fade[destination] = candidates[2]
                retained_static[destination] = candidates[3]
                retained_parent[destination] = candidates[4]
                prior_total += candidates[1].sum(axis=0)
                active_total += candidates[0].sum(axis=0)
                count_total += float(np.sum(candidates[5]))
                removed_total += float(np.sum(candidates[6]))
                distance_total += float(np.sum(candidates[7]))
                trial_sync_error = max(
                    trial_sync_error, float(np.max(candidates[8]))
                )
                probability_total += candidates[9].sum(axis=0)
                predictive_denominator += current_batch
                retained += current_batch
                attempts += current_batch
            probabilities[trial_index] = _normalize(
                probability_total / float(predictive_denominator),
                "unconditioned alive predictive probabilities",
            )
            incremental_likelihood[trial_index] = float(
                probabilities[trial_index, y[trial_index]]
            )

        probabilities[trial_index] = _normalize(
            probabilities[trial_index], "alive marginal choice probabilities"
        )
        marginal_prior[trial_index] = _normalize(
            prior_total / float(predictive_denominator),
            "alive marginal rule prior",
        )
        marginal_active[trial_index] = (
            active_total / float(predictive_denominator)
        )
        predictive_count[trial_index] = count_total / float(
            predictive_denominator
        )
        predictive_removed[trial_index] = removed_total / float(
            predictive_denominator
        )
        predictive_distance[trial_index] = distance_total / float(
            predictive_denominator
        )
        attempt_count[trial_index] = attempts
        sync_error[trial_index] = trial_sync_error

        observed_probability = float(incremental_likelihood[trial_index])
        if score[trial_index]:
            nll -= math.log(max(observed_probability, float(epsilon)))

        active = retained_active
        omega = retained_omega
        fade = retained_fade
        static = retained_static
        if trial_index == 0:
            packed = np.packbits(active, axis=1)
            unique_ancestors[trial_index] = int(
                np.unique(packed, axis=0).shape[0]
            )
        else:
            unique_ancestors[trial_index] = int(
                np.unique(retained_parent).size
            )

        omega, fade, static = _feedback_update_dense(
            active,
            omega,
            fade,
            static,
            q[trial_index],
            int(y[trial_index]),
            float(r[trial_index]),
            decoded,
            float(epsilon),
        )

    return Model0804Trace(
        nll=float(nll),
        probabilities=probabilities,
        marginal_hypothesis_prior=marginal_prior,
        marginal_active_probability=marginal_active,
        predictive_replacement_count=predictive_count,
        predictive_replacement_fraction=predictive_count / float(capacity),
        predictive_removed_mass=predictive_removed,
        predictive_newcomer_distance=predictive_distance,
        pre_choice_ess=pre_ess,
        post_choice_ess=post_ess,
        resampled=resampled,
        resampling_unique_ancestors=unique_ancestors,
        memory_sync_error=sync_error,
        final_weights=np.full(
            n_particles, 1.0 / float(n_particles), dtype=float
        ),
        particle_count=n_particles,
        capacity=int(capacity),
        filter_seed=int(filter_seed),
        transition_proposals_per_particle=1,
        integration_mode="alive_iid_categorical_choice",
        replacement_count_stratified=False,
        inference_method="alive_categorical",
        alive_attempt_count=attempt_count,
        alive_incremental_likelihood=incremental_likelihood,
    )


def _replay_model0804_window_dense(
    q: np.ndarray,
    y: np.ndarray,
    r: np.ndarray,
    p0: np.ndarray,
    kernels: TransitionKernels,
    parameters: Model0804Parameters,
    *,
    capacity: int,
    start_trial: int,
    stop_trial: int,
    condition: np.ndarray,
    particle_count: int,
    rng: np.random.Generator,
    anchor: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
    epsilon: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    """Replay an IID prior-transition path proposal over one fixed window."""

    first = int(start_trial)
    last = int(stop_trial)
    n_particles = int(particle_count)
    if first < 0 or last < first or last >= q.shape[0]:
        raise ValueError("invalid resample-move replay window")
    if first == 0:
        if anchor is not None:
            raise ValueError("a window starting at trial zero cannot have an anchor")
        active, omega, fade, static = _dense_initial_states_iid(
            p0, int(capacity), n_particles, rng
        )
    else:
        if anchor is None:
            raise ValueError("a positive-start replay window requires an anchor")
        active, omega, fade, static = (
            np.asarray(value).copy() for value in anchor
        )
        if active.shape != (n_particles, p0.size):
            raise ValueError("resample-move anchor has an incompatible shape")

    window_length = last - first + 1
    active_history = np.empty(
        (window_length, n_particles, p0.size), dtype=bool
    )
    omega_history = np.empty(
        (window_length, n_particles, p0.size), dtype=float
    )
    fade_history = np.empty_like(omega_history)
    static_history = np.empty_like(omega_history)
    log_choice_history = np.zeros((window_length, n_particles), dtype=float)
    maximum_sync_error = 0.0

    for offset, trial_index in enumerate(range(first, last + 1)):
        if trial_index > 0:
            transition_unit = rng.random(
                (
                    n_particles,
                    _transition_uniform_dimension(parameters, int(capacity)),
                )
            )
            (
                active,
                omega,
                fade,
                static,
                _,
                _,
                _,
                transition_sync,
            ) = _sample_transition_candidates_dense(
                active,
                omega,
                fade,
                static,
                p0,
                kernels,
                parameters,
                int(capacity),
                1,
                transition_unit,
            )
            maximum_sync_error = max(
                maximum_sync_error, float(np.max(transition_sync))
            )
        choice_probabilities = _choice_probability_dense(
            omega,
            q[trial_index],
            parameters.kappa,
            parameters.lapse,
        )
        if condition[trial_index]:
            log_choice_history[offset] = np.log(
                np.clip(
                    choice_probabilities[:, y[trial_index]],
                    float(epsilon),
                    1.0,
                )
            )
        omega, fade, static = _feedback_update_dense(
            active,
            omega,
            fade,
            static,
            q[trial_index],
            int(y[trial_index]),
            float(r[trial_index]),
            parameters,
            float(epsilon),
        )
        active_history[offset] = active
        omega_history[offset] = omega
        fade_history[offset] = fade
        static_history[offset] = static

    return (
        active_history,
        omega_history,
        fade_history,
        static_history,
        log_choice_history,
        np.sum(log_choice_history, axis=0),
        maximum_sync_error,
    )


def run_model0804_resample_move_particle_filter(
    q_values: np.ndarray,
    choices: np.ndarray,
    feedback: np.ndarray,
    prior: np.ndarray,
    kernels: TransitionKernels,
    *,
    model_id: str,
    parameters: Model0804Parameters,
    capacity: int,
    particle_count: int,
    filter_seed: int = 20260804,
    rejuvenation_window: int = 4,
    rejuvenation_sweeps: int = 1,
    score_mask: np.ndarray | None = None,
    condition_on_choice_mask: np.ndarray | None = None,
    epsilon: float = EPS,
) -> Model0804Trace:
    """Bootstrap-resample each choice, then rejuvenate a recent path window.

    The independent Metropolis proposal replays the original initial and
    transition kernels conditional on the fixed pre-window anchor.  Those
    proposal factors cancel from the MH ratio, leaving only the product of
    conditioned choice likelihoods inside the window.  Feedback remains a
    deterministic, post-choice state update and is never scored or leaked.
    """

    q, y, r, p0, decoded = _validate_inputs(
        q_values, choices, feedback, prior, kernels, capacity, model_id, parameters
    )
    if model_id == "FA0":
        raise ValueError(
            "resample-move is restricted to dynamic FA1/FA2; "
            "use exact initial-set integration for static FA0"
        )
    n_trials, n_hypotheses, n_categories = q.shape
    n_particles = int(particle_count)
    if n_particles < 2:
        raise ValueError("particle_count must be at least 2")
    lag = int(rejuvenation_window)
    sweeps = int(rejuvenation_sweeps)
    if lag != float(rejuvenation_window) or lag < 1:
        raise ValueError("rejuvenation_window must be a positive integer")
    if sweeps != float(rejuvenation_sweeps) or sweeps < 0:
        raise ValueError("rejuvenation_sweeps must be a non-negative integer")
    score, condition = _trial_masks(
        n_trials, score_mask, condition_on_choice_mask
    )

    probabilities = np.zeros((n_trials, n_categories), dtype=float)
    marginal_prior = np.zeros((n_trials, n_hypotheses), dtype=float)
    marginal_active = np.zeros((n_trials, n_hypotheses), dtype=float)
    predictive_count = np.zeros(n_trials, dtype=float)
    predictive_removed = np.zeros(n_trials, dtype=float)
    predictive_distance = np.zeros(n_trials, dtype=float)
    pre_ess = np.full(n_trials, float(n_particles), dtype=float)
    post_ess = np.full(n_trials, float(n_particles), dtype=float)
    resampled = np.asarray(condition, dtype=bool).copy()
    unique_ancestors = np.full(n_trials, n_particles, dtype=int)
    sync_error = np.zeros(n_trials, dtype=float)
    acceptance_rate = np.zeros(n_trials, dtype=float)
    unique_active_sets = np.zeros(n_trials, dtype=int)
    nll = 0.0

    active: np.ndarray | None = None
    omega: np.ndarray | None = None
    fade: np.ndarray | None = None
    static: np.ndarray | None = None
    history_active: np.ndarray | None = None
    history_omega: np.ndarray | None = None
    history_fade: np.ndarray | None = None
    history_static: np.ndarray | None = None
    history_log_choice: np.ndarray | None = None

    for trial_index in range(n_trials):
        transition_rng = np.random.default_rng(
            _stable_seed(filter_seed, "move_filter_transition", trial_index)
        )
        if trial_index == 0:
            active, omega, fade, static = _dense_initial_states_iid(
                p0, int(capacity), n_particles, transition_rng
            )
            candidate_count = np.zeros(n_particles, dtype=float)
            candidate_removed = np.zeros(n_particles, dtype=float)
            candidate_distance = np.zeros(n_particles, dtype=float)
            transition_sync = np.zeros(n_particles, dtype=float)
        else:
            if active is None or omega is None or fade is None or static is None:
                raise AssertionError("resample-move particle panel is uninitialized")
            transition_unit = transition_rng.random(
                (
                    n_particles,
                    _transition_uniform_dimension(decoded, int(capacity)),
                )
            )
            (
                active,
                omega,
                fade,
                static,
                candidate_count,
                candidate_removed,
                candidate_distance,
                transition_sync,
            ) = _sample_transition_candidates_dense(
                active,
                omega,
                fade,
                static,
                p0,
                kernels,
                decoded,
                int(capacity),
                1,
                transition_unit,
            )

        particle_probabilities = _choice_probability_dense(
            omega, q[trial_index], decoded.kappa, decoded.lapse
        )
        probabilities[trial_index] = _normalize(
            np.mean(particle_probabilities, axis=0),
            "resample-move marginal choice probabilities",
        )
        marginal_prior[trial_index] = _normalize(
            np.mean(omega, axis=0), "resample-move marginal rule prior"
        )
        marginal_active[trial_index] = np.mean(active, axis=0)
        predictive_count[trial_index] = float(np.mean(candidate_count))
        predictive_removed[trial_index] = float(np.mean(candidate_removed))
        predictive_distance[trial_index] = float(np.mean(candidate_distance))
        sync_error[trial_index] = float(np.max(transition_sync))

        observed_probability = float(
            probabilities[trial_index, y[trial_index]]
        )
        if score[trial_index]:
            nll -= math.log(max(observed_probability, float(epsilon)))
        if condition[trial_index]:
            choice_weights = _normalize(
                np.clip(
                    particle_probabilities[:, y[trial_index]],
                    float(epsilon),
                    1.0,
                ),
                "resample-move choice weights",
            )
            post_ess[trial_index] = effective_sample_size(choice_weights)
            resample_uniform = float(
                np.random.default_rng(
                    _stable_seed(filter_seed, "move_filter_resample", trial_index)
                ).random()
            )
            ancestors = systematic_resample(choice_weights, resample_uniform)
        else:
            ancestors = np.arange(n_particles, dtype=int)
        unique_ancestors[trial_index] = int(np.unique(ancestors).size)

        selected_log_choice = np.zeros(n_particles, dtype=float)
        if condition[trial_index]:
            selected_log_choice = np.log(
                np.clip(
                    particle_probabilities[ancestors, y[trial_index]],
                    float(epsilon),
                    1.0,
                )
            )
        active = active[ancestors].copy()
        omega = omega[ancestors].copy()
        fade = fade[ancestors].copy()
        static = static[ancestors].copy()
        if history_active is not None:
            history_active = history_active[:, ancestors].copy()
            history_omega = history_omega[:, ancestors].copy()
            history_fade = history_fade[:, ancestors].copy()
            history_static = history_static[:, ancestors].copy()
            history_log_choice = history_log_choice[:, ancestors].copy()

        omega, fade, static = _feedback_update_dense(
            active,
            omega,
            fade,
            static,
            q[trial_index],
            int(y[trial_index]),
            float(r[trial_index]),
            decoded,
            float(epsilon),
        )
        if history_active is None:
            history_active = active[None, ...].copy()
            history_omega = omega[None, ...].copy()
            history_fade = fade[None, ...].copy()
            history_static = static[None, ...].copy()
            history_log_choice = selected_log_choice[None, ...].copy()
        else:
            history_active = np.concatenate(
                [history_active, active[None, ...]], axis=0
            )
            history_omega = np.concatenate(
                [history_omega, omega[None, ...]], axis=0
            )
            history_fade = np.concatenate(
                [history_fade, fade[None, ...]], axis=0
            )
            history_static = np.concatenate(
                [history_static, static[None, ...]], axis=0
            )
            history_log_choice = np.concatenate(
                [history_log_choice, selected_log_choice[None, ...]], axis=0
            )

        window_start = max(0, trial_index - lag + 1)
        if window_start == 0:
            destination_start = 0
            anchor = None
        else:
            destination_start = 1
            anchor = (
                history_active[0],
                history_omega[0],
                history_fade[0],
                history_static[0],
            )
        current_window_log_likelihood = np.sum(
            history_log_choice[destination_start:], axis=0
        )
        accepted_total = 0
        for sweep_index in range(sweeps):
            move_rng = np.random.default_rng(
                _stable_seed(
                    filter_seed,
                    "fixed_lag_rejuvenation",
                    trial_index,
                    sweep_index,
                )
            )
            proposal = _replay_model0804_window_dense(
                q,
                y,
                r,
                p0,
                kernels,
                decoded,
                capacity=int(capacity),
                start_trial=window_start,
                stop_trial=trial_index,
                condition=condition,
                particle_count=n_particles,
                rng=move_rng,
                anchor=anchor,
                epsilon=float(epsilon),
            )
            proposal_log_likelihood = proposal[5]
            log_acceptance = np.minimum(
                0.0,
                proposal_log_likelihood - current_window_log_likelihood,
            )
            accepted = (
                np.log(
                    np.clip(move_rng.random(n_particles), float(epsilon), 1.0)
                )
                < log_acceptance
            )
            accepted_total += int(np.sum(accepted))
            if np.any(accepted):
                for proposal_offset, history_offset in enumerate(
                    range(destination_start, history_active.shape[0])
                ):
                    history_active[history_offset, accepted] = proposal[0][
                        proposal_offset, accepted
                    ]
                    history_omega[history_offset, accepted] = proposal[1][
                        proposal_offset, accepted
                    ]
                    history_fade[history_offset, accepted] = proposal[2][
                        proposal_offset, accepted
                    ]
                    history_static[history_offset, accepted] = proposal[3][
                        proposal_offset, accepted
                    ]
                    history_log_choice[history_offset, accepted] = proposal[4][
                        proposal_offset, accepted
                    ]
                current_window_log_likelihood[accepted] = (
                    proposal_log_likelihood[accepted]
                )
            sync_error[trial_index] = max(
                sync_error[trial_index], float(proposal[6])
            )

        if sweeps > 0:
            acceptance_rate[trial_index] = accepted_total / float(
                n_particles * sweeps
            )
        active = history_active[-1].copy()
        omega = history_omega[-1].copy()
        fade = history_fade[-1].copy()
        static = history_static[-1].copy()
        packed = np.packbits(active, axis=1)
        unique_active_sets[trial_index] = int(
            np.unique(packed, axis=0).shape[0]
        )
        if history_active.shape[0] > lag:
            history_active = history_active[-lag:].copy()
            history_omega = history_omega[-lag:].copy()
            history_fade = history_fade[-lag:].copy()
            history_static = history_static[-lag:].copy()
            history_log_choice = history_log_choice[-lag:].copy()

    return Model0804Trace(
        nll=float(nll),
        probabilities=probabilities,
        marginal_hypothesis_prior=marginal_prior,
        marginal_active_probability=marginal_active,
        predictive_replacement_count=predictive_count,
        predictive_replacement_fraction=predictive_count / float(capacity),
        predictive_removed_mass=predictive_removed,
        predictive_newcomer_distance=predictive_distance,
        pre_choice_ess=pre_ess,
        post_choice_ess=post_ess,
        resampled=resampled,
        resampling_unique_ancestors=unique_ancestors,
        memory_sync_error=sync_error,
        final_weights=np.full(
            n_particles, 1.0 / float(n_particles), dtype=float
        ),
        particle_count=n_particles,
        capacity=int(capacity),
        filter_seed=int(filter_seed),
        transition_proposals_per_particle=1,
        integration_mode="bootstrap_iid_fixed_lag_resample_move",
        replacement_count_stratified=False,
        inference_method="resample_move",
        rejuvenation_window=lag,
        rejuvenation_sweeps=sweeps,
        rejuvenation_acceptance_rate=acceptance_rate,
        rejuvenation_unique_active_sets=unique_active_sets,
    )


def combine_model0804_alive_islands(
    traces: Sequence[Model0804Trace],
    choices: np.ndarray,
) -> Model0804IslandEnsemble:
    """Combine independent alive filters without averaging their NLLs.

    Before trial ``t``, island ``j`` is weighted by its estimated cumulative
    evidence through ``t - 1``.  The combined observed-choice increment is
    therefore the ratio of the arithmetic-mean evidence estimates at ``t``
    and ``t - 1``.  Consequently, the final ensemble likelihood is exactly
    the arithmetic mean of the islands' likelihood estimates.
    """

    panels = list(traces)
    if not panels:
        raise ValueError("at least one alive trace is required")
    y = np.asarray(choices, dtype=int).reshape(-1)
    reference_shape = panels[0].probabilities.shape
    if len(reference_shape) != 2 or reference_shape[0] != y.size:
        raise ValueError("choices and alive probability arrays are incompatible")
    n_islands = len(panels)
    n_trials, n_categories = reference_shape
    probability_panel = np.empty(
        (n_islands, n_trials, n_categories), dtype=float
    )
    increment_panel = np.empty((n_islands, n_trials), dtype=float)
    for island_index, trace in enumerate(panels):
        if trace.inference_method != "alive_categorical":
            raise ValueError("every island must come from the categorical alive filter")
        if trace.probabilities.shape != reference_shape:
            raise ValueError("alive islands have incompatible probability shapes")
        if trace.alive_incremental_likelihood is None:
            raise ValueError("alive island is missing incremental likelihoods")
        increments = np.asarray(
            trace.alive_incremental_likelihood, dtype=float
        ).reshape(-1)
        if increments.size != n_trials or np.any(increments <= 0.0):
            raise ValueError("alive island increments are invalid")
        observed = trace.probabilities[np.arange(n_trials), y]
        if not np.allclose(observed, increments, atol=1e-14, rtol=1e-12):
            raise ValueError(
                "alive probabilities do not match their stopping-rule increments"
            )
        probability_panel[island_index] = trace.probabilities
        increment_panel[island_index] = increments

    combined_probabilities = np.zeros(reference_shape, dtype=float)
    combined_increment = np.zeros(n_trials, dtype=float)
    island_weights = np.zeros((n_trials, n_islands), dtype=float)
    effective_count = np.zeros(n_trials, dtype=float)
    log_evidence = np.zeros(n_islands, dtype=float)
    nll = 0.0
    for trial_index in range(n_trials):
        centered = log_evidence - float(np.max(log_evidence))
        weights = np.exp(centered)
        weights /= weights.sum()
        island_weights[trial_index] = weights
        effective_count[trial_index] = 1.0 / float(np.sum(np.square(weights)))
        combined_probabilities[trial_index] = _normalize(
            np.sum(
                weights[:, None] * probability_panel[:, trial_index, :],
                axis=0,
            ),
            "alive-island marginal choice probabilities",
        )
        combined_increment[trial_index] = float(
            np.sum(weights * increment_panel[:, trial_index])
        )
        observed = float(
            combined_probabilities[trial_index, y[trial_index]]
        )
        if not np.isclose(
            observed,
            combined_increment[trial_index],
            atol=1e-14,
            rtol=1e-12,
        ):
            raise AssertionError("alive-island evidence recursion is incoherent")
        nll -= math.log(combined_increment[trial_index])
        log_evidence += np.log(increment_panel[:, trial_index])

    final_center = log_evidence - float(np.max(log_evidence))
    log_mean_evidence = (
        float(np.max(log_evidence))
        + math.log(float(np.mean(np.exp(final_center))))
    )
    if not np.isclose(nll, -log_mean_evidence, atol=1e-10, rtol=1e-12):
        raise AssertionError("alive-island likelihood is not the mean evidence")
    return Model0804IslandEnsemble(
        nll=float(nll),
        probabilities=combined_probabilities,
        incremental_likelihood=combined_increment,
        pretrial_island_weights=island_weights,
        effective_island_count=effective_count,
        final_island_log_evidence=log_evidence,
        island_count=n_islands,
    )


def run_model0804_exact(
    q_values: np.ndarray,
    choices: np.ndarray,
    feedback: np.ndarray,
    prior: np.ndarray,
    kernels: TransitionKernels,
    *,
    model_id: str,
    parameters: Model0804Parameters,
    capacity: int,
    score_mask: np.ndarray | None = None,
    condition_on_choice_mask: np.ndarray | None = None,
    epsilon: float = EPS,
    max_branches: int = 250_000,
) -> ExactModel0804Trace:
    """Exactly sum all finite-set paths in a deliberately tiny rule space."""

    q, y, r, p0, decoded = _validate_inputs(
        q_values, choices, feedback, prior, kernels, capacity, model_id, parameters
    )
    n_trials, n_hypotheses, n_categories = q.shape
    score, condition = _trial_masks(
        n_trials, score_mask, condition_on_choice_mask
    )
    branches = [
        (state, float(probability))
        for state, probability in enumerate_initial_states(p0, int(capacity))
    ]
    probabilities = np.zeros((n_trials, n_categories), dtype=float)
    marginal_prior = np.zeros((n_trials, n_hypotheses), dtype=float)
    marginal_active = np.zeros((n_trials, n_hypotheses), dtype=float)
    branch_counts = np.zeros(n_trials, dtype=int)
    nll = 0.0

    for trial_index in range(n_trials):
        if trial_index > 0:
            expanded: list[tuple[HFWState, float]] = []
            for state, branch_weight in branches:
                for new_state, _, _, transition_probability in enumerate_transition_outcomes(
                    state, p0, kernels, decoded, int(capacity)
                ):
                    expanded.append(
                        (new_state, branch_weight * float(transition_probability))
                    )
                    if len(expanded) > int(max_branches):
                        raise RuntimeError(
                            "exact model_0804 branch limit exceeded; use a smaller test space"
                        )
            branches = expanded

        total_weight = sum(weight for _, weight in branches)
        if total_weight <= 0.0:
            raise RuntimeError("exact filter has zero branch mass")
        branches = [(state, weight / total_weight) for state, weight in branches]
        branch_counts[trial_index] = len(branches)
        for state, branch_weight in branches:
            prediction = _choice_probability(
                state, q[trial_index], decoded.kappa, decoded.lapse
            )
            probabilities[trial_index] += branch_weight * prediction
            marginal_prior[trial_index] += branch_weight * state.omega
            marginal_active[trial_index, state.active] += branch_weight
        probabilities[trial_index] = _normalize(
            probabilities[trial_index], "exact marginal choice probabilities"
        )
        marginal_prior[trial_index] = _normalize(
            marginal_prior[trial_index], "exact marginal rule prior"
        )
        observed_probability = float(probabilities[trial_index, y[trial_index]])
        if score[trial_index]:
            nll -= math.log(max(observed_probability, float(epsilon)))

        filtered: list[tuple[HFWState, float]] = []
        filtered_total = 0.0
        for state, branch_weight in branches:
            prediction = _choice_probability(
                state, q[trial_index], decoded.kappa, decoded.lapse
            )
            new_weight = branch_weight
            if condition[trial_index]:
                new_weight *= float(prediction[y[trial_index]])
            updated, _, _ = _feedback_update(
                state,
                q[trial_index],
                int(y[trial_index]),
                float(r[trial_index]),
                decoded,
                float(epsilon),
            )
            filtered.append((updated, new_weight))
            filtered_total += new_weight
        if filtered_total <= 0.0:
            raise RuntimeError("exact choice filtering has zero branch mass")
        branches = [
            (state, weight / filtered_total) for state, weight in filtered
        ]

    return ExactModel0804Trace(
        nll=float(nll),
        probabilities=probabilities,
        marginal_hypothesis_prior=marginal_prior,
        marginal_active_probability=marginal_active,
        branch_counts=branch_counts,
    )


def parameter_definition(model_id: str, memory_id: str = "dual") -> ParameterDefinition:
    """Return the first-gate bounded-ML schema for FA0--FA2."""

    if model_id not in HFW_MODEL_IDS:
        raise ValueError(f"unknown model_id {model_id!r}")
    if memory_id not in {"bayes", "fade", "dual"}:
        raise ValueError("memory_id must be bayes, fade, or dual")
    names: list[str] = []
    bounds: list[tuple[float, float]] = []
    starts: list[tuple[float, float]] = []
    center: list[float] = []

    def add(
        name: str,
        bound: tuple[float, float],
        start: tuple[float, float],
        center_value: float,
    ) -> None:
        names.append(name)
        bounds.append(bound)
        starts.append(start)
        center.append(float(center_value))

    if memory_id in {"fade", "dual"}:
        add("gamma", (0.02, 0.995), (0.15, 0.95), 0.70)
    if memory_id == "dual":
        add("w0", (0.005, 0.995), (0.05, 0.95), 0.40)
    add("log_kappa", (math.log(0.05), math.log(20.0)), (-1.25, 1.75), math.log(2.0))
    if model_id in {"FA1", "FA2", "FA2R"}:
        add("m", (0.0, 1.0), (0.02, 0.70), 0.15)
    if model_id in {"FA2", "FA2R"}:
        add("g", (0.0, 1.0), (0.02, 0.90), 0.35)
    if model_id == "FA2R":
        add("rho", (0.0, 0.25), (0.001, 0.10), 0.02)
    return ParameterDefinition(
        names=tuple(names),
        bounds=tuple(bounds),
        start_bounds=tuple(starts),
        center=np.asarray(center, dtype=float),
    )


def decode_parameters(
    raw_vector: np.ndarray,
    model_id: str,
    memory_id: str = "dual",
) -> tuple[Model0804Parameters, dict[str, float]]:
    definition = parameter_definition(model_id, memory_id)
    raw = np.asarray(raw_vector, dtype=float).reshape(-1)
    if raw.shape != definition.center.shape:
        raise ValueError("raw parameter vector does not match the FA schema")
    supplied = {name: float(raw[index]) for index, name in enumerate(definition.names)}
    if memory_id == "bayes":
        gamma, w0 = 1.0, 1.0
    elif memory_id == "fade":
        gamma, w0 = supplied["gamma"], 0.0
    else:
        gamma, w0 = supplied["gamma"], supplied["w0"]
    parameters = _validate_model_and_parameters(
        model_id,
        Model0804Parameters(
            gamma=float(gamma),
            w0=float(w0),
            kappa=float(math.exp(supplied["log_kappa"])),
            m=float(supplied.get("m", 0.0)),
            g=float(supplied.get("g", 0.0)),
            rho=float(supplied.get("rho", 0.0)),
        ),
    )
    reported = dict(supplied)
    reported.update(
        {
            "gamma": parameters.gamma,
            "w0": parameters.w0,
            "kappa": parameters.kappa,
            "m": parameters.m,
            "g": parameters.g,
            "rho": parameters.rho,
        }
    )
    return parameters, reported


def nested_child_start(parent: Model0804Fit, child_model: str) -> np.ndarray:
    expected = {"FA0": "FA1", "FA1": "FA2", "FA2": "FA2R"}
    if expected.get(parent.model_id) != child_model:
        raise ValueError("nested FA warm start requires a direct parent and child")
    parent_definition = parameter_definition(parent.model_id, parent.memory_id)
    child_definition = parameter_definition(child_model, parent.memory_id)
    mapping = {
        name: float(parent.raw_vector[index])
        for index, name in enumerate(parent_definition.names)
    }
    out = child_definition.center.copy()
    for index, name in enumerate(child_definition.names):
        if name in mapping:
            out[index] = mapping[name]
    if child_model == "FA1":
        out[child_definition.names.index("m")] = 0.0
    elif child_model == "FA2":
        out[child_definition.names.index("g")] = 0.0
    elif child_model == "FA2R":
        out[child_definition.names.index("rho")] = 0.0
    return out


def _sobol_starts(
    definition: ParameterDefinition,
    n_starts: int,
    seed: int,
    extra_starts: Iterable[np.ndarray] | None,
) -> list[np.ndarray]:
    if int(n_starts) < 1:
        raise ValueError("n_starts must be positive")
    starts = [definition.center.copy()]
    if extra_starts is not None:
        for value in extra_starts:
            array = np.asarray(value, dtype=float).reshape(-1)
            if array.shape == definition.center.shape:
                starts.append(
                    np.asarray(
                        [
                            np.clip(array[index], *definition.bounds[index])
                            for index in range(array.size)
                        ],
                        dtype=float,
                    )
                )
    remaining = max(0, int(n_starts) - len(starts))
    if remaining:
        exponent = int(math.ceil(math.log2(remaining)))
        unit = qmc.Sobol(
            d=len(definition.names), scramble=True, seed=int(seed)
        ).random_base2(exponent)[:remaining]
        for row in unit:
            starts.append(
                np.asarray(
                    [
                        low + float(row[index]) * (high - low)
                        for index, (low, high) in enumerate(definition.start_bounds)
                    ],
                    dtype=float,
                )
            )
    return starts[: int(n_starts)]


def fit_model0804(
    q_values: np.ndarray,
    choices: np.ndarray,
    feedback: np.ndarray,
    prior: np.ndarray,
    kernels: TransitionKernels,
    train_mask: np.ndarray,
    *,
    model_id: str,
    capacity: int,
    memory_id: str = "dual",
    particle_count: int = 256,
    filter_seed: int = 20260804,
    transition_proposals_per_particle: int = 1,
    stratify_replacement_count: bool = False,
    fa0_exact_initial_sets: bool = False,
    fa0_maximum_exact_initial_sets: int = 1_000_000,
    n_starts: int = 4,
    base_seed: int = 20260804,
    seed_parts: Sequence[object] = (),
    extra_starts: Iterable[np.ndarray] | None = None,
    maxiter: int = 250,
    epsilon: float = EPS,
) -> Model0804Fit:
    """Fit one FA candidate with a common-random-number particle objective."""

    train = np.asarray(train_mask, dtype=bool).reshape(-1)
    if train.size != np.asarray(choices).reshape(-1).size or not train.any():
        raise ValueError("train_mask must contain at least one trial")
    rows = np.flatnonzero(train)
    if not np.array_equal(rows, np.arange(rows[-1] + 1)):
        raise ValueError("model_0804 fitting requires a contiguous training prefix")
    stop = int(rows[-1]) + 1
    definition = parameter_definition(model_id, memory_id)
    start_seed = _stable_seed(base_seed, model_id, memory_id, *seed_parts)
    starts = _sobol_starts(definition, n_starts, start_seed, extra_starts)

    def objective(raw: np.ndarray) -> float:
        try:
            parameters, _ = decode_parameters(raw, model_id, memory_id)
            trace = run_model0804_particle_filter(
                np.asarray(q_values)[:stop],
                np.asarray(choices)[:stop],
                np.asarray(feedback)[:stop],
                prior,
                kernels,
                model_id=model_id,
                parameters=parameters,
                capacity=int(capacity),
                particle_count=int(particle_count),
                filter_seed=int(filter_seed),
                resample_threshold_fraction=0.0 if model_id == "FA0" else 0.5,
                transition_proposals_per_particle=int(
                    transition_proposals_per_particle
                ),
                stratify_replacement_count=bool(stratify_replacement_count),
                fa0_exact_initial_sets=bool(
                    fa0_exact_initial_sets
                ),
                fa0_maximum_exact_initial_sets=int(
                    fa0_maximum_exact_initial_sets
                ),
                epsilon=float(epsilon),
            )
            return float(trace.nll) if np.isfinite(trace.nll) else 1e100
        except (FloatingPointError, ValueError, RuntimeError):
            return 1e100

    results: list[tuple[OptimizeResult, float]] = []
    for start in starts:
        initial_nll = objective(start)
        result = minimize(
            objective,
            np.asarray(start, dtype=float),
            method="Powell",
            bounds=definition.bounds,
            options={"maxiter": int(maxiter), "xtol": 1e-5, "ftol": 1e-7},
        )
        results.append((result, initial_nll))
    finite = [item for item in results if np.isfinite(float(item[0].fun))]
    if not finite:
        raise RuntimeError(f"all model_0804 optimization starts failed for {model_id}")
    successful = [item for item in finite if bool(item[0].success)]
    pool = successful if successful else finite
    best_result, best_initial = min(pool, key=lambda item: float(item[0].fun))
    parameters, reported = decode_parameters(best_result.x, model_id, memory_id)
    boundary_names = []
    for index, name in enumerate(definition.names):
        low, high = definition.bounds[index]
        tolerance = 1e-5 * max(1.0, abs(low), abs(high))
        if abs(float(best_result.x[index]) - low) <= tolerance or abs(
            float(best_result.x[index]) - high
        ) <= tolerance:
            boundary_names.append(name)
    diagnostics = {
        "method": "Powell_CRN",
        "success": bool(best_result.success),
        "status": int(best_result.status),
        "message": str(best_result.message),
        "nfev": int(best_result.nfev),
        "nit": int(getattr(best_result, "nit", -1)),
        "n_starts": int(len(starts)),
        "successful_starts": int(len(successful)),
        "selected_initial_nll": float(best_initial),
        "boundary_parameters": boundary_names,
        "particle_count": int(particle_count),
        "filter_seed": int(filter_seed),
        "transition_proposals_per_particle": int(
            transition_proposals_per_particle
        ),
        "stratify_replacement_count": bool(stratify_replacement_count),
        "fa0_exact_initial_sets": bool(
            fa0_exact_initial_sets
        ),
    }
    return Model0804Fit(
        model_id=model_id,
        memory_id=memory_id,
        raw_vector=np.asarray(best_result.x, dtype=float),
        parameters=parameters,
        reported_parameters=reported,
        train_nll=float(best_result.fun),
        diagnostics=diagnostics,
    )


__all__ = [
    "EPS",
    "ExactModel0804Trace",
    "FA_MODEL_IDS",
    "HFW_MODEL_IDS",
    "HFWState",
    "Model0804Fit",
    "Model0804Parameters",
    "Model0804Trace",
    "TransitionSummary",
    "decode_parameters",
    "effective_sample_size",
    "enumerate_initial_states",
    "enumerate_transition_outcomes",
    "enumerate_weighted_wor",
    "fit_model0804",
    "nested_child_start",
    "parameter_definition",
    "run_model0804_exact",
    "run_model0804_particle_filter",
    "systematic_resample",
]

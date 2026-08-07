"""Deterministic condition-1 implementation of ``manuscript/model_0803.tex``.

This module implements the choice-only, full-hypothesis-space H0--H3 gate.
It deliberately stays separate from the historical finite-capacity transition
engine.  The scientific scope is narrow and explicit:

* fixed Task-1b integrated rule predictions ``q[t, h, c]``;
* fixed local/global transition kernels built from labelled rule geometry;
* the synchronized fade/static memory recursion;
* H0, H1, H2, H3-M, and H3-MG deterministic control states;
* subject-wise maximum-likelihood fitting on a frozen temporal training split.

RT, oral-report likelihoods, random H4 paths, H5 states, hierarchical posterior
sampling, and autonomous posterior-predictive validation are not implemented
here and must not be inferred from the outputs of this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.stats import qmc

from ..problems.partitions import Partition

try:  # pragma: no cover - the production environment includes numba.
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):
        def decorate(func):
            return func

        return decorate


EPS = 1e-12
ORDER_COLUMNS = ("iSession", "iBlock", "iTrial")
MODEL_IDS = ("H0", "H1", "H2", "H3_M", "H3_MG")
MEMORY_IDS = ("bayes", "fade", "dual")
MODEL_CODE = {name: index for index, name in enumerate(MODEL_IDS)}


def n_categories(condition: int) -> int:
    """Return the task category count for a condition."""

    return 2 if int(condition) == 1 else 4


def build_partition(condition: int) -> Partition:
    """Build the labelled rule space used by the frozen model_0803 analysis."""

    return Partition(
        n_dims=4,
        n_cats=n_categories(condition),
        include_label_reversals=int(condition) == 1,
    )


def partition_prior(partition: Partition, mode: str = "uniform_rule") -> np.ndarray:
    """Return either a rule-uniform or split-family-uniform prior."""

    mode = str(mode).strip().lower()
    if mode == "uniform_rule":
        return np.full(partition.length, 1.0 / partition.length, dtype=float)
    if mode != "uniform_family":
        raise ValueError("prior mode must be 'uniform_rule' or 'uniform_family'")

    families = np.asarray([split.type for split in partition.splits], dtype=object)
    unique_families = sorted(set(families.tolist()))
    prior = np.zeros(partition.length, dtype=float)
    for family in unique_families:
        indices = np.flatnonzero(families == family)
        prior[indices] = 1.0 / (len(unique_families) * len(indices))
    return prior


def expected_feedback_from_category(
    condition: int,
    choices_one_based: np.ndarray,
    categories_one_based: np.ndarray,
) -> np.ndarray:
    """Reconstruct task feedback from recorded one-based choices/categories."""

    choices = np.asarray(choices_one_based, dtype=int) - 1
    categories = np.asarray(categories_one_based, dtype=int) - 1
    if int(condition) in (1, 2):
        return (choices == categories).astype(float)
    exact = choices == categories
    same_family = (choices // 2) == (categories // 2)
    return np.where(exact, 1.0, np.where(same_family, 0.5, 0.0))


@dataclass(frozen=True)
class TransitionKernels:
    """Frozen local/global kernels and their geometry audit."""

    local: np.ndarray
    global_: np.ndarray
    distance: np.ndarray
    tau_local: float
    expected_local_distance: np.ndarray
    expected_global_distance: np.ndarray


@dataclass(frozen=True)
class FeatureScaling:
    """Training-only reference centring for H3 dynamic inputs."""

    center: np.ndarray
    scale: np.ndarray
    reference: str


@dataclass
class Model0803Trace:
    """Trialwise predictions and latent states from one deterministic run."""

    nll: float
    probabilities: np.ndarray
    pi_minus: np.ndarray
    pi_plus: np.ndarray
    fade_state: np.ndarray
    static_state: np.ndarray
    m: np.ndarray
    g: np.ndarray
    operation_weights: np.ndarray
    feedback_surprise: np.ndarray
    rule_uncertainty: np.ndarray
    memory_sync_error: np.ndarray


@dataclass(frozen=True)
class ParameterDefinition:
    """Optimizer-facing parameter schema."""

    names: tuple[str, ...]
    bounds: tuple[tuple[float, float], ...]
    start_bounds: tuple[tuple[float, float], ...]
    center: np.ndarray


@dataclass
class Model0803Fit:
    """Best multi-start fit for one subject/model/memory combination."""

    model_id: str
    memory_id: str
    raw_vector: np.ndarray
    full_parameters: np.ndarray
    parameters: dict[str, float]
    train_nll: float
    diagnostics: dict[str, Any]


def _as_probability_vector(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size < 2:
        raise ValueError(f"{name} must contain at least two hypotheses")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(f"{name} must be finite and strictly positive")
    return values / values.sum()


def build_transition_kernels(
    similarity: np.ndarray,
    prior: np.ndarray,
    *,
    tau_local: float | None = None,
) -> TransitionKernels:
    """Build the frozen :math:`K_L` and :math:`K_G` matrices.

    If ``tau_local`` is omitted, it is fixed solely from rule geometry as the
    median nearest-nonself labelled-rule distance.  No behavioural outcome is
    consulted.  Rows index the source hypothesis and columns the destination.
    """

    similarity = np.asarray(similarity, dtype=np.float64)
    prior = _as_probability_vector(prior, "prior")
    n_hypotheses = prior.size
    if similarity.shape != (n_hypotheses, n_hypotheses):
        raise ValueError(
            "similarity shape must match the prior: "
            f"{similarity.shape} vs {(n_hypotheses, n_hypotheses)}"
        )
    if not np.all(np.isfinite(similarity)):
        raise ValueError("similarity must contain only finite values")
    if np.max(np.abs(similarity - similarity.T)) > 1e-10:
        raise ValueError("similarity must be symmetric")
    if not np.allclose(np.diag(similarity), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("similarity diagonal must equal one")
    if np.any(similarity < -1e-12) or np.any(similarity > 1.0 + 1e-12):
        raise ValueError("similarity must lie in [0, 1]")

    distance = np.clip(1.0 - similarity, 0.0, 1.0)
    nonself = distance.copy()
    np.fill_diagonal(nonself, np.inf)
    nearest = np.min(nonself, axis=1)
    if tau_local is None:
        tau_local = float(np.median(nearest))
    tau_local = float(tau_local)
    if not np.isfinite(tau_local) or tau_local <= 0.0:
        raise ValueError("tau_local must be finite and positive")

    local = np.zeros_like(distance)
    global_kernel = np.zeros_like(distance)
    for source in range(n_hypotheses):
        local_weights = prior * np.exp(-distance[source] / tau_local)
        local_weights[source] = 0.0
        local_total = float(local_weights.sum())
        if local_total <= 0.0:
            raise ValueError(f"local kernel row {source} has zero mass")
        local[source] = local_weights / local_total

        global_weights = prior.copy()
        global_weights[source] = 0.0
        global_total = float(global_weights.sum())
        if global_total <= 0.0:
            raise ValueError(f"global kernel row {source} has zero mass")
        global_kernel[source] = global_weights / global_total

    if not np.allclose(local.sum(axis=1), 1.0, atol=1e-12, rtol=0.0):
        raise AssertionError("local transition kernel is not row-normalized")
    if not np.allclose(global_kernel.sum(axis=1), 1.0, atol=1e-12, rtol=0.0):
        raise AssertionError("global transition kernel is not row-normalized")
    if np.any(np.diag(local) != 0.0) or np.any(np.diag(global_kernel) != 0.0):
        raise AssertionError("transition kernels must exclude the source rule")

    expected_local = np.sum(local * distance, axis=1)
    expected_global = np.sum(global_kernel * distance, axis=1)
    return TransitionKernels(
        local=local,
        global_=global_kernel,
        distance=distance,
        tau_local=tau_local,
        expected_local_distance=expected_local,
        expected_global_distance=expected_global,
    )


@njit(cache=True, nogil=True)
def _expit_scalar(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


@njit(cache=True, nogil=True)
def _entropy_vector(values: np.ndarray, epsilon: float) -> float:
    total = 0.0
    for index in range(values.size):
        value = values[index]
        if value < epsilon:
            value = epsilon
        total -= value * math.log(value)
    return total


@njit(cache=True, nogil=True)
def _run_model0803_numba(
    q_values: np.ndarray,
    choices: np.ndarray,
    feedback: np.ndarray,
    prior: np.ndarray,
    local_kernel: np.ndarray,
    global_kernel: np.ndarray,
    model_code: int,
    full_parameters: np.ndarray,
    feature_center: np.ndarray,
    feature_scale: np.ndarray,
    score_mask: np.ndarray,
    record_states: bool,
    epsilon: float,
):
    n_trials = q_values.shape[0]
    n_hypotheses = q_values.shape[1]
    n_categories = q_values.shape[2]

    probabilities = np.empty((n_trials, n_categories), dtype=np.float64)
    m_values = np.zeros(n_trials, dtype=np.float64)
    g_values = np.zeros(n_trials, dtype=np.float64)
    operation_weights = np.zeros((n_trials, 3), dtype=np.float64)
    feedback_surprise = np.zeros(n_trials, dtype=np.float64)
    rule_uncertainty = np.zeros(n_trials, dtype=np.float64)
    memory_sync_error = np.zeros(n_trials, dtype=np.float64)

    if record_states:
        pi_minus_log = np.empty((n_trials, n_hypotheses), dtype=np.float64)
        pi_plus_log = np.empty((n_trials, n_hypotheses), dtype=np.float64)
        fade_log = np.empty((n_trials, n_hypotheses), dtype=np.float64)
        static_log = np.empty((n_trials, n_hypotheses), dtype=np.float64)
    else:
        pi_minus_log = np.empty((1, 1), dtype=np.float64)
        pi_plus_log = np.empty((1, 1), dtype=np.float64)
        fade_log = np.empty((1, 1), dtype=np.float64)
        static_log = np.empty((1, 1), dtype=np.float64)

    gamma = full_parameters[0]
    w0 = full_parameters[1]
    kappa = full_parameters[2]
    mu_m = full_parameters[3]
    mu_g = full_parameters[4]
    phi_m = full_parameters[5]
    b_m_surprise = full_parameters[6]
    b_m_uncertainty = full_parameters[7]
    phi_g = full_parameters[8]
    b_g_surprise = full_parameters[9]
    b_g_uncertainty = full_parameters[10]

    pi_plus = prior.copy()
    fade_state = np.empty(n_hypotheses, dtype=np.float64)
    static_state = np.empty(n_hypotheses, dtype=np.float64)
    for hypothesis in range(n_hypotheses):
        log_value = math.log(prior[hypothesis])
        fade_state[hypothesis] = log_value
        static_state[hypothesis] = log_value

    a_m = mu_m
    a_g = mu_g
    previous_surprise = feature_center[0]
    previous_uncertainty = feature_center[1]
    log_hypothesis_count = math.log(n_hypotheses)
    nll = 0.0

    pi_minus = np.empty(n_hypotheses, dtype=np.float64)
    local_proposal = np.empty(n_hypotheses, dtype=np.float64)
    global_proposal = np.empty(n_hypotheses, dtype=np.float64)
    fade_pre = np.empty(n_hypotheses, dtype=np.float64)
    static_pre = np.empty(n_hypotheses, dtype=np.float64)
    delta = np.empty(n_hypotheses, dtype=np.float64)
    likelihood = np.empty(n_hypotheses, dtype=np.float64)
    ell = np.empty(n_hypotheses, dtype=np.float64)

    for trial in range(n_trials):
        if model_code >= 3 and trial > 0:
            z_surprise = (previous_surprise - feature_center[0]) / feature_scale[0]
            z_uncertainty = (
                previous_uncertainty - feature_center[1]
            ) / feature_scale[1]
            a_m = (
                mu_m
                + phi_m * (a_m - mu_m)
                + b_m_surprise * z_surprise
                + b_m_uncertainty * z_uncertainty
            )
            if model_code >= 4:
                a_g = (
                    mu_g
                    + phi_g * (a_g - mu_g)
                    + b_g_surprise * z_surprise
                    + b_g_uncertainty * z_uncertainty
                )

        if model_code == 0:
            m_value = 0.0
            g_value = 0.0
        elif model_code == 1:
            # H1/H2 are explicit closed-interval boundary models.  Keeping
            # their constants on [0, 1] (rather than finite logits) preserves
            # the exact H0 -> H1 -> H2 nesting specified in the manuscript.
            m_value = a_m
            g_value = 0.0
        elif model_code == 2:
            m_value = a_m
            g_value = a_g
        else:
            m_value = _expit_scalar(a_m)
            g_value = _expit_scalar(a_g)

        m_values[trial] = m_value
        g_values[trial] = g_value
        operation_weights[trial, 0] = 1.0 - m_value
        operation_weights[trial, 1] = m_value * (1.0 - g_value)
        operation_weights[trial, 2] = m_value * g_value

        if model_code == 0:
            for destination in range(n_hypotheses):
                pi_minus[destination] = pi_plus[destination]
        else:
            for destination in range(n_hypotheses):
                local_value = 0.0
                global_value = 0.0
                for source in range(n_hypotheses):
                    local_value += pi_plus[source] * local_kernel[source, destination]
                    global_value += pi_plus[source] * global_kernel[source, destination]
                local_proposal[destination] = local_value
                global_proposal[destination] = global_value
                pi_minus[destination] = (
                    (1.0 - m_value) * pi_plus[destination]
                    + m_value * (1.0 - g_value) * local_value
                    + m_value * g_value * global_value
                )

        pi_minus_total = 0.0
        for hypothesis in range(n_hypotheses):
            pi_minus_total += pi_minus[hypothesis]
        for hypothesis in range(n_hypotheses):
            pi_minus[hypothesis] /= pi_minus_total

        for hypothesis in range(n_hypotheses):
            value = pi_minus[hypothesis]
            if value < epsilon:
                value = epsilon
            log_prior_value = math.log(value)
            delta[hypothesis] = static_state[hypothesis] - fade_state[hypothesis]
            fade_pre[hypothesis] = log_prior_value - w0 * delta[hypothesis]
            static_pre[hypothesis] = (
                log_prior_value + (1.0 - w0) * delta[hypothesis]
            )

        maximum_sync_error = 0.0
        for hypothesis in range(n_hypotheses):
            combined = (
                w0 * static_pre[hypothesis]
                + (1.0 - w0) * fade_pre[hypothesis]
            )
            expected = math.log(max(pi_minus[hypothesis], epsilon))
            sync_error = abs(combined - expected)
            if sync_error > maximum_sync_error:
                maximum_sync_error = sync_error
        memory_sync_error[trial] = maximum_sync_error

        core_total = 0.0
        for category in range(n_categories):
            core_value = 0.0
            for hypothesis in range(n_hypotheses):
                core_value += pi_minus[hypothesis] * q_values[trial, hypothesis, category]
            if core_value < epsilon:
                core_value = epsilon
            probabilities[trial, category] = core_value
            core_total += core_value
        for category in range(n_categories):
            probabilities[trial, category] /= core_total

        maximum_logit = -1e300
        for category in range(n_categories):
            value = kappa * math.log(max(probabilities[trial, category], epsilon))
            probabilities[trial, category] = value
            if value > maximum_logit:
                maximum_logit = value
        choice_total = 0.0
        for category in range(n_categories):
            value = math.exp(probabilities[trial, category] - maximum_logit)
            probabilities[trial, category] = value
            choice_total += value
        for category in range(n_categories):
            probabilities[trial, category] /= choice_total

        observed_choice = choices[trial]
        if score_mask[trial]:
            observed_probability = probabilities[trial, observed_choice]
            nll -= math.log(max(observed_probability, epsilon))

        if feedback[trial] >= 0.5:
            compatible_category = observed_choice
        else:
            compatible_category = 1 - observed_choice
        feedback_probability = 0.0
        for hypothesis in range(n_hypotheses):
            value = q_values[trial, hypothesis, compatible_category]
            if value < epsilon:
                value = epsilon
            likelihood[hypothesis] = value
            feedback_probability += pi_minus[hypothesis] * value
        current_surprise = -math.log(max(feedback_probability, epsilon))
        feedback_surprise[trial] = current_surprise

        maximum_ell = -1e300
        for hypothesis in range(n_hypotheses):
            log_likelihood = math.log(likelihood[hypothesis])
            fade_state[hypothesis] = gamma * fade_pre[hypothesis] + log_likelihood
            static_state[hypothesis] = static_pre[hypothesis] + log_likelihood
            ell[hypothesis] = (
                w0 * static_state[hypothesis]
                + (1.0 - w0) * fade_state[hypothesis]
            )
            if ell[hypothesis] > maximum_ell:
                maximum_ell = ell[hypothesis]
        posterior_total = 0.0
        for hypothesis in range(n_hypotheses):
            value = math.exp(ell[hypothesis] - maximum_ell)
            pi_plus[hypothesis] = value
            posterior_total += value
        for hypothesis in range(n_hypotheses):
            pi_plus[hypothesis] /= posterior_total

        current_uncertainty = (
            _entropy_vector(pi_plus, epsilon) / log_hypothesis_count
        )
        rule_uncertainty[trial] = current_uncertainty
        previous_surprise = current_surprise
        previous_uncertainty = current_uncertainty

        if record_states:
            for hypothesis in range(n_hypotheses):
                pi_minus_log[trial, hypothesis] = pi_minus[hypothesis]
                pi_plus_log[trial, hypothesis] = pi_plus[hypothesis]
                fade_log[trial, hypothesis] = fade_state[hypothesis]
                static_log[trial, hypothesis] = static_state[hypothesis]

    return (
        nll,
        probabilities,
        pi_minus_log,
        pi_plus_log,
        fade_log,
        static_log,
        m_values,
        g_values,
        operation_weights,
        feedback_surprise,
        rule_uncertainty,
        memory_sync_error,
    )


def _validate_sequence_inputs(
    q_values: np.ndarray,
    choices: np.ndarray,
    feedback: np.ndarray,
    prior: np.ndarray,
    kernels: TransitionKernels,
    score_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q_values = np.ascontiguousarray(np.asarray(q_values, dtype=np.float64))
    choices = np.ascontiguousarray(np.asarray(choices, dtype=np.int64).reshape(-1))
    feedback = np.ascontiguousarray(np.asarray(feedback, dtype=np.float64).reshape(-1))
    prior = np.ascontiguousarray(_as_probability_vector(prior, "prior"))
    if q_values.ndim != 3 or q_values.shape[2] != 2:
        raise ValueError("condition-1 q_values must have shape [trial, hypothesis, 2]")
    if q_values.shape[0] != choices.size or choices.size != feedback.size:
        raise ValueError("q_values, choices, and feedback must have matching trials")
    if q_values.shape[1] != prior.size:
        raise ValueError("q_values hypothesis count must match prior")
    if not np.all(np.isfinite(q_values)) or np.any(q_values < 0.0):
        raise ValueError("q_values must be finite and non-negative")
    if not np.allclose(q_values.sum(axis=2), 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("q_values must normalize across categories")
    if np.any((choices < 0) | (choices > 1)):
        raise ValueError("condition-1 choices must be zero-based values 0 or 1")
    if np.any(~np.isclose(feedback, 0.0) & ~np.isclose(feedback, 1.0)):
        raise ValueError("condition-1 feedback must contain only 0 or 1")
    if kernels.local.shape != (prior.size, prior.size):
        raise ValueError("transition kernels do not match the hypothesis space")
    if score_mask is None:
        score_mask = np.ones(choices.size, dtype=bool)
    else:
        score_mask = np.asarray(score_mask, dtype=bool).reshape(-1)
        if score_mask.size != choices.size:
            raise ValueError("score_mask must have one value per trial")
    return (
        q_values,
        choices,
        feedback,
        prior,
        np.ascontiguousarray(score_mask),
    )


def run_model0803(
    q_values: np.ndarray,
    choices: np.ndarray,
    feedback: np.ndarray,
    prior: np.ndarray,
    kernels: TransitionKernels,
    *,
    model_id: str,
    full_parameters: np.ndarray,
    feature_scaling: FeatureScaling | None = None,
    score_mask: np.ndarray | None = None,
    record_states: bool = True,
    epsilon: float = EPS,
) -> Model0803Trace:
    """Run one deterministic H0--H3 sequence in strict trial order."""

    if model_id not in MODEL_CODE:
        raise ValueError(f"unknown model_id {model_id!r}; expected one of {MODEL_IDS}")
    full_parameters = np.asarray(full_parameters, dtype=np.float64).reshape(-1)
    if full_parameters.shape != (11,):
        raise ValueError("full_parameters must have length 11")
    gamma, w0, kappa = full_parameters[:3]
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must lie in (0, 1]")
    if not 0.0 <= w0 <= 1.0:
        raise ValueError("w0 must lie in [0, 1]")
    if not np.isfinite(kappa) or kappa <= 0.0:
        raise ValueError("kappa must be finite and positive")
    if not np.isfinite(epsilon) or not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must lie in (0, 0.5)")

    q_values, choices, feedback, prior, score_mask = _validate_sequence_inputs(
        q_values, choices, feedback, prior, kernels, score_mask
    )
    if feature_scaling is None:
        feature_scaling = FeatureScaling(
            center=np.zeros(2, dtype=float),
            scale=np.ones(2, dtype=float),
            reference="identity",
        )
    center = np.asarray(feature_scaling.center, dtype=np.float64).reshape(-1)
    scale = np.asarray(feature_scaling.scale, dtype=np.float64).reshape(-1)
    if center.shape != (2,) or scale.shape != (2,):
        raise ValueError("feature scaling center and scale must have length two")
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(scale)):
        raise ValueError("feature scaling must be finite")
    if np.any(scale <= 0.0):
        raise ValueError("feature scaling values must be positive")

    raw = _run_model0803_numba(
        q_values,
        choices,
        feedback,
        prior,
        np.ascontiguousarray(kernels.local, dtype=np.float64),
        np.ascontiguousarray(kernels.global_, dtype=np.float64),
        int(MODEL_CODE[model_id]),
        np.ascontiguousarray(full_parameters),
        np.ascontiguousarray(center),
        np.ascontiguousarray(scale),
        score_mask,
        bool(record_states),
        float(epsilon),
    )
    return Model0803Trace(
        nll=float(raw[0]),
        probabilities=np.asarray(raw[1]),
        pi_minus=np.asarray(raw[2]),
        pi_plus=np.asarray(raw[3]),
        fade_state=np.asarray(raw[4]),
        static_state=np.asarray(raw[5]),
        m=np.asarray(raw[6]),
        g=np.asarray(raw[7]),
        operation_weights=np.asarray(raw[8]),
        feedback_surprise=np.asarray(raw[9]),
        rule_uncertainty=np.asarray(raw[10]),
        memory_sync_error=np.asarray(raw[11]),
    )


def reference_feature_scaling(
    q_values: np.ndarray,
    choices: np.ndarray,
    feedback: np.ndarray,
    prior: np.ndarray,
    kernels: TransitionKernels,
    train_mask: np.ndarray,
    *,
    epsilon: float = EPS,
) -> FeatureScaling:
    """Freeze H3 input scaling from a parameter-free H0 Bayes reference.

    The reference uses ``gamma=1``, ``w0=1``, ``kappa=1`` and only training
    trials.  This prevents circular, parameter-dependent standardization and
    prevents the held-out suffix from determining H3 input scales.
    """

    parameters = np.zeros(11, dtype=float)
    parameters[0] = 1.0
    parameters[1] = 1.0
    parameters[2] = 1.0
    trace = run_model0803(
        q_values,
        choices,
        feedback,
        prior,
        kernels,
        model_id="H0",
        full_parameters=parameters,
        score_mask=np.asarray(train_mask, dtype=bool),
        record_states=False,
        epsilon=epsilon,
    )
    train_mask = np.asarray(train_mask, dtype=bool)
    features = np.column_stack(
        [trace.feedback_surprise[train_mask], trace.rule_uncertainty[train_mask]]
    )
    center = np.mean(features, axis=0)
    scale = np.std(features, axis=0, ddof=0)
    scale = np.maximum(scale, 1e-6)
    return FeatureScaling(
        center=center.astype(float),
        scale=scale.astype(float),
        reference="training_H0_bayes_gamma1_w1_kappa1",
    )


def parameter_definition(model_id: str, memory_id: str) -> ParameterDefinition:
    """Return the frozen bounded-ML schema for one candidate."""

    if model_id not in MODEL_CODE:
        raise ValueError(f"unknown model_id {model_id!r}")
    if memory_id not in MEMORY_IDS:
        raise ValueError(f"unknown memory_id {memory_id!r}")

    names: list[str] = []
    bounds: list[tuple[float, float]] = []
    start_bounds: list[tuple[float, float]] = []
    center: list[float] = []

    def add(
        name: str,
        bound: tuple[float, float],
        start: tuple[float, float],
        center_value: float,
    ) -> None:
        names.append(name)
        bounds.append(bound)
        start_bounds.append(start)
        center.append(float(center_value))

    if memory_id in {"fade", "dual"}:
        add("gamma", (0.02, 0.995), (0.15, 0.95), 0.70)
    if memory_id == "dual":
        add("w0", (0.005, 0.995), (0.05, 0.95), 0.40)
    add("log_kappa", (math.log(0.05), math.log(20.0)), (-1.25, 1.75), math.log(2.0))

    if model_id in {"H1", "H2"}:
        add("m", (0.0, 1.0), (0.02, 0.70), 0.15)
    elif model_id in {"H3_M", "H3_MG"}:
        # H3 uses open-interval logits, whereas H2 explicitly includes the
        # m/g boundaries.  Wide finite optimization bounds let the H3 closure
        # reproduce an H2 boundary to numerical tolerance without infinities.
        add("mu_m", (-30.0, 30.0), (-3.5, 0.5), math.log(0.15 / 0.85))
    if model_id == "H2":
        add("g", (0.0, 1.0), (0.02, 0.90), 0.35)
    elif model_id in {"H3_M", "H3_MG"}:
        add("mu_g", (-30.0, 30.0), (-3.0, 1.0), math.log(0.35 / 0.65))
    if model_id in {"H3_M", "H3_MG"}:
        add("phi_m", (-0.95, 0.95), (-0.30, 0.85), 0.50)
        add("b_m_surprise", (-4.0, 4.0), (-1.25, 1.25), 0.40)
        add("b_m_uncertainty", (-4.0, 4.0), (-1.25, 1.25), 0.40)
    if model_id == "H3_MG":
        add("phi_g", (-0.95, 0.95), (-0.30, 0.85), 0.50)
        add("b_g_surprise", (-4.0, 4.0), (-1.25, 1.25), 0.25)
        add("b_g_uncertainty", (-4.0, 4.0), (-1.25, 1.25), 0.25)

    return ParameterDefinition(
        names=tuple(names),
        bounds=tuple(bounds),
        start_bounds=tuple(start_bounds),
        center=np.asarray(center, dtype=float),
    )


def decode_parameters(
    raw_vector: np.ndarray,
    model_id: str,
    memory_id: str,
) -> tuple[np.ndarray, dict[str, float]]:
    """Map an optimizer vector to the fixed 11-value recursion vector."""

    definition = parameter_definition(model_id, memory_id)
    raw_vector = np.asarray(raw_vector, dtype=float).reshape(-1)
    if raw_vector.shape != (len(definition.names),):
        raise ValueError(
            f"raw parameter length {raw_vector.size} does not match schema "
            f"length {len(definition.names)}"
        )
    supplied = {name: float(raw_vector[index]) for index, name in enumerate(definition.names)}

    full = np.zeros(11, dtype=float)
    if memory_id == "bayes":
        full[0] = 1.0
        full[1] = 1.0
    elif memory_id == "fade":
        full[0] = supplied["gamma"]
        full[1] = 0.0
    else:
        full[0] = supplied["gamma"]
        full[1] = supplied["w0"]
    full[2] = math.exp(supplied["log_kappa"])

    if model_id in {"H1", "H2"}:
        full[3] = supplied["m"]
    elif model_id in {"H3_M", "H3_MG"}:
        full[3] = supplied["mu_m"]
    if model_id == "H2":
        full[4] = supplied["g"]
    elif model_id in {"H3_M", "H3_MG"}:
        full[4] = supplied["mu_g"]
    if model_id in {"H3_M", "H3_MG"}:
        full[5] = supplied["phi_m"]
        full[6] = supplied["b_m_surprise"]
        full[7] = supplied["b_m_uncertainty"]
    if model_id == "H3_MG":
        full[8] = supplied["phi_g"]
        full[9] = supplied["b_g_surprise"]
        full[10] = supplied["b_g_uncertainty"]

    reported = dict(supplied)
    reported["gamma"] = float(full[0])
    reported["w0"] = float(full[1])
    reported["kappa"] = float(full[2])
    if model_id == "H0":
        reported["m"] = 0.0
        reported["g"] = float("nan")
    elif model_id == "H1":
        reported["m"] = float(full[3])
        reported["g"] = 0.0
    elif model_id == "H2":
        reported["m"] = float(full[3])
        reported["g"] = float(full[4])
    else:
        reported["initial_m"] = float(_expit_scalar(full[3]))
        reported["initial_g"] = float(_expit_scalar(full[4]))
    return full, reported


def _stable_fit_seed(base_seed: int, *parts: object) -> int:
    joined = ":".join([str(int(base_seed)), *(str(value) for value in parts)])
    digest = hashlib.blake2b(joined.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


def _sobol_starts(
    definition: ParameterDefinition,
    n_starts: int,
    seed: int,
    extra_starts: Iterable[np.ndarray] | None = None,
) -> list[np.ndarray]:
    if n_starts < 1:
        raise ValueError("n_starts must be positive")
    starts = [definition.center.copy()]
    if extra_starts is not None:
        for raw in extra_starts:
            raw = np.asarray(raw, dtype=float).reshape(-1)
            if raw.shape == definition.center.shape:
                clipped = np.asarray(
                    [
                        np.clip(value, definition.bounds[index][0], definition.bounds[index][1])
                        for index, value in enumerate(raw)
                    ],
                    dtype=float,
                )
                starts.append(clipped)
    remaining = max(0, int(n_starts) - len(starts))
    if remaining:
        exponent = int(math.ceil(math.log2(remaining)))
        unit = qmc.Sobol(
            d=len(definition.names), scramble=True, seed=int(seed)
        ).random_base2(exponent)[:remaining]
        for row in unit:
            values = np.asarray(
                [
                    low + float(row[index]) * (high - low)
                    for index, (low, high) in enumerate(definition.start_bounds)
                ],
                dtype=float,
            )
            starts.append(values)
    return starts[:n_starts]


def fit_model0803(
    q_values: np.ndarray,
    choices: np.ndarray,
    feedback: np.ndarray,
    prior: np.ndarray,
    kernels: TransitionKernels,
    train_mask: np.ndarray,
    *,
    model_id: str,
    memory_id: str,
    feature_scaling: FeatureScaling,
    n_starts: int = 16,
    base_seed: int = 20260803,
    seed_parts: Sequence[object] = (),
    extra_starts: Iterable[np.ndarray] | None = None,
    maxiter: int = 1000,
    epsilon: float = EPS,
) -> Model0803Fit:
    """Fit one candidate by deterministic multi-start bounded likelihood."""

    definition = parameter_definition(model_id, memory_id)
    train_mask = np.asarray(train_mask, dtype=bool).reshape(-1)
    if not train_mask.any():
        raise ValueError("train_mask contains no training trials")
    q_values, choices, feedback, prior, train_mask = _validate_sequence_inputs(
        q_values, choices, feedback, prior, kernels, train_mask
    )
    feature_center = np.ascontiguousarray(
        np.asarray(feature_scaling.center, dtype=np.float64).reshape(2)
    )
    feature_scale = np.ascontiguousarray(
        np.asarray(feature_scaling.scale, dtype=np.float64).reshape(2)
    )
    if np.any(feature_scale <= 0.0):
        raise ValueError("feature scaling values must be positive")
    local_kernel = np.ascontiguousarray(kernels.local, dtype=np.float64)
    global_kernel = np.ascontiguousarray(kernels.global_, dtype=np.float64)
    fit_seed = _stable_fit_seed(base_seed, model_id, memory_id, *seed_parts)
    starts = _sobol_starts(definition, n_starts, fit_seed, extra_starts)

    def objective(raw_vector: np.ndarray) -> float:
        full_parameters, _ = decode_parameters(raw_vector, model_id, memory_id)
        raw = _run_model0803_numba(
            q_values,
            choices,
            feedback,
            prior,
            local_kernel,
            global_kernel,
            int(MODEL_CODE[model_id]),
            np.ascontiguousarray(full_parameters),
            feature_center,
            feature_scale,
            train_mask,
            False,
            float(epsilon),
        )
        nll = float(raw[0])
        if not np.isfinite(nll):
            return 1e100
        return nll

    fits = []
    retained_start_count = 0
    for start in starts:
        initial_value = float(objective(np.asarray(start, dtype=float)))
        result = minimize(
            objective,
            np.asarray(start, dtype=float),
            method="L-BFGS-B",
            bounds=definition.bounds,
            options={
                "maxiter": int(maxiter),
                "ftol": 1e-11,
                "gtol": 1e-7,
                "maxls": 40,
            },
        )
        if not np.isfinite(result.fun) or float(result.fun) > initial_value:
            result = OptimizeResult(
                x=np.asarray(start, dtype=float).copy(),
                fun=initial_value,
                success=False,
                message="retained_start_because_optimizer_did_not_improve",
                nit=0,
                nfev=1,
            )
            retained_start_count += 1
        fits.append(result)

    converged_fits = [result for result in fits if bool(result.success)]
    if not converged_fits:
        raise RuntimeError(
            f"No optimizer start converged for model={model_id}, memory={memory_id}"
        )
    best_any_before_rescue = min(fits, key=lambda result: float(result.fun))
    best_converged_before_rescue = min(
        converged_fits, key=lambda result: float(result.fun)
    )
    rescue_attempted = False
    rescue_succeeded = False
    if (
        not bool(best_any_before_rescue.success)
        and float(best_any_before_rescue.fun)
        < float(best_converged_before_rescue.fun) - 1e-8
    ):
        rescue_attempted = True
        rescue = minimize(
            objective,
            np.asarray(best_any_before_rescue.x, dtype=float),
            method="L-BFGS-B",
            bounds=definition.bounds,
            options={
                "maxiter": int(maxiter) * 2,
                "ftol": 1e-12,
                "gtol": 1e-7,
                "maxls": 60,
            },
        )
        fits.append(rescue)
        if bool(rescue.success) and np.isfinite(rescue.fun):
            converged_fits.append(rescue)
            rescue_succeeded = True
    best = min(converged_fits, key=lambda result: float(result.fun))
    best_any_after_rescue = min(fits, key=lambda result: float(result.fun))
    unresolved_nonconverged_advantage = max(
        0.0, float(best.fun) - float(best_any_after_rescue.fun)
    )
    best_vector = np.asarray(best.x, dtype=float)
    full_parameters, reported = decode_parameters(best_vector, model_id, memory_id)

    objective_values = np.asarray([float(result.fun) for result in fits], dtype=float)
    converged = np.asarray([bool(result.success) for result in fits], dtype=bool)
    boundary_parameters: list[str] = []
    for index, name in enumerate(definition.names):
        low, high = definition.bounds[index]
        tolerance = max(1e-6, 1e-4 * (high - low))
        if best_vector[index] <= low + tolerance or best_vector[index] >= high - tolerance:
            boundary_parameters.append(name)
    diagnostics = {
        "optimizer": "multi_start_L-BFGS-B",
        "fit_seed": int(fit_seed),
        "n_starts": int(len(starts)),
        "n_converged": int(converged.sum()),
        "n_retained_starts": int(retained_start_count),
        "require_converged_selection": True,
        "rescue_attempted": bool(rescue_attempted),
        "rescue_succeeded": bool(rescue_succeeded),
        "best_any_before_rescue_nll": float(best_any_before_rescue.fun),
        "best_converged_before_rescue_nll": float(best_converged_before_rescue.fun),
        "nonconverged_advantage_before_rescue": float(
            best_converged_before_rescue.fun - best_any_before_rescue.fun
        ),
        "best_any_after_rescue_nll": float(best_any_after_rescue.fun),
        "unresolved_nonconverged_advantage": float(
            unresolved_nonconverged_advantage
        ),
        "success": bool(best.success),
        "message": str(best.message),
        "best_train_nll": float(best.fun),
        "objective_min": float(np.min(objective_values)),
        "objective_median": float(np.median(objective_values)),
        "objective_max": float(np.max(objective_values)),
        "n_same_optimal_region": int(np.sum(np.abs(objective_values - best.fun) <= 1e-5)),
        "boundary_parameters": boundary_parameters,
        "n_iterations": int(getattr(best, "nit", -1)),
        "n_function_evaluations": int(getattr(best, "nfev", -1)),
    }
    return Model0803Fit(
        model_id=model_id,
        memory_id=memory_id,
        raw_vector=best_vector,
        full_parameters=full_parameters,
        parameters=reported,
        train_nll=float(best.fun),
        diagnostics=diagnostics,
    )


def score_choice_predictions(
    probabilities: np.ndarray,
    choices: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """Score choice predictions without treating trials as independent n."""

    probabilities = np.asarray(probabilities, dtype=float)
    choices = np.asarray(choices, dtype=np.int64).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if probabilities.shape != (choices.size, 2) or mask.size != choices.size:
        raise ValueError("probabilities, choices, and mask shapes do not align")
    rows = np.flatnonzero(mask)
    if rows.size == 0:
        raise ValueError("score mask selects no trials")
    observed_probability = np.clip(probabilities[rows, choices[rows]], EPS, 1.0)
    one_hot = np.eye(2, dtype=float)[choices[rows]]
    predicted = probabilities[rows].argmax(axis=1)
    entropy = -np.sum(
        probabilities[rows]
        * np.log(np.clip(probabilities[rows], EPS, 1.0)),
        axis=1,
    )
    return {
        "n_trials": int(rows.size),
        "nll": float(-np.log(observed_probability).sum()),
        "nll_per_trial": float(-np.log(observed_probability).mean()),
        "brier": float(np.mean(np.sum((probabilities[rows] - one_hot) ** 2, axis=1))),
        "accuracy": float(np.mean(predicted == choices[rows])),
        "mean_confidence": float(np.mean(np.max(probabilities[rows], axis=1))),
        "mean_entropy": float(np.mean(entropy)),
    }


def raw_vector_from_mapping(
    values: Mapping[str, float], model_id: str, memory_id: str
) -> np.ndarray:
    """Build an optimizer vector from named values, filling frozen centres."""

    definition = parameter_definition(model_id, memory_id)
    out = definition.center.copy()
    for index, name in enumerate(definition.names):
        if name in values:
            out[index] = float(values[name])
    return out

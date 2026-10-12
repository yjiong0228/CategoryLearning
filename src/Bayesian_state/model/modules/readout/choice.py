"""Observable readouts from Bayesian cognitive state.

Readouts map latent hypothesis state to observable choice, reaction-time, or
oral-report distributions.  They never update the cognitive state, so the same
implementation can be reused by trajectory inference, particle filtering,
simulation, and external validation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from ..base_module import ModuleRole
from .shared import (
    _bool_from_mapping, _float_from_mapping, _mapping_get_path,
    normalize_probability_vector,
)


OUTPUT_NOISE_TARGET_UNIFORM = "uniform"
OUTPUT_NOISE_TARGET_PREVIOUS_CHOICE = "previous_choice"
OUTPUT_NOISE_TARGET_LOSE_SHIFT = "lose_shift"
OUTPUT_NOISE_TARGET_CHOICES = (
    OUTPUT_NOISE_TARGET_UNIFORM,
    OUTPUT_NOISE_TARGET_PREVIOUS_CHOICE,
    OUTPUT_NOISE_TARGET_LOSE_SHIFT,
)
OUTPUT_NOISE_KWARG_KEYS = (
    "enabled",
    "base_lapse",
    "post_error_lapse",
    "low_accuracy_lapse",
    "low_accuracy_threshold",
    "recent_accuracy_window",
    "lapse_decay",
    "max_lapse",
    "lapse_target",
    "latent_volatility_lapse",
    "latent_volatility_power",
)

CHOICE_READOUT_EXPECTATION = "expectation"
CHOICE_READOUT_SHARPENED = "sharpened_expectation"
CHOICE_READOUT_MAP = "map_hypothesis"
CHOICE_READOUT_SAMPLE = "sample_hypothesis"
CHOICE_READOUT_STICKY = "sticky_sample"
CHOICE_READOUT_STUBBORN = "stubborn_sticky"
CHOICE_READOUT_METHODS = (
    CHOICE_READOUT_EXPECTATION,
    CHOICE_READOUT_SHARPENED,
    CHOICE_READOUT_MAP,
    CHOICE_READOUT_SAMPLE,
    CHOICE_READOUT_STICKY,
    CHOICE_READOUT_STUBBORN,
)
CHOICE_READOUT_KWARG_KEYS = (
    "method",
    "power",
    "weight_floor",
    "switch_probability",
    "post_error_switch_delta",
    "low_confidence_switch_gain",
    "strategy_confidence_gain",
    "rule_commitment_confidence_gain",
)


@dataclass(frozen=True)
class DecisionRuleSnapshot:
    """The actual pre-feedback rule mixture, retained for coupled readouts.

    Columns use the model's response coordinates. In condition 3 the pairing
    state affects feedback learning; these columns remain opaque choice IDs.
    """

    hypothesis_indices: tuple[int, ...]
    rule_weights: np.ndarray
    category_probabilities: np.ndarray
    perceived_stimulus: np.ndarray


@dataclass(frozen=True)
class ChoicePrediction:
    """Pre-outcome choice readout for one prepared model trial."""

    cognitive_probabilities: np.ndarray
    observed_probabilities: np.ndarray
    readout_details: Dict[str, Any]
    output_lapse: float
    post_error_lapse_state: float
    rule_snapshot: DecisionRuleSnapshot | None = None


def resolve_output_noise_config(
    params: Mapping[str, Any] | None,
    engine_config: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    """Resolve the shared output-noise configuration from nested/flat inputs."""

    raw: Dict[str, Any] = {}
    sources = [
        _mapping_get_path(engine_config, "output_noise.kwargs"),
        _mapping_get_path(engine_config, "engine.output_noise.kwargs"),
        _mapping_get_path(params, "output_noise.kwargs"),
        _mapping_get_path(params, "engine.output_noise.kwargs"),
    ]
    for source in sources:
        if isinstance(source, Mapping):
            raw.update(dict(source))
    for source in (params, engine_config):
        if not isinstance(source, Mapping):
            continue
        for key in OUTPUT_NOISE_KWARG_KEYS:
            for prefix in ("engine.output_noise.kwargs.", "output_noise.kwargs."):
                full_key = f"{prefix}{key}"
                if full_key in source:
                    raw[key] = source[full_key]
    if not raw:
        return {"enabled": False}

    context = "output_noise.kwargs"
    cfg = {
        "enabled": _bool_from_mapping(raw, "enabled", True, context=context),
        "base_lapse": _float_from_mapping(
            raw, "base_lapse", 0.0, min_value=0.0, max_value=1.0, context=context
        ),
        "post_error_lapse": _float_from_mapping(
            raw, "post_error_lapse", 0.0, min_value=0.0, max_value=1.0, context=context
        ),
        "low_accuracy_lapse": _float_from_mapping(
            raw, "low_accuracy_lapse", 0.0, min_value=0.0, max_value=1.0, context=context
        ),
        "low_accuracy_threshold": _float_from_mapping(
            raw, "low_accuracy_threshold", 0.70, min_value=1e-9, max_value=1.0, context=context
        ),
        "recent_accuracy_window": int(raw.get("recent_accuracy_window", 8)),
        "lapse_decay": _float_from_mapping(
            raw, "lapse_decay", 0.0, min_value=0.0, max_value=1.0, context=context
        ),
        "max_lapse": _float_from_mapping(
            raw, "max_lapse", 0.40, min_value=0.0, max_value=1.0, context=context
        ),
        "lapse_target": str(raw.get("lapse_target", OUTPUT_NOISE_TARGET_UNIFORM)),
        "latent_volatility_lapse": _float_from_mapping(
            raw,
            "latent_volatility_lapse",
            0.0,
            min_value=0.0,
            max_value=1.0,
            context=context,
        ),
        "latent_volatility_power": _float_from_mapping(
            raw, "latent_volatility_power", 1.0, min_value=1e-9, context=context
        ),
    }
    if int(cfg["recent_accuracy_window"]) <= 0:
        raise ValueError(
            "output_noise.kwargs.recent_accuracy_window must be positive, "
            f"got {cfg['recent_accuracy_window']!r}"
        )
    if cfg["lapse_target"] not in OUTPUT_NOISE_TARGET_CHOICES:
        raise ValueError(
            "output_noise.kwargs.lapse_target must be one of "
            f"{OUTPUT_NOISE_TARGET_CHOICES}, got {cfg['lapse_target']!r}"
        )
    if cfg["max_lapse"] < cfg["base_lapse"]:
        raise ValueError(
            "output_noise.kwargs.max_lapse must be >= base_lapse, "
            f"got max_lapse={cfg['max_lapse']!r}, base_lapse={cfg['base_lapse']!r}"
        )
    has_lapse = any(
        cfg[key] > 0.0
        for key in (
            "base_lapse",
            "post_error_lapse",
            "low_accuracy_lapse",
            "latent_volatility_lapse",
        )
    )
    cfg["enabled"] = bool(cfg["enabled"] and has_lapse and cfg["max_lapse"] > 0.0)
    return cfg


def resolve_choice_readout_config(
    params: Mapping[str, Any] | None,
    engine_config: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    """Resolve and validate the shared hypothesis-to-choice readout."""

    raw: Dict[str, Any] = {}
    sources = [
        _mapping_get_path(engine_config, "choice_readout.kwargs"),
        _mapping_get_path(engine_config, "engine.choice_readout.kwargs"),
        _mapping_get_path(params, "choice_readout.kwargs"),
        _mapping_get_path(params, "engine.choice_readout.kwargs"),
    ]
    for source in sources:
        if isinstance(source, Mapping):
            raw.update(dict(source))
    for source in (params, engine_config):
        if not isinstance(source, Mapping):
            continue
        for key in CHOICE_READOUT_KWARG_KEYS:
            for prefix in ("engine.choice_readout.kwargs.", "choice_readout.kwargs."):
                full_key = f"{prefix}{key}"
                if full_key in source:
                    raw[key] = source[full_key]

    method = str(raw.get("method", CHOICE_READOUT_EXPECTATION))
    if method not in CHOICE_READOUT_METHODS:
        raise ValueError(
            "choice_readout.kwargs.method must be one of "
            f"{CHOICE_READOUT_METHODS}, got {method!r}"
        )
    context = "choice_readout.kwargs"
    cfg = {
        "method": method,
        "power": _float_from_mapping(raw, "power", 1.0, min_value=1e-9, context=context),
        "weight_floor": _float_from_mapping(raw, "weight_floor", 0.0, min_value=0.0, context=context),
        "switch_probability": _float_from_mapping(
            raw, "switch_probability", 0.15, min_value=0.0, max_value=1.0, context=context
        ),
        "post_error_switch_delta": _float_from_mapping(
            raw, "post_error_switch_delta", 0.0, min_value=-1.0, max_value=1.0, context=context
        ),
        "low_confidence_switch_gain": _float_from_mapping(
            raw, "low_confidence_switch_gain", 0.0, min_value=0.0, max_value=1.0, context=context
        ),
        "strategy_confidence_gain": _float_from_mapping(
            raw,
            "strategy_confidence_gain",
            0.0,
            min_value=0.0,
            max_value=10.0,
            context=context,
        ),
        "rule_commitment_confidence_gain": _float_from_mapping(
            raw,
            "rule_commitment_confidence_gain",
            0.0,
            min_value=0.0,
            max_value=10.0,
            context=context,
        ),
    }
    if method == CHOICE_READOUT_STUBBORN and "post_error_switch_delta" not in raw:
        cfg["post_error_switch_delta"] = -0.10
    if method == CHOICE_READOUT_STICKY and "post_error_switch_delta" not in raw:
        cfg["post_error_switch_delta"] = 0.10
    return cfg


def apply_strategy_conditioned_choice_confidence(
    probabilities: Sequence[float] | np.ndarray,
    *,
    mastery_evidence: float,
    failure_pressure: float,
    gain: float,
) -> tuple[np.ndarray, Dict[str, float]]:
    """Sharpen choice commitment when mastery clearly exceeds failure.

    The pre-choice controller state supplies a positive mastery advantage.
    Squaring that advantage prevents weak or initial mastery signals from
    producing premature confidence, while allowing a strong mastery state to
    make the current policy more deterministic. The transformation amplifies
    whichever category is currently preferred; it never uses the true answer.
    """

    base = normalize_probability_vector(probabilities, strict=True)
    gain_value = float(gain)
    if not np.isfinite(gain_value) or not 0.0 <= gain_value <= 10.0:
        raise ValueError(
            "strategy confidence gain must be finite and lie in [0, 10]."
        )
    if gain_value == 0.0:
        return base.copy(), {
            "strategy_confidence_signal": 0.0,
            "strategy_choice_precision": 1.0,
        }

    mastery_value = float(mastery_evidence)
    failure_value = float(failure_pressure)
    if not np.isfinite(mastery_value) or not np.isfinite(failure_value):
        raise ValueError(
            "strategy-conditioned confidence requires finite pre-choice "
            "mastery_evidence and failure_pressure."
        )
    mastery_value = float(np.clip(mastery_value, 0.0, 1.0))
    failure_value = float(np.clip(failure_value, 0.0, 1.0))
    advantage = max(mastery_value - failure_value, 0.0)
    signal = float(advantage * advantage)
    precision = float(1.0 + gain_value * signal)
    log_probability = precision * np.log(np.clip(base, 1e-300, None))
    log_probability -= float(np.max(log_probability))
    sharpened = normalize_probability_vector(
        np.exp(log_probability),
        strict=True,
    )
    return sharpened, {
        "strategy_confidence_signal": signal,
        "strategy_choice_precision": precision,
    }


def apply_rule_commitment_choice_confidence(
    probabilities: Sequence[float] | np.ndarray,
    *,
    committed: bool,
    choice_compatibility: float,
    gain: float,
) -> tuple[np.ndarray, Dict[str, float]]:
    """Sharpen only a history-supported committed rule's category output.

    The signal is computed from choices completed before the current trial.
    A disabled or inactive commitment is an exact identity transformation.
    """

    base = normalize_probability_vector(probabilities, strict=True)
    gain_value = float(gain)
    if not np.isfinite(gain_value) or not 0.0 <= gain_value <= 10.0:
        raise ValueError(
            "rule commitment confidence gain must be finite and lie in [0, 10]."
        )
    if gain_value == 0.0 or not bool(committed):
        return base.copy(), {
            "rule_commitment_confidence_signal": 0.0,
            "rule_commitment_choice_precision": 1.0,
        }
    compatibility = float(choice_compatibility)
    if not np.isfinite(compatibility) or not 0.0 <= compatibility <= 1.0:
        raise ValueError(
            "active rule commitment requires finite choice compatibility in [0, 1]."
        )
    normalized_advantage = max(compatibility - 0.5, 0.0) / 0.5
    signal = float(normalized_advantage * normalized_advantage)
    precision = float(1.0 + gain_value * signal)
    log_probability = precision * np.log(np.clip(base, 1e-300, None))
    log_probability -= float(np.max(log_probability))
    sharpened = normalize_probability_vector(
        np.exp(log_probability),
        strict=True,
    )
    return sharpened, {
        "rule_commitment_confidence_signal": signal,
        "rule_commitment_choice_precision": precision,
    }


def _one_hot(index: int, size: int) -> np.ndarray:
    result = np.zeros(size, dtype=float)
    if 0 <= int(index) < size:
        result[int(index)] = 1.0
    elif size > 0:
        result[:] = 1.0 / float(size)
    return result


def choice_readout_weights(
    distribution: Sequence[float] | np.ndarray,
    *,
    trial_idx: int,
    feedback: Sequence[float] | np.ndarray,
    config: Mapping[str, Any],
    rng: np.random.Generator,
    sticky_state: Dict[str, Any],
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Read hypothesis weights without mutating the cognitive engine."""

    method = str(config.get("method", CHOICE_READOUT_EXPECTATION))
    base = normalize_probability_vector(distribution)
    if method == CHOICE_READOUT_SHARPENED:
        base = normalize_probability_vector(
            np.power(base + float(config.get("weight_floor", 0.0)), float(config.get("power", 1.0)))
        )
    size = int(base.size)
    selected_arg = -1
    switched = False
    confidence = float(np.max(base)) if size else 0.0

    if method in (CHOICE_READOUT_EXPECTATION, CHOICE_READOUT_SHARPENED):
        return base, {
            "method": method,
            "selected_arg": selected_arg,
            "switched": switched,
            "confidence": confidence,
        }
    if method == CHOICE_READOUT_MAP:
        selected_arg = int(np.argmax(base)) if size else -1
        return _one_hot(selected_arg, size), {
            "method": method,
            "selected_arg": selected_arg,
            "switched": True,
            "confidence": confidence,
        }
    if method == CHOICE_READOUT_SAMPLE:
        selected_arg = int(rng.choice(size, p=base)) if size else -1
        return _one_hot(selected_arg, size), {
            "method": method,
            "selected_arg": selected_arg,
            "switched": True,
            "confidence": confidence,
        }
    if method in (CHOICE_READOUT_STICKY, CHOICE_READOUT_STUBBORN):
        current = sticky_state.get("selected_arg")
        force_switch = (
            current is None
            or int(current) < 0
            or int(current) >= size
            or base[int(current)] <= 0.0
        )
        feedback_array = np.asarray(feedback, dtype=float).reshape(-1)
        previous_feedback = (
            float(feedback_array[trial_idx - 1])
            if trial_idx > 0 and np.isfinite(feedback_array[trial_idx - 1])
            else 1.0
        )
        last_error = float(np.clip(1.0 - previous_feedback, 0.0, 1.0))
        switch_probability = (
            float(config.get("switch_probability", 0.15))
            + float(config.get("post_error_switch_delta", 0.0)) * last_error
            + float(config.get("low_confidence_switch_gain", 0.0)) * (1.0 - confidence)
        )
        switch_probability = float(np.clip(switch_probability, 0.0, 1.0))
        if force_switch or bool(rng.random() < switch_probability):
            selected_arg = int(rng.choice(size, p=base)) if size else -1
            sticky_state["selected_arg"] = selected_arg
            switched = True
        else:
            selected_arg = int(current)
        return _one_hot(selected_arg, size), {
            "method": method,
            "selected_arg": selected_arg,
            "switched": switched,
            "confidence": confidence,
            "switch_probability": switch_probability,
        }
    raise ValueError(f"Unsupported choice_readout method: {method!r}")


def _output_noise_target_vector(
    lapse_target: str,
    trial_idx: int,
    choices: np.ndarray,
    feedback: np.ndarray,
    n_categories: int,
) -> np.ndarray:
    uniform = np.full(n_categories, 1.0 / max(1, n_categories), dtype=float)
    if trial_idx <= 0:
        return uniform
    previous_choice = int(choices[trial_idx - 1]) - 1
    previous_feedback = (
        float(feedback[trial_idx - 1])
        if np.isfinite(feedback[trial_idx - 1])
        else 1.0
    )
    if lapse_target == OUTPUT_NOISE_TARGET_PREVIOUS_CHOICE:
        return _one_hot(previous_choice, n_categories)
    if lapse_target == OUTPUT_NOISE_TARGET_LOSE_SHIFT:
        if previous_feedback >= 1.0 or not 0 <= previous_choice < n_categories:
            return uniform
        if n_categories == 2:
            return _one_hot(1 - previous_choice, n_categories)
        result = np.ones(n_categories, dtype=float)
        result[previous_choice] = 0.0
        return normalize_probability_vector(result)
    return uniform


def apply_output_noise_to_category_prob(
    category_probabilities: Sequence[float] | np.ndarray,
    *,
    trial_idx: int,
    choices: Sequence[int] | np.ndarray,
    feedback: Sequence[float] | np.ndarray,
    n_cats: int,
    output_noise_config: Mapping[str, Any],
    post_error_lapse_state: float,
    latent_volatility_value: float = 0.0,
) -> tuple[np.ndarray, float, float]:
    """Apply the shared lapse/contamination observation model."""

    probability = normalize_probability_vector(category_probabilities, n_cats)
    if not bool(output_noise_config.get("enabled", False)):
        return probability, 0.0, 0.0
    choices_array = np.asarray(choices, dtype=int).reshape(-1)
    feedback_array = np.asarray(feedback, dtype=float).reshape(-1)
    previous_feedback = (
        float(feedback_array[trial_idx - 1])
        if trial_idx > 0 and np.isfinite(feedback_array[trial_idx - 1])
        else 1.0
    )
    error_severity = float(np.clip(1.0 - previous_feedback, 0.0, 1.0))
    post_error_state = (
        float(output_noise_config["lapse_decay"]) * float(post_error_lapse_state)
        + float(output_noise_config["post_error_lapse"]) * error_severity
    )
    start = max(0, int(trial_idx) - int(output_noise_config["recent_accuracy_window"]))
    recent = feedback_array[start:trial_idx]
    recent = recent[np.isfinite(recent)]
    recent_accuracy = float(np.clip(np.mean(recent), 0.0, 1.0)) if recent.size else 1.0
    threshold = float(output_noise_config["low_accuracy_threshold"])
    low_accuracy_scale = max(0.0, threshold - recent_accuracy) / max(threshold, 1e-12)
    latent_value = float(np.clip(latent_volatility_value, 0.0, 1.0))
    lapse = (
        float(output_noise_config["base_lapse"])
        + post_error_state
        + float(output_noise_config["low_accuracy_lapse"]) * low_accuracy_scale
        + float(output_noise_config["latent_volatility_lapse"])
        * latent_value ** float(output_noise_config["latent_volatility_power"])
    )
    lapse = float(np.clip(lapse, 0.0, float(output_noise_config["max_lapse"])))
    if lapse <= 0.0:
        return probability, 0.0, post_error_state
    target = _output_noise_target_vector(
        str(output_noise_config["lapse_target"]),
        trial_idx,
        choices_array,
        feedback_array,
        n_cats,
    )
    return (
        normalize_probability_vector((1.0 - lapse) * probability + lapse * target),
        lapse,
        post_error_state,
    )


def resolve_executed_hypothesis(engine: Any) -> int | None:
    """Return the protected overt rule when persistent execution is enabled."""
    transition = engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
    if transition is None or not bool(
        getattr(transition, "persistent_execution_enabled", False)
    ):
        return None
    executed = getattr(transition, "executed_hypothesis", None)
    if executed is None:
        raise RuntimeError(
            "persistent execution is enabled but executed_hypothesis is unset."
        )
    executed_index = int(executed)
    mask = np.asarray(engine.hypotheses_mask, dtype=float).reshape(-1)
    if not 0 <= executed_index < mask.size or mask[executed_index] <= 0.0:
        raise RuntimeError(
            "executed_hypothesis must identify an active hypothesis."
        )
    return executed_index


def resolve_continuous_choice_weights(engine: Any) -> np.ndarray | None:
    """Return Model 0923 full-v3 weights across the complete rule catalogue."""
    transition = engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
    if transition is None or not bool(getattr(transition, "continuous_choice_enabled", False)):
        return None
    weights = np.asarray(transition.execution_probabilities, dtype=float)
    mask = np.asarray(engine.hypotheses_mask, dtype=float).reshape(-1)
    if (weights.shape != mask.shape or not np.isfinite(weights).all()
            or np.any(weights < 0) or not np.isclose(weights.sum(), 1)
            or np.any(weights[mask <= 0] != 0)):
        raise RuntimeError("continuous choice weights must be normalized on active hypotheses")
    return weights.copy()


def apply_hypothesis_mapping(
    engine: Any,
    hypothesis: int,
    probabilities: Sequence[float] | np.ndarray,
    *,
    use_predictive_snapshot: bool = False,
) -> np.ndarray:
    """Apply an optional category-mapping state to one rule emission."""

    base = normalize_probability_vector(probabilities, strict=True)
    mapping = engine.get_module(ModuleRole.MAPPING)
    if mapping is None:
        return base
    orient = getattr(mapping, "orient_category_probabilities", None)
    if not callable(orient):
        raise TypeError("mapping module must provide orient_category_probabilities().")
    return normalize_probability_vector(
        orient(
            int(hypothesis),
            base,
            use_predictive_snapshot=bool(use_predictive_snapshot),
        ),
        size=base.size,
        strict=True,
    )


def read_choice_probabilities_from_model(
    model: Any,
    perceived_stimulus: Sequence[float] | np.ndarray,
    *,
    power: float = 1.0,
    lapse: float = 0.0,
    return_snapshot: bool = False,
) -> np.ndarray | tuple[np.ndarray, DecisionRuleSnapshot]:
    """Map an engine's current rule contributions to response probabilities.

    The optional snapshot retains the weights and per-rule predictions used
    in this calculation, before the aggregate output lapse. It consumes no
    random draws and can be shared with the oral observation layer.
    """

    engine = model.engine
    n_categories = int(model.n_cats)
    prior = np.asarray(engine.prior, dtype=float)
    beta = np.asarray(engine.beta, dtype=float)
    executed_hypothesis = resolve_executed_hypothesis(engine)
    continuous_weights = resolve_continuous_choice_weights(engine)
    raw_mask = getattr(engine, "hypotheses_mask", None)
    if raw_mask is None:
        active = np.arange(prior.size, dtype=int)
    else:
        mask = np.asarray(raw_mask, dtype=float).reshape(-1)
        if mask.shape != prior.shape:
            raise ValueError("hypothesis mask width does not match the prior")
        active = np.flatnonzero(mask > 0.0)
    if active.size == 0:
        raise RuntimeError("Choice readout received an empty active hypothesis set.")
    if executed_hypothesis is not None:
        active = np.asarray([executed_hypothesis], dtype=int)
    power_value = float(power)
    lapse_value = float(lapse)
    if not np.isfinite(power_value) or power_value <= 0.0:
        raise ValueError("choice readout power must be finite and positive.")
    if not np.isfinite(lapse_value) or not 0.0 <= lapse_value <= 1.0:
        raise ValueError("choice lapse must lie in [0, 1].")
    if continuous_weights is not None:
        if power_value != 1.0:
            raise ValueError("0923 full-v3 continuous choice requires readout power=1")
        readout_weights = continuous_weights[active]
    elif executed_hypothesis is None:
        log_weights = power_value * np.log(np.clip(prior[active], 1e-300, None))
        log_weights -= float(np.max(log_weights))
        readout_weights = normalize_probability_vector(
            np.exp(log_weights), strict=True
        )
    else:
        readout_weights = np.ones(1, dtype=float)
    cognitive = np.zeros(n_categories, dtype=float)
    stimulus = np.asarray(perceived_stimulus, dtype=float)
    category_by_hypothesis = (
        np.zeros((active.size, n_categories), dtype=float)
        if return_snapshot else None
    )
    for row, (weight, hypothesis) in enumerate(zip(readout_weights, active)):
        raw = model.partition_model.get_category_probabilities(
            hypo=int(hypothesis),
            data=([stimulus], [1], [1.0]),
            beta=float(beta[hypothesis]),
            distance_mode=engine.distance_mode,
        )
        probability = normalize_probability_vector(
            np.asarray(raw[:, 0], dtype=float), strict=True
        )
        probability = apply_hypothesis_mapping(
            engine,
            int(hypothesis),
            probability,
        )
        if category_by_hypothesis is not None:
            category_by_hypothesis[row] = probability
        cognitive += float(weight) * probability
    cognitive = normalize_probability_vector(cognitive, strict=True)
    observed = normalize_probability_vector(
        (1.0 - lapse_value) * cognitive + lapse_value / float(n_categories),
        strict=True,
    )
    if not return_snapshot:
        return observed
    arrays = [np.array(value, copy=True) for value in
              (readout_weights, category_by_hypothesis, stimulus)]
    for value in arrays:
        value.setflags(write=False)
    snapshot = DecisionRuleSnapshot(tuple(int(h) for h in active), *arrays)
    return observed, snapshot


def predict_choice_from_model(
    model: Any,
    perceived_stimulus: Sequence[float] | np.ndarray,
    *,
    trial_idx: int,
    choices: Sequence[int] | np.ndarray,
    feedback: Sequence[float] | np.ndarray,
    choice_readout_config: Mapping[str, Any],
    output_noise_config: Mapping[str, Any],
    rng: np.random.Generator,
    sticky_state: Dict[str, Any],
    post_error_lapse_state: float,
    latent_volatility_value: float = 0.0,
) -> ChoicePrediction:
    """Read a choice distribution from the model before the outcome is known.

    This is the live-model counterpart of the prediction calculation used for
    saved trajectories.  It deliberately stops at the observable choice
    distribution: sampling a choice and producing task feedback belong to the
    autonomous execution layer.
    """

    engine = model.engine
    hypotheses = list(model.hypotheses_set)
    n_hypotheses = len(hypotheses)
    n_categories = int(model.n_cats)
    distribution = normalize_probability_vector(
        np.asarray(engine.prior, dtype=float), n_hypotheses, strict=True
    )
    executed_hypothesis = resolve_executed_hypothesis(engine)
    continuous_weights = resolve_continuous_choice_weights(engine)
    if continuous_weights is not None:
        if (choice_readout_config.get("method", CHOICE_READOUT_EXPECTATION) != CHOICE_READOUT_EXPECTATION
                or float(choice_readout_config.get("power", 1.0)) != 1.0):
            raise ValueError("0923 full-v3 continuous choice requires expectation readout with power=1")
        transition = engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION, required=True)
        readout_weights = continuous_weights
        readout_details = {
            "method": CHOICE_READOUT_EXPECTATION, "selected_arg": -1,
            "switched": bool(transition.current_execution_switch_event),
            "confidence": float(readout_weights.max()),
            "persistent_execution_enabled": False, "continuous_choice_enabled": True,
            "dominant_hypothesis": int(transition.dominant_hypothesis),
            "effective_participation": float(1 / np.sum(readout_weights ** 2)),
        }
    elif executed_hypothesis is None:
        readout_weights, readout_details = choice_readout_weights(
            distribution,
            trial_idx=int(trial_idx),
            feedback=feedback,
            config=choice_readout_config,
            rng=rng,
            sticky_state=sticky_state,
        )
    else:
        transition = engine.get_module(
            ModuleRole.HYPOTHESIS_TRANSITION,
            required=True,
        )
        readout_weights = _one_hot(executed_hypothesis, n_hypotheses)
        readout_details = {
            "method": str(
                choice_readout_config.get("method", CHOICE_READOUT_EXPECTATION)
            ),
            "selected_arg": int(executed_hypothesis),
            "switched": bool(transition.current_execution_switch_event),
            "confidence": float(distribution[executed_hypothesis]),
            "persistent_execution_enabled": True,
            "executed_hypothesis": int(executed_hypothesis),
            "execution_switch_probability": float(
                transition.current_execution_switch_probability
            ),
            "execution_dwell_trials": int(transition.execution_dwell_trials),
        }

    beta = getattr(engine, "beta", None)
    if beta is None:
        likelihood_model = engine.observation_likelihood
        default_beta = float(likelihood_model.default_beta)
        beta_values = np.full(n_hypotheses, default_beta, dtype=float)
    else:
        beta_values = np.asarray(beta, dtype=float).reshape(-1)
        if beta_values.size != n_hypotheses:
            raise ValueError(
                "engine.beta width does not match the hypothesis space: "
                f"{beta_values.size} vs {n_hypotheses}."
            )

    hypothesis_category = np.zeros((n_hypotheses, n_categories), dtype=float)
    stimulus = np.asarray(perceived_stimulus, dtype=float).reshape(-1)
    for hypothesis_arg, hypothesis in enumerate(hypotheses):
        if readout_weights[hypothesis_arg] <= 0.0:
            continue
        raw = model.partition_model.get_category_probabilities(
            hypo=int(hypothesis),
            data=([stimulus], [1], [1.0]),
            beta=float(beta_values[hypothesis_arg]),
            distance_mode=engine.distance_mode,
        )
        probability = np.asarray(raw, dtype=float)
        if probability.ndim == 2:
            probability = probability[:, 0]
        hypothesis_category[hypothesis_arg] = apply_hypothesis_mapping(
            engine,
            int(hypothesis),
            normalize_probability_vector(
                probability, n_categories, strict=True
            ),
        )

    cognitive = normalize_probability_vector(
        np.sum(readout_weights[:, None] * hypothesis_category, axis=0),
        n_categories,
        strict=True,
    )
    strategy_gain = float(
        choice_readout_config.get("strategy_confidence_gain", 0.0)
    )
    strategy_details = {
        "strategy_confidence_signal": 0.0,
        "strategy_choice_precision": 1.0,
    }
    if strategy_gain > 0.0:
        transition = engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        mastery_evidence = getattr(transition, "mastery_evidence", np.nan)
        failure_pressure = getattr(transition, "failure_pressure", np.nan)
        cognitive, strategy_details = (
            apply_strategy_conditioned_choice_confidence(
                cognitive,
                mastery_evidence=float(mastery_evidence),
                failure_pressure=float(failure_pressure),
                gain=strategy_gain,
            )
        )
    readout_details.update(strategy_details)
    commitment_gain = float(
        choice_readout_config.get("rule_commitment_confidence_gain", 0.0)
    )
    commitment_details = {
        "rule_commitment_confidence_signal": 0.0,
        "rule_commitment_choice_precision": 1.0,
    }
    if commitment_gain > 0.0:
        transition = engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        committed = bool(
            getattr(transition, "rule_commitment_active", False)
        )
        compatibility = float("nan")
        if committed:
            executed = getattr(transition, "executed_hypothesis", None)
            values = getattr(transition, "choice_compatibility", None)
            if executed is None or values is None:
                raise RuntimeError(
                    "active rule commitment is missing its executed rule or "
                    "choice-compatibility state."
                )
            compatibility = float(np.asarray(values, dtype=float)[int(executed)])
        cognitive, commitment_details = (
            apply_rule_commitment_choice_confidence(
                cognitive,
                committed=committed,
                choice_compatibility=compatibility,
                gain=commitment_gain,
            )
        )
    readout_details.update(commitment_details)
    observed, output_lapse, next_post_error_state = apply_output_noise_to_category_prob(
        cognitive,
        trial_idx=int(trial_idx),
        choices=choices,
        feedback=feedback,
        n_cats=n_categories,
        output_noise_config=output_noise_config,
        post_error_lapse_state=float(post_error_lapse_state),
        latent_volatility_value=float(latent_volatility_value),
    )
    snapshot = None
    if getattr(model, "_oral_readout_parameters", None) is not None:
        # No new draw or re-read of beliefs: oral uses precisely this decision
        # mixture, including its sampled/MAP/persistent rule and current beta.
        arrays = [np.array(value, copy=True) for value in
                  (readout_weights, hypothesis_category, stimulus)]
        for value in arrays:
            value.setflags(write=False)
        snapshot = DecisionRuleSnapshot(tuple(int(h) for h in hypotheses), *arrays)
    return ChoicePrediction(
        cognitive_probabilities=cognitive.copy(),
        observed_probabilities=observed.copy(),
        readout_details=dict(readout_details),
        output_lapse=float(output_lapse),
        post_error_lapse_state=float(next_post_error_state),
        rule_snapshot=snapshot,
    )


__all__ = [
    "CHOICE_READOUT_EXPECTATION",
    "CHOICE_READOUT_KWARG_KEYS",
    "CHOICE_READOUT_MAP",
    "CHOICE_READOUT_METHODS",
    "CHOICE_READOUT_SAMPLE",
    "CHOICE_READOUT_SHARPENED",
    "CHOICE_READOUT_STICKY",
    "CHOICE_READOUT_STUBBORN",
    "ChoicePrediction",
    "OUTPUT_NOISE_KWARG_KEYS",
    "OUTPUT_NOISE_TARGET_CHOICES",
    "OUTPUT_NOISE_TARGET_LOSE_SHIFT",
    "OUTPUT_NOISE_TARGET_PREVIOUS_CHOICE",
    "OUTPUT_NOISE_TARGET_UNIFORM",
    "apply_output_noise_to_category_prob",
    "apply_hypothesis_mapping",
    "apply_rule_commitment_choice_confidence",
    "apply_strategy_conditioned_choice_confidence",
    "choice_readout_weights",
    "normalize_probability_vector",
    "predict_choice_from_model",
    "read_choice_probabilities_from_model",
    "resolve_executed_hypothesis",
    "resolve_choice_readout_config",
    "resolve_output_noise_config",
]

"""Online particle filtering for engine-configured ``StateModel`` objects.

The filter is a numerical integration layer and does not add cognitive state.
Each particle is one possible latent trajectory of perception noise,
active-set initialization, transition events, tie breaking, and newcomer
sampling.  The public model-agnostic entry point lives in
``Bayesian_state.inference.backends.particle_filter``.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..results import InferenceResult, ParticleFilterResult
from ...model.config import ModelContext
from ...model.modules.base_module import ModuleRole
from ...utils.seeding import stable_seed


@dataclass
class _CognitiveSnapshot:
    """Opaque engine/module snapshot used by the generic resampler."""

    payload: dict[str, Any]


def effective_sample_size(weights: Sequence[float] | np.ndarray) -> float:
    values = np.asarray(weights, dtype=float).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("weights must be a non-empty finite non-negative vector.")
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("weights must have positive mass.")
    normalized = values / total
    return 1.0 / float(np.sum(np.square(normalized)))


def systematic_resample(
    weights: Sequence[float] | np.ndarray,
    uniform: float,
) -> np.ndarray:
    """Return systematic-resampling ancestor indices.

    ``uniform`` is expressed on ``[0, 1)`` and converted to the first offset
    on ``[0, 1 / n)``.  This makes the function deterministic and easy to
    pair across candidate models.
    """

    values = np.asarray(weights, dtype=float).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("weights must be a non-empty finite non-negative vector.")
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("weights must have positive mass.")
    unit_uniform = float(uniform)
    if not np.isfinite(unit_uniform) or not 0.0 <= unit_uniform < 1.0:
        raise ValueError("uniform must be finite and lie in [0, 1).")
    normalized = values / total
    cumulative = np.cumsum(normalized)
    cumulative[-1] = 1.0
    n_particles = int(values.size)
    positions = (
        unit_uniform / float(n_particles)
        + np.arange(n_particles, dtype=float) / float(n_particles)
    )
    return np.searchsorted(cumulative, positions, side="right").astype(int)


def _normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError("Probability vector must be finite and non-negative.")
    total = float(np.sum(array))
    if total <= 0.0:
        raise ValueError("Probability vector has zero mass.")
    return array / total


def _particle_seed(filter_seed: int, particle_index: int, role: str) -> int:
    return stable_seed(
        {
            "seed_role": role,
            "filter_seed": int(filter_seed),
            "particle_index": int(particle_index),
        }
    )


def _future_seed(
    filter_seed: int,
    trial_index: int,
    particle_index: int,
    role: str,
) -> int:
    return stable_seed(
        {
            "seed_role": role,
            "filter_seed": int(filter_seed),
            "trial_index": int(trial_index),
            "particle_index": int(particle_index),
        }
    )


def _particle_config(
    engine_config: Mapping[str, Any],
    filter_seed: int,
    particle_index: int,
) -> dict[str, Any]:
    config = deepcopy(dict(engine_config))
    modules = config.setdefault("modules", {})
    transition = modules.get("hypo_transitions_mod")
    perception = modules.get("perception_mod")
    if not isinstance(transition, Mapping) or not isinstance(perception, Mapping):
        raise ValueError(
            "Particle filtering requires perception_mod and hypo_transitions_mod."
        )
    transition_kwargs = transition.setdefault("kwargs", {})
    perception_kwargs = perception.setdefault("kwargs", {})
    transition_kwargs["module_seed"] = _particle_seed(
        filter_seed, particle_index, "active_set_pf_transition_initial"
    )
    perception_kwargs["module_seed"] = _particle_seed(
        filter_seed, particle_index, "active_set_pf_perception_initial"
    )
    return config


def _snapshot(model: Any) -> _CognitiveSnapshot:
    engine = model.engine
    if not hasattr(engine, "state_dict"):
        raise TypeError("StateModel engine does not implement state_dict().")
    return _CognitiveSnapshot(payload=deepcopy(engine.state_dict()))


def _restore(
    model: Any,
    snapshot: _CognitiveSnapshot,
    *,
    filter_seed: int,
    trial_index: int,
    particle_index: int,
) -> None:
    engine = model.engine
    if not hasattr(engine, "load_state_dict"):
        raise TypeError("StateModel engine does not implement load_state_dict().")
    engine.load_state_dict(deepcopy(snapshot.payload))
    if hasattr(engine, "clear_module_logs"):
        engine.clear_module_logs()

    for module_role, seed_role in (
        (ModuleRole.HYPOTHESIS_TRANSITION, "state_pf_transition_after_resample"),
        (ModuleRole.PERCEPTION, "state_pf_perception_after_resample"),
    ):
        module = engine.get_module(module_role)
        if module is not None and hasattr(module, "reseed_future"):
            module.reseed_future(
                _future_seed(
                    filter_seed,
                    trial_index,
                    particle_index,
                    seed_role,
                )
            )


def _choice_probability(
    model: Any,
    perceived_stimulus: np.ndarray,
    rho: float,
    epsilon: float,
    *,
    strategy_confidence_gain: float = 0.0,
    rule_commitment_confidence_gain: float = 0.0,
    failure_pressure: float = np.nan,
    mastery_evidence: float = np.nan,
    return_details: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, float]]:
    """Read a choice and optionally sharpen commitment from controller state."""

    from ...model.readout import (
        apply_rule_commitment_choice_confidence,
        apply_strategy_conditioned_choice_confidence,
        read_choice_probabilities_from_model,
    )

    cognitive = read_choice_probabilities_from_model(
        model,
        perceived_stimulus,
        power=float(rho),
        lapse=0.0,
    )
    conditioned, details = apply_strategy_conditioned_choice_confidence(
        cognitive,
        mastery_evidence=float(mastery_evidence),
        failure_pressure=float(failure_pressure),
        gain=float(strategy_confidence_gain),
    )
    transition = model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
    committed = bool(getattr(transition, "rule_commitment_active", False))
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
    conditioned, commitment_details = apply_rule_commitment_choice_confidence(
        conditioned,
        committed=committed,
        choice_compatibility=compatibility,
        gain=float(rule_commitment_confidence_gain),
    )
    details.update(commitment_details)
    n_categories = int(conditioned.size)
    observed = _normalize(
        (1.0 - float(epsilon)) * conditioned
        + float(epsilon) / float(n_categories)
    )
    return (observed, details) if return_details else observed


def _map_hypothesis_choice_probability(
    model: Any,
    perceived_stimulus: np.ndarray,
    *,
    lapse: float,
) -> np.ndarray:
    """Read choice probability from the highest-prior active hypothesis only."""

    engine = model.engine
    active = np.flatnonzero(np.asarray(engine.hypotheses_mask, dtype=float) > 0.0)
    if active.size == 0:
        raise RuntimeError("MAP choice audit received an empty active set.")
    prior = np.asarray(engine.prior, dtype=float).reshape(-1)
    beta = np.asarray(engine.beta, dtype=float).reshape(-1)
    hypothesis = int(active[int(np.argmax(prior[active]))])
    raw = model.partition_model.get_category_probabilities(
        hypo=hypothesis,
        data=([np.asarray(perceived_stimulus, dtype=float)], [1], [1.0]),
        beta=float(beta[hypothesis]),
        distance_mode=getattr(engine, "distance_mode", "prototype"),
    )
    cognitive = _normalize(np.asarray(raw[:, 0], dtype=float))
    n_categories = int(cognitive.size)
    return _normalize(
        (1.0 - float(lapse)) * cognitive
        + float(lapse) / float(n_categories)
    )


def _choice_layer_audit(
    model: Any,
    perceived_stimulus: np.ndarray,
    *,
    correct_category_index: int,
    readout_power: float,
    lapse: float,
    strategy_confidence_gain: float,
    rule_commitment_confidence_gain: float,
    failure_pressure: float,
    mastery_evidence: float,
) -> dict[str, Any]:
    """Read pre-choice availability, belief, readout, and noise layers.

    ``correct-predicting`` is deliberately trial-local: it means an active
    hypothesis whose most probable category for the current perceived stimulus
    matches the task-correct category.  It does not claim that the hypothesis is
    the subject's globally true generative rule.
    """

    engine = model.engine
    active = np.flatnonzero(np.asarray(engine.hypotheses_mask, dtype=float) > 0.0)
    if active.size == 0:
        raise RuntimeError("Choice-layer audit received an empty active set.")
    prior = np.asarray(engine.prior, dtype=float).reshape(-1)
    beta = np.asarray(engine.beta, dtype=float).reshape(-1)
    n_categories = int(model.n_cats)
    correct_index = int(correct_category_index)
    if not 0 <= correct_index < n_categories:
        raise ValueError("correct_category_index falls outside the category space.")
    base_weights = _normalize(np.clip(prior[active], 0.0, None))
    power_value = float(readout_power)
    sharpened_weights = _normalize(
        np.power(np.clip(prior[active], 1e-300, None), power_value)
    )
    category_by_hypothesis = np.zeros((active.size, n_categories), dtype=float)
    stimulus = np.asarray(perceived_stimulus, dtype=float)
    for active_arg, hypothesis in enumerate(active):
        raw = model.partition_model.get_category_probabilities(
            hypo=int(hypothesis),
            data=([stimulus], [1], [1.0]),
            beta=float(beta[hypothesis]),
            distance_mode=getattr(engine, "distance_mode", "prototype"),
        )
        category_by_hypothesis[active_arg] = _normalize(
            np.asarray(raw[:, 0], dtype=float)
        )
    unsharpened = _normalize(base_weights @ category_by_hypothesis)
    sharpened_no_lapse = _normalize(
        sharpened_weights @ category_by_hypothesis
    )
    from ...model.readout import (
        apply_rule_commitment_choice_confidence,
        apply_strategy_conditioned_choice_confidence,
        resolve_executed_hypothesis,
    )

    strategy_confidence_no_lapse, strategy_details = (
        apply_strategy_conditioned_choice_confidence(
            sharpened_no_lapse,
            mastery_evidence=float(mastery_evidence),
            failure_pressure=float(failure_pressure),
            gain=float(strategy_confidence_gain),
        )
    )
    executed_hypothesis = resolve_executed_hypothesis(engine)
    persistent_execution_no_lapse = strategy_confidence_no_lapse
    if executed_hypothesis is not None:
        active_lookup = {
            int(hypothesis): int(index)
            for index, hypothesis in enumerate(active)
        }
        executed_arg = active_lookup[int(executed_hypothesis)]
        persistent_execution_no_lapse, _ = (
            apply_strategy_conditioned_choice_confidence(
                category_by_hypothesis[executed_arg],
                mastery_evidence=float(mastery_evidence),
                failure_pressure=float(failure_pressure),
                gain=float(strategy_confidence_gain),
            )
        )
    transition = engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
    committed = bool(getattr(transition, "rule_commitment_active", False))
    compatibility = float("nan")
    if committed:
        values = getattr(transition, "choice_compatibility", None)
        if executed_hypothesis is None or values is None:
            raise RuntimeError(
                "active rule commitment is missing its executed rule or "
                "choice-compatibility state."
            )
        compatibility = float(
            np.asarray(values, dtype=float)[int(executed_hypothesis)]
        )
    commitment_confidence_no_lapse, commitment_details = (
        apply_rule_commitment_choice_confidence(
            persistent_execution_no_lapse,
            committed=committed,
            choice_compatibility=compatibility,
            gain=float(rule_commitment_confidence_gain),
        )
    )
    fitted = _normalize(
        (1.0 - float(lapse)) * commitment_confidence_no_lapse
        + float(lapse) / float(n_categories)
    )
    map_arg = int(np.argmax(prior[active]))
    conditioned_map, _ = apply_strategy_conditioned_choice_confidence(
        category_by_hypothesis[map_arg],
        mastery_evidence=float(mastery_evidence),
        failure_pressure=float(failure_pressure),
        gain=float(strategy_confidence_gain),
    )
    map_with_lapse = _normalize(
        (1.0 - float(lapse)) * conditioned_map
        + float(lapse) / float(n_categories)
    )
    predicts_correct = (
        np.argmax(category_by_hypothesis, axis=1) == correct_index
    )
    return {
        "unsharpened": unsharpened,
        "sharpened_no_lapse": sharpened_no_lapse,
        "strategy_confidence_no_lapse": strategy_confidence_no_lapse,
        "persistent_execution_no_lapse": persistent_execution_no_lapse,
        "commitment_confidence_no_lapse": commitment_confidence_no_lapse,
        "commitment_confidence_signal": float(
            commitment_details["rule_commitment_confidence_signal"]
        ),
        "commitment_choice_precision": float(
            commitment_details["rule_commitment_choice_precision"]
        ),
        "fitted": fitted,
        "map_with_lapse": map_with_lapse,
        "strategy_confidence_signal": strategy_details[
            "strategy_confidence_signal"
        ],
        "strategy_choice_precision": strategy_details[
            "strategy_choice_precision"
        ],
        "correct_predicting_available": float(np.any(predicts_correct)),
        "correct_predicting_prior_mass": float(
            np.sum(base_weights[predicts_correct])
        ),
        "best_active_correct_probability": float(
            np.max(category_by_hypothesis[:, correct_index])
        ),
    }


def _weighted_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: Sequence[float],
) -> np.ndarray:
    """Return deterministic weighted empirical quantiles for one particle set."""

    sample = np.asarray(values, dtype=float).reshape(-1)
    probability = _normalize(np.asarray(weights, dtype=float))
    requested = np.asarray(quantiles, dtype=float).reshape(-1)
    if sample.size != probability.size or not np.all(np.isfinite(sample)):
        raise ValueError("weighted quantiles require matching finite samples and weights.")
    if np.any((requested < 0.0) | (requested > 1.0)):
        raise ValueError("quantiles must lie in [0, 1].")
    order = np.argsort(sample, kind="stable")
    cumulative = np.cumsum(probability[order])
    cumulative[-1] = 1.0
    indices = np.searchsorted(cumulative, requested, side="left")
    indices = np.clip(indices, 0, sample.size - 1)
    return sample[order][indices]


def _trace_ancestral_indices(parent_indices: np.ndarray) -> np.ndarray:
    """Trace every terminal particle back through the resampling genealogy.

    ``parent_indices[t, child]`` identifies the particle at trial ``t`` that
    became ``child`` at trial ``t + 1``.  Identity rows therefore represent
    trials without resampling.  The returned matrix has shape
    ``(terminal_particles, trials)`` and selects one coherent particle at each
    trial rather than independently choosing a particle trial by trial.
    """

    parents = np.asarray(parent_indices, dtype=int)
    if parents.ndim != 2 or parents.shape[0] == 0 or parents.shape[1] == 0:
        raise ValueError("parent_indices must be a non-empty 2-D matrix.")
    n_trials, n_particles = parents.shape
    if np.any((parents < 0) | (parents >= n_particles)):
        raise ValueError("parent_indices contains an out-of-range particle index.")
    paths = np.empty((n_particles, n_trials), dtype=int)
    paths[:, -1] = np.arange(n_particles, dtype=int)
    for trial_index in range(n_trials - 2, -1, -1):
        paths[:, trial_index] = parents[
            trial_index,
            paths[:, trial_index + 1],
        ]
    return paths


def run_state_model_particle_filter(
    *,
    engine_config: Mapping[str, Any],
    subject_id: int,
    stimulus: Sequence[Sequence[float]] | np.ndarray,
    choices: Sequence[int] | np.ndarray,
    feedback: Sequence[float] | np.ndarray,
    particle_count: int,
    choice_readout_power: float,
    strategy_confidence_gain: float = 0.0,
    rule_commitment_confidence_gain: float = 0.0,
    output_lapse: float = 0.0,
    output_lapse_schedule: Sequence[float] | np.ndarray | None = None,
    learning_update_probability: float = 1.0,
    filter_seed: int = 20260730,
    resample_threshold_fraction: float = 0.5,
    choice_transmission_audit: bool = False,
    valid_trial_mask: Sequence[bool] | np.ndarray | None = None,
    processed_data_dir: Path | str | None = None,
    dataset_paths: Mapping[str, Path | str] | None = None,
) -> InferenceResult:
    """Filter one observed condition-1 trajectory using bootstrap particles."""

    from ...model import StateModel

    x = np.asarray(stimulus, dtype=float)
    observed_choices = np.asarray(choices, dtype=int).reshape(-1)
    observed_feedback = np.asarray(feedback, dtype=float).reshape(-1)
    if x.ndim != 2:
        raise ValueError("stimulus must be a 2-D array.")
    n_trials = int(x.shape[0])
    if observed_choices.shape[0] != n_trials or observed_feedback.shape[0] != n_trials:
        raise ValueError("stimulus, choices, and feedback must have equal trial counts.")
    if not np.all(np.isin(observed_choices, [1, 2])):
        raise ValueError("Condition-1 choices must be encoded as 1 or 2.")
    if not np.all(np.isfinite(observed_feedback)) or np.any(
        (observed_feedback < 0.0) | (observed_feedback > 1.0)
    ):
        raise ValueError("feedback must contain finite values in [0, 1].")
    n_particles = int(particle_count)
    if n_particles < 2:
        raise ValueError("particle_count must be at least 2.")
    rho_value = float(choice_readout_power)
    confidence_gain_value = float(strategy_confidence_gain)
    commitment_gain_value = float(rule_commitment_confidence_gain)
    epsilon_value = float(output_lapse)
    threshold_fraction = float(resample_threshold_fraction)
    update_probability = float(learning_update_probability)
    audit_choice_transmission = bool(choice_transmission_audit)
    if not np.isfinite(rho_value) or rho_value <= 0.0:
        raise ValueError("choice_readout_power must be finite and positive.")
    if (
        not np.isfinite(confidence_gain_value)
        or not 0.0 <= confidence_gain_value <= 10.0
    ):
        raise ValueError("strategy_confidence_gain must lie in [0, 10].")
    if (
        not np.isfinite(commitment_gain_value)
        or not 0.0 <= commitment_gain_value <= 10.0
    ):
        raise ValueError(
            "rule_commitment_confidence_gain must lie in [0, 10]."
        )
    if not np.isfinite(epsilon_value) or not 0.0 <= epsilon_value <= 1.0:
        raise ValueError("output_lapse must lie in [0, 1].")
    if output_lapse_schedule is None:
        epsilon_by_trial = np.full(n_trials, epsilon_value, dtype=float)
    else:
        epsilon_by_trial = np.asarray(
            output_lapse_schedule, dtype=float
        ).reshape(-1)
        if epsilon_by_trial.size != n_trials:
            raise ValueError(
                "output_lapse_schedule length must match the number of trials."
            )
        if (
            not np.all(np.isfinite(epsilon_by_trial))
            or np.any(epsilon_by_trial < 0.0)
            or np.any(epsilon_by_trial > 1.0)
        ):
            raise ValueError("output_lapse_schedule values must lie in [0, 1].")
    if (
        not np.isfinite(threshold_fraction)
        or not 0.0 < threshold_fraction <= 1.0
    ):
        raise ValueError("resample_threshold_fraction must lie in (0, 1].")
    if (
        not np.isfinite(update_probability)
        or not 0.0 <= update_probability <= 1.0
    ):
        raise ValueError("learning_update_probability must lie in [0, 1].")
    if valid_trial_mask is None:
        valid = np.ones(n_trials, dtype=bool)
    else:
        valid = np.asarray(valid_trial_mask, dtype=bool).reshape(-1)
        if valid.shape[0] != n_trials:
            raise ValueError("valid_trial_mask length does not match trials.")

    models = []
    shared_partition = None
    shared_space = None
    for particle_index in range(n_particles):
        config = _particle_config(engine_config, int(filter_seed), particle_index)
        model = StateModel(
            config,
            context=ModelContext(
                condition=1,
                subject_id=int(subject_id),
                processed_data_dir=processed_data_dir,
                dataset_paths=dataset_paths,
            ),
            partition=shared_partition,
            hypotheses_set=shared_space,
        )
        if shared_partition is None:
            shared_partition = model.partition_model
            shared_space = model.hypotheses_set
        models.append(model)

    n_hypotheses = int(models[0].engine.set_size)
    marginal_hypothesis_prior = np.zeros((n_trials, n_hypotheses), dtype=float)
    marginal_active_probability = np.zeros((n_trials, n_hypotheses), dtype=float)

    execution_flags = [
        bool(
            getattr(
                model.engine.get_module(
                    ModuleRole.HYPOTHESIS_TRANSITION,
                    required=True,
                ),
                "persistent_execution_enabled",
                False,
            )
        )
        for model in models
    ]
    if any(execution_flags) and not all(execution_flags):
        raise RuntimeError("persistent execution must be configured for every particle.")
    persistent_execution_enabled = bool(all(execution_flags))
    marginal_executed_probability = (
        np.zeros((n_trials, n_hypotheses), dtype=float)
        if persistent_execution_enabled
        else None
    )
    filtered_executed_probability = (
        np.zeros((n_trials, n_hypotheses), dtype=float)
        if persistent_execution_enabled
        else None
    )

    weights = np.full(n_particles, 1.0 / float(n_particles), dtype=float)
    particle_swap_counts = np.zeros(n_particles, dtype=int)
    marginal = np.zeros((n_trials, 2), dtype=float)
    pre_ess = np.zeros(n_trials, dtype=float)
    post_ess = np.zeros(n_trials, dtype=float)
    resampled = np.zeros(n_trials, dtype=bool)
    unique_ancestors = np.full(n_trials, n_particles, dtype=int)
    filtered_swap_probability = np.zeros(n_trials, dtype=float)
    filtered_swap_event_probability = np.zeros(n_trials, dtype=float)
    filtered_transition_rate = np.zeros(n_trials, dtype=float)
    filtered_search_range = np.zeros(n_trials, dtype=float)
    filtered_replacement_count = np.zeros(n_trials, dtype=float)
    filtered_replacement_fraction = np.zeros(n_trials, dtype=float)
    filtered_removed_mass = np.zeros(n_trials, dtype=float)
    filtered_newcomer_distance = np.zeros(n_trials, dtype=float)
    filtered_feedback_surprise = np.full(n_trials, np.nan, dtype=float)
    filtered_feedback_uncertainty = np.full(n_trials, np.nan, dtype=float)
    predictive_swap_probability = np.zeros(n_trials, dtype=float)
    predictive_swap_event_probability = np.zeros(n_trials, dtype=float)
    predictive_transition_rate = np.zeros(n_trials, dtype=float)
    predictive_search_range = np.zeros(n_trials, dtype=float)
    predictive_replacement_fraction = np.zeros(n_trials, dtype=float)
    predictive_newcomer_distance = np.zeros(n_trials, dtype=float)
    predictive_strategy_exploit = np.zeros(n_trials, dtype=float)
    predictive_strategy_local_explore = np.zeros(n_trials, dtype=float)
    predictive_strategy_global_explore = np.zeros(n_trials, dtype=float)
    predictive_failure_pressure = np.full(n_trials, np.nan, dtype=float)
    predictive_mastery_evidence = np.full(n_trials, np.nan, dtype=float)
    predictive_peak_mastery_evidence = np.full(n_trials, np.nan, dtype=float)
    predictive_choice_confidence_signal = np.zeros(n_trials, dtype=float)
    predictive_strategy_choice_precision = np.ones(n_trials, dtype=float)
    predictive_exploration_target = np.full(n_trials, np.nan, dtype=float)
    predictive_global_target = np.full(n_trials, np.nan, dtype=float)
    predictive_prior_reset_strength = np.zeros(n_trials, dtype=float)
    predictive_prior_reset_mass_shift = np.zeros(n_trials, dtype=float)
    predictive_execution_switch_probability = np.zeros(n_trials, dtype=float)
    predictive_execution_switch_event_probability = np.zeros(n_trials, dtype=float)
    predictive_execution_dwell_trials = np.zeros(n_trials, dtype=float)
    predictive_executed_beta = np.full(n_trials, np.nan, dtype=float)
    filtered_executed_beta = np.full(n_trials, np.nan, dtype=float)
    filtered_execution_switch_event_probability = np.zeros(n_trials, dtype=float)
    filtered_execution_dwell_trials = np.zeros(n_trials, dtype=float)
    predictive_misconception_capture_eligible_probability = np.zeros(
        n_trials, dtype=float
    )
    predictive_misconception_capture_hold_probability = np.zeros(
        n_trials, dtype=float
    )
    predictive_misconception_capture_switch_event_probability = np.zeros(
        n_trials, dtype=float
    )
    predictive_executed_choice_compatibility = np.full(n_trials, np.nan, dtype=float)
    predictive_best_alternative_choice_compatibility = np.full(n_trials, np.nan, dtype=float)
    predictive_rule_commitment_probability = np.zeros(n_trials, dtype=float)
    predictive_rule_commitment_eligible_probability = np.zeros(n_trials, dtype=float)
    predictive_rule_commitment_entry_event_probability = np.zeros(n_trials, dtype=float)
    predictive_rule_commitment_exit_event_probability = np.zeros(n_trials, dtype=float)
    predictive_rule_commitment_age = np.zeros(n_trials, dtype=float)
    predictive_rule_commitment_disconfirmation = np.zeros(n_trials, dtype=float)
    predictive_rule_commitment_margin = np.full(n_trials, np.nan, dtype=float)
    predictive_rule_commitment_confidence_signal = np.zeros(n_trials, dtype=float)
    predictive_rule_commitment_choice_precision = np.ones(n_trials, dtype=float)

    audit_hypothesis_map = (
        np.zeros((n_trials, 2), dtype=float)
        if audit_choice_transmission
        else None
    )
    audit_adaptive_sharpening = (
        np.zeros((n_trials, 2), dtype=float)
        if audit_choice_transmission
        else None
    )
    audit_exploration_lapse = (
        np.zeros((n_trials, 2), dtype=float)
        if audit_choice_transmission
        else None
    )
    audit_unsharpened_expectation = (
        np.zeros((n_trials, 2), dtype=float)
        if audit_choice_transmission
        else None
    )
    audit_sharpened_no_lapse = (
        np.zeros((n_trials, 2), dtype=float)
        if audit_choice_transmission
        else None
    )
    audit_strategy_confidence_no_lapse = (
        np.zeros((n_trials, 2), dtype=float)
        if audit_choice_transmission
        else None
    )
    audit_persistent_execution_no_lapse = (
        np.zeros((n_trials, 2), dtype=float)
        if audit_choice_transmission and persistent_execution_enabled
        else None
    )
    audit_correct_predicting_available_probability = (
        np.zeros(n_trials, dtype=float) if audit_choice_transmission else None
    )
    audit_correct_predicting_prior_mass = (
        np.zeros(n_trials, dtype=float) if audit_choice_transmission else None
    )
    audit_best_active_correct_probability = (
        np.zeros(n_trials, dtype=float) if audit_choice_transmission else None
    )
    audit_particle_correct_q10 = (
        np.zeros(n_trials, dtype=float) if audit_choice_transmission else None
    )
    audit_particle_correct_q50 = (
        np.zeros(n_trials, dtype=float) if audit_choice_transmission else None
    )
    audit_particle_correct_q90 = (
        np.zeros(n_trials, dtype=float) if audit_choice_transmission else None
    )
    audit_parent_indices = (
        np.tile(np.arange(n_particles, dtype=int), (n_trials, 1))
        if audit_choice_transmission
        else None
    )
    audit_particle_correct_probability = (
        np.zeros((n_trials, n_particles), dtype=float)
        if audit_choice_transmission
        else None
    )
    audit_particle_strategy_exploit = (
        np.zeros((n_trials, n_particles), dtype=float)
        if audit_choice_transmission
        else None
    )
    audit_particle_strategy_local_explore = (
        np.zeros((n_trials, n_particles), dtype=float)
        if audit_choice_transmission
        else None
    )
    audit_particle_strategy_global_explore = (
        np.zeros((n_trials, n_particles), dtype=float)
        if audit_choice_transmission
        else None
    )
    audit_particle_swap_event = (
        np.zeros((n_trials, n_particles), dtype=float)
        if audit_choice_transmission
        else None
    )
    audit_particle_transition_rate = (
        np.zeros((n_trials, n_particles), dtype=float)
        if audit_choice_transmission
        else None
    )
    audit_particle_search_range = (
        np.zeros((n_trials, n_particles), dtype=float)
        if audit_choice_transmission
        else None
    )
    audit_particle_failure_pressure = (
        np.full((n_trials, n_particles), np.nan, dtype=float)
        if audit_choice_transmission
        else None
    )
    audit_particle_mastery_evidence = (
        np.full((n_trials, n_particles), np.nan, dtype=float)
        if audit_choice_transmission
        else None
    )
    audit_particle_executed_hypothesis = (
        np.full((n_trials, n_particles), -1, dtype=int)
        if audit_choice_transmission and persistent_execution_enabled
        else None
    )
    audit_particle_execution_switch_event = (
        np.zeros((n_trials, n_particles), dtype=float)
        if audit_choice_transmission and persistent_execution_enabled
        else None
    )
    audit_particle_execution_dwell_trials = (
        np.zeros((n_trials, n_particles), dtype=float)
        if audit_choice_transmission and persistent_execution_enabled
        else None
    )
    audit_terminal_weights = None
    resampling_log: list[dict[str, Any]] = []

    for trial_index in range(n_trials):
        particle_predictions = np.zeros((n_particles, 2), dtype=float)
        particle_priors = np.zeros((n_particles, n_hypotheses), dtype=float)
        particle_active = np.zeros((n_particles, n_hypotheses), dtype=float)
        swap_probabilities = np.zeros(n_particles, dtype=float)
        swap_events = np.zeros(n_particles, dtype=float)
        transition_rates = np.zeros(n_particles, dtype=float)
        search_ranges = np.zeros(n_particles, dtype=float)
        replacement_counts = np.zeros(n_particles, dtype=float)
        replacement_fractions = np.zeros(n_particles, dtype=float)
        removed_masses = np.zeros(n_particles, dtype=float)
        newcomer_distances = np.zeros(n_particles, dtype=float)
        feedback_surprises = np.full(n_particles, np.nan, dtype=float)
        feedback_uncertainties = np.full(n_particles, np.nan, dtype=float)
        failure_pressures = np.full(n_particles, np.nan, dtype=float)
        mastery_evidences = np.full(n_particles, np.nan, dtype=float)
        peak_mastery_evidences = np.full(n_particles, np.nan, dtype=float)
        choice_confidence_signals = np.zeros(n_particles, dtype=float)
        strategy_choice_precisions = np.ones(n_particles, dtype=float)
        exploration_targets = np.full(n_particles, np.nan, dtype=float)
        global_targets = np.full(n_particles, np.nan, dtype=float)
        prior_reset_strengths = np.zeros(n_particles, dtype=float)
        prior_reset_mass_shifts = np.zeros(n_particles, dtype=float)
        particle_executed = np.zeros((n_particles, n_hypotheses), dtype=float)
        executed_betas = np.full(n_particles, np.nan, dtype=float)
        execution_switch_probabilities = np.zeros(n_particles, dtype=float)
        execution_switch_events = np.zeros(n_particles, dtype=float)
        execution_dwell_trials = np.zeros(n_particles, dtype=float)
        capture_eligible = np.zeros(n_particles, dtype=float)
        capture_hold = np.zeros(n_particles, dtype=float)
        capture_switch_events = np.zeros(n_particles, dtype=float)
        executed_choice_compatibility = np.full(n_particles, np.nan, dtype=float)
        best_alternative_choice_compatibility = np.full(n_particles, np.nan, dtype=float)
        rule_commitment_active = np.zeros(n_particles, dtype=float)
        rule_commitment_eligible = np.zeros(n_particles, dtype=float)
        rule_commitment_entry_events = np.zeros(n_particles, dtype=float)
        rule_commitment_exit_events = np.zeros(n_particles, dtype=float)
        rule_commitment_ages = np.zeros(n_particles, dtype=float)
        rule_commitment_disconfirmation = np.zeros(n_particles, dtype=float)
        rule_commitment_margins = np.full(n_particles, np.nan, dtype=float)
        rule_commitment_confidence_signals = np.zeros(n_particles, dtype=float)
        rule_commitment_choice_precisions = np.ones(n_particles, dtype=float)

        if audit_choice_transmission:
            particle_hypothesis_map = np.zeros((n_particles, 2), dtype=float)
            particle_adaptive_sharpening = np.zeros((n_particles, 2), dtype=float)
            particle_exploration_lapse = np.zeros((n_particles, 2), dtype=float)
            particle_unsharpened_expectation = np.zeros(
                (n_particles, 2), dtype=float
            )
            particle_sharpened_no_lapse = np.zeros(
                (n_particles, 2), dtype=float
            )
            particle_strategy_confidence_no_lapse = np.zeros(
                (n_particles, 2), dtype=float
            )
            particle_persistent_execution_no_lapse = np.zeros(
                (n_particles, 2), dtype=float
            )
            particle_correct_predicting_available = np.zeros(
                n_particles, dtype=float
            )
            particle_correct_predicting_prior_mass = np.zeros(
                n_particles, dtype=float
            )
            particle_best_active_correct_probability = np.zeros(
                n_particles, dtype=float
            )

        observed_choice_index = int(observed_choices[trial_index]) - 1
        correct_category_index = (
            observed_choice_index
            if observed_feedback[trial_index] >= 0.5
            else 1 - observed_choice_index
        )

        for particle_index, model in enumerate(models):
            engine = model.engine
            prepared = model.begin_trial(x[trial_index])
            perceived = prepared.perceived_stimulus
            transition = engine.get_module(
                ModuleRole.HYPOTHESIS_TRANSITION,
                required=True,
            )
            event = transition.transition_log[-1]
            particle_priors[particle_index] = _normalize(
                np.asarray(engine.prior, dtype=float)
            )
            particle_active[particle_index] = (
                np.asarray(engine.hypotheses_mask, dtype=float) > 0.0
            ).astype(float)
            swap_probabilities[particle_index] = float(
                event["swap_probability"]
            )
            swap_events[particle_index] = float(bool(event["swap_event"]))
            transition_rates[particle_index] = float(
                event.get("predictive_m", event["swap_probability"])
            )
            search_ranges[particle_index] = float(
                event.get("predictive_g", event.get("g", 0.0))
            )
            replacement_counts[particle_index] = float(
                event.get("replacement_count", bool(event["swap_event"]))
            )
            replacement_fractions[particle_index] = float(
                event.get("replacement_fraction", bool(event["swap_event"]))
            )
            removed_masses[particle_index] = float(event.get("removed_mass", 0.0))
            newcomer_distances[particle_index] = float(
                event.get("newcomer_distance", 0.0)
            )
            feedback_surprises[particle_index] = float(
                event.get("feedback_surprise", np.nan)
            )
            feedback_uncertainties[particle_index] = float(
                event.get("feedback_uncertainty", np.nan)
            )
            failure_pressures[particle_index] = float(
                event.get("failure_pressure", np.nan)
            )
            mastery_evidences[particle_index] = float(
                event.get("mastery_evidence", np.nan)
            )
            peak_mastery_evidences[particle_index] = float(
                event.get("peak_mastery_evidence", np.nan)
            )
            exploration_targets[particle_index] = float(
                event.get("exploration_target", np.nan)
            )
            global_targets[particle_index] = float(
                event.get("global_target", np.nan)
            )
            prior_reset_strengths[particle_index] = float(
                event.get("prior_reset_strength", 0.0)
            )
            prior_reset_mass_shifts[particle_index] = float(
                event.get("prior_reset_mass_shift", 0.0)
            )
            execution_switch_probabilities[particle_index] = float(
                event.get("execution_switch_probability", 0.0)
            )
            execution_switch_events[particle_index] = float(
                bool(event.get("execution_switch_event", False))
            )
            execution_dwell_trials[particle_index] = float(
                event.get("execution_dwell_trials", 0.0)
            )
            capture_eligible[particle_index] = float(
                bool(event.get("misconception_capture_eligible", False))
            )
            capture_hold[particle_index] = float(
                int(event.get("misconception_capture_hold_remaining", 0)) > 0
            )
            capture_switch_events[particle_index] = float(
                bool(event.get("misconception_capture_switch_event", False))
            )
            executed_choice_compatibility[particle_index] = float(
                event.get("executed_choice_compatibility", np.nan)
            )
            best_alternative_choice_compatibility[particle_index] = float(
                event.get("best_alternative_choice_compatibility", np.nan)
            )
            rule_commitment_active[particle_index] = float(
                bool(event.get("rule_commitment_active", False))
            )
            rule_commitment_eligible[particle_index] = float(
                bool(event.get("rule_commitment_eligible", False))
            )
            rule_commitment_entry_events[particle_index] = float(
                bool(event.get("rule_commitment_entry_event", False))
            )
            rule_commitment_exit_events[particle_index] = float(
                bool(event.get("rule_commitment_exit_event", False))
            )
            rule_commitment_ages[particle_index] = float(
                event.get("rule_commitment_age", 0.0)
            )
            rule_commitment_disconfirmation[particle_index] = float(
                event.get("rule_commitment_disconfirmation", 0.0)
            )
            rule_commitment_margins[particle_index] = float(
                event.get("rule_commitment_margin", np.nan)
            )
            if persistent_execution_enabled:
                executed_hypothesis = int(event["executed_hypothesis"])
                if not 0 <= executed_hypothesis < n_hypotheses:
                    raise RuntimeError(
                        "particle executed_hypothesis falls outside the hypothesis space."
                    )
                particle_executed[
                    particle_index,
                    executed_hypothesis,
                ] = 1.0
                executed_betas[particle_index] = float(
                    engine.beta[executed_hypothesis]
                )
            particle_predictions[particle_index], confidence_details = (
                _choice_probability(
                    model,
                    perceived,
                    rho=rho_value,
                    epsilon=float(epsilon_by_trial[trial_index]),
                    strategy_confidence_gain=confidence_gain_value,
                    rule_commitment_confidence_gain=commitment_gain_value,
                    failure_pressure=failure_pressures[particle_index],
                    mastery_evidence=mastery_evidences[particle_index],
                    return_details=True,
                )
            )
            choice_confidence_signals[particle_index] = float(
                confidence_details["strategy_confidence_signal"]
            )
            strategy_choice_precisions[particle_index] = float(
                confidence_details["strategy_choice_precision"]
            )
            rule_commitment_confidence_signals[particle_index] = float(
                confidence_details["rule_commitment_confidence_signal"]
            )
            rule_commitment_choice_precisions[particle_index] = float(
                confidence_details["rule_commitment_choice_precision"]
            )
            if audit_choice_transmission:
                exploration = float(np.clip(swap_probabilities[particle_index], 0.0, 1.0))
                layer_audit = _choice_layer_audit(
                    model,
                    perceived,
                    correct_category_index=correct_category_index,
                    readout_power=rho_value,
                    lapse=float(epsilon_by_trial[trial_index]),
                    strategy_confidence_gain=confidence_gain_value,
                    rule_commitment_confidence_gain=commitment_gain_value,
                    failure_pressure=failure_pressures[particle_index],
                    mastery_evidence=mastery_evidences[particle_index],
                )
                if not np.allclose(
                    layer_audit["fitted"],
                    particle_predictions[particle_index],
                    rtol=1e-10,
                    atol=1e-12,
                ):
                    raise RuntimeError(
                        "Choice-layer audit does not reproduce the fitted readout."
                    )
                particle_hypothesis_map[particle_index] = layer_audit[
                    "map_with_lapse"
                ]
                particle_unsharpened_expectation[particle_index] = layer_audit[
                    "unsharpened"
                ]
                particle_sharpened_no_lapse[particle_index] = layer_audit[
                    "sharpened_no_lapse"
                ]
                particle_strategy_confidence_no_lapse[particle_index] = (
                    layer_audit["strategy_confidence_no_lapse"]
                )
                particle_persistent_execution_no_lapse[particle_index] = (
                    layer_audit["persistent_execution_no_lapse"]
                )
                particle_correct_predicting_available[particle_index] = float(
                    layer_audit["correct_predicting_available"]
                )
                particle_correct_predicting_prior_mass[particle_index] = float(
                    layer_audit["correct_predicting_prior_mass"]
                )
                particle_best_active_correct_probability[particle_index] = float(
                    layer_audit["best_active_correct_probability"]
                )
                adaptive_power = float(
                    1.0 + (rho_value - 1.0) * (1.0 - exploration)
                )
                particle_adaptive_sharpening[particle_index] = (
                    _choice_probability(
                        model,
                        perceived,
                        rho=adaptive_power,
                        epsilon=float(epsilon_by_trial[trial_index]),
                        strategy_confidence_gain=confidence_gain_value,
                    rule_commitment_confidence_gain=commitment_gain_value,
                        failure_pressure=failure_pressures[particle_index],
                        mastery_evidence=mastery_evidences[particle_index],
                    )
                )
                particle_exploration_lapse[particle_index] = _normalize(
                    (1.0 - exploration) * particle_predictions[particle_index]
                    + exploration * np.full(2, 0.5, dtype=float)
                )

        pre_ess[trial_index] = effective_sample_size(weights)
        marginal_hypothesis_prior[trial_index] = np.sum(
            weights[:, None] * particle_priors,
            axis=0,
        )
        marginal_hypothesis_prior[trial_index] = _normalize(
            marginal_hypothesis_prior[trial_index]
        )
        marginal_active_probability[trial_index] = np.sum(
            weights[:, None] * particle_active,
            axis=0,
        )
        if marginal_executed_probability is not None:
            marginal_executed_probability[trial_index] = _normalize(
                np.sum(weights[:, None] * particle_executed, axis=0)
            )
        marginal[trial_index] = np.sum(
            weights[:, None] * particle_predictions,
            axis=0,
        )
        marginal[trial_index] = _normalize(marginal[trial_index])

        if audit_choice_transmission:
            assert audit_hypothesis_map is not None
            assert audit_adaptive_sharpening is not None
            assert audit_exploration_lapse is not None
            assert audit_unsharpened_expectation is not None
            assert audit_sharpened_no_lapse is not None
            assert audit_strategy_confidence_no_lapse is not None
            assert audit_correct_predicting_available_probability is not None
            assert audit_correct_predicting_prior_mass is not None
            assert audit_best_active_correct_probability is not None
            assert audit_particle_correct_q10 is not None
            assert audit_particle_correct_q50 is not None
            assert audit_particle_correct_q90 is not None
            assert audit_particle_correct_probability is not None
            assert audit_particle_strategy_exploit is not None
            assert audit_particle_strategy_local_explore is not None
            assert audit_particle_strategy_global_explore is not None
            assert audit_particle_swap_event is not None
            assert audit_particle_transition_rate is not None
            assert audit_particle_search_range is not None
            assert audit_particle_failure_pressure is not None
            assert audit_particle_mastery_evidence is not None
            audit_hypothesis_map[trial_index] = _normalize(
                np.sum(weights[:, None] * particle_hypothesis_map, axis=0)
            )
            audit_adaptive_sharpening[trial_index] = _normalize(
                np.sum(weights[:, None] * particle_adaptive_sharpening, axis=0)
            )
            audit_exploration_lapse[trial_index] = _normalize(
                np.sum(weights[:, None] * particle_exploration_lapse, axis=0)
            )
            audit_unsharpened_expectation[trial_index] = _normalize(
                np.sum(
                    weights[:, None] * particle_unsharpened_expectation,
                    axis=0,
                )
            )
            audit_sharpened_no_lapse[trial_index] = _normalize(
                np.sum(
                    weights[:, None] * particle_sharpened_no_lapse,
                    axis=0,
                )
            )
            audit_strategy_confidence_no_lapse[trial_index] = _normalize(
                np.sum(
                    weights[:, None] * particle_strategy_confidence_no_lapse,
                    axis=0,
                )
            )
            if audit_persistent_execution_no_lapse is not None:
                audit_persistent_execution_no_lapse[trial_index] = _normalize(
                    np.sum(
                        weights[:, None]
                        * particle_persistent_execution_no_lapse,
                        axis=0,
                    )
                )
            audit_correct_predicting_available_probability[trial_index] = float(
                np.sum(weights * particle_correct_predicting_available)
            )
            audit_correct_predicting_prior_mass[trial_index] = float(
                np.sum(weights * particle_correct_predicting_prior_mass)
            )
            audit_best_active_correct_probability[trial_index] = float(
                np.sum(weights * particle_best_active_correct_probability)
            )
            q10, q50, q90 = _weighted_quantiles(
                particle_predictions[:, correct_category_index],
                weights,
                (0.10, 0.50, 0.90),
            )
            audit_particle_correct_q10[trial_index] = float(q10)
            audit_particle_correct_q50[trial_index] = float(q50)
            audit_particle_correct_q90[trial_index] = float(q90)
            audit_particle_correct_probability[trial_index] = (
                particle_predictions[:, correct_category_index]
            )
            audit_particle_strategy_exploit[trial_index] = 1.0 - swap_probabilities
            audit_particle_strategy_local_explore[trial_index] = (
                swap_probabilities * (1.0 - search_ranges)
            )
            audit_particle_strategy_global_explore[trial_index] = (
                swap_probabilities * search_ranges
            )
            audit_particle_swap_event[trial_index] = swap_events
            audit_particle_transition_rate[trial_index] = transition_rates
            audit_particle_search_range[trial_index] = search_ranges
            audit_particle_failure_pressure[trial_index] = failure_pressures
            audit_particle_mastery_evidence[trial_index] = mastery_evidences
            if audit_particle_executed_hypothesis is not None:
                audit_particle_executed_hypothesis[trial_index] = np.argmax(
                    particle_executed,
                    axis=1,
                )
                audit_particle_execution_switch_event[trial_index] = (
                    execution_switch_events
                )
                audit_particle_execution_dwell_trials[trial_index] = (
                    execution_dwell_trials
                )

        # These summaries describe the policy available before observing the
        # current choice.  Keeping them separate from the post-choice filtered
        # diagnostics prevents choice_t from retrospectively changing the
        # interpretation of the strategy used at trial t.
        predictive_swap_probability[trial_index] = float(
            np.sum(weights * swap_probabilities)
        )
        predictive_swap_event_probability[trial_index] = float(
            np.sum(weights * swap_events)
        )
        predictive_transition_rate[trial_index] = float(
            np.sum(weights * transition_rates)
        )
        predictive_search_range[trial_index] = float(
            np.sum(weights * search_ranges)
        )
        predictive_replacement_fraction[trial_index] = float(
            np.sum(weights * replacement_fractions)
        )
        predictive_newcomer_distance[trial_index] = float(
            np.sum(weights * newcomer_distances)
        )
        exploit_tendency = 1.0 - swap_probabilities
        local_explore_tendency = swap_probabilities * (1.0 - search_ranges)
        global_explore_tendency = swap_probabilities * search_ranges
        predictive_strategy_exploit[trial_index] = float(
            np.sum(weights * exploit_tendency)
        )
        predictive_strategy_local_explore[trial_index] = float(
            np.sum(weights * local_explore_tendency)
        )
        predictive_strategy_global_explore[trial_index] = float(
            np.sum(weights * global_explore_tendency)
        )
        predictive_choice_confidence_signal[trial_index] = float(
            np.sum(weights * choice_confidence_signals)
        )
        predictive_strategy_choice_precision[trial_index] = float(
            np.sum(weights * strategy_choice_precisions)
        )
        predictive_prior_reset_strength[trial_index] = float(
            np.sum(weights * prior_reset_strengths)
        )
        predictive_prior_reset_mass_shift[trial_index] = float(
            np.sum(weights * prior_reset_mass_shifts)
        )
        predictive_execution_switch_probability[trial_index] = float(
            np.sum(weights * execution_switch_probabilities)
        )
        predictive_execution_switch_event_probability[trial_index] = float(
            np.sum(weights * execution_switch_events)
        )
        predictive_execution_dwell_trials[trial_index] = float(
            np.sum(weights * execution_dwell_trials)
        )
        predictive_misconception_capture_eligible_probability[trial_index] = float(
            np.sum(weights * capture_eligible)
        )
        predictive_misconception_capture_hold_probability[trial_index] = float(
            np.sum(weights * capture_hold)
        )
        predictive_misconception_capture_switch_event_probability[trial_index] = float(
            np.sum(weights * capture_switch_events)
        )
        predictive_rule_commitment_probability[trial_index] = float(
            np.sum(weights * rule_commitment_active)
        )
        predictive_rule_commitment_eligible_probability[trial_index] = float(
            np.sum(weights * rule_commitment_eligible)
        )
        predictive_rule_commitment_entry_event_probability[trial_index] = float(
            np.sum(weights * rule_commitment_entry_events)
        )
        predictive_rule_commitment_exit_event_probability[trial_index] = float(
            np.sum(weights * rule_commitment_exit_events)
        )
        predictive_rule_commitment_age[trial_index] = float(
            np.sum(weights * rule_commitment_ages)
        )
        predictive_rule_commitment_disconfirmation[trial_index] = float(
            np.sum(weights * rule_commitment_disconfirmation)
        )
        predictive_rule_commitment_confidence_signal[trial_index] = float(
            np.sum(weights * rule_commitment_confidence_signals)
        )
        predictive_rule_commitment_choice_precision[trial_index] = float(
            np.sum(weights * rule_commitment_choice_precisions)
        )
        finite_commitment_margin = np.isfinite(rule_commitment_margins)
        if np.any(finite_commitment_margin):
            margin_weights = weights[finite_commitment_margin]
            margin_weights = margin_weights / float(np.sum(margin_weights))
            predictive_rule_commitment_margin[trial_index] = float(
                np.sum(
                    margin_weights
                    * rule_commitment_margins[finite_commitment_margin]
                )
            )
        if persistent_execution_enabled:
            predictive_executed_beta[trial_index] = float(
                np.sum(weights * executed_betas)
            )
        for source, target in (
            (failure_pressures, predictive_failure_pressure),
            (mastery_evidences, predictive_mastery_evidence),
            (peak_mastery_evidences, predictive_peak_mastery_evidence),
            (exploration_targets, predictive_exploration_target),
            (global_targets, predictive_global_target),
            (
                executed_choice_compatibility,
                predictive_executed_choice_compatibility,
            ),
            (
                best_alternative_choice_compatibility,
                predictive_best_alternative_choice_compatibility,
            ),
        ):
            finite = np.isfinite(source)
            if np.any(finite):
                signal_weights = weights[finite]
                signal_weights = signal_weights / float(np.sum(signal_weights))
                target[trial_index] = float(
                    np.sum(signal_weights * source[finite])
                )

        choice_index = int(observed_choices[trial_index]) - 1
        if valid[trial_index]:
            weights *= np.clip(
                particle_predictions[:, choice_index],
                1e-12,
                1.0,
            )
            weight_total = float(np.sum(weights))
            if not np.isfinite(weight_total) or weight_total <= 0.0:
                weights.fill(1.0 / float(n_particles))
            else:
                weights /= weight_total
        post_ess[trial_index] = effective_sample_size(weights)
        if audit_choice_transmission and trial_index == n_trials - 1:
            audit_terminal_weights = weights.copy()
        if filtered_executed_probability is not None:
            filtered_executed_probability[trial_index] = _normalize(
                np.sum(weights[:, None] * particle_executed, axis=0)
            )
        if persistent_execution_enabled:
            filtered_executed_beta[trial_index] = float(
                np.sum(weights * executed_betas)
            )
        filtered_execution_switch_event_probability[trial_index] = float(
            np.sum(weights * execution_switch_events)
        )
        filtered_execution_dwell_trials[trial_index] = float(
            np.sum(weights * execution_dwell_trials)
        )
        filtered_swap_probability[trial_index] = float(
            np.sum(weights * swap_probabilities)
        )
        filtered_swap_event_probability[trial_index] = float(
            np.sum(weights * swap_events)
        )
        filtered_transition_rate[trial_index] = float(
            np.sum(weights * transition_rates)
        )
        filtered_search_range[trial_index] = float(
            np.sum(weights * search_ranges)
        )
        filtered_replacement_count[trial_index] = float(
            np.sum(weights * replacement_counts)
        )
        filtered_replacement_fraction[trial_index] = float(
            np.sum(weights * replacement_fractions)
        )
        filtered_removed_mass[trial_index] = float(
            np.sum(weights * removed_masses)
        )
        filtered_newcomer_distance[trial_index] = float(
            np.sum(weights * newcomer_distances)
        )
        finite_surprise = np.isfinite(feedback_surprises)
        if np.any(finite_surprise):
            signal_weights = weights[finite_surprise]
            signal_weights = signal_weights / float(np.sum(signal_weights))
            filtered_feedback_surprise[trial_index] = float(
                np.sum(signal_weights * feedback_surprises[finite_surprise])
            )
        finite_uncertainty = np.isfinite(feedback_uncertainties)
        if np.any(finite_uncertainty):
            signal_weights = weights[finite_uncertainty]
            signal_weights = signal_weights / float(np.sum(signal_weights))
            filtered_feedback_uncertainty[trial_index] = float(
                np.sum(signal_weights * feedback_uncertainties[finite_uncertainty])
            )
        particle_swap_counts += swap_events.astype(int)

        for particle_index, model in enumerate(models):
            engine = model.engine
            update_seed = _future_seed(
                int(filter_seed),
                trial_index,
                particle_index,
                "active_set_learning_update_gate",
            )
            update_occurs = bool(
                np.random.default_rng(update_seed).random()
                < update_probability
            )
            model.complete_trial(
                int(observed_choices[trial_index]),
                float(observed_feedback[trial_index]),
                update_state=update_occurs,
            )
            if hasattr(engine, "clear_module_logs"):
                engine.clear_module_logs()

        if post_ess[trial_index] < threshold_fraction * float(n_particles):
            resampling_seed = _future_seed(
                int(filter_seed),
                trial_index,
                0,
                "active_set_pf_systematic_resampling",
            )
            uniform = float(np.random.default_rng(resampling_seed).random())
            ancestors = systematic_resample(weights, uniform)
            if audit_choice_transmission:
                assert audit_parent_indices is not None
                audit_parent_indices[trial_index] = ancestors
            snapshots = [_snapshot(models[int(index)]) for index in ancestors]
            copied_swap_counts = particle_swap_counts[ancestors].copy()
            for particle_index, snapshot in enumerate(snapshots):
                _restore(
                    models[particle_index],
                    snapshot,
                    filter_seed=int(filter_seed),
                    trial_index=trial_index,
                    particle_index=particle_index,
                )
            particle_swap_counts = copied_swap_counts
            weights.fill(1.0 / float(n_particles))
            resampled[trial_index] = True
            unique_ancestors[trial_index] = int(np.unique(ancestors).size)
            resampling_log.append(
                {
                    "trial_index": int(trial_index),
                    "post_choice_ess": float(post_ess[trial_index]),
                    "unique_ancestors": int(np.unique(ancestors).size),
                    "uniform": uniform,
                    "ancestors": ancestors.tolist(),
                }
            )

    audit_ancestral_paths = None
    if audit_choice_transmission:
        assert audit_parent_indices is not None
        assert audit_terminal_weights is not None
        assert audit_particle_correct_probability is not None
        assert audit_particle_strategy_exploit is not None
        assert audit_particle_strategy_local_explore is not None
        assert audit_particle_strategy_global_explore is not None
        assert audit_particle_swap_event is not None
        assert audit_particle_transition_rate is not None
        assert audit_particle_search_range is not None
        assert audit_particle_failure_pressure is not None
        assert audit_particle_mastery_evidence is not None
        path_indices = _trace_ancestral_indices(audit_parent_indices)
        trial_rows = np.arange(n_trials, dtype=int)[None, :]

        def trace(values: np.ndarray) -> np.ndarray:
            return np.asarray(values)[trial_rows, path_indices]

        audit_ancestral_paths = {
            "particle_indices": path_indices,
            "weights": _normalize(audit_terminal_weights),
            "correct_probability": trace(audit_particle_correct_probability),
            "strategy_exploit": trace(audit_particle_strategy_exploit),
            "strategy_local_explore": trace(
                audit_particle_strategy_local_explore
            ),
            "strategy_global_explore": trace(
                audit_particle_strategy_global_explore
            ),
            "swap_event": trace(audit_particle_swap_event),
            "transition_rate": trace(audit_particle_transition_rate),
            "search_range": trace(audit_particle_search_range),
            "failure_pressure": trace(audit_particle_failure_pressure),
            "mastery_evidence": trace(audit_particle_mastery_evidence),
        }
        if audit_particle_executed_hypothesis is not None:
            audit_ancestral_paths.update(
                {
                    "executed_hypothesis": trace(
                        audit_particle_executed_hypothesis
                    ),
                    "execution_switch_event": trace(
                        audit_particle_execution_switch_event
                    ),
                    "execution_dwell_trials": trace(
                        audit_particle_execution_dwell_trials
                    ),
                }
            )

    return ParticleFilterResult(
        marginal_probabilities=marginal,
        marginal_hypothesis_prior=marginal_hypothesis_prior,
        marginal_active_probability=marginal_active_probability,
        marginal_executed_probability=marginal_executed_probability,
        filtered_executed_probability=filtered_executed_probability,
        pre_choice_ess=pre_ess,
        post_choice_ess=post_ess,
        resampled=resampled,
        resampling_unique_ancestors=unique_ancestors,
        filtered_swap_probability=filtered_swap_probability,
        filtered_swap_event_probability=filtered_swap_event_probability,
        filtered_transition_rate=filtered_transition_rate,
        filtered_search_range=filtered_search_range,
        filtered_replacement_count=filtered_replacement_count,
        filtered_replacement_fraction=filtered_replacement_fraction,
        filtered_removed_mass=filtered_removed_mass,
        filtered_newcomer_distance=filtered_newcomer_distance,
        filtered_feedback_surprise=filtered_feedback_surprise,
        filtered_feedback_uncertainty=filtered_feedback_uncertainty,
        predictive_swap_probability=predictive_swap_probability,
        predictive_swap_event_probability=predictive_swap_event_probability,
        predictive_transition_rate=predictive_transition_rate,
        predictive_search_range=predictive_search_range,
        predictive_replacement_fraction=predictive_replacement_fraction,
        predictive_newcomer_distance=predictive_newcomer_distance,
        predictive_strategy_exploit=predictive_strategy_exploit,
        predictive_strategy_local_explore=predictive_strategy_local_explore,
        predictive_strategy_global_explore=predictive_strategy_global_explore,
        predictive_failure_pressure=predictive_failure_pressure,
        predictive_mastery_evidence=predictive_mastery_evidence,
        predictive_peak_mastery_evidence=predictive_peak_mastery_evidence,
        predictive_choice_confidence_signal=(
            predictive_choice_confidence_signal
        ),
        predictive_strategy_choice_precision=(
            predictive_strategy_choice_precision
        ),
        predictive_exploration_target=predictive_exploration_target,
        predictive_global_target=predictive_global_target,
        predictive_prior_reset_strength=predictive_prior_reset_strength,
        predictive_prior_reset_mass_shift=predictive_prior_reset_mass_shift,
        predictive_execution_switch_probability=(
            predictive_execution_switch_probability
        ),
        predictive_execution_switch_event_probability=(
            predictive_execution_switch_event_probability
        ),
        predictive_execution_dwell_trials=predictive_execution_dwell_trials,
        predictive_misconception_capture_eligible_probability=(
            predictive_misconception_capture_eligible_probability
        ),
        predictive_misconception_capture_hold_probability=(
            predictive_misconception_capture_hold_probability
        ),
        predictive_misconception_capture_switch_event_probability=(
            predictive_misconception_capture_switch_event_probability
        ),
        predictive_rule_commitment_probability=(
            predictive_rule_commitment_probability
        ),
        predictive_rule_commitment_eligible_probability=(
            predictive_rule_commitment_eligible_probability
        ),
        predictive_rule_commitment_entry_event_probability=(
            predictive_rule_commitment_entry_event_probability
        ),
        predictive_rule_commitment_exit_event_probability=(
            predictive_rule_commitment_exit_event_probability
        ),
        predictive_rule_commitment_age=predictive_rule_commitment_age,
        predictive_rule_commitment_disconfirmation=(
            predictive_rule_commitment_disconfirmation
        ),
        predictive_rule_commitment_margin=predictive_rule_commitment_margin,
        predictive_rule_commitment_confidence_signal=(
            predictive_rule_commitment_confidence_signal
        ),
        predictive_rule_commitment_choice_precision=(
            predictive_rule_commitment_choice_precision
        ),
        predictive_executed_choice_compatibility=(
            predictive_executed_choice_compatibility
        ),
        predictive_best_alternative_choice_compatibility=(
            predictive_best_alternative_choice_compatibility
        ),
        predictive_executed_beta=predictive_executed_beta,
        filtered_executed_beta=filtered_executed_beta,
        filtered_execution_switch_event_probability=(
            filtered_execution_switch_event_probability
        ),
        filtered_execution_dwell_trials=filtered_execution_dwell_trials,
        audit_hypothesis_map=audit_hypothesis_map,
        audit_adaptive_sharpening=audit_adaptive_sharpening,
        audit_exploration_lapse=audit_exploration_lapse,
        audit_unsharpened_expectation=audit_unsharpened_expectation,
        audit_sharpened_no_lapse=audit_sharpened_no_lapse,
        audit_strategy_confidence_no_lapse=(
            audit_strategy_confidence_no_lapse
        ),
        audit_persistent_execution_no_lapse=(
            audit_persistent_execution_no_lapse
        ),
        audit_correct_predicting_available_probability=(
            audit_correct_predicting_available_probability
        ),
        audit_correct_predicting_prior_mass=(
            audit_correct_predicting_prior_mass
        ),
        audit_best_active_correct_probability=(
            audit_best_active_correct_probability
        ),
        audit_particle_correct_q10=audit_particle_correct_q10,
        audit_particle_correct_q50=audit_particle_correct_q50,
        audit_particle_correct_q90=audit_particle_correct_q90,
        audit_ancestral_paths=audit_ancestral_paths,
        final_weights=weights.copy(),
        particle_swap_counts=particle_swap_counts.copy(),
        resampling_log=resampling_log,
        particle_count=n_particles,
        resample_threshold_fraction=threshold_fraction,
        filter_seed=int(filter_seed),
    )


__all__ = [
    "ParticleFilterResult",
    "effective_sample_size",
    "run_state_model_particle_filter",
    "systematic_resample",
    "_trace_ancestral_indices",
]

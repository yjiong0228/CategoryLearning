"""Validate separately versioned Model 0923 scientific contracts."""

from __future__ import annotations

import math
from typing import Any, Mapping

from ..hypothesis_space.spaces.structural_0923 import AXIS_PAIR_OVERLAP_EXTENSION, RELATIONAL_EXTENSION
from ..hypothesis_space.spaces.continuous import DEFAULT_PAIRWISE_SIMILARITY_TOLERANCE


TRANSITION_CLASS = (
    "src.Bayesian_state.model.modules.hypothesis_transition.unified_rule_search."
    "UnifiedRuleSearchHypothesisTransitionModule"
)
MECHANISTIC_TRANSITION_CLASS = (
    "src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search."
    "MechanisticRuleSearchHypothesisTransitionModule"
)
MEMORY_CLASS = "src.Bayesian_state.model.modules.memory.DualMemoryModule"
PAIRING_MEMORY_CLASS = (
    "src.Bayesian_state.model.modules.pairing_memory.HierarchicalPairingMemoryModule"
)

# Saved configurations keep their original provenance. New runs declare full.
FULL_MODEL_VARIANTS = frozenset({"full", "M1"})
REVISED_FULL_VERSION = "0923-full-v3"
FULL_MODEL_VERSIONS = frozenset({"0923-full-v1", "0923-M1-v1", "0923-full-v2", REVISED_FULL_VERSION})
RELATIONAL_CATALOGUE_VERSIONS = frozenset({"0923-B0-v3", "0923-full-v2", REVISED_FULL_VERSION})
SIMPLE_MODEL_VARIANTS = frozenset({"fixed_rule", "random_rule"})
SIMPLE_MODEL_VERSION = "0923-simple-v1"
SIMPLE_TRANSITION_CLASS = (
    "src.Bayesian_state.model.modules.hypothesis_transition.simple_rule."
    "SimpleRuleHypothesisTransitionModule"
)


def is_model_0923_full(provenance: Mapping[str, Any]) -> bool:
    """Recognize the full model, including its saved legacy provenance alias."""
    return (provenance.get("model_id") == "model_0923"
            and provenance.get("variant") in FULL_MODEL_VARIANTS)


def is_model_0923_revised(provenance: Mapping[str, Any]) -> bool:
    """Identify the 2026-10-12 contract without changing saved full-model runs."""
    return ((is_model_0923_full(provenance)
             and provenance.get("specification_version") == REVISED_FULL_VERSION)
            or (provenance.get("model_id") == "model_0923"
                and provenance.get("variant") in SIMPLE_MODEL_VARIANTS
                and provenance.get("specification_version") == SIMPLE_MODEL_VERSION))


def validate_model_0923_config(config: Mapping[str, Any]) -> None:
    """Validate the declared full-model or baseline scientific contract."""
    provenance = config.get("provenance") or {}
    simple = provenance.get("variant") in SIMPLE_MODEL_VARIANTS
    if provenance.get("model_id") != "model_0923" or provenance.get("variant") not in {"B0", *FULL_MODEL_VARIANTS, *SIMPLE_MODEL_VARIANTS}:
        raise ValueError("Model 0923 requires an explicitly declared full, B0 or simple-rule variant")
    mechanistic = is_model_0923_full(provenance) or simple
    revised = is_model_0923_revised(provenance)
    condition = provenance.get("condition")
    if isinstance(condition, bool) or condition not in (1, 2, 3):
        raise ValueError("0923 requires an explicit condition in {1, 2, 3}")
    partition = config.get("partition") or {}
    partition_kwargs = partition.get("kwargs") or {}
    if partition_kwargs.get("n_cats") != (2 if condition == 1 else 4):
        raise ValueError("0923 category count disagrees with its condition")
    # Version the representation separately from the learning equations.
    # Saved 116/128-rule configurations retain their original catalogues.
    version = provenance.get("specification_version", "0923-B0-v1")
    versions = ({SIMPLE_MODEL_VERSION} if simple else FULL_MODEL_VERSIONS if mechanistic
                else {"0923-B0-v1", "0923-B0-v2", "0923-B0-v3"})
    if version not in versions:
        raise ValueError("unsupported 0923 specification_version")
    adopted = version != "0923-B0-v1"
    extension = None if condition == 1 or not adopted else (
        RELATIONAL_EXTENSION if version in RELATIONAL_CATALOGUE_VERSIONS or simple else AXIS_PAIR_OVERLAP_EXTENSION
    )
    if partition_kwargs.get("structural_extension") != extension:
        raise ValueError("0923 structural_extension disagrees with its specification_version/condition")
    if adopted:
        if (partition.get("class") != "src.Bayesian_state.hypothesis_space.observation_model.continuous_partition.ContinuousPartition"
                or partition_kwargs.get("n_dims") != 4
                or partition_kwargs.get("label_permutation_policy", "identity_only") != "identity_only"):
            raise ValueError("0923 requires the shared four-dimensional partition with fixed labels")
    modules = config.get("modules") or {}
    if set(modules) != {"perception_mod", "hypo_transitions_mod", "memory_mod", "beta_mod"}:
        raise ValueError("0923 requires exactly perception, transition, memory and beta modules")
    if config.get("agenda") != ["perception_mod", "hypo_transitions_mod", "memory_mod", "beta_mod"]:
        raise ValueError("0923 requires its documented trial agenda")
    expected_transition = (SIMPLE_TRANSITION_CLASS if simple else
                           MECHANISTIC_TRANSITION_CLASS if mechanistic else TRANSITION_CLASS)
    if modules["hypo_transitions_mod"].get("class") != expected_transition:
        raise ValueError("0923 transition class disagrees with its declared variant")
    beta = modules["beta_mod"]
    beta_kwargs = beta.get("kwargs") or {}
    if beta.get("class") != "src.Bayesian_state.model.modules.beta.BetaModule":
        raise ValueError("0923 requires the shared BetaModule")
    if not mechanistic and (beta_kwargs.get("increase_rate") != 0.0
        or beta_kwargs.get("decrease_rate") != 0.0
        or beta_kwargs.get("use_prior_scaling") is not False
        or "correct_additive" in beta_kwargs):
        raise ValueError("0923 B0 requires fixed rule precision without prior scaling")
    if mechanistic:
        if (beta_kwargs.get("use_prior_scaling") is not False
                or beta_kwargs.get("update_scope") != "active_hypotheses"
                or "correct_additive" in beta_kwargs):
            raise ValueError("0923 full requires unscaled newcomer precision and active feedback updates")
        transition = modules["hypo_transitions_mod"].get("kwargs") or {}
        if simple:
            allowed = {"capacity", "m", "g", "prior_assignment", "module_seed"}
            if (set(transition)-allowed or isinstance(transition.get("capacity"), bool)
                    or transition.get("capacity") != 1
                    or transition.get("g") != 1.0
                    or transition.get("prior_assignment") != {"method": "pairwise_mass_transfer"}):
                raise ValueError("0923 simple variants require one rule, global replacement and pairwise transfer")
            rate = transition.get("m")
            if (isinstance(rate, bool) or not isinstance(rate, (int, float))
                    or not math.isfinite(rate) or not 0 <= rate <= 1
                    or (provenance["variant"] == "fixed_rule" and rate != 0)):
                raise ValueError("0923 simple replacement probability must be in [0, 1], or zero for fixed_rule")
        else:
            expected_mechanism = REVISED_FULL_VERSION if revised else "legacy"
            if transition.get("mechanism_version", "legacy") != expected_mechanism:
                raise ValueError("0923 transition mechanism_version disagrees with its specification_version")
            similarity = provenance.get("hypothesis_similarity") or {}
            if (similarity.get("sample_count") != transition.get("relation_samples", 100_000)
                    or similarity.get("computation_seed") != transition.get("relation_seed", 0)
                    or similarity.get("version") != "confusion_tensor_v1"):
                raise ValueError("0923 full relation sampling disagrees with its provenance")
    memory = modules["memory_mod"]
    expected_memory = PAIRING_MEMORY_CLASS if condition == 3 else MEMORY_CLASS
    memory_kwargs = memory.get("kwargs") or {}
    if (memory.get("class") != expected_memory
        or memory_kwargs.get("w0") != 0.0
        or memory_kwargs.get("feedback_gain") != 1.0):
        raise ValueError("0923 requires one fading evidence memory and zero static weight")
    gamma = float(memory_kwargs.get("gamma", float("nan")))
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("0923 gamma must be finite and in [0, 1]")
    if condition == 3:
        expected_forgetting = "factorized" if revised else "joint"
        if memory_kwargs.get("forgetting_mode", "joint") != expected_forgetting:
            raise ValueError("0923 condition-3 forgetting_mode disagrees with its specification_version")
    else:
        expected_forgetting = "exact_power" if revised else "dual_trace"
        if memory_kwargs.get("forgetting_mode", "dual_trace") != expected_forgetting:
            raise ValueError("0923 binary forgetting_mode disagrees with its specification_version")
    likelihood = config.get("likelihood") or {}
    feedback_mode = likelihood.get("feedback_likelihood_mode", "category_feedback")
    if feedback_mode != ("hierarchical_pairing" if condition == 3 else "category_feedback"):
        raise ValueError("0923 feedback likelihood disagrees with its condition")
    if likelihood.get("beta_source") != "action":
        raise ValueError("0923 uses the same rule precision for choice and evidence")
    if revised:
        if (beta_kwargs.get("beta_update_mode") != "feedback_gradient"
                or any(key in beta_kwargs for key in ("increase_rate", "decrease_rate", "correct_additive"))):
            raise ValueError("0923 full-v3 requires feedback_gradient with one learning_rate")
        rate = beta_kwargs.get("learning_rate")
        if (isinstance(rate, bool) or not isinstance(rate, (int, float))
                or not math.isfinite(rate) or not 0.0 <= rate <= 1.0):
            raise ValueError("0923 full-v3 requires an explicit finite learning_rate in [0, 1]")
        if (beta_kwargs.get("beta_min") != 0.1 or beta_kwargs.get("beta_max") != 25.0
                or beta_kwargs.get("probabilistic_feedback_lapse", 0.0) != 0.0
                or beta_kwargs.get("feedback_lapse", 0.0) != 0.0
                or likelihood.get("distance_mode") != "boundary"):
            raise ValueError("0923 full-v3 requires boundary distances, beta bounds [0.1, 25] and zero feedback lapse")
    elif beta_kwargs.get("beta_update_mode") == "feedback_gradient" or "learning_rate" in beta_kwargs:
        raise ValueError("0923 feedback_gradient requires specification_version='0923-full-v3'")
    readout = (config.get("choice_readout") or {}).get("kwargs") or {}
    if readout != {"method": "expectation", "power": 1.0, "strategy_confidence_gain": 0.0}:
        raise ValueError("0923 requires its documented choice readout settings")
    noise = (config.get("output_noise") or {}).get("kwargs") or {}
    if noise != {"enabled": False}:
        raise ValueError("0923 requires disabled additional output noise")


def validate_model_0923_partition(config: Mapping[str, Any], partition: Any) -> None:
    """Reject an injected partition that bypasses the declared catalogue."""
    provenance = config.get("provenance") or {}
    version = provenance.get("specification_version")
    if version not in {"0923-B0-v2", *RELATIONAL_CATALOGUE_VERSIONS, *FULL_MODEL_VERSIONS, SIMPLE_MODEL_VERSION}:
        return
    binary = provenance["condition"] == 1
    relational = version in {*RELATIONAL_CATALOGUE_VERSIONS, SIMPLE_MODEL_VERSION} and not binary
    expected_extension = None if binary else (RELATIONAL_EXTENSION if relational else AXIS_PAIR_OVERLAP_EXTENSION)
    if (getattr(partition, "structural_extension", None) != expected_extension
            or partition.length != (29 if binary else 152 if relational else 128)):
        raise ValueError("0923 partition does not match its adopted catalogue")
    if relational:
        kwargs = (config.get("partition") or {}).get("kwargs") or {}
        expected = float(kwargs.get("pairwise_similarity_tolerance", DEFAULT_PAIRWISE_SIMILARITY_TOLERANCE))
        actual = partition.hypothesis_space.parameters["pairwise_similarity_tolerance"]
        if actual != expected:
            raise ValueError("0923 injected partition disagrees with its configured similarity tolerance")

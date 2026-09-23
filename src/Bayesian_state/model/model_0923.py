"""Validate the explicitly limited Model 0923 B0 scientific contract."""

from __future__ import annotations

import math
from typing import Any, Mapping

from ..hypothesis_space.spaces.structural_0923 import AXIS_PAIR_OVERLAP_EXTENSION


TRANSITION_CLASS = (
    "src.Bayesian_state.model.modules.hypothesis_transition.unified_rule_search."
    "UnifiedRuleSearchHypothesisTransitionModule"
)
MEMORY_CLASS = "src.Bayesian_state.model.modules.memory.DualMemoryModule"
PAIRING_MEMORY_CLASS = (
    "src.Bayesian_state.model.modules.pairing_memory.HierarchicalPairingMemoryModule"
)


def validate_model_0923_config(config: Mapping[str, Any]) -> None:
    """Reject hidden extra mechanisms under the B0 model name.

    Future extensions must receive their own declared variant and contract;
    an unrecognized variant is not silently treated as the minimal model.
    """
    provenance = config.get("provenance") or {}
    if provenance.get("model_id") != "model_0923" or provenance.get("variant") != "B0":
        raise ValueError("Model 0923 currently implements only variant B0")
    condition = provenance.get("condition")
    if isinstance(condition, bool) or condition not in (1, 2, 3):
        raise ValueError("0923 requires an explicit condition in {1, 2, 3}")
    partition = config.get("partition") or {}
    partition_kwargs = partition.get("kwargs") or {}
    if partition_kwargs.get("n_cats") != (2 if condition == 1 else 4):
        raise ValueError("0923 category count disagrees with its condition")
    # Keep saved v1 configurations executable; v2 explicitly adopts the 12
    # feature-overlap rules. The catalogue change is not a new search mechanism.
    version = provenance.get("specification_version", "0923-B0-v1")
    if version not in {"0923-B0-v1", "0923-B0-v2"}:
        raise ValueError("unsupported 0923 specification_version")
    extension = AXIS_PAIR_OVERLAP_EXTENSION if version == "0923-B0-v2" and condition != 1 else None
    if partition_kwargs.get("structural_extension") != extension:
        raise ValueError("0923 structural_extension disagrees with its specification_version/condition")
    if version == "0923-B0-v2":
        if (partition.get("class") != "src.Bayesian_state.hypothesis_space.observation_model.continuous_partition.ContinuousPartition"
                or partition_kwargs.get("n_dims") != 4
                or partition_kwargs.get("label_permutation_policy", "identity_only") != "identity_only"):
            raise ValueError("0923 B0 v2 requires the shared four-dimensional partition with fixed labels")
    modules = config.get("modules") or {}
    if set(modules) != {"perception_mod", "hypo_transitions_mod", "memory_mod", "beta_mod"}:
        raise ValueError("0923 B0 requires exactly perception, transition, memory and beta modules")
    if config.get("agenda") != ["perception_mod", "hypo_transitions_mod", "memory_mod", "beta_mod"]:
        raise ValueError("0923 B0 requires its documented trial agenda")
    if modules["hypo_transitions_mod"].get("class") != TRANSITION_CLASS:
        raise ValueError("0923 B0 requires the unified single-candidate transition")
    beta = modules["beta_mod"]
    beta_kwargs = beta.get("kwargs") or {}
    if beta.get("class") != "src.Bayesian_state.model.modules.beta.BetaModule":
        raise ValueError("0923 B0 requires the shared BetaModule")
    if (beta_kwargs.get("increase_rate") != 0.0
        or beta_kwargs.get("decrease_rate") != 0.0
        or beta_kwargs.get("use_prior_scaling") is not False
        or "correct_additive" in beta_kwargs):
        raise ValueError("0923 B0 requires fixed rule precision without prior scaling")
    memory = modules["memory_mod"]
    expected_memory = PAIRING_MEMORY_CLASS if condition == 3 else MEMORY_CLASS
    memory_kwargs = memory.get("kwargs") or {}
    if (memory.get("class") != expected_memory
        or memory_kwargs.get("w0") != 0.0
        or memory_kwargs.get("feedback_gain") != 1.0):
        raise ValueError("0923 B0 requires one fading evidence memory and zero static weight")
    gamma = float(memory_kwargs.get("gamma", float("nan")))
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("0923 gamma must be finite and in [0, 1]")
    likelihood = config.get("likelihood") or {}
    feedback_mode = likelihood.get("feedback_likelihood_mode", "category_feedback")
    if feedback_mode != ("hierarchical_pairing" if condition == 3 else "category_feedback"):
        raise ValueError("0923 feedback likelihood disagrees with its condition")
    if likelihood.get("beta_source") != "action":
        raise ValueError("0923 B0 uses the same fixed precision for choice and evidence")
    readout = (config.get("choice_readout") or {}).get("kwargs") or {}
    if readout != {"method": "expectation", "power": 1.0, "strategy_confidence_gain": 0.0}:
        raise ValueError("0923 B0 requires the belief-mixture choice readout")
    noise = (config.get("output_noise") or {}).get("kwargs") or {}
    if noise != {"enabled": False}:
        raise ValueError("0923 B0 has no additional output-noise mechanism")


def validate_model_0923_partition(config: Mapping[str, Any], partition: Any) -> None:
    """Reject an injected partition that silently bypasses the v2 catalogue."""
    provenance = config.get("provenance") or {}
    if provenance.get("specification_version") != "0923-B0-v2":
        return
    binary = provenance["condition"] == 1
    expected_extension = None if binary else AXIS_PAIR_OVERLAP_EXTENSION
    if (getattr(partition, "structural_extension", None) != expected_extension
            or partition.length != (29 if binary else 128)):
        raise ValueError("0923 B0 v2 partition does not match its adopted catalogue")

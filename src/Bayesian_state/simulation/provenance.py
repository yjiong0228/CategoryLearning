"""Compact, machine-independent provenance for scientific model runs."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Mapping

from src.Bayesian_state.hypothesis_space.geometry import BoundaryGeometry
from ..model.model_0923 import is_model_0923_full, is_model_0923_revised


def _mapping_path(mapping: Mapping[str, Any], path: str) -> Any:
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_model_provenance(
    engine_config: Mapping[str, Any],
    *,
    repeat_aggregation: str,
) -> dict[str, Any]:
    """Describe the resolved model structure and its initialization policy."""

    config = deepcopy(dict(engine_config))
    transition = _mapping_path(
        config,
        "modules.hypo_transitions_mod.kwargs",
    )
    transition = dict(transition) if isinstance(transition, Mapping) else {}
    init_hypotheses = transition.get("init_hypotheses")
    initialization = {
        "method": (
            "fixed_indices"
            if init_hypotheses is not None
            else "prior_weighted_without_replacement"
        ),
        "hypothesis_pool": (
            "declared_indices" if init_hypotheses is not None else "full_space"
        ),
        "fixed_indices": (
            None
            if init_hypotheses is None
            else [int(value) for value in init_hypotheses]
        ),
        "integrated_by_particle_filter": bool(
            _mapping_path(config, "inference.backend") == "particle_filter"
            and init_hypotheses is None
        ),
        "seed_policy": "deterministic_per_filter_seed_and_particle",
    }
    declared = config.get("provenance", {})
    if declared is not None and not isinstance(declared, Mapping):
        raise ValueError("engine_config.provenance must be a mapping when provided.")
    declared = dict(declared or {})
    mechanistic = is_model_0923_full(declared)
    if mechanistic:
        initialization.update(method="single_prior_weighted_candidate", active_count=1)
    declared_similarity = declared.get("hypothesis_similarity", {})
    if declared_similarity is not None and not isinstance(
        declared_similarity, Mapping
    ):
        raise ValueError(
            "engine_config.provenance.hypothesis_similarity must be a mapping "
            "when provided."
        )

    resolved = {
        "partition": deepcopy(config.get("partition", {})),
        "likelihood": deepcopy(config.get("likelihood", {})),
        "inference": deepcopy(config.get("inference", {})),
        "workspace_capacity": transition.get("capacity"),
        "initialization": initialization,
        "action_beta": deepcopy(
            _mapping_path(config, "modules.beta_mod.kwargs") or {}
        ),
        "choice_readout": deepcopy(config.get("choice_readout", {})),
        "repeat_aggregation": str(repeat_aggregation),
    }
    if mechanistic:
        resolved.update(
            workspace_capacity=transition.get("max_capacity"),
            workspace_capacity_semantics="upper_bound_variable_occupancy",
            search_control=deepcopy(transition.get("search_controller")),
            candidate_management={name: deepcopy(transition.get(name)) for name in (
                "exit_rate", "local_structure_weight", "transfer_scale",
                "execution_inertia", "edit_costs")},
            transition_event_semantics={
                "search_probability": "search_initiation",
                "swap_probability": "search_probability_times_conditional_admission_probability",
                "swap_event": "newcomer_admission",
                "replacement_count": "capacity_competition_replacements_only",
                "replacement_fraction": "capacity_replacements_divided_by_previous_active_count",
            },
        )
    if is_model_0923_revised(declared):
        resolved["evidence_memory"] = deepcopy(_mapping_path(config, "modules.memory_mod.kwargs") or {})
        resolved["candidate_management"].update(
            mechanism_version=transition.get("mechanism_version"),
            exit_policy="all_candidates_relative_belief_with_nonempty_fallback",
            capacity_policy="remove_lowest_belief_before_single_transfer",
            transfer_policy="survivor_mass_split_by_one_to_affinity",
        )
        resolved["choice_readout"].update(
            rule_weights="belief_times_previous_dominant_inertia_normalized",
            dominant_rule="maximum_weight_previous_dominant_tie_preference",
        )
        resolved["transition_event_semantics"].update(
            swap_probability="search_probability",
            replacement_count="lowest_belief_capacity_replacements_only",
        )
        intervention = transition.get("intervention", {"kind": "B"})
        kind = intervention.get("kind", "B")
        if kind != "B":
            resolved["intervention"] = deepcopy(intervention)
            resolved["scientific_status"] = "fixed_parameter_intervention_not_refitted"
            if kind == "S1":
                resolved["search_control"]["event_signal"] = "failure_accumulates_success_multiplies_by_decay"
                resolved["search_control"]["range_signal"] = "original_bounded_failure_average"
            elif kind == "S_control":
                resolved["search_control"]["odds_multiplier"] = intervention["factor"]
            elif kind == "R_control":
                resolved["candidate_management"]["exit_rate_multiplier"] = intervention["factor"]
            elif kind == "R1":
                resolved["candidate_management"]["exit_policy"] = "relative_belief_gated_by_negative_recent_log_feedback_ratio"
            elif kind == "E":
                resolved['candidate_management'].update(
                    exit_policy='relative_belief_gated_by_negative_discounted_log_feedback_ratio',
                    evidence_decay=intervention['decay'], capacity_policy=intervention['capacity'])
            elif kind == "J":
                resolved['candidate_management'].update(exit_policy='joint_nonempty_capacity_constrained_subset',
                    capacity_policy='same_subset_distribution', evidence_decay=intervention['decay'],
                    retention_cost=intervention['cost'], refresh=intervention['refresh'])
                resolved['transition_event_semantics'].update(
                    swap_probability='search_probability_times_newcomer_inclusion_marginal',
                    replacement_count='legacy_capacity_replacements_zero_joint_dropped_separate')
            elif kind == "C1":
                resolved["candidate_management"]["capacity_policy"] = "compare_all_temporary_weights_then_single_final_transfer"
                resolved["transition_event_semantics"]["swap_probability"] = "search_probability_times_conditional_admission_probability"
                resolved["transition_event_semantics"]["replacement_count"] = "old_rule_loses_joint_temporary_weight_competition"
    partition_kwargs = _mapping_path(config, "partition.kwargs") or {}
    partition_class = str(_mapping_path(config, "partition.class") or "")
    is_continuous = partition_class.endswith("ContinuousPartition")
    likelihood = config.get("likelihood", {})
    # The boundary solver is fixed: the exact active-set enumeration.  These
    # fields stay in the encoding record so previously generated provenance and
    # downstream readers keep the same keys.
    boundary_solver = BoundaryGeometry.SOLVER_NAME if is_continuous else None
    resolved["encoding"] = {
        "distance_mode": _mapping_path(config, "likelihood.distance_mode"),
        "boundary_distance_method": partition_kwargs.get(
            "boundary_distance_method", "dykstra_iterative_projection"
        ) if is_continuous else None,
        "boundary_distance_tolerance": partition_kwargs.get(
            "boundary_distance_tolerance", 1.0e-9
        ) if is_continuous else None,
        "boundary_projection_iterations": partition_kwargs.get(
            "boundary_projection_iterations", 100
        ) if is_continuous else None,
        "boundary_solver": boundary_solver,
        "label_permutation_policy": partition_kwargs.get(
            "label_permutation_policy", "identity_only"
        ) if is_continuous else None,
        "feedback_likelihood_mode": (
            likelihood.get("feedback_likelihood_mode")
            if isinstance(likelihood, Mapping)
            else None
        ),
    }
    tau_local = transition.get("tau_local")
    if declared_similarity or tau_local is not None:
        resolved["hypothesis_similarity"] = {
            **deepcopy(dict(declared_similarity or {})),
            "tau_local": tau_local,
        }
        if mechanistic:
            resolved["hypothesis_similarity"].pop("tau_local")
            resolved["hypothesis_similarity"].update(
                tau_proposal=transition.get("tau_proposal"),
                transfer_scale=transition.get("transfer_scale"))
    return {
        "schema_version": 2,
        "model_config_sha256": _canonical_sha256(config),
        "declared": deepcopy(declared),
        "resolved": resolved,
    }


__all__ = ["build_model_provenance"]

"""Compact, machine-independent provenance for scientific model runs."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Mapping


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
    tau_local = transition.get("tau_local")
    if declared_similarity or tau_local is not None:
        resolved["hypothesis_similarity"] = {
            **deepcopy(dict(declared_similarity or {})),
            "tau_local": tau_local,
        }
    return {
        "schema_version": 1,
        "model_config_sha256": _canonical_sha256(config),
        "declared": deepcopy(declared),
        "resolved": resolved,
    }


__all__ = ["build_model_provenance"]

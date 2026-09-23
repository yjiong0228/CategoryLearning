"""Build a configured :class:`StateModel` engine and its cognitive modules.

Configuration parsing and dynamic imports belong here. The inference engine
receives concrete objects and therefore remains independent of YAML and Python
class-path conventions.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from ..hypothesis_space import ContinuousPartition, ObservationLikelihood
from ..utils.logging import LOGGER
from .config import ModelContext
from .engine import BayesianStateEngine, IndexedSet
from .modules.base_module import ModuleRole
from .modules.pairing_memory import HierarchicalPairingMemoryModule
from .modules.hypothesis_transition.unified_rule_search import (
    UnifiedRuleSearchHypothesisTransitionModule,
)


def resolve_class(class_reference: str | type) -> type:
    """Resolve a class object or an absolute dotted class path."""

    if isinstance(class_reference, type):
        return class_reference
    if not isinstance(class_reference, str) or "." not in class_reference:
        raise TypeError("class reference must be a class or an absolute dotted path.")
    module_path, class_name = class_reference.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        resolved = getattr(module, class_name)
    except (ImportError, AttributeError, ValueError) as error:
        raise ImportError(f"cannot import class {class_reference!r}.") from error
    if not isinstance(resolved, type):
        raise TypeError(f"configured object {class_reference!r} is not a class.")
    return resolved


def build_partition(engine_config: Mapping[str, Any], condition: int) -> Any:
    """Build the configured observation partition."""

    partition_config = engine_config.get("partition")
    if partition_config is None:
        dimensions = int(engine_config.get("n_dims", 4))
        categories = int(engine_config.get("n_cats", 2 if condition == 1 else 4))
        return ContinuousPartition(dimensions, categories)
    if not isinstance(partition_config, Mapping):
        raise ValueError("engine_config.partition must be a mapping when provided.")
    class_reference = partition_config.get("class")
    if class_reference is None:
        raise ValueError("engine_config.partition must include a class path.")
    partition_class = resolve_class(class_reference)
    return partition_class(**dict(partition_config.get("kwargs", {}) or {}))


def build_observation_likelihood(
    engine_config: Mapping[str, Any],
    partition: Any,
) -> ObservationLikelihood:
    """Build the mandatory observation-likelihood evaluator."""

    likelihood_config = engine_config.get("likelihood", {})
    if likelihood_config is None:
        likelihood_config = {}
    if not isinstance(likelihood_config, Mapping):
        raise ValueError("engine_config.likelihood must be a mapping when provided.")
    partition_config = engine_config.get("partition", {})
    partition_kwargs = (
        partition_config.get("kwargs", {})
        if isinstance(partition_config, Mapping)
        else {}
    )
    if (
        isinstance(partition, ContinuousPartition)
        and likelihood_config.get("distance_mode") == "prototype"
        and isinstance(partition_kwargs, Mapping)
    ):
        boundary_only = {
            "boundary_distance_method",
            "boundary_distance_tolerance",
            "boundary_projection_iterations",
            "boundary_dykstra_backend",
        }.intersection(partition_kwargs)
        if boundary_only:
            raise ValueError(
                "Prototype encoding must not configure boundary-only partition "
                f"parameters: {sorted(boundary_only)}."
            )
    return ObservationLikelihood(partition, **dict(likelihood_config))


def build_engine(
    engine_config: Mapping[str, Any],
    *,
    hypotheses_set: IndexedSet,
    partition: Any,
    observation_likelihood: ObservationLikelihood,
    context: ModelContext,
) -> BayesianStateEngine:
    """Construct an engine and instantiate configured cognitive modules."""

    provenance = engine_config.get("provenance") or {}
    model_0923 = provenance.get("model_id") == "model_0923"
    if model_0923:
        from .model_0923 import validate_model_0923_config

        # Recheck after StateModel module overrides and seed injection.
        validate_model_0923_config(engine_config)
        if provenance.get("condition") != context.condition:
            raise ValueError("0923 declared condition disagrees with the trial context")
    agenda = engine_config.get("agenda", [])
    if not isinstance(agenda, list):
        raise ValueError("engine_config.agenda must be a list.")
    module_configs = engine_config.get("modules", {})
    if not isinstance(module_configs, Mapping):
        raise ValueError("engine_config.modules must be a mapping.")

    engine = BayesianStateEngine(
        agenda=agenda,
        hypotheses_set=hypotheses_set,
        partition=partition,
        observation_likelihood=observation_likelihood,
        context=context,
    )

    for name, raw_config in module_configs.items():
        if not isinstance(raw_config, Mapping):
            raise ValueError(f"module configuration {name!r} must be a mapping.")
        if "class" not in raw_config:
            raise ValueError(f"module configuration {name!r} must include class.")
        module_class = resolve_class(raw_config["class"])
        module_kwargs = dict(raw_config.get("kwargs", {}) or {})
        LOGGER.debug("Building module %s with kwargs=%s", name, module_kwargs)
        engine.register_module(name, module_class(engine=engine, **module_kwargs))
    engine.validate_agenda()
    memory = engine.get_module(ModuleRole.MEMORY)
    hierarchical = observation_likelihood.feedback_likelihood_mode == "hierarchical_pairing"
    if context.condition == 3 or hierarchical or isinstance(memory, HierarchicalPairingMemoryModule):
        if context.condition != 3 or not hierarchical or not isinstance(memory, HierarchicalPairingMemoryModule):
            raise ValueError("condition 3 requires hierarchical_pairing likelihood and HierarchicalPairingMemoryModule together.")
        beta = engine.get_module(ModuleRole.BETA)
        transition = engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        # The graded score trace is an explicit 0923 change. All existing
        # condition-3 profiles retain the full-success search contract.
        search_feedback = (
            "graded" if model_0923 and isinstance(
                transition, UnifiedRuleSearchHypothesisTransitionModule
            ) else "full_success"
        )
        if (observation_likelihood.beta_source != "action"
                or observation_likelihood.distance_mode != "boundary"
                or getattr(beta, "beta_update_mode", None) != "hierarchical_feedback"
                or getattr(transition, "feedback_interpretation", None) != search_feedback
                or not hasattr(transition, "prior_assignment_method")
                or engine.get_module(ModuleRole.MAPPING) is not None):
            raise ValueError(f"condition 3 requires boundary/action emissions, hierarchical_feedback beta, {search_feedback} workspace search and fixed response coordinates.")
        names = {module.role: name for name, module in engine.modules.items()}
        if engine.agenda.index(names[ModuleRole.MEMORY]) > engine.agenda.index(names[ModuleRole.BETA]):
            raise ValueError("condition 3 memory must precede beta in the post-choice agenda.")
        memory.initialize_prior()
    return engine


__all__ = [
    "build_engine",
    "build_observation_likelihood",
    "build_partition",
    "resolve_class",
]

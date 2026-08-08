"""Deterministic seed derivation shared across pipeline layers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import numpy as np


SEED_MODULUS = 2**32


def _seedable(obj: Any) -> Any:
    """Convert common Python, NumPy, and path values to stable JSON payloads."""
    if isinstance(obj, np.ndarray):
        return [_seedable(value) for value in obj.tolist()]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, Mapping):
        return {
            str(key): _seedable(value)
            for key, value in sorted(obj.items(), key=lambda item: str(item[0]))
        }
    if isinstance(obj, (list, tuple)):
        return [_seedable(value) for value in obj]
    return obj


def stable_seed(payload: Any) -> int:
    """Derive a deterministic uint32 seed from a JSON-serializable payload."""
    encoded = json.dumps(
        _seedable(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % SEED_MODULUS


def derive_hyper_candidate_seed(
    hyper_base_seed: int,
    stage: str,
    combination_index: int,
    hyperparams: Mapping[str, Any],
    extra_context: Mapping[str, Any] | None = None,
) -> int:
    payload: dict[str, Any] = {
        "seed_role": "hyper_candidate_seed",
        "hyper_base_seed": int(hyper_base_seed),
        "stage": str(stage),
        "combination_index": int(combination_index),
        "hyperparams": dict(hyperparams),
    }
    if extra_context:
        payload["extra_context"] = dict(extra_context)
    return stable_seed(payload)


def derive_simulation_point_seed(
    hyper_candidate_seed: int,
    subject_id: int,
    params: Mapping[str, Any],
) -> int:
    return stable_seed(
        {
            "seed_role": "simulation_point_seed",
            "hyper_candidate_seed": int(hyper_candidate_seed),
            "subject_id": int(subject_id),
            "params": dict(params),
        }
    )


def derive_trajectory_seed(
    simulation_point_seed: int,
    phase: str,
    repeat_index: int,
) -> int:
    return stable_seed(
        {
            "seed_role": "trajectory_seed",
            "simulation_point_seed": int(simulation_point_seed),
            "phase": str(phase),
            "repeat_index": int(repeat_index),
        }
    )


def derive_module_seed(
    trajectory_seed: int,
    module_name: str = "hypo_transitions_mod",
) -> int:
    return stable_seed(
        {
            "seed_role": "module_seed",
            "trajectory_seed": int(trajectory_seed),
            "module_name": str(module_name),
        }
    )


def inject_module_seed_from_trajectory(
    engine_config: MutableMapping[str, Any],
    trajectory_seed: int | None,
    module_name: str = "hypo_transitions_mod",
) -> int | None:
    """Inject the derived seed into one configured module, if present."""
    if trajectory_seed is None:
        return None
    module_seed = derive_module_seed(int(trajectory_seed), module_name=module_name)
    modules = engine_config.get("modules")
    if not isinstance(modules, MutableMapping) or module_name not in modules:
        return None
    module_cfg = modules[module_name]
    if not isinstance(module_cfg, MutableMapping):
        return None
    kwargs = module_cfg.setdefault("kwargs", {})
    if not isinstance(kwargs, MutableMapping):
        return None
    kwargs["module_seed"] = int(module_seed)
    return int(module_seed)


__all__ = [
    "derive_hyper_candidate_seed",
    "derive_module_seed",
    "derive_simulation_point_seed",
    "derive_trajectory_seed",
    "inject_module_seed_from_trajectory",
    "stable_seed",
]

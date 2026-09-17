"""Configuration and pure utilities for the opt-in Hyper-CD 2.0 workflow."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from numbers import Real
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.Bayesian_state.optimization.artifacts import to_builtin
from src.Bayesian_state.utils.provenance import search_dependencies
from src.Bayesian_state.optimization.objectives import (
    ObjectiveSpec,
    compare_objective_values,
    first_objective_value,
)


def canonical_point_key(point: Mapping[str, Any]) -> str:
    """Return a stable cache key for a complete hyperparameter point."""

    return json.dumps(
        to_builtin(dict(point)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def project_point_to_space(
    point: Mapping[str, Any],
    space: Mapping[str, Sequence[Any]],
) -> dict[str, Any]:
    """Project numeric scalars to declared support and preserve mappings exactly."""

    if set(point) != set(space):
        raise ValueError("point and fine space must contain identical coordinates")

    projected: dict[str, Any] = {}
    for name, raw_value in point.items():
        values = list(space[name])
        if not values:
            raise ValueError(f"fine space coordinate {name!r} cannot be empty")

        if isinstance(raw_value, Mapping):
            target_key = canonical_point_key({"value": raw_value})
            matches = [
                value
                for value in values
                if isinstance(value, Mapping)
                and canonical_point_key({"value": value}) == target_key
            ]
            if not matches:
                raise ValueError(
                    f"mapping-valued coordinate {name!r} requires an exact fine candidate"
                )
            projected[name] = deepcopy(matches[0])
            continue

        if raw_value in values:
            projected[name] = deepcopy(raw_value)
            continue
        if isinstance(raw_value, Real) and not isinstance(raw_value, bool) and all(
            isinstance(value, Real) and not isinstance(value, bool) for value in values
        ):
            projected[name] = deepcopy(
                min(values, key=lambda value: (abs(float(value) - float(raw_value)), float(value)))
            )
            continue
        raise ValueError(
            f"non-numeric coordinate {name!r} requires an exact fine candidate"
        )
    return projected


def candidate_improves(
    current_values: Mapping[str, Any],
    candidate_values: Mapping[str, Any],
    objective_order: Sequence[ObjectiveSpec],
    min_delta: float,
) -> bool:
    """Return whether a candidate is ordered-better by at least ``min_delta``."""

    threshold = float(min_delta)
    if threshold < 0.0:
        raise ValueError("cd.min_delta must be non-negative")
    if compare_objective_values(
        candidate_values,
        current_values,
        objective_order,
    ) >= 0:
        return False
    improvement = first_objective_value(
        current_values, objective_order
    ) - first_objective_value(candidate_values, objective_order)
    rounding_slack = max(1.0, abs(improvement), abs(threshold)) * 1e-15
    return bool(improvement + rounding_slack >= threshold)


def atomic_write_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably replace one JSON checkpoint without exposing a partial file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(
                to_builtin(dict(payload)),
                stream,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load one checkpoint mapping and reject malformed top-level values."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load Hyper-CD checkpoint: {source}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"Hyper-CD checkpoint must contain a mapping: {source}")
    return dict(payload)


def search_context_fingerprint(
    search_config: Mapping[str, Any],
    base_sim_config: Mapping[str, Any],
    subjects: Sequence[int],
    requested_stage: str,
    *,
    config_dir: Path | None = None,
) -> str:
    """Hash every context element that makes cached candidate scores valid."""

    payload = {
        "search_config": to_builtin(dict(search_config)),
        "base_sim_config": to_builtin(dict(base_sim_config)),
        "subjects": [int(subject_id) for subject_id in subjects],
        "requested_stage": str(requested_stage),
        "dependencies": to_builtin(search_dependencies(
            search_config, base_sim_config, subjects,
            Path.cwd() if config_dir is None else Path(config_dir),
        )),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CDV2Config:
    """Validated behavior switches for schema-v2 coordinate descent."""

    enabled: bool
    resume_mode: str
    checkpoint_every_coordinate: bool
    fine_initialization: str

    @classmethod
    def from_search_config(cls, config: Mapping[str, Any]) -> "CDV2Config":
        schema_version = int(config.get("search_schema_version", 1))
        if schema_version not in {1, 2}:
            raise ValueError("search_schema_version must be 1 or 2")
        if schema_version == 1:
            return cls(
                enabled=False,
                resume_mode="legacy",
                checkpoint_every_coordinate=False,
                fine_initialization="legacy",
            )

        cd = config.get("cd")
        if not isinstance(cd, Mapping):
            raise ValueError("schema-v2 Hyper-CD requires a cd mapping")
        if "resume_mode" not in cd:
            raise ValueError("schema-v2 Hyper-CD requires cd.resume_mode")
        resume_mode = str(cd["resume_mode"]).strip().lower()
        if resume_mode != "explicit":
            raise ValueError("cd.resume_mode must be 'explicit' for schema-v2 Hyper-CD")

        refine_policy = config.get("refine_policy") or {}
        if not isinstance(refine_policy, Mapping):
            raise ValueError("refine_policy must be a mapping when provided")
        fine_initialization = str(
            refine_policy.get("fine_initialization", "coarse_shortlist")
        ).strip().lower()
        if fine_initialization != "coarse_shortlist":
            raise ValueError(
                "refine_policy.fine_initialization must be 'coarse_shortlist' "
                "for schema-v2 Hyper-CD"
            )
        return cls(
            enabled=True,
            resume_mode=resume_mode,
            checkpoint_every_coordinate=bool(
                cd.get("checkpoint_every_coordinate", True)
            ),
            fine_initialization=fine_initialization,
        )


__all__ = [
    "CDV2Config",
    "atomic_write_checkpoint",
    "candidate_improves",
    "canonical_point_key",
    "load_checkpoint",
    "project_point_to_space",
    "search_context_fingerprint",
]

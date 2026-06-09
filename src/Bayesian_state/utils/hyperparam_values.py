"""Hyperparameter value-source helpers for hyper-grid and hyper-CD."""
from __future__ import annotations

import json
import itertools
from copy import deepcopy
from pathlib import Path
from typing import Any, List, Mapping


def values_from_json(spec: Mapping[str, Any], config_dir: Path) -> List[Any]:
    source = spec.get("values_from_json")
    if not isinstance(source, Mapping):
        raise ValueError("values_from_json must be a mapping with path, key, and value_key.")

    raw_path = source.get("path")
    if not raw_path:
        raise ValueError("values_from_json.path is required.")
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = (config_dir / path).resolve()
    if not path.is_file():
        raise ValueError(f"values_from_json.path does not exist or is not a file: {path}")

    key = source.get("key")
    if not isinstance(key, str) or not key:
        raise ValueError("values_from_json.key must be a non-empty string.")
    value_key = source.get("value_key")
    if not isinstance(value_key, str) or not value_key:
        raise ValueError("values_from_json.value_key must be a non-empty string.")

    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"values_from_json.path is not valid JSON: {path}") from exc

    if not isinstance(payload, Mapping):
        raise ValueError(f"values_from_json JSON root must be a mapping: {path}")
    if key not in payload:
        raise ValueError(f"values_from_json key '{key}' not found in {path}")
    candidates = payload[key]
    if not isinstance(candidates, list):
        raise ValueError(f"values_from_json key '{key}' must contain a list.")
    if not candidates:
        raise ValueError(f"values_from_json key '{key}' contains an empty list.")

    values = []
    for idx, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise ValueError(f"values_from_json candidate #{idx} under '{key}' must be a mapping.")
        if value_key not in candidate:
            raise ValueError(
                f"values_from_json candidate #{idx} under '{key}' is missing value_key '{value_key}'."
            )
        values.append(deepcopy(candidate[value_key]))
    return values


def values_product(spec: Mapping[str, Any]) -> List[Any]:
    source = spec.get("values_product")
    if not isinstance(source, Mapping):
        raise ValueError("values_product must be a non-empty mapping.")
    if not source:
        raise ValueError("values_product must contain at least one factor.")

    names: List[str] = []
    value_lists: List[List[Any]] = []
    for name, raw_values in source.items():
        if not isinstance(name, str) or not name:
            raise ValueError("values_product factor names must be non-empty strings.")
        if not isinstance(raw_values, list):
            raise ValueError(f"values_product.{name} must be a non-empty list.")
        if not raw_values:
            raise ValueError(f"values_product.{name} cannot be empty.")
        names.append(name)
        value_lists.append([deepcopy(value) for value in raw_values])

    values = [dict(zip(names, combo)) for combo in itertools.product(*value_lists)]
    if not values:
        raise ValueError("values_product produced no values.")
    return values


def validate_no_nested_hyperparam_paths(param_specs: Mapping[str, Any]) -> None:
    names = sorted(str(name) for name in param_specs.keys())
    for idx, left in enumerate(names):
        prefix = f"{left}."
        for right in names[idx + 1:]:
            if right.startswith(prefix):
                raise ValueError(
                    "hyperparam_space cannot contain both a parent path and its child path: "
                    f"'{left}' and '{right}'."
                )

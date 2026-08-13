"""Helpers for subject-specific YAML config overrides."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

SUBJECT_OVERRIDE_KEYS = ("subject_overrides", "subject_configs", "per_subject")


def deep_update(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def without_subject_overrides(config: Mapping[str, Any]) -> dict[str, Any]:
    return {k: deepcopy(v) for k, v in config.items() if k not in SUBJECT_OVERRIDE_KEYS}


def _key_matches_subject(key: Any, subject_id: int) -> bool:
    if isinstance(key, int):
        return key == subject_id

    text = str(key).strip()
    if text == str(subject_id):
        return True

    for sep in (",", " "):
        if sep in text:
            parts = [part.strip() for part in text.split(sep) if part.strip()]
            if parts and all(part.lstrip("+-").isdigit() for part in parts):
                return str(subject_id) in parts

    if "-" in text:
        left, right = [part.strip() for part in text.split("-", 1)]
        if left.isdigit() and right.isdigit():
            return int(left) <= subject_id <= int(right)

    return False


def subject_override_for(config: Mapping[str, Any], subject_id: int) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for override_key in SUBJECT_OVERRIDE_KEYS:
        overrides = config.get(override_key)
        if not isinstance(overrides, Mapping):
            continue
        for key, value in overrides.items():
            if _key_matches_subject(key, subject_id):
                if not isinstance(value, Mapping):
                    raise ValueError(
                        f"{override_key}[{key!r}] must be a mapping, got {type(value).__name__}"
                    )
                merged = deep_update(merged, value)
    return merged


def resolve_subject_config(config: Mapping[str, Any], subject_id: int) -> dict[str, Any]:
    base = without_subject_overrides(config)
    override = subject_override_for(config, subject_id)
    return deep_update(base, override)

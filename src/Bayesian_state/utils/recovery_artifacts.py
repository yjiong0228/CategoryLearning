"""Atomic artifacts and fingerprints for recovery workflows."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import pandas as pd
import yaml
from src.Bayesian_state.optimization.artifacts import to_builtin


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Cannot read Model0826 recovery config: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid Model0826 recovery YAML: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Model0826 recovery YAML root must be a mapping")
    return deepcopy(dict(payload))


def _canonical_fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        to_builtin(dict(payload)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
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
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(
                to_builtin(dict(payload)),
                stream,
                sort_keys=False,
                allow_unicode=True,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_immutable_yaml(
    path: Path,
    payload: Mapping[str, Any],
    *,
    resume: bool,
) -> None:
    if path.exists():
        existing = _load_yaml(path)
        if _canonical_fingerprint(existing) != _canonical_fingerprint(payload):
            raise ValueError(f"recovery config fingerprint does not match: {path}")
        if not resume:
            raise FileExistsError(f"recovery config already exists: {path}")
        return
    _atomic_yaml(path, payload)

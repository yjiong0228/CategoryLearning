"""Portable offline handoff reader; copy this file as python1031_io.py.

Only the standard library and NumPy are required. Reading never fits, simulates,
plots, or imports model code. Integrity verification is explicit, not automatic.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys
from typing import Any

import numpy as np


def _local_file(root: Path, relative: str) -> Path:
    """Accept canonical relative POSIX paths, without following symlinks."""
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or ":" in relative or "\0" in relative
            or PurePosixPath(relative).is_absolute()
            or PureWindowsPath(relative).drive
            or any(part in ("", ".", "..") for part in relative.split("/"))):
        raise ValueError(f"Invalid relative path: {relative!r}")
    path = root
    for part in relative.split("/"):
        path = path / part
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise ValueError(f"Symlink/junction path is forbidden: {relative}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes bundle: {relative}")
    return path


def _subject_dir(bundle_root: str | Path, condition: int | str, subject: int | str) -> Path:
    condition_text = str(condition)
    subject_text = str(subject)
    if condition_text.startswith("cond"):
        condition_text = condition_text[4:]
    if subject_text.startswith("subject_"):
        subject_text = subject_text[8:]
    if condition_text not in ("1", "2", "3") or not re.fullmatch(r"[0-9]+", subject_text):
        raise ValueError("condition must be 1/2/3 (or condN); subject must be an integer ID")
    root = Path(bundle_root).absolute()
    return _local_file(root, f"subjects/cond{condition_text}/subject_{subject_text}")


def load_subject(bundle_root: str | Path, condition: int | str, subject: int | str) -> dict[str, Any]:
    """Return metadata fields plus fit/generation dictionaries of saved arrays.

    All arrays in these two summary archives are loaded. Particle contents use
    load_particles separately. No aliases, estimates or metadata are synthesized.
    """
    directory = _subject_dir(bundle_root, condition, subject)
    metadata_path = _local_file(directory, "python1031_metadata.json")
    with metadata_path.open(encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict) or "fit" in result or "generation" in result:
        raise ValueError("Metadata must be an object without reserved fit/generation keys")
    for name in ("fit", "generation"):
        with np.load(_local_file(directory, f"python1031_{name}.npz"), allow_pickle=False) as packet:
            result[name] = {key: packet[key] for key in packet.files}
    return result


class ParticleMapping(Mapping[str, np.ndarray]):
    """Lazy, non-caching NPZ mapping; callers own arrays they request.

    Iteration, length and membership inspect names only. Repeated reads decompress
    again, so retain a requested array locally when reusing it. close() releases
    the archive; use as a context manager for deterministic release.
    """

    def __init__(self, path: Path):
        self._packet = np.load(path, allow_pickle=False)
        self._closed = False
        self._keys: dict[str, str] = {}
        try:
            for raw in self._packet.files:
                name = raw.removeprefix("particle_")
                if name in self._keys:
                    raise ValueError(f"Duplicate particle mapping key: {name}")
                self._keys[name] = raw
            if "template_category_belief" in self._keys:
                alias = "conditional_template_category_belief"
                if alias in self._keys:
                    raise ValueError(f"Duplicate particle mapping key: {alias}")
                self._keys[alias] = self._keys["template_category_belief"]
        except Exception:
            self.close()
            raise

    def __getitem__(self, key: str) -> np.ndarray:
        if self._closed:
            raise ValueError("Particle archive is closed")
        value = self._packet[self._keys[key]]
        if key in ("active", "template_mask"):
            return value.astype(bool, copy=False)
        if key == "template_category_belief":
            if value.ndim != 5:
                raise ValueError("template_category_belief must have shape [T,P,K,G,C]")
            if value.shape[-2] == 1:
                return np.squeeze(value, axis=-2)
        return value

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, key: object) -> bool:
        return key in self._keys

    def close(self) -> None:
        """Release the underlying file handle; safe to call repeatedly."""
        if not self._closed:
            self._packet.close()
            self._closed = True

    def __enter__(self) -> ParticleMapping:
        if self._closed:
            raise ValueError("Particle archive is closed")
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def validate_weights(self) -> None:
        """Explicitly load/check supplied weights; invalid trials are excluded.

        No weights, ESS or missing fields are reconstructed. Without a valid
        mask, every trial is checked. This operation does not read content arrays.
        """
        for key in ("weights", "predictive_weights"):
            if key not in self:
                continue
            weights = self[key]
            if weights.ndim != 2 or weights.shape[1] == 0:
                raise ValueError(f"{key} must have shape [T,P] with P > 0")
            valid = self["valid"].astype(bool) if "valid" in self else np.ones(len(weights), bool)
            if valid.shape != (len(weights),):
                raise ValueError("valid must have shape [T]")
            selected = weights[valid]
            if (not np.isfinite(selected).all() or np.any(selected < 0)
                    or not np.allclose(selected.sum(axis=1), 1., rtol=1e-7, atol=1e-10)):
                raise ValueError(f"{key} must be finite, nonnegative and normalized on valid trials")


def load_particles(bundle_root: str | Path, condition: int | str, subject: int | str) -> ParticleMapping:
    """Open the first-PF-seed packet lazily, preserving observed-input fields."""
    directory = _subject_dir(bundle_root, condition, subject)
    return ParticleMapping(_local_file(directory, "native/conditional_particle_contents.npz"))


def verify_bundle(root: str | Path) -> dict[str, int]:
    """Stream SHA256 and byte-count checks for files listed in manifest.json.

    Raises ValueError for invalid records or mismatches, OSError for missing or
    unreadable files. Unlisted files are outside the integrity inventory. This
    checks integrity against the supplied manifest, not scientific validity or
    authenticity of that manifest.
    """
    root = Path(root).absolute()
    with _local_file(root, "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        raise ValueError("Manifest files must be a list")
    records = []
    seen = set()
    for record in manifest["files"]:
        if not isinstance(record, dict):
            raise ValueError("Manifest file record must be an object")
        relative = record.get("path")
        path = _local_file(root, relative)
        identity = path.resolve()
        if identity in seen:
            raise ValueError(f"Duplicate manifest path: {relative}")
        seen.add(identity)
        digest, size = record.get("sha256"), record.get("bytes")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ValueError(f"Invalid SHA256: {relative}")
        if type(size) is not int or size < 0:
            raise ValueError(f"Invalid bytes: {relative}")
        records.append((path, relative, digest.lower(), size))
    total = 0
    for path, relative, expected, size in records:
        digest = hashlib.sha256()
        count = 0
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                count += len(chunk)
        if count != size:
            raise ValueError(f"Bytes mismatch: {relative}: expected {size}, got {count}")
        if digest.hexdigest() != expected:
            raise ValueError(f"SHA256 mismatch: {relative}")
        total += count
    return dict(files_checked=len(records), bytes_checked=total)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify portable bundle integrity only; no plots or model runs.")
    parser.add_argument("--verify", required=True, metavar="ROOT")
    args = parser.parse_args(argv)
    try:
        report = verify_bundle(args.verify)
    except (OSError, ValueError) as error:
        print(f"Verification failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

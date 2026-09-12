"""Publish new simulation artifacts without replacing existing research outputs."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
import os
from pathlib import Path
import tempfile
from typing import Iterator, Sequence


def _require_absent(path: Path) -> None:
    # A dangling symlink also reserves a name and must not be replaced.
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Simulation output already exists: {path}. Choose a new output directory.")


@contextmanager
def new_artifact_path(path: Path) -> Iterator[Path]:
    """Yield a temporary path, then atomically publish it only if the target is absent.

    The temporary file lives on the same filesystem. Linking, rather than
    replacing, also rejects a destination created by another writer after the
    initial check. Filesystems without hard-link support fail without replacing
    any existing artifact.
    """
    path = Path(path)
    _require_absent(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink()


def _check_subject_artifacts(output: Path, subject_id: int) -> None:
    _require_absent(output / "subjects" / f"subject_{subject_id}.json")
    _require_absent(output / "cache" / f"subject_{subject_id}_raw_runs.gz")


@contextmanager
def reserve_subject_outputs(outputs: Sequence[tuple[Path, int]]) -> Iterator[None]:
    """Check and reserve every selected subject before any model computation.

    Only locks created by this call are released. An interrupted process may
    leave a lock; it is never automatically removed or treated as permission to
    reuse the directory.
    """
    normalized = [(Path(output).resolve(), int(sid)) for output, sid in outputs]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Duplicate subjects target the same simulation output directory")
    for output, sid in normalized:
        _check_subject_artifacts(output, sid)
        _require_absent(output / f".subject_{sid}.lock")
    with ExitStack() as cleanup:
        for output, sid in sorted(normalized):
            output.mkdir(parents=True, exist_ok=True)
            lock = output / f".subject_{sid}.lock"
            with lock.open("x", encoding="utf-8") as stream:
                cleanup.callback(lock.unlink)
                stream.write(f"pid={os.getpid()}\n")
            # Recheck after acquiring the lock to close the preflight race.
            _check_subject_artifacts(output, sid)
        yield

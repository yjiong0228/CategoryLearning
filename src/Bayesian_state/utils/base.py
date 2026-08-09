"""Legacy runtime helpers with import-safe defaults.

Importing this module never creates directories, opens log files, or scans
configuration.  Executable entry points may call :func:`configure_logging`
explicitly when console logging is desired.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .paths import CONFIGS_DIR, LOGS_DIR, ROOT_DIR, SRC_DIR, UTILS_DIR


PATHS = {
    "utils": UTILS_DIR,
    "src": SRC_DIR,
    "root": ROOT_DIR,
    "configs": CONFIGS_DIR,
    "logs": LOGS_DIR,
}

LOGGER = logging.getLogger("cat-learning")


def configure_logging(
    *,
    level: int = logging.INFO,
    log_path: Path | str | None = None,
    force: bool = False,
) -> None:
    """Configure runtime logging explicitly for a CLI or application.

    A file handler is created only when ``log_path`` is provided.  Library
    imports therefore remain free of filesystem side effects.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path is not None:
        resolved = Path(log_path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(resolved, mode="a", encoding="utf-8"))
    logging.basicConfig(
        level=int(level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=bool(force),
    )


__all__ = ["LOGGER", "PATHS", "configure_logging"]

"""Import-safe YAML configuration loading and legacy lazy model lookup."""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml

from .paths import CONFIGS_DIR


def load_config(filename: str | Path) -> Any:
    """Load one YAML document without mutating module globals."""
    path = Path(filename)
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


class _LazyModelStruct(Mapping[str, Any]):
    """Compatibility mapping that scans model YAML only on first access."""

    def __init__(self, config_dir: Path) -> None:
        self._config_dir = Path(config_dir)
        self._values: dict[str, Any] | None = None

    def _load(self) -> dict[str, Any]:
        if self._values is None:
            self._values = {
                path.stem: load_config(path)
                for path in sorted(self._config_dir.glob("*.yaml"))
            }
        return self._values

    def __getitem__(self, key: str) -> Any:
        return self._load()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._load())

    def __len__(self) -> int:
        return len(self._load())


MODEL_STRUCT: Mapping[str, Any] = _LazyModelStruct(CONFIGS_DIR / "model_struct")


__all__ = ["MODEL_STRUCT", "load_config"]

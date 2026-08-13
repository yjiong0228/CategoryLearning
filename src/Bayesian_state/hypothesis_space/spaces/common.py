"""Small data structures shared by continuous and discrete spaces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class FrozenParameters(Mapping[str, object]):
    """Immutable and pickle-safe mapping for hypothesis-space metadata."""

    entries: tuple[tuple[str, object], ...]

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "FrozenParameters":
        return cls(tuple(sorted((str(key), value) for key, value in values.items())))

    def __getitem__(self, key: str) -> object:
        for candidate, value in self.entries:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self.entries)

    def __len__(self) -> int:
        return len(self.entries)


__all__ = ["FrozenParameters"]

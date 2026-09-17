"""Instance-local bounded memoization of exact stimulus distances.

Only numerical distances are retained, never beta, probabilities or learner
state. The byte budget counts stimulus/result payload; the entry bound also
bounds Python bookkeeping. Large batches bypass the cache before copying keys.
"""
from __future__ import annotations

from collections import OrderedDict
from numbers import Integral
from threading import RLock
from typing import Callable

import numpy as np


class ExactDistanceCache:
    """LRU cache with immutable buffers and empty-on-pickle process ownership."""

    DEFAULT_MAX_ENTRIES = 4096
    DEFAULT_MAX_BYTES = 4 * 1024 * 1024

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES,
                 max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        for name, value in (("max_entries", max_entries), ("max_bytes", max_bytes)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"distance cache {name} must be a nonnegative integer")
        self._max_entries = int(max_entries)
        self._max_bytes = int(max_bytes)
        self._lock = RLock()
        self.clear()

    def clear(self) -> None:
        """Release retained inputs/results and reset diagnostic counters."""
        with self._lock:
            self._entries: OrderedDict[tuple, tuple[tuple[int, ...], bytes]] = OrderedDict()
            self._context = None
            self._payload_bytes = 0
            self._hits = self._misses = self._bypasses = self._invalidations = 0

    def info(self) -> dict[str, int]:
        with self._lock:
            return {"entries": len(self._entries), "payload_bytes": self._payload_bytes,
                    "max_entries": self._max_entries, "max_bytes": self._max_bytes,
                    "hits": self._hits, "misses": self._misses, "bypasses": self._bypasses,
                    "invalidations": self._invalidations}

    def __getstate__(self) -> dict[str, int]:
        # A new process/copy owns a fresh cache, not the previous learner's inputs.
        return {"max_entries": self._max_entries, "max_bytes": self._max_bytes}

    def __setstate__(self, state: dict[str, int]) -> None:
        self.__init__(**state)

    @staticmethod
    def _view(entry: tuple[tuple[int, ...], bytes]) -> np.ndarray:
        shape, payload = entry
        # Bytes cannot be made writable. Give each caller its own array header so
        # reshaping/retyping a returned view cannot change the cached shape/dtype.
        return np.frombuffer(payload, dtype=np.float64).reshape(shape)

    def get_or_compute(self, context: tuple, hypothesis: int, stimuli: np.ndarray,
                       n_categories: int, compute: Callable[[], np.ndarray]) -> np.ndarray:
        payload_size = stimuli.nbytes + stimuli.shape[0] * n_categories * 8
        if not self._max_entries or not self._max_bytes or payload_size > self._max_bytes:
            with self._lock:
                self._bypasses += 1
            return compute()
        key = (int(hypothesis), stimuli.shape, stimuli.tobytes())
        with self._lock:
            if context != self._context:
                if self._context is not None:
                    self._invalidations += 1
                self._entries.clear()
                self._payload_bytes = 0
                self._context = context
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                self._hits += 1
                return self._view(entry)
            self._misses += 1
        # Keep expensive projection outside the lock. Duplicate computation on a
        # concurrent miss is safe; insertion below accounts for an entry once.
        result = compute()
        entry = (result.shape, result.tobytes())
        with self._lock:
            if context != self._context:
                return result
            existing = self._entries.get(key)
            if existing is not None:
                self._entries.move_to_end(key)
                return self._view(existing)
            size = len(key[2]) + len(entry[1])
            if size > self._max_bytes:
                self._bypasses += 1
                return result
            while self._entries and (len(self._entries) >= self._max_entries
                                     or self._payload_bytes + size > self._max_bytes):
                old_key, old_entry = self._entries.popitem(last=False)
                self._payload_bytes -= len(old_key[2]) + len(old_entry[1])
            self._entries[key] = entry
            self._payload_bytes += size
        return self._view(entry)

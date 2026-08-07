"""
Base Module
"""

from abc import ABC
from typing import Any, Mapping
from ..partitions import *
from ..base_problem import *


class BaseModule(ABC):

    def __init__(self, engine, **kwargs):
        """
        Initialize
        """
        self.engine = engine

    def process(self, **kwargs):
        """
        """
        pass

    def state_dict(self) -> dict[str, Any]:
        """Return mutable cognitive state for particle resampling.

        Stateless modules may keep the default empty payload. Stateful modules
        override this together with :meth:`load_state_dict`.
        """

        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a payload produced by :meth:`state_dict`."""

        if state:
            raise ValueError(
                f"{type(self).__name__} does not implement state restoration."
            )

    def clear_logs(self) -> None:
        """Discard trajectory logs after a particle state is copied."""

    def reseed_future(self, module_seed: int) -> None:
        """Assign a future RNG stream when the module owns stochastic state."""

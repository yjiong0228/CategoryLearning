"""Compatibility exports for the pre-refactor simulation runner path."""

from ..simulation.repeated_simulation import (
    StateModelSimulationRunner,
    aggregate_simulation_runs,
)

__all__ = ["StateModelSimulationRunner", "aggregate_simulation_runs"]

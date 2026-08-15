"""Stable result contracts shared by simulation and optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


@dataclass
class SimulationResult:
    """Repeated runs under one fixed parameter setting."""

    params: Dict[str, Any]
    mean_error: float
    metrics_by_mode: Dict[str, Dict[str, np.ndarray | float]]
    selection_prediction_mode: str
    state_log: Optional[Dict[str, Sequence[np.ndarray]]] = None
    trial_events: Optional[Sequence[Dict[str, Any]]] = None
    transition_counts: Optional[Sequence[Dict[str, Any]]] = None
    raw_runs: Optional[Sequence[Dict[str, Any]]] = None
    sample_errors: Optional[Sequence[float]] = None
    best_error: Optional[float] = None
    representative_run_index: Optional[int] = None
    simulation_repeats: int = 0
    simulation_point_seed: Optional[int] = None
    std_error: float = 0.0
    statistics_summary: Optional[Dict[str, Any]] = None
    repeat_aggregation: str = "mean_loss"
    aggregation_diagnostics: Optional[Dict[str, Any]] = None
    model_provenance: Optional[Dict[str, Any]] = None

    @property
    def gamma(self) -> float:
        memory_kwargs = self.params.get("engine.modules.memory_mod.kwargs")
        if isinstance(memory_kwargs, Mapping) and "gamma" in memory_kwargs:
            return memory_kwargs["gamma"]
        return self.params.get(
            "gamma",
            self.params.get("engine.modules.memory_mod.kwargs.gamma", float("nan")),
        )

    @property
    def w0(self) -> float:
        memory_kwargs = self.params.get("engine.modules.memory_mod.kwargs")
        if isinstance(memory_kwargs, Mapping) and "w0" in memory_kwargs:
            return memory_kwargs["w0"]
        return self.params.get(
            "w0",
            self.params.get("engine.modules.memory_mod.kwargs.w0", float("nan")),
        )


@dataclass
class SingleRunResult:
    """Normalized output of one trajectory or particle-marginal run."""

    params: Dict[str, Any]
    mean_error: float
    metrics_by_mode: Dict[str, Dict[str, np.ndarray | float]]
    selection_prediction_mode: str
    loss_metric: str
    loss_delta: Optional[float]
    state_log: Optional[Dict[str, Sequence[np.ndarray]]] = None
    trial_events: Optional[Sequence[Dict[str, Any]]] = None
    transition_counts: Optional[Sequence[Dict[str, Any]]] = None
    simulation_point_seed: Optional[int] = None
    trajectory_seed: Optional[int] = None
    module_seed: Optional[int] = None
    seed_context: Optional[Dict[str, Any]] = None
    posterior_log: Optional[Any] = None
    prior_log: Optional[Any] = None
    beta_log: Optional[Any] = None
    step_log: Optional[Any] = None
    strategy_counts_log: Optional[Any] = None


__all__ = ["SimulationResult", "SingleRunResult"]

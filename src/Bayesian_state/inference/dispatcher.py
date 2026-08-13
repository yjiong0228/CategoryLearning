"""Resolve and execute the inference backend selected by model configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .backends.particle_filter import run_state_model_particle_filter
from .backends.trajectory import run_state_model_trajectory
from .results import InferenceResult, ensure_inference_result


BACKEND_TRAJECTORY = "trajectory"
BACKEND_PARTICLE_FILTER = "particle_filter"


@dataclass(frozen=True)
class InferenceBackendConfig:
    backend: str
    particle_count: int | None = None
    resample_threshold_fraction: float | None = None
    choice_transmission_audit: bool = False


def resolve_inference_backend(
    engine_config: Mapping[str, Any],
) -> InferenceBackendConfig:
    """Normalize ``engine_config.inference`` without changing the YAML schema."""
    raw = engine_config.get("inference")
    if raw is None:
        return InferenceBackendConfig(backend=BACKEND_TRAJECTORY)
    if not isinstance(raw, Mapping):
        raise ValueError("engine_config.inference must be a mapping when provided.")
    backend = str(raw.get("backend", BACKEND_TRAJECTORY)).strip().lower()
    if backend in {BACKEND_TRAJECTORY, "single_trajectory", "simulation"}:
        return InferenceBackendConfig(backend=BACKEND_TRAJECTORY)
    if backend not in {BACKEND_PARTICLE_FILTER, "bootstrap_particle_filter"}:
        raise ValueError(
            "engine_config.inference.backend must be 'trajectory' or "
            f"'particle_filter', got {backend!r}."
        )
    particle_count = int(raw.get("particle_count", 512))
    threshold = float(raw.get("resample_threshold_fraction", 0.5))
    choice_transmission_audit = bool(raw.get("choice_transmission_audit", False))
    if particle_count < 2:
        raise ValueError("particle-filter particle_count must be at least 2.")
    if not 0.0 < threshold <= 1.0:
        raise ValueError(
            "particle-filter resample_threshold_fraction must lie in (0, 1]."
        )
    return InferenceBackendConfig(
        backend=BACKEND_PARTICLE_FILTER,
        particle_count=particle_count,
        resample_threshold_fraction=threshold,
        choice_transmission_audit=choice_transmission_audit,
    )


def run_inference_backend(
    *,
    engine_config: Mapping[str, Any],
    subject_id: int,
    condition: int,
    stimulus: Sequence[Sequence[float]] | np.ndarray,
    choices: Sequence[int] | np.ndarray,
    feedback: Sequence[float] | np.ndarray,
    inference_seed: int | None = None,
    choice_readout_power: float = 1.0,
    strategy_confidence_gain: float = 0.0,
    rule_commitment_confidence_gain: float = 0.0,
    output_lapse: float = 0.0,
    valid_trial_mask: Sequence[bool] | np.ndarray | None = None,
    processed_data_dir: Path | str | None = None,
    dataset_paths: Mapping[str, Path | str] | None = None,
) -> InferenceResult:
    """Run the configured backend and return inference-level outputs."""
    config = resolve_inference_backend(engine_config)
    if config.backend == BACKEND_TRAJECTORY:
        result = run_state_model_trajectory(
            engine_config=engine_config,
            subject_id=int(subject_id),
            condition=int(condition),
            stimulus=stimulus,
            choices=choices,
            feedback=feedback,
            trajectory_seed=inference_seed,
            processed_data_dir=processed_data_dir,
            dataset_paths=dataset_paths,
        )
        return ensure_inference_result(result, backend=BACKEND_TRAJECTORY)
    if int(condition) != 1:
        raise ValueError("the current StateModel particle backend supports condition 1 only.")
    assert config.particle_count is not None
    assert config.resample_threshold_fraction is not None
    result = run_state_model_particle_filter(
        engine_config=engine_config,
        subject_id=int(subject_id),
        stimulus=stimulus,
        choices=choices,
        feedback=feedback,
        particle_count=config.particle_count,
        choice_readout_power=float(choice_readout_power),
        strategy_confidence_gain=float(strategy_confidence_gain),
        rule_commitment_confidence_gain=float(
            rule_commitment_confidence_gain
        ),
        output_lapse=float(output_lapse),
        filter_seed=int(inference_seed if inference_seed is not None else 20260806),
        resample_threshold_fraction=config.resample_threshold_fraction,
        choice_transmission_audit=config.choice_transmission_audit,
        valid_trial_mask=valid_trial_mask,
        processed_data_dir=processed_data_dir,
        dataset_paths=dataset_paths,
    )
    return ensure_inference_result(result, backend=BACKEND_PARTICLE_FILTER)


__all__ = [
    "BACKEND_PARTICLE_FILTER",
    "BACKEND_TRAJECTORY",
    "InferenceBackendConfig",
    "resolve_inference_backend",
    "run_inference_backend",
]

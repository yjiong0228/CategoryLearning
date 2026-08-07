"""Particle integration backend for engine-configured ``StateModel`` objects.

The implementation is shared with the active-set workflows for backward
compatibility.  This module supplies the model-agnostic public entry point used
by optimization and simulation code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..active_set.particle_filter import (
    ActiveSetParticleFilterResult,
    effective_sample_size,
    run_active_set_particle_filter,
    systematic_resample,
)


StateModelParticleFilterResult = ActiveSetParticleFilterResult


def run_state_model_particle_filter(
    *,
    engine_config: Mapping[str, Any],
    subject_id: int,
    stimulus: Sequence[Sequence[float]] | np.ndarray,
    choices: Sequence[int] | np.ndarray,
    feedback: Sequence[float] | np.ndarray,
    particle_count: int,
    choice_readout_power: float = 1.0,
    output_lapse: float = 0.0,
    filter_seed: int = 20260806,
    resample_threshold_fraction: float = 0.5,
    valid_trial_mask: Sequence[bool] | np.ndarray | None = None,
    processed_data_dir: Path | str | None = None,
    dataset_paths: Mapping[str, Path | str] | None = None,
) -> StateModelParticleFilterResult:
    """Integrate latent ``StateModel`` trajectories with bootstrap particles.

    ``choice_readout_power`` is the standard sharpened-expectation power, and
    ``output_lapse`` is the standard uniform response lapse.  The legacy names
    ``rho`` and ``epsilon`` remain confined to the compatibility implementation.
    """

    power = float(choice_readout_power)
    lapse = float(output_lapse)
    if not np.isfinite(power) or power <= 0.0:
        raise ValueError("choice_readout_power must be finite and positive.")
    if not np.isfinite(lapse) or not 0.0 <= lapse <= 1.0:
        raise ValueError("output_lapse must lie in [0, 1].")
    return run_active_set_particle_filter(
        engine_config=engine_config,
        subject_id=int(subject_id),
        stimulus=stimulus,
        choices=choices,
        feedback=feedback,
        particle_count=int(particle_count),
        rho=power,
        epsilon=lapse,
        filter_seed=int(filter_seed),
        resample_threshold_fraction=float(resample_threshold_fraction),
        valid_trial_mask=valid_trial_mask,
        processed_data_dir=processed_data_dir,
        dataset_paths=dataset_paths,
    )


__all__ = [
    "StateModelParticleFilterResult",
    "effective_sample_size",
    "run_state_model_particle_filter",
    "systematic_resample",
]

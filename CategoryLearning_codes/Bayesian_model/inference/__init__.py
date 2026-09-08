"""Public inference-backend interfaces."""

from .dispatcher import (
    BACKEND_PARTICLE_FILTER,
    BACKEND_TRAJECTORY,
    InferenceBackendConfig,
    resolve_inference_backend,
    run_inference_backend,
)
from .results import (
    InferenceResult,
    ParticleFilterResult,
    TrajectoryInferenceResult,
    ensure_inference_result,
)

__all__ = [
    "BACKEND_PARTICLE_FILTER",
    "BACKEND_TRAJECTORY",
    "InferenceBackendConfig",
    "InferenceResult",
    "ParticleFilterResult",
    "TrajectoryInferenceResult",
    "ensure_inference_result",
    "resolve_inference_backend",
    "run_inference_backend",
]

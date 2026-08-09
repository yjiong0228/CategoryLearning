"""Compatibility exports for simulation configuration helpers.

New code should import these names from
:mod:`src.Bayesian_state.simulation.simulation_config`.
"""

from ..simulation.simulation_config import (
    DEFAULT_DATA_PATH,
    DEFAULT_OUTPUT_DIR,
    EVALUATION_ROLE_OPTIMIZATION,
    EVALUATION_ROLE_SIMULATION,
    EVALUATION_ROLES,
    dump_stream,
    load_yaml,
    recursive_to_builtin,
    resolve_engine_config,
    resolve_evaluation_score_mask,
    resolve_loss_delta,
    resolve_loss_metric,
    resolve_path,
    resolve_prediction_modes,
    resolve_simulation_repeats,
    resolve_subjects,
    resolve_window_size,
    save_json,
    stream_ref_relative_to,
)

_dump_stream = dump_stream
_recursive_to_builtin = recursive_to_builtin
_resolve_path = resolve_path
_stream_ref_relative_to = stream_ref_relative_to

__all__ = [
    "DEFAULT_DATA_PATH",
    "DEFAULT_OUTPUT_DIR",
    "EVALUATION_ROLE_OPTIMIZATION",
    "EVALUATION_ROLE_SIMULATION",
    "EVALUATION_ROLES",
    "dump_stream",
    "load_yaml",
    "recursive_to_builtin",
    "resolve_engine_config",
    "resolve_evaluation_score_mask",
    "resolve_loss_delta",
    "resolve_loss_metric",
    "resolve_path",
    "resolve_prediction_modes",
    "resolve_simulation_repeats",
    "resolve_subjects",
    "resolve_window_size",
    "save_json",
    "stream_ref_relative_to",
]

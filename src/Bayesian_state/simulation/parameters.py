"""固定仿真参数的提取、覆盖与可复现种子解析。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping

from .config import expand_profile_candidate_hyperparams
from ..utils.seeding import derive_hyper_candidate_seed

DEFAULT_FIXED_HYPERPARAM_PATHS = (
    "engine.modules.memory_mod.kwargs.gamma",
    "engine.modules.memory_mod.kwargs.w0",
    "engine.modules.memory_mod.kwargs.feedback_gain",
    "engine.modules.hypo_transitions_mod.kwargs.strategies",
    "engine.modules.hypo_transitions_mod.kwargs.state_controller",
    "engine.modules.hypo_transitions_mod.kwargs.post_to_prior",
    "engine.modules.hypo_transitions_mod.kwargs.capacity",
    "engine.modules.hypo_transitions_mod.kwargs.m",
    "engine.modules.hypo_transitions_mod.kwargs.g",
    "engine.modules.hypo_transitions_mod.kwargs.rate_controller",
    "engine.modules.hypo_transitions_mod.kwargs.range_controller",
    "engine.modules.hypo_transitions_mod.kwargs.continuous_controller",
    "engine.modules.hypo_transitions_mod.kwargs.nested_feedback_accumulator_controller",
    "engine.modules.hypo_transitions_mod.kwargs.prior_assignment",
    "engine.modules.hypo_transitions_mod.kwargs.persistent_execution",
    "engine.modules.hypo_transitions_mod.kwargs.tau_local",
    "engine.modules.hypo_transitions_mod.kwargs.theta",
    "engine.modules.hypo_transitions_mod.kwargs.max_active_hypotheses",
    "engine.modules.hypo_transitions_mod.kwargs.init_num",
    "engine.modules.hypo_transitions_mod.kwargs.prior_reset_base",
    "engine.modules.hypo_transitions_mod.kwargs.prior_reset_post_error",
    "engine.modules.hypo_transitions_mod.kwargs.prior_reset_low_accuracy",
    "engine.modules.hypo_transitions_mod.kwargs.prior_reset_threshold",
    "engine.modules.hypo_transitions_mod.kwargs.prior_reset_window",
    "engine.modules.hypo_transitions_mod.kwargs.prior_reset_decay",
    "engine.modules.hypo_transitions_mod.kwargs.prior_reset_max",
    "engine.modules.hypo_transitions_mod.kwargs.prior_reset_target",
    "engine.modules.hypo_transitions_mod.kwargs.prior_reset_source",
    "engine.modules.hypo_transitions_mod.kwargs.prior_reset_volatility_gain",
    "engine.modules.hypo_transitions_mod.kwargs.latent_volatility_base",
    "engine.modules.hypo_transitions_mod.kwargs.latent_volatility_error_gain",
    "engine.modules.hypo_transitions_mod.kwargs.latent_volatility_low_accuracy_gain",
    "engine.modules.hypo_transitions_mod.kwargs.latent_volatility_threshold",
    "engine.modules.hypo_transitions_mod.kwargs.latent_volatility_window",
    "engine.modules.hypo_transitions_mod.kwargs.latent_volatility_decay",
    "engine.modules.hypo_transitions_mod.kwargs.latent_volatility_max",
    "engine.modules.hypo_transitions_mod.kwargs.latent_volatility_feedback_mode",
    "engine.modules.hypo_transitions_mod.kwargs.latent_volatility_signal",
    "engine.modules.hypo_transitions_mod.kwargs.latent_volatility_pressure_slope",
    "engine.modules.beta_mod.kwargs.beta_init",
    "engine.modules.beta_mod.kwargs.beta_min",
    "engine.modules.beta_mod.kwargs.beta_max",
    "engine.modules.beta_mod.kwargs.decrease_rate",
    "engine.modules.beta_mod.kwargs.increase_rate",
    "engine.modules.beta_mod.kwargs.prior_beta_scale",
    "engine.modules.beta_mod.kwargs.correct_additive",
    "engine.modules.beta_mod.kwargs.beta_update_mode",
    "engine.modules.beta_mod.kwargs.update_scope",
    "engine.modules.beta_mod.kwargs.probabilistic_feedback_lapse",
    "engine.likelihood.distance_mode",
    "engine.likelihood.beta_source",
    "engine.likelihood.default_beta",
    "engine.output_noise.kwargs.base_lapse",
    "engine.output_noise.kwargs.post_error_lapse",
    "engine.output_noise.kwargs.low_accuracy_lapse",
    "engine.output_noise.kwargs.low_accuracy_threshold",
    "engine.output_noise.kwargs.recent_accuracy_window",
    "engine.output_noise.kwargs.lapse_decay",
    "engine.output_noise.kwargs.max_lapse",
    "engine.output_noise.kwargs.lapse_target",
    "engine.output_noise.kwargs.latent_volatility_lapse",
    "engine.output_noise.kwargs.latent_volatility_power",
    "engine.choice_readout.kwargs",
    "engine.choice_readout.kwargs.method",
    "engine.choice_readout.kwargs.power",
    "engine.choice_readout.kwargs.weight_floor",
    "engine.choice_readout.kwargs.switch_probability",
    "engine.choice_readout.kwargs.post_error_switch_delta",
    "engine.choice_readout.kwargs.low_confidence_switch_gain",
    "engine.choice_readout.kwargs.strategy_confidence_gain",
    "engine.choice_readout.kwargs.rule_commitment_confidence_gain",
)


def resolve_hyper_base_seed(cfg: Mapping[str, Any]) -> int:
    if "hyper_base_seed" not in cfg:
        raise ValueError("Config must include hyper_base_seed.")
    return int(cfg["hyper_base_seed"])


def resolve_hyper_candidate_seed(
    cfg: Mapping[str, Any],
    hyper_base_seed: int,
    subject_id: int,
    fixed_hyperparams: Mapping[str, Any],
) -> int:
    raw = cfg.get("hyper_candidate_seed")
    if raw is not None:
        return int(raw)
    return derive_hyper_candidate_seed(
        hyper_base_seed=hyper_base_seed,
        stage="simulation_direct",
        combination_index=0,
        hyperparams=dict(fixed_hyperparams),
        extra_context={"subject_id": int(subject_id)},
    )


def _set_by_path(root: Dict[str, Any], path: str, value: Any) -> None:
    curr = root
    parts = path.split(".")
    for part in parts[:-1]:
        next_value = curr.setdefault(part, {})
        if not isinstance(next_value, dict):
            raise ValueError(f"Cannot set nested path through non-mapping segment: {path}")
        curr = next_value
    curr[parts[-1]] = deepcopy(value)


def _get_by_path(root: Mapping[str, Any], path: str) -> Any:
    curr: Any = root
    for part in path.split("."):
        if not isinstance(curr, Mapping) or part not in curr:
            return None
        curr = curr[part]
    return curr


def infer_fixed_hyperparams_from_engine_config(engine_config: Mapping[str, Any]) -> Dict[str, Any]:
    inferred: Dict[str, Any] = {}
    for key in DEFAULT_FIXED_HYPERPARAM_PATHS:
        engine_path = key[len("engine."):]
        value = _get_by_path(engine_config, engine_path)
        if value is not None:
            inferred[key] = deepcopy(value)
    return inferred


def apply_fixed_hyperparams_to_subject_config(
    subject_cfg: Mapping[str, Any],
    fixed_hyperparams: Mapping[str, Any],
) -> Dict[str, Any]:
    resolved = deepcopy(dict(subject_cfg))
    for key, value in expand_profile_candidate_hyperparams(fixed_hyperparams).items():
        if key.startswith("simulation."):
            _set_by_path(resolved, key[len("simulation."):], value)
        elif key.startswith("engine."):
            continue
        else:
            raise ValueError(
                "fixed_hyperparams key must start with 'engine.' or "
                f"'simulation.': {key}"
            )
    return resolved


def apply_fixed_hyperparams_to_engine_config(
    engine_config: Mapping[str, Any],
    fixed_hyperparams: Mapping[str, Any],
) -> Dict[str, Any]:
    resolved = deepcopy(dict(engine_config))
    for key, value in expand_profile_candidate_hyperparams(fixed_hyperparams).items():
        if key.startswith("engine."):
            _set_by_path(resolved, key[len("engine."):], value)
        elif key.startswith("simulation."):
            continue
        else:
            raise ValueError(
                "fixed_hyperparams key must start with 'engine.' or "
                f"'simulation.': {key}"
            )
    return resolved


__all__ = [
    "DEFAULT_FIXED_HYPERPARAM_PATHS",
    "apply_fixed_hyperparams_to_engine_config",
    "apply_fixed_hyperparams_to_subject_config",
    "infer_fixed_hyperparams_from_engine_config",
    "resolve_hyper_base_seed",
    "resolve_hyper_candidate_seed",
]

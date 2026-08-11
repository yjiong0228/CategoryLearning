"""Batch repeated simulations for fixed StateModel hyperparameters.

Usage:
    python -m src.Bayesian_state.run_simulation \
        --config configs/simulation_cfg/pmh_cond1_simulation.yaml
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from src.Bayesian_state.simulation.simulation_config import (
    DEFAULT_DATA_PATH,
    DEFAULT_OUTPUT_DIR,
    PROFILE_CANDIDATE_KEY,
    dump_stream,
    expand_profile_candidate_hyperparams,
    load_yaml,
    recursive_to_builtin,
    resolve_engine_config,
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
from src.Bayesian_state.utils.config_subjects import resolve_subject_config
from src.Bayesian_state.utils.base import configure_logging
from src.Bayesian_state.utils.datasets import resolve_dataset_paths
from src.Bayesian_state.simulation.repeated_simulation import StateModelSimulationRunner
from src.Bayesian_state.utils.seeding import derive_hyper_candidate_seed
from src.Bayesian_state.utils.paths import ROOT_DIR


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
    "engine.modules.beta_mod.kwargs.decrease_rate",
    "engine.modules.beta_mod.kwargs.prior_beta_scale",
    "engine.modules.beta_mod.kwargs.correct_additive",
    "engine.modules.beta_mod.kwargs.beta_update_mode",
    "engine.modules.beta_mod.kwargs.update_scope",
    "engine.modules.beta_mod.kwargs.probabilistic_feedback_lapse",
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
            raise ValueError(f"fixed_hyperparams key must start with 'engine.' or 'simulation.': {key}")
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
            raise ValueError(f"fixed_hyperparams key must start with 'engine.' or 'simulation.': {key}")
    return resolved


def serialize_result(
    subject_id: int,
    condition: int,
    result: Dict[str, Any],
    output_dir: Path,
    subject_json_dir: Path | None = None,
) -> Dict[str, Any]:
    best = result["best"]
    fixed_hyperparams = dict(result.get("fixed_hyperparams") or getattr(best, "params", {}) or {})
    best_params = _compact_hyperparams(fixed_hyperparams)
    mean_error = float(getattr(best, "mean_error"))
    std_error = float(getattr(best, "std_error", 0.0))
    best_error = float(getattr(best, "best_error", mean_error))
    sample_errors = list(getattr(best, "sample_errors", []) or [])
    statistics_summary = dict(getattr(best, "statistics_summary", {}) or {})

    raw_runs = getattr(best, "raw_runs", None)
    raw_runs_ref = None
    if raw_runs:
        raw_runs_ref = dump_stream(raw_runs, output_dir, subject_id, "raw_runs")
        subject_json_dir = subject_json_dir or (output_dir / "subjects")
        raw_runs_ref = stream_ref_relative_to(raw_runs_ref, output_dir, subject_json_dir)

    metrics_by_mode = getattr(best, "metrics_by_mode", None) or {}
    selection_meta = result.get("selection_meta", {}) or {}
    selection_mode = str(selection_meta.get("selection_prediction_mode", "posterior_t_minus_1"))
    available_modes = sorted(metrics_by_mode.keys())

    simulation = {
        "mean_error": mean_error,
        "best_error": best_error,
        "std_error": std_error,
        "simulation_repeats": int(selection_meta.get("simulation_repeats", getattr(best, "simulation_repeats", 0))),
        "window_size": selection_meta.get("window_size"),
        "sample_errors": sample_errors,
    }

    data = {
        "result_type": "simulation",
        "subject_id": subject_id,
        "condition": condition,
        "best_params": best_params,
        "fixed_hyperparams": fixed_hyperparams,
        "simulation": simulation,
        "statistics": statistics_summary,
        "selection": {
            "prediction_mode": selection_meta.get("prediction_mode"),
            "selection_prediction_mode": selection_mode,
            "available_prediction_modes": available_modes,
            "representative_run_index": getattr(best, "representative_run_index", 0),
            "loss_metric": selection_meta.get("loss_metric"),
            "loss_delta": selection_meta.get("loss_delta"),
            "hyper_base_seed": selection_meta.get("hyper_base_seed"),
            "hyper_candidate_seed": selection_meta.get("hyper_candidate_seed"),
            "simulation_point_seed": selection_meta.get("simulation_point_seed"),
            "selection_meta": selection_meta,
        },
        "representative_run": {
            "metrics_by_mode": metrics_by_mode,
            "state_log": getattr(best, "state_log", None),
            "trial_events": getattr(best, "trial_events", None),
            "transition_counts": getattr(best, "transition_counts", None),
        },
        "raw_runs_ref": raw_runs_ref,
    }
    return recursive_to_builtin(data)


def _compact_hyperparams(hyperparams: Mapping[str, Any]) -> Dict[str, Any]:
    expanded_hyperparams = expand_profile_candidate_hyperparams(hyperparams)
    summary = dict(expanded_hyperparams)
    if PROFILE_CANDIDATE_KEY in hyperparams:
        summary[PROFILE_CANDIDATE_KEY] = deepcopy(hyperparams[PROFILE_CANDIDATE_KEY])
    shortcuts = {
        "engine.modules.memory_mod.kwargs.gamma": "gamma",
        "engine.modules.memory_mod.kwargs.w0": "w0",
        "engine.modules.beta_mod.kwargs.beta_init": "beta_init",
        "engine.modules.beta_mod.kwargs.decrease_rate": "decrease_rate",
        "engine.modules.beta_mod.kwargs.prior_beta_scale": "prior_beta_scale",
        "engine.modules.beta_mod.kwargs.correct_additive": "correct_additive",
        "engine.modules.beta_mod.kwargs.beta_update_mode": "beta_update_mode",
        "engine.modules.beta_mod.kwargs.update_scope": "beta_update_scope",
        "engine.modules.beta_mod.kwargs.probabilistic_feedback_lapse": "probabilistic_feedback_lapse",
        "engine.modules.hypo_transitions_mod.kwargs.prior_reset_base": "prior_reset_base",
        "engine.modules.hypo_transitions_mod.kwargs.prior_reset_post_error": "prior_reset_post_error",
        "engine.modules.hypo_transitions_mod.kwargs.prior_reset_low_accuracy": "prior_reset_low_accuracy",
        "engine.modules.hypo_transitions_mod.kwargs.prior_reset_threshold": "prior_reset_threshold",
        "engine.modules.hypo_transitions_mod.kwargs.prior_reset_window": "prior_reset_window",
        "engine.modules.hypo_transitions_mod.kwargs.prior_reset_decay": "prior_reset_decay",
        "engine.modules.hypo_transitions_mod.kwargs.prior_reset_max": "prior_reset_max",
        "engine.modules.hypo_transitions_mod.kwargs.prior_reset_target": "prior_reset_target",
        "engine.modules.hypo_transitions_mod.kwargs.prior_reset_source": "prior_reset_source",
        "engine.modules.hypo_transitions_mod.kwargs.prior_reset_volatility_gain": "prior_reset_volatility_gain",
        "engine.modules.hypo_transitions_mod.kwargs.latent_volatility_base": "latent_volatility_base",
        "engine.modules.hypo_transitions_mod.kwargs.latent_volatility_error_gain": "latent_volatility_error_gain",
        "engine.modules.hypo_transitions_mod.kwargs.latent_volatility_low_accuracy_gain": "latent_volatility_low_accuracy_gain",
        "engine.modules.hypo_transitions_mod.kwargs.latent_volatility_threshold": "latent_volatility_threshold",
        "engine.modules.hypo_transitions_mod.kwargs.latent_volatility_window": "latent_volatility_window",
        "engine.modules.hypo_transitions_mod.kwargs.latent_volatility_decay": "latent_volatility_decay",
        "engine.modules.hypo_transitions_mod.kwargs.latent_volatility_max": "latent_volatility_max",
        "engine.modules.hypo_transitions_mod.kwargs.latent_volatility_feedback_mode": "latent_volatility_feedback_mode",
        "engine.modules.hypo_transitions_mod.kwargs.latent_volatility_signal": "latent_volatility_signal",
        "engine.modules.hypo_transitions_mod.kwargs.latent_volatility_pressure_slope": "latent_volatility_pressure_slope",
        "engine.output_noise.kwargs.base_lapse": "output_base_lapse",
        "engine.output_noise.kwargs.post_error_lapse": "output_post_error_lapse",
        "engine.output_noise.kwargs.low_accuracy_lapse": "output_low_accuracy_lapse",
        "engine.output_noise.kwargs.low_accuracy_threshold": "output_low_accuracy_threshold",
        "engine.output_noise.kwargs.recent_accuracy_window": "output_recent_accuracy_window",
        "engine.output_noise.kwargs.lapse_decay": "output_lapse_decay",
        "engine.output_noise.kwargs.max_lapse": "output_max_lapse",
        "engine.output_noise.kwargs.lapse_target": "output_lapse_target",
        "engine.output_noise.kwargs.latent_volatility_lapse": "output_latent_volatility_lapse",
        "engine.output_noise.kwargs.latent_volatility_power": "output_latent_volatility_power",
        "engine.modules.hypo_transitions_mod.kwargs.init_num": "init_num",
        "engine.modules.hypo_transitions_mod.kwargs.max_active_hypotheses": "max_active_hypotheses",
        "engine.modules.hypo_transitions_mod.kwargs.capacity": "capacity",
        "engine.modules.hypo_transitions_mod.kwargs.theta": "theta",
        "simulation.window_size": "window_size",
    }
    for source, target in shortcuts.items():
        if source in expanded_hyperparams:
            summary[target] = expanded_hyperparams[source]
    transition_kwargs = expanded_hyperparams.get("engine.modules.hypo_transitions_mod.kwargs")
    if isinstance(transition_kwargs, Mapping):
        for source, target in (
            ("capacity", "capacity"),
            ("theta", "theta"),
            ("init_num", "init_num"),
            ("max_active_hypotheses", "max_active_hypotheses"),
            ("prior_reset_base", "prior_reset_base"),
            ("prior_reset_post_error", "prior_reset_post_error"),
            ("prior_reset_low_accuracy", "prior_reset_low_accuracy"),
            ("prior_reset_threshold", "prior_reset_threshold"),
            ("prior_reset_window", "prior_reset_window"),
            ("prior_reset_decay", "prior_reset_decay"),
            ("prior_reset_max", "prior_reset_max"),
            ("prior_reset_target", "prior_reset_target"),
            ("prior_reset_source", "prior_reset_source"),
            ("prior_reset_volatility_gain", "prior_reset_volatility_gain"),
            ("latent_volatility_base", "latent_volatility_base"),
            ("latent_volatility_error_gain", "latent_volatility_error_gain"),
            ("latent_volatility_low_accuracy_gain", "latent_volatility_low_accuracy_gain"),
            ("latent_volatility_threshold", "latent_volatility_threshold"),
            ("latent_volatility_window", "latent_volatility_window"),
            ("latent_volatility_decay", "latent_volatility_decay"),
            ("latent_volatility_max", "latent_volatility_max"),
            ("latent_volatility_feedback_mode", "latent_volatility_feedback_mode"),
            ("latent_volatility_signal", "latent_volatility_signal"),
            ("latent_volatility_pressure_slope", "latent_volatility_pressure_slope"),
        ):
            if source in transition_kwargs:
                summary[target] = transition_kwargs[source]
    output_noise = expanded_hyperparams.get("engine.output_noise.kwargs")
    if isinstance(output_noise, Mapping):
        for source, target in (
            ("base_lapse", "output_base_lapse"),
            ("post_error_lapse", "output_post_error_lapse"),
            ("low_accuracy_lapse", "output_low_accuracy_lapse"),
            ("low_accuracy_threshold", "output_low_accuracy_threshold"),
            ("recent_accuracy_window", "output_recent_accuracy_window"),
            ("lapse_decay", "output_lapse_decay"),
            ("max_lapse", "output_max_lapse"),
            ("lapse_target", "output_lapse_target"),
            ("latent_volatility_lapse", "output_latent_volatility_lapse"),
            ("latent_volatility_power", "output_latent_volatility_power"),
        ):
            if source in output_noise:
                summary[target] = output_noise[source]
    readout = expanded_hyperparams.get("engine.choice_readout.kwargs")
    if isinstance(readout, Mapping):
        for source, target in (
            ("method", "choice_readout_method"),
            ("power", "choice_readout_power"),
            ("switch_probability", "choice_readout_switch_probability"),
            ("post_error_switch_delta", "choice_readout_post_error_switch_delta"),
            ("low_confidence_switch_gain", "choice_readout_low_confidence_switch_gain"),
            ("strategy_confidence_gain", "strategy_confidence_gain"),
        ):
            if source in readout:
                summary[target] = readout[source]
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch repeated simulation for fixed StateModel hyperparameters")
    p.add_argument("--config", required=True, type=Path, help="Simulation YAML config")
    p.add_argument("--subjects", nargs="+", type=int, help="Subject IDs; overrides subjects/subject_range in YAML")
    p.add_argument("--subject-range", nargs=2, type=int, metavar=("START", "END"), help="Inclusive subject range")
    return p.parse_args()


def run_simulation(
    config_path: str | Path,
    *,
    subjects: Sequence[int] | None = None,
    subject_range: Sequence[int] | None = None,
) -> list[Path]:
    """Run the configured fixed-hyperparameter simulations and return saved files."""
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = (ROOT_DIR / cfg_path).resolve()
    cfg = load_yaml(cfg_path)

    selected_subjects = resolve_subjects(subjects, subject_range, cfg)
    saved_paths: list[Path] = []

    for sid in selected_subjects:
        subject_cfg = resolve_subject_config(cfg, sid)
        explicit_fixed_hyperparams = dict(subject_cfg.get("fixed_hyperparams") or {})
        subject_cfg = apply_fixed_hyperparams_to_subject_config(subject_cfg, explicit_fixed_hyperparams)
        engine_config = resolve_engine_config(subject_cfg, cfg_path.parent, subject_id=sid)
        fixed_hyperparams = {
            **infer_fixed_hyperparams_from_engine_config(engine_config),
            **explicit_fixed_hyperparams,
        }
        seed_hyperparams = explicit_fixed_hyperparams or fixed_hyperparams
        engine_config = apply_fixed_hyperparams_to_engine_config(engine_config, fixed_hyperparams)
        prediction_mode, selection_prediction_mode = resolve_prediction_modes(subject_cfg)
        loss_metric = resolve_loss_metric(subject_cfg)
        loss_delta = resolve_loss_delta(subject_cfg, loss_metric)

        dataset_paths = resolve_dataset_paths(subject_cfg, cfg_path.parent, DEFAULT_DATA_PATH)
        data_path = dataset_paths["learning_data"]
        output_dir = resolve_path(cfg_path.parent, subject_cfg.get("output_dir"), DEFAULT_OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        n_jobs = int(subject_cfg.get("n_jobs", 4))
        simulation_repeats = resolve_simulation_repeats(subject_cfg)
        hyper_base_seed = resolve_hyper_base_seed(subject_cfg)
        hyper_candidate_seed = resolve_hyper_candidate_seed(
            subject_cfg,
            hyper_base_seed,
            sid,
            seed_hyperparams,
        )
        stop_at = float(subject_cfg.get("stop_at", 1.0))
        max_trials_val = subject_cfg.get("max_trials")
        max_trials = int(max_trials_val) if max_trials_val is not None else None
        keep_logs = bool(subject_cfg.get("keep_logs", False))
        window_size = resolve_window_size(subject_cfg, sid, selected_subjects)
        representative_run_selection = str(
            subject_cfg.get("representative_run_selection", "min_error")
        )
        representative_choice_fraction = float(
            subject_cfg.get("representative_choice_fraction", 0.10)
        )
        statistics_config = subject_cfg.get("statistics_config")

        runner = StateModelSimulationRunner(
            engine_config=engine_config,
            processed_data_dir=dataset_paths["processed_dir"],
            dataset_paths=dataset_paths,
            n_jobs=n_jobs,
        )
        runner.prepare_data(data_path)

        print(f"\n{'=' * 60}")
        print(f"Subject {sid}")
        print(f"{'=' * 60}")

        result: Dict[str, Any] = runner.simulate_subject(
            subject_id=sid,
            simulation_repeats=simulation_repeats,
            fixed_hyperparams=fixed_hyperparams,
            window_size=window_size,
            stop_at=stop_at,
            max_trials=max_trials,
            keep_logs=keep_logs,
            prediction_mode=prediction_mode,
            selection_prediction_mode=selection_prediction_mode,
            loss_metric=loss_metric,
            loss_delta=loss_delta,
            hyper_candidate_seed=hyper_candidate_seed,
            seed_hyperparams=seed_hyperparams,
            representative_run_selection=representative_run_selection,
            representative_choice_fraction=representative_choice_fraction,
            statistics_config=statistics_config,
            evaluation_protocol=subject_cfg.get("evaluation_protocol"),
            evaluation_role="simulation",
        )
        result["selection_meta"]["hyper_base_seed"] = hyper_base_seed
        if statistics_config is not None:
            result["selection_meta"]["statistics_config"] = statistics_config

        best: Any = result["best"]
        print(f"  Fixed hyperparams: {fixed_hyperparams}")
        print(f"  Mean error ({selection_prediction_mode}): {float(best.mean_error):.6f} +/- {float(best.std_error):.6f}")
        print(f"  Best run error ({selection_prediction_mode}): {float(best.best_error):.6f}")
        print(f"  Loss metric: {loss_metric}")
        print(f"  Seeds: hyper_base_seed={hyper_base_seed}, hyper_candidate_seed={hyper_candidate_seed}")

        subjects_dir = output_dir / "subjects"
        payload = serialize_result(sid, int(result["condition"]), result, output_dir, subject_json_dir=subjects_dir)
        save_path = subjects_dir / f"subject_{sid}.json"
        save_json(payload, save_path)
        saved_paths.append(save_path)
        print(f"  Saved -> {save_path}")

    print("\nAll subjects done.")
    return saved_paths


def main() -> None:
    configure_logging()
    args = parse_args()
    run_simulation(
        args.config,
        subjects=args.subjects,
        subject_range=args.subject_range,
    )


if __name__ == "__main__":
    main()

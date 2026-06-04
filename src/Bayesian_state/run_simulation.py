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

from src.Bayesian_state.utils.optimization_config import (
    DEFAULT_DATA_PATH,
    DEFAULT_OUTPUT_DIR,
    load_yaml,
    resolve_engine_config,
    resolve_loss_delta,
    resolve_loss_metric,
    resolve_prediction_modes,
    resolve_subjects,
    resolve_window_size,
    save_json,
    _dump_stream,
    _recursive_to_builtin,
    _resolve_path,
    _stream_ref_relative_to,
)
from src.Bayesian_state.utils.config_subjects import resolve_subject_config
from src.Bayesian_state.utils.datasets import resolve_dataset_paths
from src.Bayesian_state.utils.optimizer_common import derive_hyper_candidate_seed
from src.Bayesian_state.utils.optimizer_simulation import StateModelSimulationRunner
from src.Bayesian_state.utils.paths import ROOT_DIR


DEFAULT_FIXED_HYPERPARAM_PATHS = (
    "engine.modules.memory_mod.kwargs.gamma",
    "engine.modules.memory_mod.kwargs.w0",
    "engine.modules.hypo_transitions_mod.kwargs.strategies",
    "engine.modules.hypo_transitions_mod.kwargs.max_active_hypotheses",
    "engine.modules.hypo_transitions_mod.kwargs.init_num",
    "engine.modules.beta_mod.kwargs.beta_init",
    "engine.modules.beta_mod.kwargs.decrease_rate",
    "engine.modules.beta_mod.kwargs.prior_beta_scale",
)


def resolve_simulation_repeats(cfg: Mapping[str, Any]) -> int:
    if "simulation_repeats" not in cfg:
        raise ValueError("Config must include simulation_repeats.")
    repeats = int(cfg["simulation_repeats"])
    if repeats <= 0:
        raise ValueError(f"simulation_repeats must be positive, got {repeats}")
    return repeats


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
    for key, value in fixed_hyperparams.items():
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
    for key, value in fixed_hyperparams.items():
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

    raw_runs = getattr(best, "raw_runs", None)
    raw_runs_ref = None
    if raw_runs:
        raw_runs_ref = _dump_stream(raw_runs, output_dir, subject_id, "raw_runs")
        subject_json_dir = subject_json_dir or (output_dir / "subjects")
        raw_runs_ref = _stream_ref_relative_to(raw_runs_ref, output_dir, subject_json_dir)

    metrics_by_mode = getattr(best, "metrics_by_mode", None) or {}
    selection_meta = result.get("selection_meta", {}) or {}
    selection_mode = str(selection_meta.get("selection_prediction_mode", "posterior_t_minus_1"))
    available_modes = sorted(metrics_by_mode.keys())

    data = {
        "schema_version": 6,
        "result_type": "simulation",
        "subject_id": subject_id,
        "condition": condition,
        "best_params": best_params,
        "fixed_hyperparams": fixed_hyperparams,
        "mean_error": mean_error,
        "best_error": best_error,
        "std_error": std_error,
        "simulation_repeats": int(selection_meta.get("simulation_repeats", getattr(best, "simulation_repeats", 0))),
        "sample_errors": sample_errors,
        "prediction_mode": selection_meta.get("prediction_mode"),
        "selection_prediction_mode": selection_mode,
        "available_prediction_modes": available_modes,
        "metrics_by_mode": metrics_by_mode,
        "best_step_results": getattr(best, "step_results", None),
        "strategy_counts_log": getattr(best, "strategy_counts_log", None),
        "posterior_log": getattr(best, "posterior_log", None),
        "prior_log": getattr(best, "prior_log", None),
        "beta_log": getattr(best, "beta_log", None),
        "representative_run_index": getattr(best, "representative_run_index", 0),
        "selection_meta": selection_meta,
        "loss_metric": selection_meta.get("loss_metric"),
        "loss_delta": selection_meta.get("loss_delta"),
        "hyper_base_seed": selection_meta.get("hyper_base_seed"),
        "hyper_candidate_seed": selection_meta.get("hyper_candidate_seed"),
        "simulation_point_seed": selection_meta.get("simulation_point_seed"),
        "raw_runs_ref": raw_runs_ref,
    }
    return _recursive_to_builtin(data)


def _compact_hyperparams(hyperparams: Mapping[str, Any]) -> Dict[str, Any]:
    summary = dict(hyperparams)
    shortcuts = {
        "engine.modules.memory_mod.kwargs.gamma": "gamma",
        "engine.modules.memory_mod.kwargs.w0": "w0",
        "engine.modules.beta_mod.kwargs.beta_init": "beta_init",
        "engine.modules.beta_mod.kwargs.decrease_rate": "decrease_rate",
        "engine.modules.beta_mod.kwargs.prior_beta_scale": "prior_beta_scale",
        "engine.modules.hypo_transitions_mod.kwargs.init_num": "init_num",
        "engine.modules.hypo_transitions_mod.kwargs.max_active_hypotheses": "max_active_hypotheses",
        "simulation.window_size": "window_size",
    }
    for source, target in shortcuts.items():
        if source in hyperparams:
            summary[target] = hyperparams[source]
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch repeated simulation for fixed StateModel hyperparameters")
    p.add_argument("--config", required=True, type=Path, help="Simulation YAML config")
    p.add_argument("--subjects", nargs="+", type=int, help="Subject IDs; overrides subjects/subject_range in YAML")
    p.add_argument("--subject-range", nargs=2, type=int, metavar=("START", "END"), help="Inclusive subject range")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = args.config
    if not cfg_path.is_absolute():
        cfg_path = (ROOT_DIR / cfg_path).resolve()
    cfg = load_yaml(cfg_path)

    subjects = resolve_subjects(args.subjects, args.subject_range, cfg)

    for sid in subjects:
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
        output_dir = _resolve_path(cfg_path.parent, subject_cfg.get("output_dir"), DEFAULT_OUTPUT_DIR)
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
        window_size = resolve_window_size(subject_cfg, sid, subjects)

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
        )
        result["selection_meta"]["hyper_base_seed"] = hyper_base_seed

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
        print(f"  Saved -> {save_path}")

    print("\nAll subjects done.")


if __name__ == "__main__":
    main()

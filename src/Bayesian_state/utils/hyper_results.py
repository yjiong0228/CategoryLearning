"""Shared helpers for hyperparameter-selection result payloads."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence


def to_builtin(obj: Any) -> Any:
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None

    if np is not None:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
    if isinstance(obj, Mapping):
        return {str(key): to_builtin(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(value) for value in obj]
    return obj


def compact_hyperparams(hyperparams: Mapping[str, Any]) -> dict[str, Any]:
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

    memory_kwargs = hyperparams.get("engine.modules.memory_mod.kwargs")
    if isinstance(memory_kwargs, Mapping):
        if "gamma" in memory_kwargs:
            summary["gamma"] = memory_kwargs["gamma"]
        if "w0" in memory_kwargs:
            summary["w0"] = memory_kwargs["w0"]
    return summary


def _safe_number(value: Any) -> Any:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return value
    if out.is_integer():
        return int(out)
    return out


def compact_metric_summary(
    metrics: Mapping[str, Any] | None,
    *,
    include_sample_errors: bool = False,
) -> dict[str, Any]:
    if not isinstance(metrics, Mapping):
        return {}

    keys = (
        "condition",
        "mean_error",
        "best_error",
        "best10_mean_error",
        "q10_error",
        "selection_error",
        "accuracy_shape_score",
        "accuracy_shape_choice_error",
        "accuracy_shape_repeat_index",
        "accuracy_shape_acc_mae",
        "accuracy_shape_acc_rmse",
        "accuracy_shape_acc_corr",
        "accuracy_shape_vol_ratio",
        "accuracy_shape_range_ratio",
        "accuracy_shape_slope_agree",
        "accuracy_shape_run_choice_cutoff",
        "accuracy_shape_eligible_run_count",
        "accuracy_shape_all_run_count",
        "accuracy_shape_score_mean",
        "accuracy_shape_score_q10",
        "accuracy_shape_eligible_score_mean",
        "lower_tail_fraction",
        "lower_tail_count",
        "std_error",
        "simulation_repeats",
    )
    out = {key: _safe_number(metrics[key]) for key in keys if key in metrics}
    if include_sample_errors and "sample_errors" in metrics:
        out["sample_errors"] = list(metrics.get("sample_errors") or [])
    return out


def combination_metrics_summary(subject_metrics: Mapping[int, Mapping[str, Any]] | None) -> dict[str, Any]:
    if not isinstance(subject_metrics, Mapping) or not subject_metrics:
        return {}
    if len(subject_metrics) == 1:
        metrics = next(iter(subject_metrics.values()))
        return compact_metric_summary(metrics)
    return {
        str(subject_id): compact_metric_summary(metrics)
        for subject_id, metrics in subject_metrics.items()
        if isinstance(metrics, Mapping)
    }


def _clean_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def build_subject_artifacts(
    subject_dir: Path,
    *,
    include_cd: bool = False,
) -> dict[str, str]:
    artifacts = {
        "output_dir": str(subject_dir),
        "all_combinations": str(subject_dir / "all_combinations.jsonl"),
        "stage_summary": str(subject_dir / "stage_summary.json"),
        "best_hyperparams": str(subject_dir / "best_hyperparams.json"),
    }
    if include_cd:
        artifacts["restart_summary"] = str(subject_dir / "restart_summary.json")
        artifacts["coordinate_trace"] = str(subject_dir / "coordinate_trace.jsonl")
    return artifacts


def build_subject_best_payload(
    *,
    subject_id: int,
    backend: str,
    hyper_base_seed: int,
    selection_metric: str,
    best_stage: str,
    best_combination_index: int,
    best_hyperparams: Mapping[str, Any],
    aggregated_error: float,
    hyper_candidate_seed: int,
    metrics: Mapping[str, Any] | None = None,
    search_context: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, Any] | None = None,
    full_subject_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    simulation_summary = compact_metric_summary(metrics, include_sample_errors=True)
    selection_error = (
        simulation_summary.get("selection_error")
        if simulation_summary.get("selection_error") is not None
        else aggregated_error
    )
    payload: dict[str, Any] = {
        "result_type": "hyper_subject_best",
        "subject_id": int(subject_id),
        "hyper": {
            "backend": str(backend),
            "hyper_base_seed": int(hyper_base_seed),
        },
        "selected": {
            "best_hyperparams": deepcopy(dict(best_hyperparams)),
            "best_params": compact_hyperparams(best_hyperparams),
        },
        "selection": {
            "selection_metric": str(selection_metric),
            "best_stage": str(best_stage),
            "best_combination_index": int(best_combination_index),
            "selection_error": float(selection_error),
            "aggregated_error": float(aggregated_error),
            "hyper_candidate_seed": int(hyper_candidate_seed),
        },
        "simulation_summary": simulation_summary,
        "search_context": _clean_mapping(dict(search_context or {})),
        "artifacts": dict(artifacts or {}),
    }
    if full_subject_metrics is not None:
        payload["details"] = {"subject_metrics": full_subject_metrics}
    return to_builtin(payload)


def build_root_best_payload(
    *,
    backend: str,
    config_path: Path,
    output_dir: Path,
    base_sim_config_path: Path,
    hyper_base_seed: int,
    selection_metric: str,
    save_level: str,
    subjects: Sequence[int],
    per_subject_best: Mapping[str, Any],
    per_subject_outputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    backend = str(backend)
    payload = {
        "result_type": "hyper_selection",
        "hyper": {
            "backend": backend,
            "config_path": str(config_path),
            "output_dir": str(output_dir),
            "base_sim_config_path": str(base_sim_config_path),
            "hyper_base_seed": int(hyper_base_seed),
        },
        "selection": {
            "selection_metric": str(selection_metric),
            "save_level": str(save_level),
        },
        "subjects": sorted(int(subject_id) for subject_id in subjects),
        "per_subject_best": dict(
            sorted(((str(key), value) for key, value in per_subject_best.items()), key=lambda item: int(item[0]))
        ),
        "per_subject_outputs": dict(
            sorted(((str(key), value) for key, value in (per_subject_outputs or {}).items()), key=lambda item: int(item[0]))
        ),
    }
    return to_builtin(payload)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def root_hyper(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(payload.get("hyper"))


def root_selection(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(payload.get("selection"))


def root_backend(payload: Mapping[str, Any]) -> Any:
    return root_hyper(payload).get("backend", payload.get("hyper_backend"))


def root_base_sim_config_path(payload: Mapping[str, Any]) -> Any:
    return root_hyper(payload).get("base_sim_config_path", payload.get("base_sim_config_path"))


def root_hyper_base_seed(payload: Mapping[str, Any]) -> Any:
    return root_hyper(payload).get("hyper_base_seed", payload.get("hyper_base_seed"))


def subject_selected(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(payload.get("selected"))


def subject_selection(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(payload.get("selection"))


def subject_best_stage(payload: Mapping[str, Any]) -> Any:
    return subject_selection(payload).get("best_stage", payload.get("best_stage"))


def subject_best_hyperparams(payload: Mapping[str, Any]) -> Any:
    return subject_selected(payload).get("best_hyperparams", payload.get("best_hyperparams"))


def subject_hyper_candidate_seed(payload: Mapping[str, Any]) -> Any:
    return subject_selection(payload).get("hyper_candidate_seed", payload.get("hyper_candidate_seed"))


def subject_hyper_base_seed(payload: Mapping[str, Any]) -> Any:
    return root_hyper(payload).get("hyper_base_seed", payload.get("hyper_base_seed"))


def subject_simulation_summary(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(payload.get("simulation_summary"), Mapping):
        return payload["simulation_summary"]
    keys = (
        "mean_error",
        "best_error",
        "best10_mean_error",
        "q10_error",
        "selection_error",
        "lower_tail_fraction",
        "lower_tail_count",
        "std_error",
        "simulation_repeats",
        "sample_errors",
    )
    return {key: payload[key] for key in keys if key in payload}

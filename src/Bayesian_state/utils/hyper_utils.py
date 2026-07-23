"""Shared helpers for hyperparameter search, result payloads, and serialization."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
from typing import Any, List, Mapping, Sequence


HYPER_RESULT_SCHEMA_VERSION = "hyper_result.v2"
PROFILE_CANDIDATE_KEY = "__profile_candidate__"


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
            value = float(obj)
            return value if math.isfinite(value) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Mapping):
        return {str(key): to_builtin(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(value) for value in obj]
    return obj


def file_sha256(path: Path | str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: Path | None = None) -> str | None:
    cwd = root or Path.cwd()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    value = result.stdout.strip()
    return value or None


def build_hyper_provenance(
    *,
    config_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    base_sim_config_path: Path | str | None = None,
) -> dict[str, Any]:
    config = Path(config_path) if config_path is not None else None
    base_sim = Path(base_sim_config_path) if base_sim_config_path is not None else None
    output = Path(output_dir) if output_dir is not None else None
    git_root = None
    for candidate in (config, base_sim, output):
        if candidate is not None:
            git_root = candidate.parent if candidate.suffix else candidate
            break
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config) if config is not None else None,
        "config_sha256": file_sha256(config),
        "base_sim_config_path": str(base_sim) if base_sim is not None else None,
        "base_sim_config_sha256": file_sha256(base_sim),
        "output_dir": str(output) if output is not None else None,
        "git_commit": git_commit(git_root),
    }


def values_from_json(spec: Mapping[str, Any], config_dir: Path) -> List[Any]:
    source = spec.get("values_from_json")
    if not isinstance(source, Mapping):
        raise ValueError("values_from_json must be a mapping with path, key, and value_key.")

    raw_path = source.get("path")
    if not raw_path:
        raise ValueError("values_from_json.path is required.")
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = (config_dir / path).resolve()
    if not path.is_file():
        raise ValueError(f"values_from_json.path does not exist or is not a file: {path}")

    key = source.get("key")
    if not isinstance(key, str) or not key:
        raise ValueError("values_from_json.key must be a non-empty string.")
    value_key = source.get("value_key")
    if not isinstance(value_key, str) or not value_key:
        raise ValueError("values_from_json.value_key must be a non-empty string.")

    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"values_from_json.path is not valid JSON: {path}") from exc

    if not isinstance(payload, Mapping):
        raise ValueError(f"values_from_json JSON root must be a mapping: {path}")
    if key not in payload:
        raise ValueError(f"values_from_json key '{key}' not found in {path}")
    candidates = payload[key]
    if not isinstance(candidates, list):
        raise ValueError(f"values_from_json key '{key}' must contain a list.")
    if not candidates:
        raise ValueError(f"values_from_json key '{key}' contains an empty list.")

    values = []
    for idx, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise ValueError(f"values_from_json candidate #{idx} under '{key}' must be a mapping.")
        if value_key not in candidate:
            raise ValueError(
                f"values_from_json candidate #{idx} under '{key}' is missing value_key '{value_key}'."
            )
        values.append(deepcopy(candidate[value_key]))
    return values


def values_product(spec: Mapping[str, Any]) -> List[Any]:
    source = spec.get("values_product")
    if not isinstance(source, Mapping):
        raise ValueError("values_product must be a non-empty mapping.")
    if not source:
        raise ValueError("values_product must contain at least one factor.")

    names: List[str] = []
    value_lists: List[List[Any]] = []
    for name, raw_values in source.items():
        if not isinstance(name, str) or not name:
            raise ValueError("values_product factor names must be non-empty strings.")
        if not isinstance(raw_values, list):
            raise ValueError(f"values_product.{name} must be a non-empty list.")
        if not raw_values:
            raise ValueError(f"values_product.{name} cannot be empty.")
        names.append(name)
        value_lists.append([deepcopy(value) for value in raw_values])

    values = [dict(zip(names, combo)) for combo in itertools.product(*value_lists)]
    if not values:
        raise ValueError("values_product produced no values.")
    return values


def validate_no_nested_hyperparam_paths(param_specs: Mapping[str, Any]) -> None:
    names = sorted(str(name) for name in param_specs.keys() if str(name) != PROFILE_CANDIDATE_KEY)
    for idx, left in enumerate(names):
        prefix = f"{left}."
        for right in names[idx + 1:]:
            if right.startswith(prefix):
                raise ValueError(
                    "hyperparam_space cannot contain both a parent path and its child path: "
                    f"'{left}' and '{right}'."
                )


def expand_profile_candidate_hyperparams(hyperparams: Mapping[str, Any]) -> dict[str, Any]:
    expanded: dict[str, Any] = {}
    for key, value in hyperparams.items():
        key_text = str(key)
        if key_text != PROFILE_CANDIDATE_KEY:
            expanded[key_text] = deepcopy(value)
            continue
        if not isinstance(value, Mapping):
            raise ValueError(f"{PROFILE_CANDIDATE_KEY} value must be a mapping of hyperparameter paths.")
        for nested_key, nested_value in value.items():
            if not isinstance(nested_key, str) or not nested_key:
                raise ValueError(f"{PROFILE_CANDIDATE_KEY} nested keys must be non-empty strings.")
            if nested_key == PROFILE_CANDIDATE_KEY:
                raise ValueError(f"{PROFILE_CANDIDATE_KEY} cannot contain itself.")
            expanded[nested_key] = deepcopy(nested_value)
    return expanded


def compact_hyperparams(hyperparams: Mapping[str, Any]) -> dict[str, Any]:
    expanded_hyperparams = expand_profile_candidate_hyperparams(hyperparams)

    summary = dict(expanded_hyperparams)
    shortcuts = {
        "engine.modules.memory_mod.kwargs.gamma": "gamma",
        "engine.modules.memory_mod.kwargs.w0": "w0",
        "engine.modules.beta_mod.kwargs.beta_init": "beta_init",
        "engine.modules.beta_mod.kwargs.decrease_rate": "decrease_rate",
        "engine.modules.beta_mod.kwargs.prior_beta_scale": "prior_beta_scale",
        "engine.modules.beta_mod.kwargs.correct_additive": "correct_additive",
        "engine.modules.beta_mod.kwargs.beta_update_mode": "beta_update_mode",
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
        "simulation.window_size": "window_size",
    }
    for source, target in shortcuts.items():
        if source in expanded_hyperparams:
            summary[target] = expanded_hyperparams[source]

    memory_kwargs = expanded_hyperparams.get("engine.modules.memory_mod.kwargs")
    if isinstance(memory_kwargs, Mapping):
        if "gamma" in memory_kwargs:
            summary["gamma"] = memory_kwargs["gamma"]
        if "w0" in memory_kwargs:
            summary["w0"] = memory_kwargs["w0"]
    transition_kwargs = expanded_hyperparams.get("engine.modules.hypo_transitions_mod.kwargs")
    if isinstance(transition_kwargs, Mapping):
        for source, target in (
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
        ):
            if source in readout:
                summary[target] = readout[source]
    return summary


def _safe_number(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
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

    out: dict[str, Any] = {}
    for key in (
        "condition",
        "dataset_paths",
        "fixed_hyperparams",
        "hyper_candidate_seed",
        "simulation_point_seed",
    ):
        if key in metrics:
            out[key] = deepcopy(metrics[key])

    simulation = deepcopy(metrics.get("simulation") or {})
    if isinstance(simulation, Mapping):
        if not include_sample_errors:
            simulation.pop("sample_errors", None)
        out["simulation"] = dict(simulation)

    statistics = metrics.get("statistics")
    if isinstance(statistics, Mapping):
        out["statistics"] = deepcopy(dict(statistics))

    objectives = metrics.get("objectives")
    if isinstance(objectives, Mapping):
        out["objectives"] = deepcopy(dict(objectives))

    selection = metrics.get("selection")
    if isinstance(selection, Mapping):
        if isinstance(selection.get("primary"), Mapping):
            out["selection"] = deepcopy(dict(selection))
        else:
            out["selection"] = {
                "primary": {
                    "metric": selection.get("metric"),
                    "value": selection.get("value"),
                }
            }
    return out


def _objective_values(metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    objectives = metrics.get("objectives")
    if isinstance(objectives, Mapping):
        values = objectives.get("values")
        if isinstance(values, Mapping):
            return values
    return {}


def _selection_primary_value(metrics: Mapping[str, Any]) -> Any:
    selection = metrics.get("selection")
    if not isinstance(selection, Mapping):
        return None
    primary = selection.get("primary")
    if isinstance(primary, Mapping):
        return primary.get("value")
    return selection.get("value")


def combination_metrics_summary(
    subject_metrics: Mapping[int, Mapping[str, Any]] | None,
    *,
    aggregated_error: float | None = None,
    objective_values: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(subject_metrics, Mapping) or not subject_metrics:
        return {}
    subject_summaries = {
        str(subject_id): compact_metric_summary(metrics)
        for subject_id, metrics in subject_metrics.items()
        if isinstance(metrics, Mapping)
    }
    if objective_values is None:
        values = []
        for metrics in subject_metrics.values():
            if not isinstance(metrics, Mapping):
                continue
            value = _selection_primary_value(metrics)
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                values.append(numeric)
        aggregate_value = aggregated_error
        if aggregate_value is None and values:
            aggregate_value = sum(values) / len(values)
        return {
            "aggregate": {
                "subject_count": int(len(subject_summaries)),
                "selection": {
                    "value": aggregate_value,
                },
            },
            "subjects": subject_summaries,
        }

    aggregate_value = aggregated_error
    if aggregate_value is None:
        first_value = next(iter(objective_values.values()), None)
        try:
            aggregate_value = float(first_value)
        except (TypeError, ValueError):
            aggregate_value = None
    return {
        "aggregate": {
            "subject_count": int(len(subject_summaries)),
            "objectives": {
                "values": deepcopy(dict(objective_values or {})),
            },
            "selection": {
                "method": "objective_order",
                "value": aggregate_value,
            },
        },
        "subjects": subject_summaries,
    }


def _clean_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _final_selection_metric(final_context: Mapping[str, Any], primary_metric: str) -> str:
    if not final_context.get("enabled", False):
        return primary_metric
    mode = str(final_context.get("mode") or "")
    by_mode = {
        "accuracy_shape": "statistics.scores.accuracy_shape.value",
        "history_kernel": "statistics.scores.history_kernel.value",
        "switch_behavior": "statistics.scores.switch_behavior.value",
        "distribution_ppc_interval": "statistics.scores.distribution.ppc_interval.score",
        "distribution_intersection": "statistics.scores.distribution.intersection.score",
        "distribution_multiobjective": "selection.rank.distribution_minimax",
        "multiobjective": "selection.rank.multiobjective",
    }
    return by_mode.get(mode, primary_metric)


def build_subject_artifacts(
    subject_dir: Path,
    *,
    include_cd: bool = False,
    include_accepted: bool = False,
) -> dict[str, str]:
    artifacts = {
        "output_dir": str(subject_dir),
        "all_combinations": str(subject_dir / "all_combinations.jsonl"),
        "stage_summary": str(subject_dir / "stage_summary.json"),
        "best_hyperparams": str(subject_dir / "best_hyperparams.json"),
    }
    if include_accepted:
        artifacts["accepted_hyperparams"] = str(subject_dir / "accepted_hyperparams.jsonl")
    if include_cd:
        artifacts["restart_summary"] = str(subject_dir / "restart_summary.json")
        artifacts["coordinate_trace"] = str(subject_dir / "coordinate_trace.jsonl")
    return artifacts


def build_subject_best_payload(
    *,
    subject_id: int,
    backend: str,
    hyper_base_seed: int,
    best_stage: str,
    best_combination_index: int,
    best_hyperparams: Mapping[str, Any],
    aggregated_error: float,
    hyper_candidate_seed: int,
    selection_metric: str | None = None,
    objective_order: Sequence[Mapping[str, Any]] | None = None,
    objective_values: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    search_context: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, Any] | None = None,
    full_subject_metrics: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metric_summary = compact_metric_summary(metrics, include_sample_errors=True)
    simulation = deepcopy(metric_summary.get("simulation") or {})
    statistics = deepcopy(metric_summary.get("statistics") or {})
    final_context = {}
    if isinstance(search_context, Mapping):
        maybe_final = search_context.get("final_selection")
        if isinstance(maybe_final, Mapping):
            final_context = dict(maybe_final)

    using_objectives = objective_order is not None or objective_values is not None
    if using_objectives:
        objective_payload = {
            "order": deepcopy(list(objective_order or [])),
            "values": deepcopy(dict(objective_values or {})),
        }
        metadata = {
            key: deepcopy(value)
            for key, value in metric_summary.items()
            if key not in {"simulation", "statistics", "selection", "objectives"}
        }
        payload: dict[str, Any] = {
            "schema_version": HYPER_RESULT_SCHEMA_VERSION,
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
                "method": final_context.get("selected_by", "objective_order"),
                "value": aggregated_error,
                "objectives": objective_payload,
                "candidate": {
                    "stage": str(best_stage),
                    "combination_index": int(best_combination_index),
                    "aggregated_error": float(aggregated_error),
                    "hyper_candidate_seed": int(hyper_candidate_seed),
                },
            },
            "simulation": simulation,
            "statistics": statistics,
            "objectives": objective_payload,
            "metadata": metadata,
            "provenance": dict(provenance or {}),
            "search_context": _clean_mapping(dict(search_context or {})),
            "artifacts": dict(artifacts or {}),
        }
        if full_subject_metrics is not None:
            payload["details"] = {"subject_metrics": full_subject_metrics}
        return to_builtin(payload)

    if selection_metric is None:
        raise ValueError("selection_metric is required when objective_order is not provided.")
    metric_selection = deepcopy(metric_summary.get("selection") or {})
    primary_selection = deepcopy(metric_selection.get("primary") or {})
    if not primary_selection:
        primary_selection = {
            "metric": str(selection_metric),
            "value": aggregated_error,
        }
    selection_value = (
        primary_selection.get("value")
        if primary_selection.get("value") is not None
        else aggregated_error
    )
    final_selection = {
        "method": final_context.get("selected_by", "primary_selection_metric"),
        "metric": final_context.get(
            "final_metric",
            _final_selection_metric(final_context, str(selection_metric)),
        ),
        "value": final_context.get(
            "final_value",
            final_context.get("selected_secondary_score", selection_value),
        ),
    }
    if not final_context.get("enabled", False):
        final_selection["metric"] = str(selection_metric)
        final_selection["value"] = selection_value
    metadata = {
        key: deepcopy(value)
        for key, value in metric_summary.items()
        if key not in {"simulation", "statistics", "selection"}
    }
    payload: dict[str, Any] = {
        "schema_version": HYPER_RESULT_SCHEMA_VERSION,
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
            "primary": {
                "metric": str(primary_selection.get("metric") or selection_metric),
                "value": selection_value,
            },
            "final": final_selection,
            "candidate": {
                "stage": str(best_stage),
                "combination_index": int(best_combination_index),
                "aggregated_error": float(aggregated_error),
                "hyper_candidate_seed": int(hyper_candidate_seed),
            },
        },
        "simulation": simulation,
        "statistics": statistics,
        "metadata": metadata,
        "provenance": dict(provenance or {}),
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
    selection_metric: str | None = None,
    objective_order: Sequence[Mapping[str, Any]] | None = None,
    save_level: str,
    subjects: Sequence[int],
    per_subject_best: Mapping[str, Any],
    per_subject_outputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    backend = str(backend)
    payload = {
        "schema_version": HYPER_RESULT_SCHEMA_VERSION,
        "result_type": "hyper_selection",
        "hyper": {
            "backend": backend,
            "config_path": str(config_path),
            "output_dir": str(output_dir),
            "base_sim_config_path": str(base_sim_config_path),
            "hyper_base_seed": int(hyper_base_seed),
        },
        "selection": {
            "save_level": str(save_level),
        },
        "provenance": build_hyper_provenance(
            config_path=config_path,
            output_dir=output_dir,
            base_sim_config_path=base_sim_config_path,
        ),
        "subjects": sorted(int(subject_id) for subject_id in subjects),
        "per_subject_best": dict(
            sorted(((str(key), value) for key, value in per_subject_best.items()), key=lambda item: int(item[0]))
        ),
        "per_subject_outputs": dict(
            sorted(((str(key), value) for key, value in (per_subject_outputs or {}).items()), key=lambda item: int(item[0]))
        ),
    }
    if objective_order is not None:
        payload["selection"].update(
            {
                "method": "objective_order",
                "objectives": {
                    "order": deepcopy(list(objective_order)),
                },
            }
        )
    elif selection_metric is not None:
        payload["selection"]["metric"] = str(selection_metric)
    else:
        raise ValueError("Either selection_metric or objective_order is required.")
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


def subject_selection_candidate(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(subject_selection(payload).get("candidate"))


def subject_best_stage(payload: Mapping[str, Any]) -> Any:
    return subject_selection_candidate(payload).get(
        "stage",
        subject_selection(payload).get("best_stage", payload.get("best_stage")),
    )


def subject_best_hyperparams(payload: Mapping[str, Any]) -> Any:
    return subject_selected(payload).get("best_hyperparams", payload.get("best_hyperparams"))


def subject_hyper_candidate_seed(payload: Mapping[str, Any]) -> Any:
    return subject_selection_candidate(payload).get(
        "hyper_candidate_seed",
        subject_selection(payload).get("hyper_candidate_seed", payload.get("hyper_candidate_seed")),
    )


def subject_hyper_base_seed(payload: Mapping[str, Any]) -> Any:
    return root_hyper(payload).get("hyper_base_seed", payload.get("hyper_base_seed"))


def subject_simulation_summary(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(payload.get("simulation"), Mapping):
        return payload["simulation"]
    if isinstance(payload.get("simulation_summary"), Mapping):
        return payload["simulation_summary"]
    return {}

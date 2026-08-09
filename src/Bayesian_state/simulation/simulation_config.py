"""Simulation configuration loading, validation, and parameter expansion."""
from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import yaml

from src.Bayesian_state.metrics.losses import (
    LOSS_METRIC_BERHU,
    LOSS_METRIC_CHOICES,
)
from src.Bayesian_state.simulation.state_model_execution import (
    PREDICTION_MODE_CHOICES,
    PREDICTION_MODE_POSTERIOR_T_MINUS_1,
)
from src.Bayesian_state.utils.stream import StreamList
from src.Bayesian_state.utils.paths import (
    TASK2_PROCESSED_PATH,
    SIMULATION_RESULTS_DIR,
)
from src.Bayesian_state.utils.config_subjects import (
    deep_update,
    subject_override_for,
    without_subject_overrides,
)

DEFAULT_DATA_PATH = TASK2_PROCESSED_PATH
DEFAULT_OUTPUT_DIR = SIMULATION_RESULTS_DIR
PROFILE_CANDIDATE_KEY = "__profile_candidate__"
EVALUATION_ROLE_OPTIMIZATION = "optimization"
EVALUATION_ROLE_SIMULATION = "simulation"
EVALUATION_ROLES = (
    EVALUATION_ROLE_OPTIMIZATION,
    EVALUATION_ROLE_SIMULATION,
)


# Parameter payloads
def expand_profile_candidate_hyperparams(
    hyperparams: Mapping[str, Any],
) -> dict[str, Any]:
    """Expand a packed profile candidate into ordinary parameter paths."""

    expanded: dict[str, Any] = {}
    for key, value in hyperparams.items():
        key_text = str(key)
        if key_text != PROFILE_CANDIDATE_KEY:
            expanded[key_text] = deepcopy(value)
            continue
        if not isinstance(value, Mapping):
            raise ValueError(
                f"{PROFILE_CANDIDATE_KEY} value must be a mapping of hyperparameter paths."
            )
        for nested_key, nested_value in value.items():
            if not isinstance(nested_key, str) or not nested_key:
                raise ValueError(
                    f"{PROFILE_CANDIDATE_KEY} nested keys must be non-empty strings."
                )
            if nested_key == PROFILE_CANDIDATE_KEY:
                raise ValueError(f"{PROFILE_CANDIDATE_KEY} cannot contain itself.")
            expanded[nested_key] = deepcopy(nested_value)
    return expanded


# YAML and simulation settings
def resolve_simulation_repeats(cfg: Mapping[str, Any]) -> int:
    """Validate the repeated-run budget shared by simulation and optimizers."""

    if "simulation_repeats" not in cfg:
        raise ValueError("Config must include simulation_repeats.")
    repeats = int(cfg["simulation_repeats"])
    if repeats <= 0:
        raise ValueError(f"simulation_repeats must be positive, got {repeats}")
    return repeats


def resolve_evaluation_score_mask(
    n_trials: int,
    protocol: Mapping[str, Any] | None,
    *,
    role: str,
) -> tuple[np.ndarray | None, Dict[str, Any]]:
    """Resolve the trials scored for optimization or frozen simulation.

    A sequential holdout always executes the complete observed trial sequence.
    The returned mask changes only which trials contribute to metrics/loss, so
    evaluation-trial predictions remain causally conditioned on the preceding
    observed history.
    """
    total = int(n_trials)
    if total <= 0:
        raise ValueError(f"n_trials must be positive, got {n_trials!r}")
    role_name = str(role).strip().lower()
    if role_name not in EVALUATION_ROLES:
        raise ValueError(f"evaluation role must be one of {EVALUATION_ROLES}, got {role!r}")

    if protocol is None:
        return None, {
            "enabled": False,
            "mode": "all",
            "role": role_name,
            "partition": "all",
            "n_trials": total,
            "split_index": None,
            "train_trial_count": total,
            "evaluation_trial_count": total,
            "score_trial_count": total,
        }
    if not isinstance(protocol, Mapping):
        raise ValueError("evaluation_protocol must be a mapping when provided")

    raw = dict(protocol)
    enabled = bool(raw.get("enabled", True))
    mode = str(raw.get("mode", "sequential_holdout" if enabled else "all")).strip().lower()
    if not enabled or mode in {"all", "full", "disabled", "none"}:
        return None, {
            "enabled": False,
            "mode": "all",
            "role": role_name,
            "partition": "all",
            "n_trials": total,
            "split_index": None,
            "train_trial_count": total,
            "evaluation_trial_count": total,
            "score_trial_count": total,
        }
    if mode not in {"sequential_holdout", "prefix_suffix", "temporal_holdout"}:
        raise ValueError(
            "evaluation_protocol.mode must be 'sequential_holdout' or 'all'"
        )

    train_trials_raw = raw.get("train_trials")
    train_fraction_raw = raw.get("train_fraction")
    if train_trials_raw is not None and train_fraction_raw is not None:
        raise ValueError(
            "evaluation_protocol must set only one of train_trials or train_fraction"
        )
    if train_trials_raw is None and train_fraction_raw is None:
        raise ValueError(
            "sequential_holdout requires evaluation_protocol.train_trials or train_fraction"
        )
    if train_trials_raw is not None:
        split_index = int(train_trials_raw)
        if float(train_trials_raw) != float(split_index):
            raise ValueError("evaluation_protocol.train_trials must be an integer")
        requested_fraction = None
    else:
        requested_fraction = float(train_fraction_raw)
        if not np.isfinite(requested_fraction) or not 0.0 < requested_fraction < 1.0:
            raise ValueError("evaluation_protocol.train_fraction must lie in (0, 1)")
        split_index = int(np.floor(total * requested_fraction))

    min_train = int(raw.get("min_train_trials", 1))
    min_evaluation = int(raw.get("min_evaluation_trials", 1))
    if min_train <= 0 or min_evaluation <= 0:
        raise ValueError(
            "evaluation_protocol min_train_trials and min_evaluation_trials must be positive"
        )
    if split_index < min_train or total - split_index < min_evaluation:
        raise ValueError(
            "sequential holdout cannot satisfy minimum partition sizes: "
            f"n_trials={total}, split_index={split_index}, "
            f"min_train_trials={min_train}, min_evaluation_trials={min_evaluation}"
        )

    partition_key = f"{role_name}_partition"
    default_partition = "train" if role_name == EVALUATION_ROLE_OPTIMIZATION else "evaluation"
    partition = str(raw.get(partition_key, default_partition)).strip().lower()
    aliases = {
        "fit": "train",
        "training": "train",
        "eval": "evaluation",
        "test": "evaluation",
        "holdout": "evaluation",
        "full": "all",
    }
    partition = aliases.get(partition, partition)
    if partition not in {"train", "evaluation", "all"}:
        raise ValueError(
            f"evaluation_protocol.{partition_key} must be train, evaluation, or all"
        )

    mask = np.ones(total, dtype=bool)
    if partition == "train":
        mask[split_index:] = False
    elif partition == "evaluation":
        mask[:split_index] = False

    context = {
        "enabled": True,
        "mode": "sequential_holdout",
        "role": role_name,
        "partition": partition,
        "n_trials": total,
        "split_index": int(split_index),
        "train_trial_count": int(split_index),
        "evaluation_trial_count": int(total - split_index),
        "score_trial_count": int(np.sum(mask)),
        "requested_train_fraction": requested_fraction,
        "realized_train_fraction": float(split_index / total),
    }
    return mask, context


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML must be a mapping: {path}")
    return data


def save_json(obj: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(recursive_to_builtin(obj), f, ensure_ascii=False, indent=2)


def resolve_path(base: Path, maybe_path: Any, default: Path) -> Path:
    if maybe_path is None:
        return default
    p = Path(maybe_path)
    if not p.is_absolute():
        p = (base / p).resolve()
    return p


def resolve_engine_config(
    cfg: Mapping[str, Any],
    yaml_dir: Path,
    subject_id: int | None = None,
) -> Dict[str, Any]:
    inline_cfg = cfg.get("engine_config")
    path_cfg = cfg.get("engine_config_path")

    if inline_cfg is not None and not isinstance(inline_cfg, dict):
        raise ValueError("engine_config must be a mapping when provided")

    base_cfg: Dict[str, Any] = {}
    if path_cfg:
        engine_path = Path(path_cfg)
        if not engine_path.is_absolute():
            engine_path = (yaml_dir / engine_path).resolve()
        loaded = load_yaml(engine_path)
        if not isinstance(loaded, dict):
            raise ValueError(f"Engine config must be a mapping: {engine_path}")
        base_cfg = loaded

    if inline_cfg is None and not path_cfg:
        raise ValueError("Config must provide engine_config or engine_config_path")
    resolved = base_cfg if inline_cfg is None else deep_update(base_cfg, inline_cfg)
    if subject_id is None:
        return without_subject_overrides(resolved)

    subject_override = subject_override_for(resolved, subject_id)
    return deep_update(without_subject_overrides(resolved), subject_override)


def resolve_prediction_modes(cfg: Mapping[str, Any]) -> tuple[str, str]:
    prediction_mode = str(cfg.get("prediction_mode", PREDICTION_MODE_POSTERIOR_T_MINUS_1))
    selection_prediction_mode = str(cfg.get("selection_prediction_mode", PREDICTION_MODE_POSTERIOR_T_MINUS_1))
    if prediction_mode not in PREDICTION_MODE_CHOICES:
        raise ValueError(f"Unsupported prediction_mode '{prediction_mode}'. Valid: {PREDICTION_MODE_CHOICES}")
    if selection_prediction_mode not in (
        PREDICTION_MODE_POSTERIOR_T_MINUS_1,
        "prior_t",
    ):
        raise ValueError("selection_prediction_mode must be 'posterior_t_minus_1' or 'prior_t'")
    if prediction_mode != "both" and selection_prediction_mode != prediction_mode:
        raise ValueError(
            "When prediction_mode is not 'both', selection_prediction_mode must equal prediction_mode."
        )
    return prediction_mode, selection_prediction_mode


def resolve_loss_metric(cfg: Mapping[str, Any]) -> str:
    raw = cfg.get("loss_metric")
    if raw is None:
        raise ValueError(f"Config must include loss_metric. Valid: {LOSS_METRIC_CHOICES}")
    metric = str(raw).strip().lower()
    if metric not in LOSS_METRIC_CHOICES:
        raise ValueError(f"Unsupported loss_metric '{metric}'. Valid: {LOSS_METRIC_CHOICES}")
    return metric


def resolve_loss_delta(cfg: Mapping[str, Any], loss_metric: str) -> float | None:
    raw = cfg.get("loss_delta")
    if str(loss_metric).strip().lower() == LOSS_METRIC_BERHU:
        if raw is None:
            raise ValueError("Config must include loss_delta when loss_metric='accuracy_curve_berhu'")
        delta = float(raw)
        if delta <= 0:
            raise ValueError(f"loss_delta must be > 0 when loss_metric='accuracy_curve_berhu', got {delta}")
        return delta
    return None


def resolve_window_size(cfg: Mapping[str, Any], subject_id: int, subjects: Sequence[int]) -> int:
    raw_ws = cfg.get("window_size", 16)
    overrides = {int(k): int(v) for k, v in (cfg.get("window_size_overrides") or {}).items()}
    if subject_id in overrides:
        return overrides[subject_id]
    if isinstance(raw_ws, (list, tuple)):
        ws_list = [int(x) for x in raw_ws]
        if len(ws_list) != len(subjects):
            raise ValueError("window_size list length must match subjects list length")
        return dict(zip(subjects, ws_list))[subject_id]
    return int(raw_ws)


def resolve_subjects(args_subjects: Sequence[int] | None, args_subject_range: Sequence[int] | None, cfg: Mapping[str, Any]) -> list[int]:
    if args_subjects is not None:
        return [int(x) for x in args_subjects]
    if args_subject_range is not None:
        start, end = map(int, args_subject_range)
        return list(range(start, end + 1))
    subjects = cfg.get("subjects")
    if subjects is None:
        range_cfg = cfg.get("subject_range")
        if not (isinstance(range_cfg, (list, tuple)) and len(range_cfg) == 2):
            raise ValueError("Config must provide subjects/subject_range, or pass --subjects/--subject-range")
        start, end = map(int, range_cfg)
        return list(range(start, end + 1))
    return [int(x) for x in subjects]


def recursive_to_builtin(obj: Any) -> Any:
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (list, tuple)):
        return [recursive_to_builtin(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): recursive_to_builtin(v) for k, v in obj.items()}
    return obj


def dump_stream(items: Sequence[Any] | None, output_dir: Path, subject_id: int, tag: str) -> Dict[str, Any] | None:
    if not items:
        return None
    rel_path = Path("cache") / f"subject_{subject_id}_{tag}.gz"
    abs_path = output_dir / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    if abs_path.exists():
        abs_path.unlink()

    stream = StreamList(str(abs_path), 0)
    stream.extend(items)
    return {
        "format": "stream-gzip-pickle",
        "path": rel_path.as_posix(),
        "count": len(stream),
    }


def stream_ref_relative_to(ref: Dict[str, Any] | None, output_dir: Path, ref_base_dir: Path) -> Dict[str, Any] | None:
    if not ref or "path" not in ref:
        return ref
    adjusted = dict(ref)
    abs_path = (output_dir / str(ref["path"])).resolve()
    adjusted["path"] = os.path.relpath(abs_path, ref_base_dir.resolve())
    return adjusted


__all__ = [
    "DEFAULT_DATA_PATH",
    "DEFAULT_OUTPUT_DIR",
    "EVALUATION_ROLE_OPTIMIZATION",
    "EVALUATION_ROLE_SIMULATION",
    "EVALUATION_ROLES",
    "PROFILE_CANDIDATE_KEY",
    "dump_stream",
    "expand_profile_candidate_hyperparams",
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

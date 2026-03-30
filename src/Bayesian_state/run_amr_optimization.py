"""Batch AMR optimization over subjects (single YAML config)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import pandas as pd
import yaml
from joblib import Parallel, delayed

from src.Bayesian_state.utils.optimizer_amr import StateModelAMROptimizer  # noqa: E402
from src.Bayesian_state.utils.paths import (
    ROOT_DIR,
    TASK2_PROCESSED_PATH,
    AMR_RESULTS_DIR,
)

DEFAULT_DATA_PATH = TASK2_PROCESSED_PATH
DEFAULT_OUTPUT_DIR = AMR_RESULTS_DIR

# Per-process cache to avoid loading the same CSV repeatedly in worker tasks.
_LEARNING_DATA_CACHE: dict[str, pd.DataFrame] = {}


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_path(base: Path, maybe_path: Any, default: Path | None = None) -> Path:
    if maybe_path is None:
        if default is None:
            raise ValueError("path is required")
        return default
    p = Path(maybe_path)
    if not p.is_absolute():
        p = (base / p).resolve()
    return p


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def resolve_engine_config(opt_cfg: Dict[str, Any], yaml_dir: Path) -> Dict[str, Any]:
    inline_cfg = opt_cfg.get("engine_config")
    path_cfg = opt_cfg.get("engine_config_path")

    if inline_cfg is not None and not isinstance(inline_cfg, dict):
        raise ValueError("engine_config must be a mapping when provided")

    base_cfg: Dict[str, Any] = {}
    if path_cfg:
        engine_path = _resolve_path(yaml_dir, path_cfg)
        loaded = load_yaml(engine_path)
        if not isinstance(loaded, dict):
            raise ValueError(f"Engine config must be a mapping: {engine_path}")
        base_cfg = loaded

    if inline_cfg is None and not path_cfg:
        raise ValueError("opt-config must provide engine_config or engine_config_path")
    if inline_cfg is None:
        return base_cfg
    return _deep_update(base_cfg, inline_cfg)


def resolve_param_grid(opt_cfg: Dict[str, Any]) -> Dict[str, Sequence[Any]]:
    pg = opt_cfg.get("param_grid")
    if pg is None:
        raise ValueError("opt-config must include param_grid (mapping name -> list)")
    if not isinstance(pg, dict):
        raise ValueError("param_grid must be a mapping")
    return {k: list(v) for k, v in pg.items()}


def resolve_amr_kwargs(opt_cfg: Dict[str, Any]) -> Dict[str, Any]:
    if "amr_kwargs" in opt_cfg and isinstance(opt_cfg["amr_kwargs"], dict):
        return dict(opt_cfg["amr_kwargs"])
    return {
        "max_evals": 50,
        "coarse_grid_per_dim": 3,
        "split_factor": 2,
        "refine_top_k": 3,
    }


def _recursive_to_builtin(obj: Any) -> Any:
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (list, tuple)):
        return [_recursive_to_builtin(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _recursive_to_builtin(v) for k, v in obj.items()}
    return obj


def serialize_result(subject_id: int, condition: int, result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert optimizer output to JSON-serializable dict."""
    best = result["best"]
    data = {
        "subject_id": subject_id,
        "condition": condition,
        "best_params": best.params,
        "mean_error": best.mean_error,
        "std_error": getattr(best, "std_error", 0.0),
        "n_repeats": getattr(best, "n_repeats", 1),
        "metrics": best.metrics,
        "param_grid": result.get("param_grid", {}),
        "best_step_results": getattr(best, "step_results", None),
        "strategy_counts_log": getattr(best, "strategy_counts_log", None),
        "posterior_log": getattr(best, "posterior_log", None),
        "prior_log": getattr(best, "prior_log", None),
    }
    return _recursive_to_builtin(data)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _get_learning_data(data_path: Path) -> pd.DataFrame:
    key = str(data_path.resolve())
    if key not in _LEARNING_DATA_CACHE:
        _LEARNING_DATA_CACHE[key] = pd.read_csv(data_path)
    return _LEARNING_DATA_CACHE[key]


def run_single_subject(
    subject_id: int,
    engine_config: Dict[str, Any],
    param_grid: Dict[str, Sequence[Any]],
    amr_kwargs: Dict[str, Any],
    data_path: Path,
    output_dir: Path,
    n_repeats: int,
    refit_repeats: int,
    window_size: int,
    stop_at: float,
    max_trials: int | None,
    n_jobs_inner: int,
    keep_logs: bool,
) -> None:
    opt = StateModelAMROptimizer(
        engine_config=engine_config,
        processed_data_dir=data_path.parent,
        amr_kwargs=amr_kwargs,
        n_jobs=n_jobs_inner,
    )
    # Avoid repeated CSV I/O for every subject task inside worker processes.
    opt.learning_data = _get_learning_data(data_path)

    res = opt.optimize_subject(
        subject_id=subject_id,
        param_grid=param_grid,
        n_repeats=n_repeats,
        refit_repeats=refit_repeats,
        window_size=window_size,
        stop_at=stop_at,
        max_trials=max_trials,
        keep_logs=keep_logs,
    )

    payload = serialize_result(subject_id, res["condition"], res)
    save_path = output_dir / f"subject_{subject_id}.json"
    save_json(payload, save_path)
    print(f"Saved subject {subject_id} result to {save_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch AMR optimization (single YAML config)")
    # Keep --opt-config as backward-compatible alias; unify UX to --config.
    p.add_argument("--config", "--opt-config", dest="config", required=True, type=Path, help="Optimization YAML config")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = args.config
    if not cfg_path.is_absolute():
        cfg_path = (ROOT_DIR / cfg_path).resolve()
    cfg = load_yaml(cfg_path)

    engine_config = resolve_engine_config(cfg, cfg_path.parent)
    param_grid = resolve_param_grid(cfg)
    amr_kwargs = resolve_amr_kwargs(cfg)

    # Subjects
    subjects = cfg.get("subjects")
    if subjects is None:
        range_cfg = cfg.get("subject_range")
        if not (isinstance(range_cfg, (list, tuple)) and len(range_cfg) == 2):
            raise ValueError("opt-config must provide subjects: [...] or subject_range: [start, end]")
        start, end = map(int, range_cfg)
        subjects = list(range(start, end + 1))
    else:
        subjects = [int(x) for x in subjects]

    # Paths and runtime args
    data_path = _resolve_path(cfg_path.parent, cfg.get("data_path"), DEFAULT_DATA_PATH)
    output_dir = _resolve_path(cfg_path.parent, cfg.get("output_dir"), DEFAULT_OUTPUT_DIR)
    n_jobs_subjects = int(cfg.get("n_jobs_subjects", 2))
    n_jobs_inner = int(cfg.get("n_jobs_inner", 4))
    n_repeats = int(cfg.get("n_repeats", 4))
    refit_repeats = int(cfg.get("refit_repeats", 8))
    raw_window_size = cfg.get("window_size", 16)
    overrides_raw = cfg.get("window_size_overrides") or {}
    window_size_overrides = {int(k): int(v) for k, v in overrides_raw.items()}
    stop_at = float(cfg.get("stop_at", 1.0))
    max_trials_val = cfg.get("max_trials")
    max_trials = int(max_trials_val) if max_trials_val is not None else None
    keep_logs = bool(cfg.get("keep_logs", False))

    output_dir.mkdir(parents=True, exist_ok=True)

    # Window size (scalar or per-subject list + per-subject overrides)
    if isinstance(raw_window_size, (list, tuple)):
        window_size_list = [int(x) for x in raw_window_size]
        if len(window_size_list) != len(subjects):
            raise ValueError("window_size list length must match number of subjects")
        window_size_map = {sid: window_size_list[idx] for idx, sid in enumerate(subjects)}

        def resolve_window_size(sid: int) -> int:
            return window_size_overrides.get(sid, window_size_map[sid])
    else:
        default_window_size = int(raw_window_size)

        def resolve_window_size(sid: int) -> int:
            return window_size_overrides.get(sid, default_window_size)

    Parallel(n_jobs=n_jobs_subjects)(
        delayed(run_single_subject)(
            subject_id=sid,
            engine_config=engine_config,
            param_grid=param_grid,
            amr_kwargs=amr_kwargs,
            data_path=data_path,
            output_dir=output_dir,
            n_repeats=n_repeats,
            refit_repeats=refit_repeats,
            window_size=resolve_window_size(sid),
            stop_at=stop_at,
            max_trials=max_trials,
            n_jobs_inner=n_jobs_inner,
            keep_logs=keep_logs,
        )
        for sid in subjects
    )


if __name__ == "__main__":
    main()

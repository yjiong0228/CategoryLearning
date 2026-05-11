"""Batch grid-search optimization over subjects (driven by a single YAML config).

Usage:
    python -m src.Bayesian_state.run_grid_optimization \
        --config configs/grid_opt_cfg/pmh_cond1.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import yaml

from src.Bayesian_state.utils.optimizer_grid import StateModelGridOptimizer  # noqa: E402
from src.Bayesian_state.utils.optimizer_common import (
    PREDICTION_MODE_CHOICES,
    PREDICTION_MODE_POSTERIOR_T_MINUS_1,
)
from src.Bayesian_state.utils.stream import StreamList
from src.Bayesian_state.utils.paths import (
    ROOT_DIR,
    TASK2_PROCESSED_PATH,
    GRID_RESULTS_DIR,
)
from src.Bayesian_state.utils.datasets import resolve_dataset_paths
from src.Bayesian_state.utils.config_subjects import (
    deep_update,
    resolve_subject_config,
    subject_override_for,
    without_subject_overrides,
)

DEFAULT_DATA_PATH = TASK2_PROCESSED_PATH
DEFAULT_OUTPUT_DIR = GRID_RESULTS_DIR


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_path(base: Path, maybe_path: Any, default: Path) -> Path:
    if maybe_path is None:
        return default
    p = Path(maybe_path)
    if not p.is_absolute():
        p = (base / p).resolve()
    return p


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    return deep_update(base, override)


def resolve_engine_config(
    cfg: Dict[str, Any],
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
    resolved = base_cfg if inline_cfg is None else _deep_update(base_cfg, inline_cfg)
    if subject_id is None:
        return without_subject_overrides(resolved)

    subject_override = subject_override_for(resolved, subject_id)
    return _deep_update(without_subject_overrides(resolved), subject_override)


def resolve_param_grid(cfg: Dict[str, Any]) -> Dict[str, Sequence[Any]]:
    pg = cfg.get("param_grid")
    if pg is None or not isinstance(pg, dict):
        raise ValueError("Config must include param_grid (mapping name -> list)")
    return {k: list(v) for k, v in pg.items()}


def resolve_prediction_modes(cfg: Dict[str, Any]) -> tuple[str, str]:
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


def resolve_window_size(cfg: Dict[str, Any], subject_id: int, subjects: Sequence[int]) -> int:
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


def _dump_stream(items: Sequence[Any] | None, output_dir: Path, subject_id: int, tag: str) -> Dict[str, Any] | None:
    if not items:
        return None
    rel_path = Path("cache") / f"subject_{subject_id}_{tag}.gz"
    abs_path = output_dir / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)

    # StreamList appends to existing files, so truncate first when rewriting a
    # subject result. Otherwise reruns silently mix old and new fit logs.
    if abs_path.exists():
        abs_path.unlink()

    stream = StreamList(str(abs_path), 0)
    stream.extend(items)
    return {
        "format": "stream-gzip-pickle",
        "path": rel_path.as_posix(),
        "count": len(stream),
    }


def _build_grid_errors(result: Dict[str, Any]) -> list[Dict[str, Any]]:
    records: list[Dict[str, Any]] = []
    selection_mode = str(result.get("selection_meta", {}).get("selection_prediction_mode", "posterior_t_minus_1"))
    for gp in result.get("grid", []) or []:
        records.append(
            {
                "params": gp.params,
                "errors": list(getattr(gp, "sample_errors", []) or []),
                "mean_error": gp.mean_error,
                "std_error": gp.std_error,
                "best_error": getattr(gp, "best_error", gp.mean_error),
                "selection_prediction_mode": selection_mode,
            }
        )
    return records


def serialize_result(subject_id: int, condition: int, result: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    best = result["best"]
    best_error = float(getattr(best, "best_error", best.mean_error))
    refit_mean_error = float(getattr(best, "refit_mean_error", best.mean_error))
    refit_std_error = float(getattr(best, "refit_std_error", best.std_error))
    sample_errors = list(getattr(best, "sample_errors", []) or [])
    raw_runs = getattr(best, "raw_runs", None)
    if not raw_runs:
        raise ValueError(
            f"No run-level records available for subject {subject_id}. "
            "Enable keep_logs to store per-run objects."
        )
    raw_runs_ref = _dump_stream(raw_runs, output_dir, subject_id, "raw_runs")

    metrics_by_mode = getattr(best, "metrics_by_mode", None) or {}
    selection_mode = str(result.get("selection_meta", {}).get("selection_prediction_mode", "posterior_t_minus_1"))
    available_modes = sorted(metrics_by_mode.keys())

    data = {
        "schema_version": 4,
        "subject_id": subject_id,
        "condition": condition,
        "best_params": best.params,
        "mean_error": best_error,
        "best_error": best_error,
        "refit_mean_error": refit_mean_error,
        "std_error": refit_std_error,
        "refit_std_error": refit_std_error,
        "n_repeats": getattr(best, "n_repeats", 1),
        "sample_errors": sample_errors,
        "prediction_mode": result.get("selection_meta", {}).get("prediction_mode"),
        "selection_prediction_mode": selection_mode,
        "available_prediction_modes": available_modes,
        "metrics_by_mode": metrics_by_mode,
        "param_grid": result.get("param_grid", {}),
        "best_step_results": getattr(best, "step_results", None),
        "strategy_counts_log": getattr(best, "strategy_counts_log", None),
        "posterior_log": getattr(best, "posterior_log", None),
        "prior_log": getattr(best, "prior_log", None),
        "representative_run_index": getattr(best, "representative_run_index", 0),
        "selection_meta": result.get("selection_meta", {}),
        "raw_runs_ref": raw_runs_ref,
        "grid_errors": _build_grid_errors(result),
        "grid_summary": [
            {
                "params": gp.params,
                "mean_error": gp.mean_error,
                "std_error": gp.std_error,
                "best_error": getattr(gp, "best_error", gp.mean_error),
            }
            for gp in result.get("grid", [])
        ],
    }
    return _recursive_to_builtin(data)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def resolve_subjects(args: argparse.Namespace, cfg: Dict[str, Any]) -> list[int]:
    if args.subjects is not None:
        return [int(x) for x in args.subjects]

    if args.subject_range is not None:
        start, end = map(int, args.subject_range)
        return list(range(start, end + 1))

    subjects = cfg.get("subjects")
    if subjects is None:
        range_cfg = cfg.get("subject_range")
        if not (isinstance(range_cfg, (list, tuple)) and len(range_cfg) == 2):
            raise ValueError("Config must provide subjects/subject_range, or pass --subjects/--subject-range")
        start, end = map(int, range_cfg)
        return list(range(start, end + 1))

    return [int(x) for x in subjects]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch grid-search optimization (single YAML config)")
    p.add_argument("--config", required=True, type=Path, help="Grid optimization YAML config")
    p.add_argument("--subjects", nargs="+", type=int, help="Subject IDs; overrides subjects/subject_range in YAML")
    p.add_argument("--subject-range", nargs=2, type=int, metavar=("START", "END"), help="Inclusive subject range; overrides YAML when --subjects is not provided")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = args.config
    if not cfg_path.is_absolute():
        cfg_path = (ROOT_DIR / cfg_path).resolve()
    cfg = load_yaml(cfg_path)

    subjects = resolve_subjects(args, cfg)

    for sid in subjects:
        subject_cfg = resolve_subject_config(cfg, sid)
        engine_config = resolve_engine_config(subject_cfg, cfg_path.parent, subject_id=sid)
        param_grid = resolve_param_grid(subject_cfg)
        prediction_mode, selection_prediction_mode = resolve_prediction_modes(subject_cfg)

        dataset_paths = resolve_dataset_paths(subject_cfg, cfg_path.parent, DEFAULT_DATA_PATH)
        data_path = dataset_paths["learning_data"]
        output_dir = _resolve_path(cfg_path.parent, subject_cfg.get("output_dir"), DEFAULT_OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        n_jobs = int(subject_cfg.get("n_jobs", 4))
        n_repeats = int(subject_cfg.get("n_repeats", 4))
        refit_repeats = int(subject_cfg.get("refit_repeats", 64))
        stop_at = float(subject_cfg.get("stop_at", 1.0))
        max_trials_val = subject_cfg.get("max_trials")
        max_trials = int(max_trials_val) if max_trials_val is not None else None
        keep_logs = bool(subject_cfg.get("keep_logs", False))
        window_size = resolve_window_size(subject_cfg, sid, subjects)

        optimizer = StateModelGridOptimizer(
            engine_config=engine_config,
            processed_data_dir=dataset_paths["processed_dir"],
            dataset_paths=dataset_paths,
            n_jobs=n_jobs,
        )
        optimizer.prepare_data(data_path)

        print(f"\n{'='*60}")
        print(f"Subject {sid}")
        print(f"{'='*60}")

        result: Dict[str, Any] = optimizer.optimize_subject(
            subject_id=sid,
            param_grid=param_grid,
            n_repeats=n_repeats,
            refit_repeats=refit_repeats,
            window_size=window_size,
            stop_at=stop_at,
            max_trials=max_trials,
            keep_logs=keep_logs,
            prediction_mode=prediction_mode,
            selection_prediction_mode=selection_prediction_mode,
        )

        best: Any = result["best"]
        print(f"  Best params: {best.params}")
        best_error = float(getattr(best, "best_error", best.mean_error))
        refit_mean = float(getattr(best, "refit_mean_error", best.mean_error))
        refit_std = float(getattr(best, "refit_std_error", best.std_error))
        print(f"  Best run error ({selection_prediction_mode}): {best_error:.6f}")
        print(f"  Refit mean ({selection_prediction_mode}):     {refit_mean:.6f} +/- {refit_std:.6f}")

        payload = serialize_result(sid, int(result["condition"]), result, output_dir)
        save_path = output_dir / f"subject_{sid}.json"
        save_json(payload, save_path)
        print(f"  Saved -> {save_path}")

    print("\nAll subjects done.")


if __name__ == "__main__":
    main()

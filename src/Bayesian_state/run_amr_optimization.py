"""Batch AMR optimization over subjects (single YAML config)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Sequence

import pandas as pd
import yaml
from joblib import Parallel, delayed

from src.Bayesian_state.utils.optimizer_amr import StateModelAMROptimizer  # noqa: E402
from src.Bayesian_state.utils.optimizer_common import (
    PREDICTION_MODE_CHOICES,
    PREDICTION_MODE_POSTERIOR_T_MINUS_1,
    LOSS_METRIC_BERHU,
    LOSS_METRIC_CHOICES,
)
from src.Bayesian_state.utils.stream import StreamList
from src.Bayesian_state.utils.paths import (
    ROOT_DIR,
    TASK2_PROCESSED_PATH,
    AMR_RESULTS_DIR,
)
from src.Bayesian_state.utils.datasets import resolve_dataset_paths
from src.Bayesian_state.utils.config_subjects import (
    deep_update,
    resolve_subject_config,
    subject_override_for,
    without_subject_overrides,
)

DEFAULT_DATA_PATH = TASK2_PROCESSED_PATH
DEFAULT_OUTPUT_DIR = AMR_RESULTS_DIR

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
    return deep_update(base, override)


def resolve_engine_config(
    opt_cfg: Dict[str, Any],
    yaml_dir: Path,
    subject_id: int | None = None,
) -> Dict[str, Any]:
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
    resolved = base_cfg if inline_cfg is None else _deep_update(base_cfg, inline_cfg)
    if subject_id is None:
        return without_subject_overrides(resolved)

    subject_override = subject_override_for(resolved, subject_id)
    return _deep_update(without_subject_overrides(resolved), subject_override)


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


def resolve_prediction_modes(cfg: Dict[str, Any]) -> tuple[str, str]:
    prediction_mode = str(cfg.get("prediction_mode", PREDICTION_MODE_POSTERIOR_T_MINUS_1))
    selection_prediction_mode = str(cfg.get("selection_prediction_mode", PREDICTION_MODE_POSTERIOR_T_MINUS_1))
    if prediction_mode not in PREDICTION_MODE_CHOICES:
        raise ValueError(f"Unsupported prediction_mode '{prediction_mode}'. Valid: {PREDICTION_MODE_CHOICES}")
    if selection_prediction_mode not in ("posterior_t_minus_1", "prior_t"):
        raise ValueError("selection_prediction_mode must be 'posterior_t_minus_1' or 'prior_t'")
    if prediction_mode != "both" and selection_prediction_mode != prediction_mode:
        raise ValueError(
            "When prediction_mode is not 'both', selection_prediction_mode must equal prediction_mode."
        )
    return prediction_mode, selection_prediction_mode


def resolve_loss_metric(cfg: Dict[str, Any]) -> str:
    raw = cfg.get("loss_metric")
    if raw is None:
        raise ValueError(f"Config must include loss_metric. Valid: {LOSS_METRIC_CHOICES}")
    metric = str(raw).strip().lower()
    if metric not in LOSS_METRIC_CHOICES:
        raise ValueError(f"Unsupported loss_metric '{metric}'. Valid: {LOSS_METRIC_CHOICES}")
    return metric


def resolve_loss_delta(cfg: Dict[str, Any], loss_metric: str) -> float | None:
    raw = cfg.get("loss_delta")
    if str(loss_metric).strip().lower() == LOSS_METRIC_BERHU:
        if raw is None:
            raise ValueError("Config must include loss_delta when loss_metric='accuracy_curve_berhu'")
        delta = float(raw)
        if delta <= 0:
            raise ValueError(f"loss_delta must be > 0 when loss_metric='accuracy_curve_berhu', got {delta}")
        return delta
    return None


def resolve_window_size(cfg: Dict[str, Any], subject_id: int, subjects: Sequence[int]) -> int:
    raw_ws = cfg.get("window_size", 16)
    overrides = {int(k): int(v) for k, v in (cfg.get("window_size_overrides") or {}).items()}
    if subject_id in overrides:
        return overrides[subject_id]
    if isinstance(raw_ws, (list, tuple)):
        ws_list = [int(x) for x in raw_ws]
        if len(ws_list) != len(subjects):
            raise ValueError("window_size list length must match number of subjects")
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

    stream = StreamList(str(abs_path), 0)
    stream.extend(items)
    return {
        "format": "stream-gzip-pickle",
        "path": rel_path.as_posix(),
        "count": len(stream),
    }


def _stream_ref_relative_to(ref: Dict[str, Any] | None, output_dir: Path, ref_base_dir: Path) -> Dict[str, Any] | None:
    if not ref or "path" not in ref:
        return ref
    adjusted = dict(ref)
    abs_path = (output_dir / str(ref["path"])).resolve()
    adjusted["path"] = os.path.relpath(abs_path, ref_base_dir.resolve())
    return adjusted


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


def serialize_result(
    subject_id: int,
    condition: int,
    result: Dict[str, Any],
    output_dir: Path,
    subject_json_dir: Path | None = None,
) -> Dict[str, Any]:
    best = result["best"]
    best_error = float(getattr(best, "best_error", best.mean_error))
    refit_mean_error = float(getattr(best, "refit_mean_error", best.mean_error))
    refit_std_error = float(getattr(best, "refit_std_error", best.std_error))
    sample_errors = list(getattr(best, "sample_errors", []) or [])
    raw_step_ref = _dump_stream(getattr(best, "raw_step_results", None), output_dir, subject_id, "raw_step_results")
    subject_json_dir = subject_json_dir or (output_dir / "subjects")
    raw_step_ref = _stream_ref_relative_to(raw_step_ref, output_dir, subject_json_dir)

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
        "grid_repeats": getattr(best, "grid_repeats", 1),
        "refit_repeats": getattr(best, "refit_repeats", 0),
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
        "beta_log": getattr(best, "beta_log", None),
        "representative_run_index": getattr(best, "representative_run_index", 0),
        "selection_meta": result.get("selection_meta", {}),
        "loss_metric": result.get("selection_meta", {}).get("loss_metric"),
        "loss_delta": result.get("selection_meta", {}).get("loss_delta"),
        "raw_step_results_ref": raw_step_ref,
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
    dataset_paths: Dict[str, Path],
    output_dir: Path,
    grid_repeats: int,
    refit_repeats: int,
    window_size: int,
    stop_at: float,
    max_trials: int | None,
    n_jobs_inner: int,
    keep_logs: bool,
    prediction_mode: str,
    selection_prediction_mode: str,
    loss_metric: str,
    loss_delta: float | None,
    random_seed: int | None,
) -> None:
    opt = StateModelAMROptimizer(
        engine_config=engine_config,
        processed_data_dir=str(dataset_paths["processed_dir"]),
        dataset_paths=dataset_paths,
        amr_kwargs=amr_kwargs,
        n_jobs=n_jobs_inner,
    )
    opt.learning_data = _get_learning_data(data_path)

    res: Dict[str, Any] = opt.optimize_subject(
        subject_id=subject_id,
        param_grid=param_grid,
        grid_repeats=grid_repeats,
        refit_repeats=refit_repeats,
        window_size=window_size,
        stop_at=stop_at,
        max_trials=max_trials,
        keep_logs=keep_logs,
        prediction_mode=prediction_mode,
        selection_prediction_mode=selection_prediction_mode,
        loss_metric=loss_metric,
        loss_delta=loss_delta,
        random_seed=random_seed,
    )

    subjects_dir = output_dir / "subjects"
    payload = serialize_result(
        subject_id,
        int(res["condition"]),
        res,
        output_dir,
        subject_json_dir=subjects_dir,
    )
    save_path = subjects_dir / f"subject_{subject_id}.json"
    save_json(payload, save_path)
    print(f"Saved subject {subject_id} result to {save_path}")


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
            raise ValueError("opt-config must provide subjects/subject_range, or pass --subjects/--subject-range")
        start, end = map(int, range_cfg)
        return list(range(start, end + 1))

    return [int(x) for x in subjects]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch AMR optimization (single YAML config)")
    p.add_argument("--config", "--opt-config", dest="config", required=True, type=Path, help="Optimization YAML config")
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

    base_n_jobs_subjects = int(cfg.get("n_jobs_subjects", 2))

    jobs = []
    for sid in subjects:
        subject_cfg = resolve_subject_config(cfg, sid)
        engine_config = resolve_engine_config(subject_cfg, cfg_path.parent, subject_id=sid)
        param_grid = resolve_param_grid(subject_cfg)
        amr_kwargs = resolve_amr_kwargs(subject_cfg)
        prediction_mode, selection_prediction_mode = resolve_prediction_modes(subject_cfg)
        loss_metric = resolve_loss_metric(subject_cfg)
        loss_delta = resolve_loss_delta(subject_cfg, loss_metric)

        dataset_paths = resolve_dataset_paths(subject_cfg, cfg_path.parent, DEFAULT_DATA_PATH)
        data_path = dataset_paths["learning_data"]
        output_dir = _resolve_path(cfg_path.parent, subject_cfg.get("output_dir"), DEFAULT_OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        max_trials_val = subject_cfg.get("max_trials")
        if "grid_repeats" not in subject_cfg:
            raise ValueError("Config must include grid_repeats. The old n_repeats field is no longer supported.")
        jobs.append(dict(
            subject_id=sid,
            engine_config=engine_config,
            param_grid=param_grid,
            amr_kwargs=amr_kwargs,
            data_path=data_path,
            dataset_paths=dataset_paths,
            output_dir=output_dir,
            grid_repeats=int(subject_cfg["grid_repeats"]),
            refit_repeats=int(subject_cfg.get("refit_repeats", 8)),
            window_size=resolve_window_size(subject_cfg, sid, subjects),
            stop_at=float(subject_cfg.get("stop_at", 1.0)),
            max_trials=int(max_trials_val) if max_trials_val is not None else None,
            n_jobs_inner=int(subject_cfg.get("n_jobs_inner", 4)),
            keep_logs=bool(subject_cfg.get("keep_logs", False)),
            prediction_mode=prediction_mode,
            selection_prediction_mode=selection_prediction_mode,
            loss_metric=loss_metric,
            loss_delta=loss_delta,
            random_seed=subject_cfg.get("random_seed"),
        ))

    Parallel(n_jobs=base_n_jobs_subjects)(
        delayed(run_single_subject)(**job)
        for job in jobs
    )


if __name__ == "__main__":
    main()

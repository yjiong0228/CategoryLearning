"""Batch grid-search optimization over subjects (driven by a single YAML config).

Usage:
    python -m src.Bayesian_state.run_grid_optimization \
        --config configs/grid_opt_cfg/pmh_cond1.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import yaml

def _add_project_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root

PROJECT_ROOT = _add_project_root()

from src.Bayesian_state.utils.state_grid_optimizer import StateModelGridOptimizer  # noqa: E402

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "Task2_processed.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "state-based-grid-result"


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------
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


def resolve_engine_config(cfg: Dict[str, Any], yaml_dir: Path) -> Dict[str, Any]:
    if "engine_config" in cfg and isinstance(cfg["engine_config"], dict):
        return cfg["engine_config"]
    engine_path = cfg.get("engine_config_path")
    if not engine_path:
        raise ValueError("Config must provide engine_config or engine_config_path")
    engine_path = Path(engine_path)
    if not engine_path.is_absolute():
        engine_path = (yaml_dir / engine_path).resolve()
    return load_yaml(engine_path)


def resolve_param_grid(cfg: Dict[str, Any]) -> Dict[str, Sequence[Any]]:
    pg = cfg.get("param_grid")
    if pg is None or not isinstance(pg, dict):
        raise ValueError("Config must include param_grid (mapping name -> list)")
    return {k: list(v) for k, v in pg.items()}


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------
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
        "sample_errors": getattr(best, "sample_errors", None),
        "metrics": best.metrics,
        "param_grid": result.get("param_grid", {}),
        "best_step_results": getattr(best, "step_results", None),
        "strategy_counts_log": getattr(best, "strategy_counts_log", None),
        "posterior_log": getattr(best, "posterior_log", None),
        "prior_log": getattr(best, "prior_log", None),
        # grid summary (compact)
        "grid_summary": [
            {"params": gp.params, "mean_error": gp.mean_error, "std_error": gp.std_error}
            for gp in result.get("grid", [])
        ],
    }
    return _recursive_to_builtin(data)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch grid-search optimization (single YAML config)")
    p.add_argument("--config", required=True, type=Path, help="Grid optimization YAML config")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = args.config
    if not cfg_path.is_absolute():
        cfg_path = (PROJECT_ROOT / cfg_path).resolve()
    cfg = load_yaml(cfg_path)

    engine_config = resolve_engine_config(cfg, cfg_path.parent)
    param_grid = resolve_param_grid(cfg)

    # Subjects
    subjects = cfg.get("subjects")
    if subjects is None:
        sr = cfg.get("subject_range")
        if not (isinstance(sr, (list, tuple)) and len(sr) == 2):
            raise ValueError("Config needs subjects: [...] or subject_range: [start, end]")
        subjects = list(range(int(sr[0]), int(sr[1]) + 1))
    else:
        subjects = [int(x) for x in subjects]

    # Paths
    data_path = _resolve_path(cfg_path.parent, cfg.get("data_path"), DEFAULT_DATA_PATH)
    output_dir = _resolve_path(cfg_path.parent, cfg.get("output_dir"), DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Optimizer settings
    n_jobs = int(cfg.get("n_jobs", 4))
    n_repeats = int(cfg.get("n_repeats", 4))
    refit_repeats = int(cfg.get("refit_repeats", 64))
    stop_at = float(cfg.get("stop_at", 1.0))
    max_trials_val = cfg.get("max_trials")
    max_trials = int(max_trials_val) if max_trials_val is not None else None
    keep_logs = bool(cfg.get("keep_logs", False))

    # Window size: scalar or per-subject list
    raw_ws = cfg.get("window_size", 16)
    overrides = {int(k): int(v) for k, v in (cfg.get("window_size_overrides") or {}).items()}

    if isinstance(raw_ws, (list, tuple)):
        ws_list = [int(x) for x in raw_ws]
        if len(ws_list) != len(subjects):
            raise ValueError("window_size list length must match subjects list length")
        ws_map = dict(zip(subjects, ws_list))
    else:
        ws_map = {sid: int(raw_ws) for sid in subjects}

    def get_ws(sid: int) -> int:
        return overrides.get(sid, ws_map.get(sid, 16))

    # ---- Run grid search for each subject sequentially (inner parallelism) ----
    optimizer = StateModelGridOptimizer(
        engine_config=engine_config,
        processed_data_dir=data_path.parent,
        n_jobs=n_jobs,
    )
    optimizer.prepare_data(data_path)

    for sid in subjects:
        print(f"\n{'='*60}")
        print(f"Subject {sid}")
        print(f"{'='*60}")

        result = optimizer.optimize_subject(
            subject_id=sid,
            param_grid=param_grid,
            n_repeats=n_repeats,
            refit_repeats=refit_repeats,
            window_size=get_ws(sid),
            stop_at=stop_at,
            max_trials=max_trials,
            keep_logs=keep_logs,
        )

        best = result["best"]
        print(f"  Best params: {best.params}")
        print(f"  Mean error:  {best.mean_error:.6f} ± {best.std_error:.6f}")

        payload = serialize_result(sid, result["condition"], result)
        save_path = output_dir / f"subject_{sid}.json"
        save_json(payload, save_path)
        print(f"  Saved → {save_path}")

    print(f"\nAll subjects done. Results in {output_dir}")


if __name__ == "__main__":
    main()

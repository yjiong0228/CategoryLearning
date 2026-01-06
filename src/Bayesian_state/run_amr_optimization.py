"""Batch AMR optimization over subjects (driven by a single YAML config).

Usage example:

    python -m src.Bayesian_state.run_amr_optimization \
        --opt-config configs/amr_opt_cfg/pmh_amr_example.yaml

All runtime参数（模型YAML、被试列表、参数网格、AMR超参、并行设置等）都从一个YAML读取，
无需再传一堆 CLI 参数。可在 YAML 中覆盖 param_grid / amr_kwargs，否则用默认。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import yaml
from joblib import Parallel, delayed


def _add_project_root() -> Path:
	"""Ensure project root is on sys.path."""
	root = Path(__file__).resolve().parents[2]
	if str(root) not in sys.path:
		sys.path.insert(0, str(root))
	return root


PROJECT_ROOT = _add_project_root()

from src.Bayesian_state.utils.state_amr_optimizer import StateModelAMROptimizer  # noqa: E402


DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "Task2_processed.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "state-based-AMR-result"


def load_yaml(path: Path) -> Dict[str, Any]:
	with path.open("r", encoding="utf-8") as f:
		return yaml.safe_load(f) or {}


def resolve_engine_config(opt_cfg: Dict[str, Any]) -> Dict[str, Any]:
	# Prefer inline engine_config; otherwise load via engine_config_path
	if "engine_config" in opt_cfg and isinstance(opt_cfg["engine_config"], dict):
		return opt_cfg["engine_config"]

	engine_path = opt_cfg.get("engine_config_path")
	if not engine_path:
		raise ValueError("opt-config must provide engine_config or engine_config_path")
	engine_path = Path(engine_path)
	if not engine_path.is_absolute():
		engine_path = (PROJECT_ROOT / engine_path).resolve()
	engine_cfg = load_yaml(engine_path)
	if not isinstance(engine_cfg, dict):
		raise ValueError(f"Engine config must be a mapping: {engine_path}")
	return engine_cfg


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


def serialize_result(subject_id: int, condition: int, result: Dict[str, Any]) -> Dict[str, Any]:
	"""Convert optimizer output to JSON-serializable dict."""
	best = result["best"]
	metrics = best.metrics

	def _tolist(x):
		if x is None:
			return None
		if isinstance(x, (list, tuple)):
			return list(x)
		try:
			import numpy as np  # local import to keep top clean

			if isinstance(x, np.ndarray):
				return x.tolist()
		except Exception:
			pass
		return x

	return {
		"subject_id": subject_id,
		"condition": condition,
		"best_params": best.params,
		"mean_error": best.mean_error,
		"std_error": getattr(best, "std_error", 0.0),
		"n_repeats": getattr(best, "n_repeats", 1),
		"metrics": {k: _tolist(v) for k, v in metrics.items()},
		"param_grid": result.get("param_grid", {}),
	}


def save_json(obj: Dict[str, Any], path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as f:
		json.dump(obj, f, ensure_ascii=False, indent=2)


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
	opt.prepare_data(data_path)
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
	p = argparse.ArgumentParser(description="Batch AMR optimization over subjects (single YAML config)")
	p.add_argument("--opt-config", required=True, type=Path, help="Full optimization yaml (engine config path + param_grid + subjects etc.)")
	return p.parse_args()


def _resolve_path(base: Path, maybe_path: Any, default: Path | None = None) -> Path:
	if maybe_path is None:
		if default is None:
			raise ValueError("path is required")
		return default
	p = Path(maybe_path)
	if not p.is_absolute():
		p = (base / p).resolve()
	return p


def main() -> None:
	args = parse_args()
	op_cfg_path = args.opt_config
	if not op_cfg_path.is_absolute():
		op_cfg_path = (PROJECT_ROOT / op_cfg_path).resolve()
	op_cfg = load_yaml(op_cfg_path)

	engine_config = resolve_engine_config(op_cfg)
	param_grid = resolve_param_grid(op_cfg)
	amr_kwargs = resolve_amr_kwargs(op_cfg)

	# Subjects list
	subjects = op_cfg.get("subjects")
	if subjects is None:
		range_cfg = op_cfg.get("subject_range")
		if not (isinstance(range_cfg, (list, tuple)) and len(range_cfg) == 2):
			raise ValueError("opt-config must provide subjects: [...] or subject_range: [start, end]")
		start, end = map(int, range_cfg)
		subjects = list(range(start, end + 1))
	else:
		subjects = [int(x) for x in subjects]

	data_path = _resolve_path(op_cfg_path.parent, op_cfg.get("data_path", DEFAULT_DATA_PATH), DEFAULT_DATA_PATH)
	output_dir = _resolve_path(op_cfg_path.parent, op_cfg.get("output_dir", DEFAULT_OUTPUT_DIR), DEFAULT_OUTPUT_DIR)
	n_jobs_subjects = int(op_cfg.get("n_jobs_subjects", 2))
	n_jobs_inner = int(op_cfg.get("n_jobs_inner", 4))
	n_repeats = int(op_cfg.get("n_repeats", 4))
	refit_repeats = int(op_cfg.get("refit_repeats", 8))
	window_size = int(op_cfg.get("window_size", 16))
	stop_at = float(op_cfg.get("stop_at", 1.0))
	max_trials_val = op_cfg.get("max_trials", None)
	max_trials = int(max_trials_val) if max_trials_val is not None else None
	keep_logs = bool(op_cfg.get("keep_logs", False))

	output_dir.mkdir(parents=True, exist_ok=True)

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
			window_size=window_size,
			stop_at=stop_at,
			max_trials=max_trials,
			n_jobs_inner=n_jobs_inner,
			keep_logs=keep_logs,
		)
		for sid in subjects
	)


if __name__ == "__main__":
	main()

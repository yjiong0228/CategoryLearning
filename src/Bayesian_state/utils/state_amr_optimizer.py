"""Adaptive mesh refinement optimizer for StateModel parameters.

This module mirrors StateModelGridOptimizer but replaces the exhaustive
grid enumeration with the AMRGridSearch heuristic defined in amr_optimizer.py.
It keeps the same input/output contract so callers can switch optimizers
without changing downstream code.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any
from itertools import product

from joblib import Parallel, delayed

import numpy as np
import pandas as pd

from src.Bayesian_state import problems

from .amr_optimizer import AMRGridSearch
from .state_grid_optimizer import GridPointResult
from ..utils.base import LOGGER


# ---------------------------------------------------------------------------
# Shared helpers (copied to avoid tight coupling to state_grid_optimizer internals)
# ---------------------------------------------------------------------------
def _prepare_trial_sequence(
	stimulus: np.ndarray,
	choices: np.ndarray,
	feedback: np.ndarray,
) -> List[List[float]]:
	trials: List[List[float]] = []
	for stim, choice, fb in zip(stimulus, choices, feedback):
		trial: List[float] = [stim, int(choice), float(fb)]
		trials.append(trial)
	return trials

from ..problems.model import StateModel
def _compute_prediction_metrics(
	model: StateModel,
	post_log: Sequence[np.ndarray],
	stimulus: np.ndarray,
	choices: np.ndarray,
	feedback: np.ndarray,
	categories: np.ndarray,
	window_size: int,
) -> Dict[str, np.ndarray | float]:
	# Memory weighting is assumed to be embedded in the prior trajectory
	# produced during fitting.
	partition = model.partition_model
	hypotheses = list(model.hypotheses_set)

	# Try to find beta in likelihood_mod, default to 10.0
	beta_param = 10.0
	if hasattr(model.engine, "likelihood_mod"):
		lik_mod = getattr(model.engine, "likelihood_mod")
		beta_param = float(lik_mod.kwargs.get("beta", 10.0))

	post_arr = np.asarray(post_log, dtype=float)
	if post_arr.ndim == 1:
		post_arr = post_arr.reshape(1, -1)

	n_trials = len(feedback)
	if post_arr.shape[0] != n_trials:
		raise ValueError(
			"Post log length does not match number of trials: "
			f"{post_arr.shape[0]} vs {n_trials}"
		)

	true_acc = (feedback == 1.0).astype(float)
	pred_acc = np.full(n_trials, np.nan, dtype=float)

	for trial_idx in range(n_trials):
		current_post = post_arr[trial_idx]
		weighted_prob = 0.0

		trial_slice = (
			[stimulus[trial_idx]],
			[choices[trial_idx]],
			[feedback[trial_idx]],
			[categories[trial_idx]],
		)

		for weight, hypo in zip(current_post, hypotheses):
			if weight <= 0:
				continue
			lik = partition.calc_trueprob_entry(
				hypo,
				trial_slice,
				beta_param,
				use_cached_dist=False,
			)
			weighted_prob += weight * float(np.ravel(lik)[0])

		pred_acc[trial_idx] = weighted_prob

	sliding_true_acc: List[float] = []
	sliding_pred_acc: List[float] = []
	sliding_pred_std: List[float] = []

	for start in range(1, n_trials - window_size + 2):
		end = start + window_size
		true_window = true_acc[start:end]
		pred_window = pred_acc[start:end]

		sliding_true_acc.append(float(np.mean(true_window)))
		sliding_pred_acc.append(float(np.nanmean(pred_window)))

		valid = pred_window[~np.isnan(pred_window)]
		if valid.size == 0:
			sliding_pred_std.append(np.nan)
		else:
			sliding_pred_std.append(
				float(np.sqrt(np.sum(valid * (1 - valid))) / window_size)
			)

	error = np.abs(np.array(sliding_true_acc) - np.array(sliding_pred_acc))
	mean_error = float(np.nanmean(error)) if error.size else float("nan")

	return {
		"true_acc": true_acc,
		"pred_acc": pred_acc,
		"sliding_true_acc": np.asarray(sliding_true_acc, dtype=float),
		"sliding_pred_acc": np.asarray(sliding_pred_acc, dtype=float),
		"sliding_pred_acc_std": np.asarray(sliding_pred_std, dtype=float),
		"mean_error": mean_error,
	}


def _inject_params(config: Dict, params: Dict[str, Any]) -> None:
	"""
	Inject parameters into the configuration dictionary.
	Supports dot notation (e.g. 'modules.memory_mod.kwargs.gamma')
	and shortcuts for common parameters.
	"""
	shortcuts = {
		"gamma": "modules.memory_mod.kwargs.gamma",
		"w0": "modules.memory_mod.kwargs.w0",
		"beta": "modules.likelihood_mod.kwargs.beta",
	}

	def set_by_path(root: Dict, path: str, value: Any):
		parts = path.split(".")
		curr = root
		for part in parts[:-1]:
			curr = curr.setdefault(part, {})
		curr[parts[-1]] = value

	for key, value in params.items():
		path = shortcuts.get(key, key)
		set_by_path(config, path, value)


def _evaluate_single_run(
	subject_id: int,
	condition: int,
	arrays: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
	params: Dict[str, Any],
	engine_config_template: Dict,
	processed_data_dir: Path,
	window_size: int,
) -> Tuple[Dict[str, Any], float, Dict[str, Any], Sequence[np.ndarray], Sequence[np.ndarray]]:
	stimulus, choices, feedback, categories = arrays
	trial_sequence = _prepare_trial_sequence(stimulus, choices, feedback)

	engine_config = deepcopy(engine_config_template)
	_inject_params(engine_config, params)

	from ..problems import StateModel

	model = StateModel(
		engine_config,
		condition=condition,
		subject_id=subject_id,
		processed_data_dir=processed_data_dir,
	)

	model.precompute_distances(stimulus)
	posterior_log, prior_log = model.fit_step_by_step(trial_sequence)

	metrics = _compute_prediction_metrics(
		model,
		posterior_log,
		stimulus,
		choices,
		feedback,
		categories,
		window_size,
	)

	return params, float(metrics["mean_error"]), metrics, posterior_log, prior_log


# ---------------------------------------------------------------------------
# AMR optimizer
# ---------------------------------------------------------------------------
class StateModelAMROptimizer:
	"""
	Adaptive-mesh replacement for StateModelGridOptimizer.

	Input and output follow the grid optimizer: optimize_subject returns
	a dict with keys {"subject_id", "condition", "best", "grid", "param_grid"}.
	"""

	def __init__(
		self,
		engine_config: Dict,
		processed_data_dir: Optional[Path | str] = None,
		n_jobs: int = 1,
		amr_kwargs: Optional[Dict[str, Any]] = None,
	) -> None:
		self._engine_config_template = deepcopy(engine_config)
		self._processed_data_dir = (
			Path(processed_data_dir).resolve()
			if processed_data_dir is not None
			else Path(__file__).resolve().parents[3] / "data" / "processed"
		)
		self.learning_data: Optional[pd.DataFrame] = None
		self.n_jobs = n_jobs
		self._amr_kwargs = amr_kwargs or {}

	def prepare_data(self, data_path: Path | str) -> None:
		data_path = Path(data_path).resolve()
		if not data_path.exists():
			raise FileNotFoundError(f"Dataset not found: {data_path}")
		self.learning_data = pd.read_csv(data_path)

	def _get_subject_frame(self, subject_id: int, stop_at: float) -> pd.DataFrame:
		if self.learning_data is None:
			# Lazy-load default
			default_path = Path(__file__).resolve().parents[3] / "data" / "processed" / "Task2_processed.csv"
			self.prepare_data(default_path)
		assert self.learning_data is not None

		subject_frame = self.learning_data[self.learning_data["iSub"] == subject_id]
		if subject_frame.empty:
			raise ValueError(f"Subject {subject_id} not found in dataset")

		stop_index = max(1, int(len(subject_frame) * stop_at + 0.5))
		return subject_frame.iloc[:stop_index].copy()

	def _extract_arrays(
		self,
		subject_frame: pd.DataFrame,
		max_trials: Optional[int],
	) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		stimulus = subject_frame[["feature1", "feature2", "feature3", "feature4"]].to_numpy(dtype=float)
		choices = subject_frame["choice"].to_numpy(dtype=int)
		feedback = subject_frame["feedback"].to_numpy(dtype=float)
		categories = subject_frame["category"].to_numpy(dtype=int)

		if max_trials is not None:
			usable = min(max_trials, stimulus.shape[0])
			stimulus = stimulus[:usable]
			choices = choices[:usable]
			feedback = feedback[:usable]
			categories = categories[:usable]

		return stimulus, choices, feedback, categories

	def _evaluate_params(
		self,
		subject_id: int,
		condition: int,
		arrays: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
		params: Dict[str, Any],
		window_size: int,
		n_repeats: int,
		keep_logs: bool,
	) -> GridPointResult:
		if n_repeats <= 1:
			runs = [
				_evaluate_single_run(
					subject_id,
					condition,
					arrays,
					params,
					self._engine_config_template,
					self._processed_data_dir,
					window_size,
				)
			]
		else:
			runs = Parallel(n_jobs=self.n_jobs)(
				delayed(_evaluate_single_run)(
					subject_id,
					condition,
					arrays,
					params,
					self._engine_config_template,
					self._processed_data_dir,
					window_size,
				)
				for _ in range(n_repeats)
			)

		errors = [r[1] for r in runs]
		mean_error = float(np.mean(errors))
		std_error = float(np.std(errors)) if len(errors) > 1 else 0.0

		best_idx = int(np.argmin([abs(e - mean_error) for e in errors]))
		_, _, metrics, posterior_log, prior_log = runs[best_idx]

		if not keep_logs:
			posterior_log = None
			prior_log = None

		return GridPointResult(
			params=dict(params),
			mean_error=mean_error,
			metrics=metrics,
			posterior_log=posterior_log,
			prior_log=prior_log,
			n_repeats=n_repeats,
			std_error=std_error,
		)

	def optimize_subject(
		self,
		subject_id: int,
		param_grid: Dict[str, Sequence[Any]],
		n_repeats: int = 1,
		refit_repeats: int = 0,
		window_size: int = 16,
		stop_at: float = 1.0,
		max_trials: Optional[int] = None,
		keep_logs: bool = False,
	) -> Dict[str, object]:
		subject_frame = self._get_subject_frame(subject_id, stop_at)
		condition = int(subject_frame["condition"].iloc[0])
		arrays = self._extract_arrays(subject_frame, max_trials)

		param_names = list(param_grid.keys())
		param_values = list(param_grid.values())
		combinations = list(product(*param_values)) if param_values else []

		# Derive bounds from provided grids
		bounds: Dict[str, Tuple[float, float]] = {}
		for name, values in param_grid.items():
			if len(values) == 0:
				raise ValueError(f"Param grid for {name} is empty")
			lo = float(np.min(values))
			hi = float(np.max(values))
			if hi == lo:
				hi = lo + 1e-6
			bounds[name] = (lo, hi)

		# Cache to avoid duplicate evaluations
		cache: Dict[Tuple[Tuple[str, float], ...], GridPointResult] = {}
		grid_results: List[GridPointResult] = []
		best_result: Optional[GridPointResult] = None

		def _key(params: Dict[str, float]) -> Tuple[Tuple[str, float], ...]:
			return tuple(sorted((k, float(v)) for k, v in params.items()))

		def objective(params: Dict[str, float]) -> float:
			nonlocal best_result
			k = _key(params)
			if k in cache:
				return cache[k].mean_error

			gp = self._evaluate_params(
				subject_id,
				condition,
				arrays,
				params,
				window_size,
				n_repeats,
				keep_logs,
			)
			cache[k] = gp
			grid_results.append(gp)

			if best_result is None or gp.mean_error < best_result.mean_error:
				best_result = gp
			return gp.mean_error

		# Configure AMR defaults relative to grid size for a fair budget
		amr_options = dict(self._amr_kwargs)
		approx_budget = max(50, len(combinations) * n_repeats) if combinations else 200
		amr_options.setdefault("max_evals", approx_budget)
		amr_options.setdefault("refine_top_k", 2)
		amr_options.setdefault("split_factor", 2)
		amr_options.setdefault(
			"coarse_grid_per_dim",
			max(2, min(8, max(len(v) for v in param_grid.values()))),
		)

		optimizer = AMRGridSearch(bounds=bounds, objective=objective, **amr_options)
		optimizer.run()

		if best_result is None:
			raise RuntimeError("AMR optimizer produced no evaluations.")

		# Refit stage (same semantics as grid optimizer)
		if refit_repeats > 0:
			LOGGER.info(f"Refitting best params for subject {subject_id} with {refit_repeats} repeats.")
			refit_gp = self._evaluate_params(
				subject_id,
				condition,
				arrays,
				best_result.params,
				window_size,
				refit_repeats,
				keep_logs,
			)
			best_result = refit_gp

		# Return structure mirrors grid optimizer
		return {
			"subject_id": subject_id,
			"condition": condition,
			"best": best_result,
			"grid": grid_results,
			"param_grid": param_grid,
		}


__all__ = ["StateModelAMROptimizer"]

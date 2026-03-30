"""Adaptive mesh refinement optimizer for StateModel parameters.

This module mirrors StateModelGridOptimizer but replaces the exhaustive
grid enumeration with an inline AMRGridSearch heuristic.
It keeps the same input/output contract so callers can switch optimizers
without changing downstream code.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Any
from itertools import product

from joblib import Parallel, delayed

import numpy as np

from .optimization_common import (
	BaseStateOptimizer,
	GridPointResult,
	evaluate_state_model_run,
)
from ..utils.base import LOGGER

Params = Dict[str, float]
Bounds = Dict[str, Tuple[float, float]]


@dataclass(order=True)
class Cell:
	priority: float
	depth: int
	center: Params = field(compare=False)
	half_span: Params = field(compare=False)
	corners: List[Params] = field(compare=False, default_factory=list)
	scores: List[float] = field(compare=False, default_factory=list)


@dataclass
class AMRResult:
	best_params: Params
	best_score: float
	history: List[Tuple[Params, float]]


class AMRGridSearch:
	"""Adaptive Mesh Refinement for continuous parameter spaces.

	Supports optional coarse-grid seeding and configurable axis-aligned splits.
	"""

	def __init__(
		self,
		bounds: Bounds,
		objective: Callable[[Params], float],
		max_evals: int = 200,
		max_depth: int = 8,
		min_half_span: float = 1e-3,
		refine_top_k: int = 1,
		split_factor: int = 2,
		coarse_grid_per_dim: int | None = None,
		coarse_keep_top_k: int = 4,
	) -> None:
		self.bounds = bounds
		self.objective = objective
		self.max_evals = max_evals
		self.max_depth = max_depth
		self.min_half_span = min_half_span
		self.refine_top_k = refine_top_k
		self.split_factor = max(2, int(split_factor))
		self.coarse_grid_per_dim = coarse_grid_per_dim
		self.coarse_keep_top_k = max(1, int(coarse_keep_top_k))
		self.dim_names = list(bounds.keys())
		self._history: List[Tuple[Params, float]] = []
		self.best_params: Params | None = None
		self.best_score: float = math.inf

	def _initial_cell(self) -> Cell:
		center: Params = {}
		half_span: Params = {}
		corners: List[Params] = []
		for k, (lo, hi) in self.bounds.items():
			c = 0.5 * (lo + hi)
			h = 0.5 * (hi - lo)
			center[k] = c
			half_span[k] = h
		for mask in range(2 ** len(self.dim_names)):
			corner: Params = {}
			for i, name in enumerate(self.dim_names):
				sign = 1 if (mask >> i) & 1 else -1
				corner[name] = center[name] + sign * half_span[name]
			corners.append(corner)
		return Cell(priority=math.inf, depth=0, center=center, half_span=half_span, corners=corners)

	def _coarse_seed_cells(self) -> List[Cell]:
		if self.coarse_grid_per_dim is None or self.coarse_grid_per_dim <= 1:
			return [self._initial_cell()]

		per_dim = int(self.coarse_grid_per_dim)
		grids = []
		for name in self.dim_names:
			lo, hi = self.bounds[name]
			step = (hi - lo) / per_dim
			grids.append(np.linspace(lo + 0.5 * step, hi - 0.5 * step, per_dim))

		seeds: List[Cell] = []
		for combo in product(*grids):
			center: Params = {name: float(val) for name, val in zip(self.dim_names, combo)}
			half_span: Params = {}
			for name in self.dim_names:
				lo, hi = self.bounds[name]
				half_span[name] = 0.5 * (hi - lo) / per_dim
			corners: List[Params] = []
			for mask in range(2 ** len(self.dim_names)):
				corner: Params = {}
				for i, name in enumerate(self.dim_names):
					sign = 1 if (mask >> i) & 1 else -1
					corner[name] = center[name] + sign * half_span[name]
				corners.append(corner)
			seeds.append(Cell(priority=math.inf, depth=0, center=center, half_span=half_span, corners=corners))

		for cell in seeds:
			self._eval_cell(cell)
		seeds = sorted(seeds, key=lambda c: c.priority)[: self.coarse_keep_top_k]
		return seeds

	def _eval_point(self, params: Params) -> float:
		score = float(self.objective(params))
		self._history.append((dict(params), score))
		if score < self.best_score:
			self.best_score = score
			self.best_params = dict(params)
		return score

	def _eval_cell(self, cell: Cell) -> None:
		center_score = self._eval_point(cell.center)
		scores = [center_score]
		for corner in cell.corners:
			scores.append(self._eval_point(corner))
		cell.scores = scores
		cell.priority = min(scores)

	def _should_stop(self, cell: Cell, evals_done: int) -> bool:
		if evals_done >= self.max_evals:
			return True
		if cell.depth >= self.max_depth:
			return True
		for h in cell.half_span.values():
			if h < self.min_half_span:
				return True
		return False

	def _split_cell(self, cell: Cell) -> List[Cell]:
		m = self.split_factor
		children: List[Cell] = []
		for name in self.dim_names:
			new_half_span = dict(cell.half_span)
			new_half_span[name] = cell.half_span[name] / m
			for step in range(-m + 1, m, 2):
				new_center = dict(cell.center)
				new_center[name] = cell.center[name] + step * new_half_span[name]
				corners: List[Params] = []
				for mask in range(2 ** len(self.dim_names)):
					corner: Params = {}
					for i, dim in enumerate(self.dim_names):
						s = 1 if (mask >> i) & 1 else -1
						corner[dim] = new_center[dim] + s * new_half_span[dim]
					corners.append(corner)
				child = Cell(
					priority=math.inf,
					depth=cell.depth + 1,
					center=new_center,
					half_span=dict(new_half_span),
					corners=corners,
				)
				children.append(child)
		return children

	def run(self) -> AMRResult:
		import heapq

		seeds = self._coarse_seed_cells()
		evals_done = 0
		heap: List[Cell] = []
		for cell in seeds:
			evals_done += len(cell.scores)
			heapq.heappush(heap, cell)

		while heap and evals_done < self.max_evals:
			cell = heapq.heappop(heap)
			if self._should_stop(cell, evals_done):
				continue

			children = self._split_cell(cell)
			for child in children:
				self._eval_cell(child)
				evals_done += len(child.scores)
				heapq.heappush(heap, child)
				if evals_done >= self.max_evals:
					break

			if len(heap) > self.refine_top_k * len(self.dim_names) * 4:
				heap = heapq.nsmallest(self.refine_top_k * len(self.dim_names) * 4, heap)
				heapq.heapify(heap)

		return AMRResult(best_params=self.best_params or {}, best_score=self.best_score, history=self._history)


# ---------------------------------------------------------------------------
# AMR optimizer
# ---------------------------------------------------------------------------
class StateModelAMROptimizer(BaseStateOptimizer):
	"""
	Adaptive-mesh replacement for StateModelGridOptimizer.

	Input and output follow the grid optimizer: optimize_subject returns
	a dict with keys {"subject_id", "condition", "best", "grid", "param_grid"}.
	"""

	def __init__(
		self,
		engine_config: Dict,
		processed_data_dir: Optional[str] = None,
		n_jobs: int = 1,
		amr_kwargs: Optional[Dict[str, Any]] = None,
	) -> None:
		super().__init__(engine_config, processed_data_dir=processed_data_dir, n_jobs=n_jobs)
		self._amr_kwargs = amr_kwargs or {}

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
				evaluate_state_model_run(
					subject_id,
					condition,
					arrays,
					params,
					self._engine_config_template,
					self._processed_data_dir,
					window_size,
					keep_logs,
					True,
				)
			]
		else:
			runs = Parallel(n_jobs=self.n_jobs)(
				delayed(evaluate_state_model_run)(
					subject_id,
					condition,
					arrays,
					params,
					self._engine_config_template,
					self._processed_data_dir,
					window_size,
					keep_logs,
					True,
				)
				for _ in range(n_repeats)
			)

		errors = [r.mean_error for r in runs]
		mean_error = float(np.mean(errors))
		std_error = float(np.std(errors)) if len(errors) > 1 else 0.0

		best_idx = int(np.argmin([abs(e - mean_error) for e in errors]))
		best_run = runs[best_idx]
		metrics = best_run.metrics
		posterior_log = best_run.posterior_log
		prior_log = best_run.prior_log
		step_log = best_run.step_log
		strategy_log = best_run.strategy_counts_log

		if not keep_logs:
			posterior_log = None
			prior_log = None
			step_log = None
			strategy_log = None

		return GridPointResult(
			params=dict(params),
			mean_error=mean_error,
			metrics=metrics,
			posterior_log=posterior_log,
			prior_log=prior_log,
			step_results=step_log,
			strategy_counts_log=strategy_log,
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

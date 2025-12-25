"""Adaptive Mesh Refinement (AMR) search framework.

This module is self-contained and does not depend on existing grid search
code. It provides a reusable AMR-style optimizer suitable for expensive
black-box objectives (e.g., model fitting). The algorithm refines promising
regions of the parameter space by recursively subdividing hyper-rectangles
until a stopping criterion is met.

Usage outline:
    1) Implement an objective function: f(params: dict[str, float]) -> float
       that returns a scalar error (lower is better).
    2) Define parameter bounds as a dict: {"gamma": (0.05, 1.0), "w0": (0.05, 0.9)}.
    3) Instantiate AMRGridSearch(bounds, objective, ...). Call run().
    4) Retrieve best result via optimizer.best or optimizer.history.

Notes:
- The implementation is deliberately light-weight and pure Python.
- Parallelism is not included; can be wrapped externally if needed.
- The refinement strategy is axis-aligned splitting of the current best cell.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Any
import math
import numpy as np
from itertools import product


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

    支持可选的初始粗网格播种与可配置的分裂因子。流程：
      1) 可先用粗网格在每维均匀采样，评估并保留得分最好的播种点，作为初始单元中心。
      2) 在细化阶段，按可配置的 split_factor 对单元做轴向等距多分裂（默认二分）。
    """

    def __init__(
        self,
        bounds: Bounds,
        objective: Callable[[Params], float],
        max_evals: int = 200,
        max_depth: int = 8,
        min_half_span: float = 1e-3,
        refine_top_k: int = 1,
        # 新增：分裂因子与粗网格控制
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
        """Default根单元：整个参数边界。"""
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
        """可选粗网格播种：每维等距采样，保留 top-K 作为初始单元。"""
        if self.coarse_grid_per_dim is None or self.coarse_grid_per_dim <= 1:
            return [self._initial_cell()]

        per_dim = int(self.coarse_grid_per_dim)
        grids = []
        for name in self.dim_names:
            lo, hi = self.bounds[name]
            step = (hi - lo) / per_dim
            # Use cell centers offset by half-step to keep corners inside bounds
            grids.append(np.linspace(lo + 0.5 * step, hi - 0.5 * step, per_dim))

        seeds: List[Cell] = []
        for combo in product(*grids):
            center: Params = {name: float(val) for name, val in zip(self.dim_names, combo)}
            half_span: Params = {}
            for name in self.dim_names:
                lo, hi = self.bounds[name]
                half_span[name] = 0.5 * (hi - lo) / per_dim  # half of coarse cell width
            corners: List[Params] = []
            for mask in range(2 ** len(self.dim_names)):
                corner: Params = {}
                for i, name in enumerate(self.dim_names):
                    sign = 1 if (mask >> i) & 1 else -1
                    corner[name] = center[name] + sign * half_span[name]
                corners.append(corner)
            seeds.append(Cell(priority=math.inf, depth=0, center=center, half_span=half_span, corners=corners))

        # 评估所有种子，保留最优 top-K
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
        # Evaluate center
        center_score = self._eval_point(cell.center)
        scores = [center_score]
        # Evaluate corners (cheap uniform sampling of cell)
        for corner in cell.corners:
            scores.append(self._eval_point(corner))
        cell.scores = scores
        cell.priority = min(scores)

    def _should_stop(self, cell: Cell, evals_done: int) -> bool:
        if evals_done >= self.max_evals:
            return True
        if cell.depth >= self.max_depth:
            return True
        # Check minimal span in any dimension
        for h in cell.half_span.values():
            if h < self.min_half_span:
                return True
        return False

    def _split_cell(self, cell: Cell) -> List[Cell]:
        """Axis-aligned multi-split controlled by split_factor (>=2)."""
        m = self.split_factor
        children: List[Cell] = []
        for name in self.dim_names:
            new_half_span = dict(cell.half_span)
            new_half_span[name] = cell.half_span[name] / m
            for step in range(-m + 1, m, 2):  # symmetric positions
                new_center = dict(cell.center)
                new_center[name] = cell.center[name] + step * new_half_span[name]
                # Recompute corners for child
                corners: List[Params] = []
                for mask in range(2 ** len(self.dim_names)):
                    corner: Params = {}
                    for i, dim in enumerate(self.dim_names):
                        s = 1 if (mask >> i) & 1 else -1
                        corner[dim] = new_center[dim] + s * new_half_span[dim]
                    corners.append(corner)
                child = Cell(priority=math.inf, depth=cell.depth + 1, center=new_center, half_span=dict(new_half_span), corners=corners)
                children.append(child)
        return children

    def run(self) -> AMRResult:
        import heapq

        # 1) 粗网格播种（可选）
        seeds = self._coarse_seed_cells()
        evals_done = 0
        heap: List[Cell] = []
        for cell in seeds:
            evals_done += len(cell.scores)
            heapq.heappush(heap, cell)

        # 2) 主循环细化
        while heap and evals_done < self.max_evals:
            # Pop best cell (lowest priority = lowest score)
            cell = heapq.heappop(heap)
            if self._should_stop(cell, evals_done):
                continue

            # Refine this cell
            children = self._split_cell(cell)
            for child in children:
                self._eval_cell(child)
                evals_done += len(child.scores)
                heapq.heappush(heap, child)
                if evals_done >= self.max_evals:
                    break

            # Keep only top-K promising cells to control branching
            if len(heap) > self.refine_top_k * len(self.dim_names) * 4:
                heap = heapq.nsmallest(self.refine_top_k * len(self.dim_names) * 4, heap)
                heapq.heapify(heap)

        return AMRResult(best_params=self.best_params or {}, best_score=self.best_score, history=self._history)


__all__ = ["AMRGridSearch", "AMRResult"]

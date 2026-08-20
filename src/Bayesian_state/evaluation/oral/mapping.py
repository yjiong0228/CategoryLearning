"""口述中心和区域到共享 hypothesis space 的映射。"""

from __future__ import annotations

import ast
import json
from typing import Any, List, Optional, Tuple

import numpy as np

from ...hypothesis_space import (
    CategoryRegion,
    ContinuousHypothesisSpace,
    ContinuousPartition,
    Polytope,
)
from ...hypothesis_space.similarity import region_overlap


class OralRegionMapper:
    """Region-based oral analysis with overlap scoring.

    Oral reports use convex ``A @ x <= b`` regions. Model categories may
    additionally be unions of such convex components.
    """

    VALID_OVERLAP_METRICS = {"iou", "intersection", "precision_like", "recall_like"}

    @staticmethod
    def _parse_region(region: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Parse a region into (A, b) with robust shape checks.

        Accepted forms:
        - dict: {"A": ..., "b": ...}
        - tuple/list: (A, b)
        - JSON string of either form
        """
        if region is None:
            return None, None

        if isinstance(region, Polytope):
            return region.A, region.b

        if isinstance(region, str):
            try:
                region = json.loads(region)
            except json.JSONDecodeError:
                return None, None

        if isinstance(region, dict):
            if "A" not in region or "b" not in region:
                return None, None
            A = np.asarray(region["A"], dtype=float)
            b = np.asarray(region["b"], dtype=float)
        elif isinstance(region, (list, tuple)) and len(region) == 2:
            raw_A, raw_b = region
            if isinstance(raw_A, str):
                try:
                    raw_A = json.loads(raw_A)
                except json.JSONDecodeError:
                    return None, None
            if isinstance(raw_b, str):
                try:
                    raw_b = json.loads(raw_b)
                except json.JSONDecodeError:
                    return None, None
            A = np.asarray(raw_A, dtype=float)
            b = np.asarray(raw_b, dtype=float)
        else:
            return None, None

        if np.isnan(A).any() or np.isnan(b).any():
            return None, None
        if A.ndim == 1:
            A = np.atleast_2d(A)
        b = np.asarray(b).reshape(-1)
        if A.ndim != 2 or b.ndim != 1 or A.shape[0] != b.shape[0]:
            return None, None
        return A, b

    @staticmethod
    def _region_components(region: Any) -> list[Any]:
        """Return one or more convex region components."""
        if isinstance(region, CategoryRegion):
            return list(region.components)
        if isinstance(region, dict) and "components" in region:
            components = region.get("components")
            return list(components) if isinstance(components, (list, tuple)) else []
        return [region]

    @classmethod
    def _points_in_region_object(
        cls,
        points: np.ndarray,
        region: Any,
        dist_tol: float,
    ) -> np.ndarray:
        """Return membership in a convex region or a union of components."""
        mask = np.zeros(points.shape[0], dtype=bool)
        for component in cls._region_components(region):
            A, b = cls._parse_region(component)
            mask |= cls._points_in_region(points, A, b, dist_tol=dist_tol)
        return mask

    @staticmethod
    def _points_in_region(
        points: np.ndarray,
        A: Optional[np.ndarray],
        b: Optional[np.ndarray],
        dist_tol: float,
    ) -> np.ndarray:
        """Return mask of points satisfying canonical ``A @ x <= b + tol``."""
        if A is None or b is None:
            return np.zeros(points.shape[0], dtype=bool)
        if A.size == 0:
            return np.ones(points.shape[0], dtype=bool)
        lhs = points @ A.T
        return np.all(lhs <= (b + dist_tol), axis=1)

    @classmethod
    def _estimate_overlap_score(
        cls,
        region1: Any,
        region2: Any,
        metric: str,
        n_samples: int,
        bounds: Tuple[float, float],
        random_state: Optional[int],
        dist_tol: float,
    ) -> float:
        """Estimate overlap score between two regions via Monte Carlo sampling."""
        parsed_by_region = []
        for region in (region1, region2):
            parsed = [cls._parse_region(item) for item in cls._region_components(region)]
            valid = [(A, b) for A, b in parsed if A is not None and b is not None]
            if not valid:
                return float("nan")
            parsed_by_region.append(valid)

        rng = np.random.default_rng(random_state)
        d = parsed_by_region[0][0][0].shape[1]
        low, high = bounds
        points = rng.uniform(low, high, size=(n_samples, d))

        in_r1 = cls._points_in_region_object(points, region1, dist_tol=dist_tol)
        in_r2 = cls._points_in_region_object(points, region2, dist_tol=dist_tol)
        box_volume = (high - low) ** d

        # Convert point-wise inclusion rates into geometric volumes.
        vol1 = float(np.mean(in_r1) * box_volume)
        vol2 = float(np.mean(in_r2) * box_volume)
        intersection = float(np.mean(in_r1 & in_r2) * box_volume)
        union = float(np.mean(in_r1 | in_r2) * box_volume)

        if metric == "iou":
            return intersection / union if union > 0 else 0.0
        if metric == "intersection":
            return intersection
        if metric == "precision_like":
            return intersection / vol1 if vol1 > 0 else 0.0
        if metric == "recall_like":
            return intersection / vol2 if vol2 > 0 else 0.0
        raise ValueError(f"Unsupported overlap metric: {metric}")

    @staticmethod
    def _true_region(
        hypothesis_space: ContinuousHypothesisSpace,
        hypo_idx: int,
        cat_idx: int,
    ) -> CategoryRegion:
        """Fetch one typed category region from the canonical catalogue."""
        return hypothesis_space[int(hypo_idx)].categories[int(cat_idx)]


class RegionOverlapScorer:
    """Fast Monte Carlo scorer for oral regions against all hypothesis regions.

    It fixes one point cloud per partition/category and precomputes the
    hypothesis-region inclusion masks. Per oral trial, the only expensive work
    left is computing the oral mask once, then vectorized boolean overlap
    against all hypothesis masks.
    """

    def __init__(
        self,
        partition: ContinuousPartition,
        n_samples: int = 1000,
        bounds: Tuple[float, float] = (0.0, 1.0),
        random_state: Optional[int] = 42,
        dist_tol: float = 1e-9,
    ):
        self.partition = partition
        self.n_samples = int(n_samples)
        self.bounds = tuple(float(x) for x in bounds)
        self.random_state = random_state
        self.dist_tol = float(dist_tol)
        self.n_hypos = int(len(partition.hypothesis_space))
        self.n_cats = int(partition.n_cats)
        self.n_dims = int(partition.n_dims)
        low, high = self.bounds
        rng = np.random.default_rng(random_state)
        self.points = rng.uniform(low, high, size=(self.n_samples, self.n_dims))
        self.box_volume = float((high - low) ** self.n_dims)
        self.hypothesis_masks = self._precompute_hypothesis_masks()

    def _precompute_hypothesis_masks(self) -> List[np.ndarray]:
        masks_by_cat: List[np.ndarray] = []
        for cat_idx in range(self.n_cats):
            cat_masks = np.zeros((self.n_hypos, self.n_samples), dtype=bool)
            for hypo_idx in range(self.n_hypos):
                region = OralRegionMapper._true_region(
                    self.partition.hypothesis_space,
                    hypo_idx,
                    cat_idx,
                )
                cat_masks[hypo_idx] = OralRegionMapper._points_in_region_object(
                    self.points,
                    region,
                    dist_tol=self.dist_tol,
                )
            masks_by_cat.append(cat_masks)
        return masks_by_cat

    def score_all(
        self,
        oral_region: Any,
        cat_idx: int,
        metric: str = "iou",
    ) -> np.ndarray:
        """Score one oral region against every hypothesis for one category."""
        if metric not in OralRegionMapper.VALID_OVERLAP_METRICS:
            raise ValueError(
                f"Unsupported overlap_metric={metric}. "
                f"Choose from {sorted(OralRegionMapper.VALID_OVERLAP_METRICS)}."
            )
        if cat_idx < 0 or cat_idx >= self.n_cats:
            return np.full(self.n_hypos, np.nan, dtype=float)

        A, b = OralRegionMapper._parse_region(oral_region)
        if A is None or b is None:
            return np.full(self.n_hypos, np.nan, dtype=float)

        oral_mask = OralRegionMapper._points_in_region(
            self.points,
            A,
            b,
            dist_tol=self.dist_tol,
        )
        hypo_masks = self.hypothesis_masks[int(cat_idx)]
        return region_overlap(
            hypo_masks,
            oral_mask,
            metric=metric,
            box_volume=self.box_volume,
        )


class OralCenterMapper:
    """Center-based oral analysis with nearest-hypothesis matching."""

    @staticmethod
    def _parse_center(value: Any, n_dims: int = 4) -> np.ndarray:
        """Parse one oral_center value into a numeric vector.

        Invalid or empty values become an all-NaN vector so callers can keep
        trial alignment while marking the oral report as unusable.
        """
        if value is None:
            return np.full(n_dims, np.nan, dtype=float)
        if isinstance(value, float) and np.isnan(value):
            return np.full(n_dims, np.nan, dtype=float)

        parsed = value
        if isinstance(value, str):
            text = value.strip()
            if not text or text.lower() in {"nan", "none"}:
                return np.full(n_dims, np.nan, dtype=float)
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(text)
                except (SyntaxError, ValueError):
                    return np.full(n_dims, np.nan, dtype=float)

        try:
            arr = np.asarray(parsed, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return np.full(n_dims, np.nan, dtype=float)

        if arr.size != n_dims or np.isnan(arr).any():
            return np.full(n_dims, np.nan, dtype=float)
        return arr


__all__ = ["OralCenterMapper", "OralRegionMapper", "RegionOverlapScorer"]

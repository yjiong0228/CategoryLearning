"""Oral report post-processing utilities.

This module provides two analysis paths:
1) ``Oral_region_analysis``: compare reported regions (A, b) against each
   hypothesis region using Monte Carlo overlap metrics.
2) ``Oral_center_analysis``: compare reported feature centers against each
   hypothesis prototype using Euclidean distance.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..problems.partitions import Partition


def _resolve_top_k(condition: int, top_k: Optional[int]) -> int:
    """Resolve default top-k per condition when user does not provide one."""
    if top_k is not None and top_k > 0:
        return int(top_k)
    return 4 if int(condition) == 1 else 10


class Oral_region_analysis:
    """Region-based oral analysis with overlap scoring."""

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
            A = np.asarray(region[0], dtype=float)
            b = np.asarray(region[1], dtype=float)
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
    def _points_in_region(
        points: np.ndarray,
        A: Optional[np.ndarray],
        b: Optional[np.ndarray],
        dist_tol: float,
    ) -> np.ndarray:
        """Return mask of points satisfying A x > b - tol."""
        if A is None or b is None:
            return np.zeros(points.shape[0], dtype=bool)
        if A.size == 0:
            return np.ones(points.shape[0], dtype=bool)
        lhs = points @ A.T
        return np.all(lhs > (b - dist_tol), axis=1)

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
        A1, b1 = cls._parse_region(region1)
        A2, b2 = cls._parse_region(region2)
        if A1 is None or A2 is None:
            return float("nan")

        rng = np.random.default_rng(random_state)
        d = A1.shape[1] if A1.size > 0 else A2.shape[1]
        low, high = bounds
        points = rng.uniform(low, high, size=(n_samples, d))

        in_r1 = cls._points_in_region(points, A1, b1, dist_tol=dist_tol)
        in_r2 = cls._points_in_region(points, A2, b2, dist_tol=dist_tol)
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
    def _true_region(regions: Any, hypo_idx: int, cat_idx: int) -> Any:
        """Fetch one hypothesis region for one category from partition storage."""
        if isinstance(regions, np.ndarray):
            return regions[hypo_idx, 1, cat_idx, :]
        if isinstance(regions, (list, tuple)):
            return regions[hypo_idx][cat_idx]
        raise TypeError(f"Unsupported partition_model.regions type: {type(regions)}")

    def get_oral_hypos_list(
        self,
        condition: int,
        oral_region: Sequence[Any],
        choices: np.ndarray,
        partition: Partition,
        region_valid_mask: Optional[np.ndarray] = None,
        dist_tol: float = 1e-9,
        top_k: Optional[int] = None,
        n_samples: int = 100,
        bounds: Tuple[float, float] = (0.0, 1.0),
        random_state: Optional[int] = 42,
        overlap_metric: str = "iou",
    ) -> List[Dict[str, Any]]:
        """Return per-trial top hypotheses and overlap scores.

        Output per trial includes:
        - ``top_hypos``: ranked hypothesis indices.
        - ``top_scores``: corresponding overlap scores for those hypotheses.
        """
        if overlap_metric not in self.VALID_OVERLAP_METRICS:
            raise ValueError(
                f"Unsupported overlap_metric={overlap_metric}. "
                f"Choose from {sorted(self.VALID_OVERLAP_METRICS)}."
            )

        n_trials = len(choices)
        if region_valid_mask is None:
            # Default validity rule: both A and b must be non-empty.
            region_valid_list: List[bool] = []
            for region in oral_region:
                valid = False
                if isinstance(region, (list, tuple)) and len(region) == 2:
                    raw_A, raw_b = region
                    if isinstance(raw_A, str):
                        try:
                            raw_A = json.loads(raw_A)
                        except json.JSONDecodeError:
                            raw_A = None
                    if isinstance(raw_b, str):
                        try:
                            raw_b = json.loads(raw_b)
                        except json.JSONDecodeError:
                            raw_b = None
                    try:
                        A_size = np.asarray(raw_A, dtype=float).size if raw_A is not None else 0
                        b_size = np.asarray(raw_b, dtype=float).size if raw_b is not None else 0
                        valid = A_size > 0 and b_size > 0
                    except (TypeError, ValueError):
                        valid = False
                region_valid_list.append(valid)
            region_valid_mask = np.asarray(region_valid_list, dtype=bool)

        resolved_top_k = _resolve_top_k(condition, top_k)
        n_hypos = len(partition.regions)
        regions = partition.regions
        out: List[Dict[str, Any]] = []

        for trial_idx in range(n_trials):
            # Invalid oral report -> keep empty result for this trial.
            if not bool(region_valid_mask[trial_idx]):
                out.append(
                    {
                        "trial_idx": trial_idx,
                        "choice": int(choices[trial_idx]),
                        "reported_region": None,
                        "top_hypos": [],
                        "top_scores": [],
                    }
                )
                continue

            cat_idx = int(choices[trial_idx]) - 1
            reported_region = oral_region[trial_idx]
            overlap_map: List[Dict[str, Any]] = []

            for hypo_idx in range(n_hypos):
                # Seed design keeps run-level reproducibility while separating
                # trials/hypotheses.
                score = self._estimate_overlap_score(
                    reported_region,
                    self._true_region(regions, hypo_idx, cat_idx),
                    metric=overlap_metric,
                    n_samples=n_samples,
                    bounds=bounds,
                    random_state=None if random_state is None else random_state + trial_idx * 100000 + hypo_idx,
                    dist_tol=dist_tol,
                )
                overlap_map.append(
                    {
                        "hypo_idx": hypo_idx,
                        "overlap_score": score,
                    }
                )

            overlap_map.sort(key=lambda x: x["overlap_score"], reverse=True)
            top_results = overlap_map[:resolved_top_k]
            out.append(
                {
                    "trial_idx": trial_idx,
                    "choice": int(choices[trial_idx]),
                    "reported_region": reported_region,
                    "top_hypos": [item["hypo_idx"] for item in top_results],
                    "top_scores": [item["overlap_score"] for item in top_results],
                }
            )

        return out

    def get_oral_hypo_hits(
        self,
        data: pd.DataFrame,
        top_k: Optional[int] = None,
        window_size: int = 16,
        n_samples: int = 50000,
        bounds: Tuple[float, float] = (0.0, 1.0),
        random_state: Optional[int] = 42,
        overlap_metric: str = "iou",
    ) -> Dict[int, Dict[str, Any]]:
        """Compute hit trajectories per subject for region-based oral reports."""
        
        learning_data = data.copy()
        results: Dict[int, Dict[str, Any]] = {}

        for _, subj_df in learning_data.groupby("iSub"):
            subj_df = subj_df.reset_index(drop=True)
            sid = int(subj_df["iSub"].iloc[0])
            cond = int(subj_df["condition"].iloc[0])
            n_cats = 2 if cond == 1 else 4
            partition = Partition(n_dims=4, n_cats=n_cats)

            oral_region = [(row["A"], row["b"]) for _, row in subj_df.iterrows()]
            region_valid_mask = []
            for A_val, b_val in oral_region:
                parsed_A = A_val
                parsed_b = b_val
                if isinstance(parsed_A, str):
                    try:
                        parsed_A = json.loads(parsed_A)
                    except json.JSONDecodeError:
                        parsed_A = None
                if isinstance(parsed_b, str):
                    try:
                        parsed_b = json.loads(parsed_b)
                    except json.JSONDecodeError:
                        parsed_b = None
                try:
                    a_size = np.asarray(parsed_A, dtype=float).size if parsed_A is not None else 0
                    b_size = np.asarray(parsed_b, dtype=float).size if parsed_b is not None else 0
                    region_valid_mask.append(bool(a_size > 0 and b_size > 0))
                except (TypeError, ValueError):
                    region_valid_mask.append(False)

            choices = subj_df["choice"].to_numpy()
            trial_results = self.get_oral_hypos_list(
                condition=cond,
                oral_region=oral_region,
                choices=choices,
                partition=partition,
                region_valid_mask=np.asarray(region_valid_mask, dtype=bool),
                top_k=top_k,
                n_samples=n_samples,
                bounds=bounds,
                random_state=random_state,
                overlap_metric=overlap_metric,
            )

            target_value = 0 if cond == 1 else 42
            top_hypos_per_trial: List[List[int]] = []
            top_scores_per_trial: List[List[float]] = []
            hits: List[float] = []

            for idx, tr in enumerate(trial_results):
                # top_hypos/top_scores are aligned by position.
                hypos = tr["top_hypos"]
                scores = tr["top_scores"]
                top_hypos_per_trial.append(hypos)
                top_scores_per_trial.append(scores)
                if len(hypos) == 0:
                    hits.append(np.nan)
                else:
                    hits.append(1.0 if target_value in hypos else 0.0)

            rolling_hits = pd.Series(hits).rolling(window=window_size, min_periods=window_size).mean().tolist()
            results[sid] = {
                "iSub": sid,
                "condition": cond,
                "target_hypo": target_value,
                "hits": hits,
                "rolling_hits": rolling_hits,
                "top_hypos_per_trial": top_hypos_per_trial,
                "top_scores_per_trial": top_scores_per_trial,
            }

        return results


class Oral_center_analysis:
    """Center-based oral analysis with nearest-hypothesis matching."""

    @staticmethod
    def get_oral_hypos_list(
        condition: int,
        data: Tuple[np.ndarray, np.ndarray],
        partition: Partition,
        center_valid_mask: Optional[np.ndarray] = None,
        dist_tol: float = 1e-9,
        top_k: Optional[int] = None,
    ) -> List[List[int]]:
        """Return candidate hypotheses per trial from oral center reports."""
        oral_centers, choices = data
        n_trials = len(choices)
        if center_valid_mask is None:
            # Default validity rule: center vector is non-empty and not all-NaN.
            center_valid_mask = np.ones(n_trials, dtype=bool)
            for idx in range(n_trials):
                center_arr = np.asarray(oral_centers[idx], dtype=float)
                center_valid_mask[idx] = bool(center_arr.size > 0 and not np.all(np.isnan(center_arr)))

        resolved_top_k = _resolve_top_k(condition, top_k)
        n_hypos = partition.prototypes_np.shape[0]
        out: List[List[int]] = []

        for trial_idx in range(n_trials):
            if not bool(center_valid_mask[trial_idx]):
                out.append([])
                continue

            reported_center = oral_centers[trial_idx]
            cat_idx = int(choices[trial_idx]) - 1
            distance_map = []
            for hypo_idx in range(n_hypos):
                # Compare oral center with each hypothesis prototype center.
                true_center = partition.prototypes_np[hypo_idx, 0, cat_idx, :]
                distance_val = float(np.linalg.norm(reported_center - true_center))
                distance_map.append((distance_val, hypo_idx))

            # Keep exact matches if present; otherwise take nearest top-k.
            exact_matches = [h for (d, h) in distance_map if d <= dist_tol]
            if exact_matches:
                out.append(exact_matches)
            else:
                distance_map.sort(key=lambda x: x[0])
                out.append([h for (_, h) in distance_map[:resolved_top_k]])

        return out

    def get_oral_hypo_hits(self, data: pd.DataFrame, window_size: int = 16) -> Dict[int, Dict[str, Any]]:
        """Compute hit trajectories per subject for center-based oral reports."""
        learning_data = data.copy()
        results: Dict[int, Dict[str, Any]] = {}

        for _, subj_df in learning_data.groupby("iSub"):
            sid = int(subj_df["iSub"].iloc[0])
            cond = int(subj_df["condition"].iloc[0])

            n_cats = 2 if cond == 1 else 4
            partition = Partition(n_dims=4, n_cats=n_cats)

            oral_value_cols = ["feature1_oralvalue", "feature2_oralvalue", "feature3_oralvalue", "feature4_oralvalue"]
            centers = subj_df[oral_value_cols].to_numpy(dtype=float)
            center_valid_mask = np.array(
                [bool(np.asarray(center, dtype=float).size > 0 and not np.all(np.isnan(center))) for center in centers],
                dtype=bool,
            )

            choices = subj_df["choice"].to_numpy()
            hypos = self.get_oral_hypos_list(
                condition=cond,
                data=(centers, choices),
                partition=partition,
                center_valid_mask=center_valid_mask,
            )

            target_value = 0 if cond == 1 else 42
            hits: List[float] = []
            for trial_hypos in hypos:
                if len(trial_hypos) == 0:
                    hits.append(np.nan)
                else:
                    hits.append(1.0 if target_value in trial_hypos else 0.0)

            rolling_hits = pd.Series(hits).rolling(window=window_size, min_periods=window_size).mean().tolist()
            results[sid] = {
                "iSub": sid,
                "condition": cond,
                "hits": hits,
                "rolling_hits": rolling_hits,
            }

        return results

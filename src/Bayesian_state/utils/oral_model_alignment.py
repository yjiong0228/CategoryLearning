"""Oral report and model-alignment utilities.

This module provides two analysis paths:
1) ``Oral_region_analysis``: compare reported regions (A, b) against each
   hypothesis region using Monte Carlo overlap metrics.
2) ``Oral_center_analysis``: compare reported feature centers against each
   hypothesis prototype using Euclidean distance.

It also contains ``OralModelAlignmentMixin``, the oral/model alignment surface
mixed into ``ModelEval``. Keeping these methods here makes the main model
evaluation facade easier to scan.
"""

from __future__ import annotations

import ast
import json
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..problems.partitions import Partition


logger = logging.getLogger(__name__)


__all__ = [
    "OralModelAlignmentMixin",
    "Oral_center_analysis",
    "Oral_region_analysis",
]


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

            oral_region = [(row["oral_A"], row["oral_b"]) for _, row in subj_df.iterrows()]
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
        n_hypos = partition.prototypes.shape[0]
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
                true_center = partition.prototypes[hypo_idx, 0, cat_idx, :]
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
        if "oral_center" not in learning_data.columns:
            logger.warning("Skipping oral center analysis; missing oral_center column.")
            return results

        for _, subj_df in learning_data.groupby("iSub"):
            sid = int(subj_df["iSub"].iloc[0])
            cond = int(subj_df["condition"].iloc[0])

            n_cats = 2 if cond == 1 else 4
            partition = Partition(n_dims=4, n_cats=n_cats)

            centers = np.asarray([self._parse_center(value) for value in subj_df["oral_center"]], dtype=float)
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


class OralModelAlignmentMixin:
    """Oral/model alignment methods mixed into ``ModelEval``.

    The host class is expected to provide ``_filter_results`` and
    ``_layout_by_condition``. ``ModelEval`` supplies both.
    """

    @staticmethod
    def _normalize_distribution(values):
        """Return a valid probability vector or an all-NaN vector."""
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size == 0 or np.isnan(arr).all():
            return np.full(arr.shape, np.nan, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, 0.0, None)
        total = float(arr.sum())
        if total <= 0:
            return np.full(arr.shape, np.nan, dtype=float)
        return arr / total

    @staticmethod
    def _adaptive_softmax_from_distances(distances):
        """Convert distances to a distribution without a fixed top-k cutoff."""
        d = np.asarray(distances, dtype=float).reshape(-1)
        if d.size == 0 or np.isnan(d).all():
            return np.full(d.shape, np.nan, dtype=float)

        finite = d[np.isfinite(d)]
        if finite.size == 0:
            return np.full(d.shape, np.nan, dtype=float)

        d = np.where(np.isfinite(d), d, np.nanmax(finite))
        d_min = float(np.min(d))
        spread = float(np.median(d) - d_min)
        if spread <= 1e-12:
            spread = float(np.std(d))
        if spread <= 1e-12:
            exact = np.isclose(d, d_min)
            return exact.astype(float) / float(np.sum(exact))

        score = np.exp(-(d - d_min) / spread)
        return OralModelAlignmentMixin._normalize_distribution(score)

    @staticmethod
    def _js_similarity(p, q):
        """Return 1 - normalized Jensen-Shannon divergence."""
        p = OralModelAlignmentMixin._normalize_distribution(p)
        q = OralModelAlignmentMixin._normalize_distribution(q)
        if np.isnan(p).any() or np.isnan(q).any() or p.shape != q.shape:
            return np.nan

        m = 0.5 * (p + q)

        def kl(a, b):
            mask = a > 0
            return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

        js = 0.5 * kl(p, m) + 0.5 * kl(q, m)
        return float(1.0 - min(js / np.log(2.0), 1.0))

    @staticmethod
    def _effective_sample_size(prob):
        """Return distribution effective sample size, or NaN if invalid."""
        p = OralModelAlignmentMixin._normalize_distribution(prob)
        if np.isnan(p).any():
            return np.nan
        return float(1.0 / np.sum(p ** 2))

    @staticmethod
    def _extract_prior_log(info):
        """Use prior_t as the model state aligned with oral_t."""
        prior_log = info.get("prior_log") or []
        if prior_log:
            return [np.asarray(x, dtype=float) for x in prior_log]

        priors = []
        for step in info.get("best_step_results", []) or []:
            prior = step.get("prior")
            if prior is None:
                return []
            priors.append(np.asarray(prior, dtype=float))
        return priors

    @staticmethod
    def _center_oral_distribution(center, choice, partition):
        """Map one oral center report to a full hypothesis distribution."""
        center = np.asarray(center, dtype=float).reshape(-1)
        if center.size == 0 or np.isnan(center).any():
            return np.full(partition.length, np.nan, dtype=float)

        cat_idx = int(choice) - 1
        distances = np.linalg.norm(partition.prototypes[:, 0, cat_idx, :] - center, axis=1)
        return OralModelAlignmentMixin._adaptive_softmax_from_distances(distances)

    @staticmethod
    def _region_oral_distribution(region, choice, partition, n_samples=1000, random_state=42):
        """Map one oral region report to a full hypothesis distribution."""
        cat_idx = int(choice) - 1
        scores = []
        for hypo_idx in range(len(partition.regions)):
            score = Oral_region_analysis._estimate_overlap_score(
                region,
                partition.regions[hypo_idx][cat_idx],
                metric="iou",
                n_samples=int(n_samples),
                bounds=(0.0, 1.0),
                random_state=None if random_state is None else int(random_state) + hypo_idx,
                dist_tol=1e-9,
            )
            scores.append(0.0 if np.isnan(score) else float(score))
        return OralModelAlignmentMixin._normalize_distribution(scores)

    @staticmethod
    def _choice_conditioned_prior(partition, prior, stimulus, choice, beta=10.0):
        """Condition prior_t on the category choice made before oral report."""
        prior = OralModelAlignmentMixin._normalize_distribution(prior)
        if np.isnan(prior).any():
            return np.full_like(prior, np.nan, dtype=float)

        choice_idx = int(choice) - 1
        likelihood = np.zeros_like(prior, dtype=float)
        data = ([np.asarray(stimulus, dtype=float)], [int(choice)], [1.0], [choice_idx + 1])
        for hypo_idx, weight in enumerate(prior):
            if weight <= 0:
                continue
            prob = partition.get_category_probabilities(
                hypo=hypo_idx,
                data=data,
                beta=float(beta),
                distance_mode=partition.DISTANCE_MODE_PROTOTYPE,
            )[:, 0]
            if 0 <= choice_idx < len(prob):
                likelihood[hypo_idx] = float(prob[choice_idx])

        conditioned = prior * likelihood
        return OralModelAlignmentMixin._normalize_distribution(conditioned)

    @staticmethod
    def _expected_center_similarity(partition, model_dist, oral_center, choice):
        """Compare oral center with the model's choice-conditioned expected center."""
        model_dist = OralModelAlignmentMixin._normalize_distribution(model_dist)
        center = np.asarray(oral_center, dtype=float).reshape(-1)
        if np.isnan(model_dist).any() or center.size == 0 or np.isnan(center).any():
            return np.nan

        cat_idx = int(choice) - 1
        centers = partition.prototypes[:, 0, cat_idx, :]
        expected_center = np.sum(model_dist[:, None] * centers, axis=0)
        dist = float(np.linalg.norm(center - expected_center))
        max_dist = float(np.sqrt(partition.n_dims))
        if max_dist <= 0:
            return np.nan
        return float(np.clip(1.0 - dist / max_dist, 0.0, 1.0))

    def compute_oral_model_alignment(
        self,
        model_results,
        oral_df,
        oral_mode="center",
        subjects=None,
        region_n_samples=1000,
    ):
        """Compute prior_t vs oral_t alignment metrics per subject."""
        model_res = self._filter_results(model_results, subjects)
        oral_df = oral_df.copy()
        out = {}

        for iSub, info in model_res.items():
            subj_df = oral_df[oral_df["iSub"] == iSub].reset_index(drop=True)
            if subj_df.empty:
                continue

            condition = int(info.get("condition", subj_df["condition"].iloc[0]))
            n_cats = 2 if condition == 1 else 4
            target_hypo = 0 if condition == 1 else 42
            partition = Partition(n_dims=4, n_cats=n_cats)
            prior_log = self._extract_prior_log(info)
            n_trials = min(len(subj_df), len(prior_log))

            target_model_prior = []
            target_oral_score = []
            model_mass_on_oral = []
            model_oral_similarity = []
            oral_ess = []
            valid_oral = []

            for trial_idx in range(n_trials):
                prior = self._normalize_distribution(prior_log[trial_idx])
                choice = int(subj_df.loc[trial_idx, "choice"])

                if oral_mode == "center":
                    center = Oral_center_analysis._parse_center(subj_df.loc[trial_idx, "oral_center"])
                    oral_dist = self._center_oral_distribution(center, choice, partition)
                elif oral_mode == "region":
                    region = (subj_df.loc[trial_idx, "oral_A"], subj_df.loc[trial_idx, "oral_b"])
                    oral_dist = self._region_oral_distribution(
                        region,
                        choice,
                        partition,
                        n_samples=region_n_samples,
                        random_state=42 + trial_idx * 100000,
                    )
                else:
                    raise ValueError(f"Unsupported oral_mode: {oral_mode}")

                valid = not (np.isnan(prior).any() or np.isnan(oral_dist).any())
                valid_oral.append(bool(valid))
                if not valid:
                    target_model_prior.append(np.nan)
                    target_oral_score.append(np.nan)
                    model_mass_on_oral.append(np.nan)
                    model_oral_similarity.append(np.nan)
                    oral_ess.append(np.nan)
                    continue

                target_model_prior.append(float(prior[target_hypo]) if target_hypo < len(prior) else np.nan)
                target_oral_score.append(float(oral_dist[target_hypo]) if target_hypo < len(oral_dist) else np.nan)
                model_mass_on_oral.append(float(np.dot(prior, oral_dist)))
                model_oral_similarity.append(self._js_similarity(prior, oral_dist))
                oral_ess.append(self._effective_sample_size(oral_dist))

            out[iSub] = {
                "iSub": int(iSub),
                "condition": condition,
                "target_hypo": target_hypo,
                "alignment_mode": "oral_t_vs_prior_t",
                "oral_mode": oral_mode,
                "target_model_prior": target_model_prior,
                "target_oral_score": target_oral_score,
                "model_mass_on_oral": model_mass_on_oral,
                "model_oral_similarity": model_oral_similarity,
                "oral_ess": oral_ess,
                "valid_oral": valid_oral,
            }
        return out

    def plot_oral_model_alignment(
        self,
        alignment_results,
        subjects=None,
        save_path=None,
        window_size=16,
        **kwargs,
    ):
        """Plot rolling model-oral alignment metrics by subject."""
        results = self._filter_results(alignment_results, subjects)
        grouped = defaultdict(list)
        for iSub, info in results.items():
            grouped[info["condition"]].append((iSub, info))

        if not grouped:
            raise RuntimeError("No oral-model alignment results to plot.")

        n_rows, n_cols, rows_by_condition = self._layout_by_condition(grouped, kwargs)
        fig = plt.figure(figsize=(n_cols * 8, n_rows * 5))
        fig.suptitle(
            "Oral-Model Alignment (oral_t vs prior_t)",
            fontsize=kwargs.get("fontsize", 16),
            y=kwargs.get("y", 0.99),
        )

        def rolling(values):
            return pd.Series(values, dtype=float).rolling(window=window_size, min_periods=window_size).mean().to_numpy()

        row_offset = 0
        for condition, subs in sorted(grouped.items()):
            for idx, (iSub, info) in enumerate(subs):
                local_row = idx // n_cols
                col = idx % n_cols
                ax = fig.add_subplot(n_rows, n_cols, (row_offset + local_row) * n_cols + col + 1)
                n = len(info.get("model_oral_similarity", []))
                x = np.arange(1, n + 1)
                ax.plot(x, rolling(info.get("model_oral_similarity", [])), lw=2, label="1 - JS(prior, oral)")
                ax.plot(x, rolling(info.get("model_mass_on_oral", [])), lw=2, label="Prior mass on oral")
                ax.plot(x, rolling(info.get("target_model_prior", [])), lw=1.5, alpha=0.8, label="Target prior")
                ax.plot(x, rolling(info.get("target_oral_score", [])), lw=1.5, alpha=0.8, label="Target oral score")
                ax.set_ylim(0, 1)
                ax.set(title=f"Subject {iSub} (Cond {condition})", xlabel="Trial", ylabel="Alignment")
                ax.legend()
            row_offset += rows_by_condition[condition]

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("Oral-model alignment saved to %s", save_path)

    def compute_choice_conditioned_oral_alignment(
        self,
        model_results,
        oral_df,
        oral_mode="center",
        subjects=None,
        region_n_samples=1000,
        beta=10.0,
    ):
        """Compute oral_t alignment with prior_t conditioned on current choice.

        Task timing is stimulus -> choice -> oral report -> feedback, so this
        model state is prior_t after observing stimulus and choice but before
        the feedback-driven posterior update.
        """
        model_res = self._filter_results(model_results, subjects)
        oral_df = oral_df.copy()
        out = {}

        for iSub, info in model_res.items():
            subj_df = oral_df[oral_df["iSub"] == iSub].reset_index(drop=True)
            if subj_df.empty:
                continue

            condition = int(info.get("condition", subj_df["condition"].iloc[0]))
            n_cats = 2 if condition == 1 else 4
            target_hypo = 0 if condition == 1 else 42
            partition = Partition(n_dims=4, n_cats=n_cats)
            prior_log = self._extract_prior_log(info)
            steps = info.get("best_step_results") or info.get("step_results") or []
            n_trials = min(len(subj_df), len(prior_log), len(steps) if steps else len(subj_df))

            choice_conditioned_similarity = []
            choice_conditioned_mass_on_oral = []
            choice_conditioned_target_prior = []
            target_oral_score = []
            expected_center_similarity = []
            valid_oral = []

            for trial_idx in range(n_trials):
                choice = int(subj_df.loc[trial_idx, "choice"])
                step = steps[trial_idx] if trial_idx < len(steps) else {}
                stimulus = step.get("perceived_stimulus")
                if stimulus is None:
                    stimulus = subj_df.loc[trial_idx, ["feature1", "feature2", "feature3", "feature4"]].to_numpy(
                        dtype=float
                    )

                conditioned = self._choice_conditioned_prior(
                    partition=partition,
                    prior=prior_log[trial_idx],
                    stimulus=stimulus,
                    choice=choice,
                    beta=beta,
                )

                oral_center = None
                if oral_mode == "center":
                    oral_center = Oral_center_analysis._parse_center(subj_df.loc[trial_idx, "oral_center"])
                    oral_dist = self._center_oral_distribution(oral_center, choice, partition)
                elif oral_mode == "region":
                    region = (subj_df.loc[trial_idx, "oral_A"], subj_df.loc[trial_idx, "oral_b"])
                    oral_dist = self._region_oral_distribution(
                        region,
                        choice,
                        partition,
                        n_samples=region_n_samples,
                        random_state=4242 + trial_idx * 100000,
                    )
                else:
                    raise ValueError(f"Unsupported oral_mode: {oral_mode}")

                valid = not (np.isnan(conditioned).any() or np.isnan(oral_dist).any())
                valid_oral.append(bool(valid))
                if not valid:
                    choice_conditioned_similarity.append(np.nan)
                    choice_conditioned_mass_on_oral.append(np.nan)
                    choice_conditioned_target_prior.append(np.nan)
                    target_oral_score.append(np.nan)
                    expected_center_similarity.append(np.nan)
                    continue

                choice_conditioned_similarity.append(self._js_similarity(conditioned, oral_dist))
                choice_conditioned_mass_on_oral.append(float(np.dot(conditioned, oral_dist)))
                choice_conditioned_target_prior.append(
                    float(conditioned[target_hypo]) if target_hypo < len(conditioned) else np.nan
                )
                target_oral_score.append(float(oral_dist[target_hypo]) if target_hypo < len(oral_dist) else np.nan)
                if oral_mode == "center":
                    expected_center_similarity.append(
                        self._expected_center_similarity(partition, conditioned, oral_center, choice)
                    )
                else:
                    expected_center_similarity.append(np.nan)

            out[iSub] = {
                "iSub": int(iSub),
                "condition": condition,
                "target_hypo": target_hypo,
                "alignment_mode": "oral_t_vs_choice_conditioned_prior_t",
                "oral_mode": oral_mode,
                "choice_conditioned_similarity": choice_conditioned_similarity,
                "choice_conditioned_mass_on_oral": choice_conditioned_mass_on_oral,
                "choice_conditioned_target_prior": choice_conditioned_target_prior,
                "target_oral_score": target_oral_score,
                "expected_center_similarity": expected_center_similarity,
                "valid_oral": valid_oral,
            }
        return out

    def plot_choice_conditioned_oral_alignment(
        self,
        alignment_results,
        subjects=None,
        save_path=None,
        window_size=16,
        **kwargs,
    ):
        """Plot oral alignment with choice-conditioned prior_t."""
        results = self._filter_results(alignment_results, subjects)
        grouped = defaultdict(list)
        for iSub, info in results.items():
            grouped[info["condition"]].append((iSub, info))

        if not grouped:
            raise RuntimeError("No choice-conditioned oral alignment results to plot.")

        n_rows, n_cols, rows_by_condition = self._layout_by_condition(grouped, kwargs)
        fig = plt.figure(figsize=(n_cols * 8, n_rows * 5))
        fig.suptitle(
            "Oral Alignment with Choice-Conditioned Prior",
            fontsize=kwargs.get("fontsize", 16),
            y=kwargs.get("y", 0.99),
        )

        def rolling(values):
            return pd.Series(values, dtype=float).rolling(window=window_size, min_periods=window_size).mean().to_numpy()

        row_offset = 0
        for condition, subs in sorted(grouped.items()):
            for idx, (iSub, info) in enumerate(subs):
                local_row = idx // n_cols
                col = idx % n_cols
                ax = fig.add_subplot(n_rows, n_cols, (row_offset + local_row) * n_cols + col + 1)
                n = len(info.get("choice_conditioned_similarity", []))
                x = np.arange(1, n + 1)
                ax.plot(
                    x,
                    rolling(info.get("choice_conditioned_similarity", [])),
                    lw=2,
                    label="1 - JS(choice-conditioned, oral)",
                )
                ax.plot(
                    x,
                    rolling(info.get("choice_conditioned_mass_on_oral", [])),
                    lw=2,
                    label="Choice-cond. mass on oral",
                )
                center_vals = info.get("expected_center_similarity", [])
                if center_vals and not np.all(np.isnan(np.asarray(center_vals, dtype=float))):
                    ax.plot(x, rolling(center_vals), lw=2, label="Expected center similarity")
                ax.plot(
                    x,
                    rolling(info.get("choice_conditioned_target_prior", [])),
                    lw=1.5,
                    alpha=0.8,
                    label="Target choice-cond. prior",
                )
                ax.plot(x, rolling(info.get("target_oral_score", [])), lw=1.5, alpha=0.8, label="Target oral score")
                ax.set_ylim(0, 1)
                ax.set(title=f"Subject {iSub} (Cond {condition})", xlabel="Trial", ylabel="Alignment")
                ax.legend()
            row_offset += rows_by_condition[condition]

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("Choice-conditioned oral alignment saved to %s", save_path)

    def plot_k_oral_comparison(
        self,
        model_results,
        oral_results,
        subjects=None,
        save_path=None,
        window_size=16,
        **kwargs,
    ):
        """Compare true-hypothesis posterior with oral hit trajectories."""

        def _get_post_max(hypo_details, k_special):
            if not isinstance(hypo_details, dict):
                return 0.0
            entry = hypo_details.get(k_special)
            if entry is None:
                entry = hypo_details.get(str(k_special))
            if not isinstance(entry, dict):
                return 0.0
            return entry.get("post_max", 0.0)

        def extract_model_ma(step_results, k_special, win):
            posts = []
            for sr in step_results:
                p = _get_post_max(sr.get("hypo_details", {}), k_special)
                try:
                    p = float(p)
                except (TypeError, ValueError):
                    p = 0.0
                posts.append(p)
            return pd.Series(posts, dtype=float).rolling(window=win, min_periods=win).mean().to_numpy()

        def extract_oral_ma(hits, win):
            rolling = []
            n = len(hits)
            for i in range(n):
                if i + 1 < win:
                    rolling.append(np.nan)
                    continue
                window = np.asarray(hits[i - win + 1 : i + 1], dtype=float)
                if np.all(np.isnan(window)):
                    rolling.append(np.nan)
                else:
                    rolling.append(float(np.nanmean(window)))
            return np.array(rolling)

        model_res = self._filter_results(model_results, subjects)
        oral_res = self._filter_results(oral_results, subjects)

        grouped = defaultdict(list)
        for iSub, info in model_res.items():
            grouped[info["condition"]].append(iSub)

        if not grouped:
            raise RuntimeError("No model/oral comparison results to plot.")

        n_rows, n_cols, rows_by_condition = self._layout_by_condition(grouped, kwargs)
        fig = plt.figure(figsize=(n_cols * 8, n_rows * 5))
        fig.suptitle(
            "Model k vs Oral k (Filtered & Smoothed)",
            fontsize=kwargs.get("fontsize", 16),
            y=kwargs.get("y", 0.99),
        )

        row_offset = 0
        for condition, subs in sorted(grouped.items()):
            for idx, iSub in enumerate(subs):
                local_row = idx // n_cols
                col = idx % n_cols
                ax = fig.add_subplot(n_rows, n_cols, (row_offset + local_row) * n_cols + col + 1)

                info = model_res[iSub]
                step_results = info.get("step_results", info.get("best_step_results", []))
                target_hypo = 0 if condition == 1 else 42
                oral_hits = oral_res[iSub]["hits"]

                rolling_model = extract_model_ma(step_results, target_hypo, window_size)
                valid_idx = np.arange(len(rolling_model))
                x_model = np.array(valid_idx)[window_size - 1 :] + 1
                ax.plot(x_model, rolling_model[window_size - 1 :], lw=2, label="Model k")

                rolling_oral = extract_oral_ma(oral_hits, window_size)
                x_oral = np.arange(1, len(rolling_oral) + 1)
                ax.plot(x_oral, rolling_oral, lw=2, label="Oral k")

                ax.set_ylim(0, 1)
                ax.set(title=f"Subject {iSub} (Cond {condition})", xlabel="Trial", ylabel="Probability")
                ax.legend()
            row_offset += rows_by_condition[condition]

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("Filtered comparison saved to %s", save_path)

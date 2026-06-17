"""Oral report and model-alignment utilities.

The module has two layers.

1. Oral report -> hypothesis mappings:
   - ``Oral_region_mapping`` maps reported regions (A, b) to candidate
     hypotheses using Monte Carlo overlap metrics.
   - ``Oral_center_mapping`` maps reported feature centers to candidate
     hypotheses using prototype distance.

2. Oral/model alignment methods mixed into ``ModelEval``:
   - Distribution-based alignment: project oral reports into hypothesis-space
     distributions and compare them with model belief distributions.
   - Oral-based alignment: project model belief distributions into the same
     representation as the oral report itself: centers for center reports and
     fuzzy regions for region reports.
   - Target-based alignment: compare target-hypothesis prior probability with
     target-hypothesis oral mass.
   - Hit-based alignment: compare binary target hits in the model active set
     and the oral top-N set, where N is the model active-set size.
   - Coverage-based alignment: compare how well the model active set covers
     the oral top-N set as a whole.

The five blocks above are the intended analysis spine.
"""

from __future__ import annotations

import ast
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    from statsmodels.stats.anova import AnovaRM
except ImportError:  # pragma: no cover - optional dependency in some environments.
    AnovaRM = None

from ..problems.partitions import Partition


logger = logging.getLogger(__name__)


_REGION_SCORER_CACHE: Dict[Tuple[Any, ...], "RegionOverlapScorer"] = {}
_REGION_DISTRIBUTION_CACHE: Dict[Tuple[Any, ...], np.ndarray] = {}
_ORAL_EQUIVALENCE_GROUP_CACHE: Dict[Tuple[Any, ...], Tuple[np.ndarray, Tuple[str, ...]]] = {}


__all__ = [
    "OralModelAlignmentMixin",
    "Oral_center_mapping",
    "Oral_region_mapping",
]


# ---------------------------------------------------------------------------
# Oral report -> hypothesis mappings used by oral mass and alignment methods
# ---------------------------------------------------------------------------


class Oral_region_mapping:
    """Region-based oral analysis with overlap scoring.

    Oral reports and model partition regions use the same canonical constraint
    form: ``A @ x <= b``.
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
            candidates = (
                (hypo_idx, cat_idx),
                (hypo_idx, 0, cat_idx),
                (hypo_idx, 1, cat_idx),
                (hypo_idx, 1, cat_idx, slice(None)),
            )
            for index in candidates:
                try:
                    candidate = regions[index]
                except (IndexError, TypeError):
                    continue
                A, _ = Oral_region_mapping._parse_region(candidate)
                if A is not None:
                    return candidate
            raise TypeError(
                f"Could not locate parseable region in ndarray with shape {regions.shape}."
            )
        if isinstance(regions, (list, tuple)):
            return regions[hypo_idx][cat_idx]
        raise TypeError(f"Unsupported partition_model.regions type: {type(regions)}")

class RegionOverlapScorer:
    """Fast Monte Carlo scorer for oral regions against all hypothesis regions.

    It fixes one point cloud per partition/category and precomputes the
    hypothesis-region inclusion masks. Per oral trial, the only expensive work
    left is computing the oral mask once, then vectorized boolean overlap
    against all hypothesis masks.
    """

    def __init__(
        self,
        partition: Partition,
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
        self.n_hypos = int(len(partition.regions))
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
                region = Oral_region_mapping._true_region(self.partition.regions, hypo_idx, cat_idx)
                A, b = Oral_region_mapping._parse_region(region)
                cat_masks[hypo_idx] = Oral_region_mapping._points_in_region(
                    self.points,
                    A,
                    b,
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
        if metric not in Oral_region_mapping.VALID_OVERLAP_METRICS:
            raise ValueError(
                f"Unsupported overlap_metric={metric}. "
                f"Choose from {sorted(Oral_region_mapping.VALID_OVERLAP_METRICS)}."
            )
        if cat_idx < 0 or cat_idx >= self.n_cats:
            return np.full(self.n_hypos, np.nan, dtype=float)

        A, b = Oral_region_mapping._parse_region(oral_region)
        if A is None or b is None:
            return np.full(self.n_hypos, np.nan, dtype=float)

        oral_mask = Oral_region_mapping._points_in_region(
            self.points,
            A,
            b,
            dist_tol=self.dist_tol,
        )
        hypo_masks = self.hypothesis_masks[int(cat_idx)]
        intersection_count = np.sum(hypo_masks & oral_mask[None, :], axis=1).astype(float)
        oral_count = float(np.sum(oral_mask))
        hypo_count = np.sum(hypo_masks, axis=1).astype(float)
        union_count = hypo_count + oral_count - intersection_count
        total_weight = float(self.n_samples)

        if metric == "iou":
            return np.divide(
                intersection_count,
                union_count,
                out=np.zeros_like(intersection_count, dtype=float),
                where=union_count > 0,
            )
        if metric == "intersection":
            return (intersection_count / total_weight) * self.box_volume
        if metric == "precision_like":
            return np.divide(
                intersection_count,
                oral_count,
                out=np.zeros_like(intersection_count, dtype=float),
                where=oral_count > 0,
            )
        if metric == "recall_like":
            return np.divide(
                intersection_count,
                hypo_count,
                out=np.zeros_like(intersection_count, dtype=float),
                where=hypo_count > 0,
            )
        raise ValueError(f"Unsupported overlap metric: {metric}")


class Oral_center_mapping:
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

class OralModelAlignmentMixin:
    """Oral/model alignment methods mixed into ``ModelEval``.

    Public methods are organized around five intended analysis families:

    1. Distribution-based alignment:
       ``compute_distribution_based_alignment`` and its group/subject plots.
       Oral reports are first mapped into hypothesis-space distributions.
    2. Oral-based alignment:
       ``compute_oral_based_alignment`` and its group/subject plots. Model
       belief is mapped into the native oral representation: an expected center
       for center mode, or a fuzzy region field for region mode.
    3. Target-based alignment:
       ``compute_target_based_alignment`` and its group/subject plots. These
       compare ``prior_t[target]`` with ``oral_t[target]`` directly.
    4. Hit-based alignment:
       ``compute_hit_based_alignment`` and its group/subject plots. These
       binarize the target signal: model hit = target in active set; oral hit =
       target in oral top-N, where N is the active-set size.
    5. Coverage-based alignment:
       ``compute_coverage_based_alignment`` and its group/subject plots. These
       compare model active-set coverage of the whole oral top-N set.

    Full oral mass display is kept as a shared utility for the main alignment
    blocks.

    The host class is expected to provide ``_filter_results`` and
    ``_layout_by_condition``. ``ModelEval`` supplies both.
    """

    @staticmethod
    def _subjectwise_grid_layout(
        subjects,
        n_cols,
        *,
        panel_width=8.0,
        panel_height=5.0,
    ):
        """Return a subject-wise grid using the oral-mass plot layout scale."""
        n_subjects = len(subjects)
        if n_subjects <= 0:
            raise RuntimeError("No subjects available for subject-wise plot.")

        requested_cols = max(1, int(n_cols))
        actual_cols = max(1, min(requested_cols, n_subjects))
        n_rows = int(np.ceil(n_subjects / actual_cols))
        return n_rows, actual_cols, (actual_cols * float(panel_width), n_rows * float(panel_height))

    SUBJECTWISE_SUPTITLE_FONTSIZE = 16
    SUBJECTWISE_TITLE_FONTSIZE = 12
    SUBJECTWISE_LABEL_FONTSIZE = 10
    SUBJECTWISE_TICK_FONTSIZE = 10
    SUBJECTWISE_LEGEND_FONTSIZE = 10

    def _style_subjectwise_grid_axes(self, axes, n_rows, n_cols, ylabel, xlabel="Normalized trial"):
        axes = np.asarray(axes)
        for ax in axes.flat:
            ax.tick_params(axis="both", labelsize=self.SUBJECTWISE_TICK_FONTSIZE)
        for row in range(n_rows):
            axes[row, 0].set_ylabel(ylabel, fontsize=self.SUBJECTWISE_LABEL_FONTSIZE)
        for col in range(n_cols):
            axes[-1, col].set_xlabel(xlabel, fontsize=self.SUBJECTWISE_LABEL_FONTSIZE)

    DISTRIBUTION_ALIGNMENT_SPACES = ("full", "active", "union_topn")
    DISTRIBUTION_ALIGNMENT_LABELS = {
        "full": "Full hypothesis space",
        "active": "Model active set",
        "union_topn": "Active + oral top-N union",
    }
    DISTRIBUTION_ALIGNMENT_SHORT_LABELS = {
        "full": "Full",
        "active": "Active",
        "union_topn": "Union",
    }
    DISTRIBUTION_ALIGNMENT_COLORS = {
        "full": "#4c78a8",
        "active": "#f58518",
        "union_topn": "#54a24b",
    }
    ORAL_BASED_PRIMARY_METRIC = {
        "center": "expected_center_similarity",
        "region": "fuzzy_iou_similarity",
    }
    ORAL_BASED_METRIC_LABELS = {
        "expected_center_similarity": "Expected center similarity",
        "fuzzy_iou_similarity": "Fuzzy region IoU",
        "fuzzy_cosine_similarity": "Fuzzy region cosine",
        "model_mass_inside_oral": "Model mass inside oral region",
        "oral_region_covered_by_model": "Oral region covered by model",
    }
    ORAL_BASED_METRIC_COLORS = {
        "expected_center_similarity": "#4c78a8",
        "fuzzy_iou_similarity": "#54a24b",
        "fuzzy_cosine_similarity": "#f58518",
        "model_mass_inside_oral": "#b279a2",
        "oral_region_covered_by_model": "#e45756",
    }
    TARGET_BASED_METRICS = ("pearson_r", "spearman_rho", "cosine_similarity")
    TARGET_BASED_METRIC_LABELS = {
        "pearson_r": "Pearson r",
        "spearman_rho": "Spearman rho",
        "cosine_similarity": "Cosine similarity",
    }
    TARGET_BASED_METRIC_COLORS = {
        "pearson_r": "#8e44ad",
        "spearman_rho": "#c0392b",
        "cosine_similarity": "#7f8c8d",
    }
    TARGET_BASED_LINE_COLORS = {
        "model": "#8e44ad",
        "oral": "#c0392b",
    }
    TARGET_ALIGNMENT_SPACES = ("full", "active", "union_topn")
    TARGET_ALIGNMENT_LABELS = {
        "full": "Full hypothesis space",
        "active": "Model active set",
        "union_topn": "Active + oral top-N union",
    }
    TARGET_ALIGNMENT_SUFFIXES = {
        "full": "full",
        "active": "active",
        "union_topn": "union",
    }
    HIT_BASED_METRICS = ("phi_correlation", "cohen_kappa", "hit_agreement_rate", "positive_hit_jaccard")
    HIT_BASED_METRIC_LABELS = {
        "phi_correlation": "Phi correlation",
        "cohen_kappa": "Cohen kappa",
        "hit_agreement_rate": "Agreement rate",
        "positive_hit_jaccard": "Positive-hit Jaccard",
    }
    HIT_BASED_METRIC_COLORS = {
        "phi_correlation": "#2d3436",
        "cohen_kappa": "#6c5ce7",
        "hit_agreement_rate": "#e17055",
        "positive_hit_jaccard": "#00cec9",
    }
    HIT_BASED_LINE_COLORS = {
        "model": "#2d3436",
        "oral": "#d35400",
    }
    COVERAGE_BASED_METRICS = ("active_capture_ratio", "active_topn_overlap")
    COVERAGE_BASED_LABELS = {
        "active_capture_ratio": "Active/oral top-N mass ratio",
        "active_topn_overlap": "Active/oral top-N overlap",
    }
    COVERAGE_BASED_COLORS = {
        "active_capture_ratio": "#1f77b4",
        "active_topn_overlap": "#ff7f0e",
    }

    # -----------------------------------------------------------------------
    # Shared distribution and plotting helpers
    # -----------------------------------------------------------------------

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
    def _active_hypothesis_indices(values, active_threshold=1e-12):
        """Return indices that form the current model hypothesis set."""
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size == 0:
            return np.asarray([], dtype=int)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return np.flatnonzero(arr > float(active_threshold)).astype(int)

    @staticmethod
    def _oral_topn_indices(oral_dist, n_top):
        """Return oral top-N hypothesis indices."""
        oral = np.asarray(oral_dist, dtype=float).reshape(-1)
        if oral.size == 0 or int(n_top) <= 0 or np.isnan(oral).any():
            return np.asarray([], dtype=int)
        n_top = min(int(n_top), oral.size)
        return np.argsort(oral)[::-1][:n_top].astype(int)

    @staticmethod
    def _target_rank(values, target_hypo, min_value=0.0):
        """Return the 1-based descending rank of target_hypo, or NaN if absent."""
        arr = np.asarray(values, dtype=float).reshape(-1)
        target = int(target_hypo)
        if target < 0 or target >= arr.size or np.isnan(arr).all():
            return np.nan
        arr = np.nan_to_num(arr, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
        target_value = float(arr[target])
        if not np.isfinite(target_value) or target_value <= float(min_value):
            return np.nan
        return float(1 + np.sum(arr > target_value))

    @staticmethod
    def _resolve_rank_top_k(rank_top_k, condition):
        """Resolve fixed or condition-specific rank-hit K."""
        if rank_top_k is None:
            return None
        if isinstance(rank_top_k, dict):
            value = rank_top_k.get(int(condition))
        else:
            value = rank_top_k
        if value is None:
            return None
        value = int(value)
        if value <= 0:
            raise ValueError(f"rank_top_k must be positive, got {value}")
        return value

    @staticmethod
    def _rounded_signature(values, decimals=12):
        """Return a stable signature for numeric oral-representation values."""
        arr = np.asarray(values, dtype=float)
        arr = np.round(arr, int(decimals))
        return tuple(arr.reshape(-1).tolist())

    @staticmethod
    def _region_signature(region, decimals=12):
        """Return a stable signature for one hypothesis region."""
        A, b = Oral_region_mapping._parse_region(region)
        if A is None or b is None:
            return ("invalid",)
        A = np.round(np.asarray(A, dtype=float), int(decimals))
        b = np.round(np.asarray(b, dtype=float), int(decimals))
        return (
            A.shape,
            tuple(A.reshape(-1).tolist()),
            b.shape,
            tuple(b.reshape(-1).tolist()),
        )

    @staticmethod
    def _oral_equivalence_groups(partition, choice, oral_mode="center", decimals=12):
        """Group hypotheses that are indistinguishable in the oral representation.

        The grouping is trial-specific through ``choice``: hypotheses are
        grouped by the category representation that the participant is
        reporting about. For center mode the key is the prototype center; for
        region mode the key is the boundary region ``(A, b)``. Both oral mass
        and model prior can then be summed over the same groups.
        """
        mode = str(oral_mode).strip().lower()
        cat_idx = int(choice) - 1
        key = (
            partition.__class__.__name__,
            int(partition.n_dims),
            int(partition.n_cats),
            int(cat_idx),
            mode,
            int(decimals),
        )
        cached = _ORAL_EQUIVALENCE_GROUP_CACHE.get(key)
        if cached is not None:
            group_ids, labels = cached
            return group_ids.copy(), tuple(labels)

        if cat_idx < 0 or cat_idx >= int(partition.n_cats):
            return np.full(int(partition.length), -1, dtype=int), tuple()

        signature_to_group: Dict[Any, int] = {}
        labels: List[str] = []
        group_ids = np.full(int(partition.length), -1, dtype=int)

        for hypo_idx in range(int(partition.length)):
            if mode == "center":
                signature = OralModelAlignmentMixin._rounded_signature(
                    partition.prototypes[hypo_idx, 0, cat_idx, :],
                    decimals=decimals,
                )
            elif mode == "region":
                region = Oral_region_mapping._true_region(partition.regions, hypo_idx, cat_idx)
                signature = OralModelAlignmentMixin._region_signature(region, decimals=decimals)
            else:
                raise ValueError(f"Unsupported oral_mode for equivalence groups: {oral_mode}")

            if signature not in signature_to_group:
                signature_to_group[signature] = len(signature_to_group)
                labels.append(str(signature))
            group_ids[hypo_idx] = signature_to_group[signature]

        out_labels = tuple(labels)
        _ORAL_EQUIVALENCE_GROUP_CACHE[key] = (group_ids.copy(), out_labels)
        return group_ids, out_labels

    @staticmethod
    def _project_distribution_to_groups(values, group_ids, normalize=True):
        """Sum a hypothesis distribution over oral-equivalence groups."""
        arr = np.asarray(values, dtype=float).reshape(-1)
        groups = np.asarray(group_ids, dtype=int).reshape(-1)
        n = min(arr.size, groups.size)
        if n <= 0 or np.isnan(arr[:n]).all():
            return np.asarray([np.nan], dtype=float)

        arr = arr[:n]
        groups = groups[:n]
        valid_group = groups >= 0
        if not np.any(valid_group):
            return np.asarray([np.nan], dtype=float)

        n_groups = int(np.max(groups[valid_group])) + 1
        out = np.zeros(n_groups, dtype=float)
        clean = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        for value, group in zip(clean[valid_group], groups[valid_group]):
            if value > 0:
                out[int(group)] += float(value)

        if not normalize:
            return out
        return OralModelAlignmentMixin._normalize_distribution(out)

    @staticmethod
    def _comparison_space_distributions(
        model_dist,
        oral_dist,
        alignment_space="active",
        active_idx=None,
    ):
        """Project model/oral distributions onto the requested comparison space."""
        model_arr = np.asarray(model_dist, dtype=float).reshape(-1)
        oral_arr = np.asarray(oral_dist, dtype=float).reshape(-1)
        n_hypos = min(model_arr.size, oral_arr.size)
        if n_hypos <= 0:
            return (
                np.asarray([np.nan], dtype=float),
                np.asarray([np.nan], dtype=float),
                np.asarray([], dtype=int),
            )

        if alignment_space == "full":
            compare_idx = np.arange(n_hypos, dtype=int)
        elif alignment_space == "active":
            if active_idx is None:
                active_idx = OralModelAlignmentMixin._active_hypothesis_indices(model_arr)
            compare_idx = np.asarray(active_idx, dtype=int).reshape(-1)
            compare_idx = compare_idx[(compare_idx >= 0) & (compare_idx < n_hypos)]
        elif alignment_space == "union_topn":
            if active_idx is None:
                active_idx = OralModelAlignmentMixin._active_hypothesis_indices(model_arr)
            active_idx = np.asarray(active_idx, dtype=int).reshape(-1)
            active_idx = active_idx[(active_idx >= 0) & (active_idx < n_hypos)]
            oral_topn_idx = OralModelAlignmentMixin._oral_topn_indices(oral_arr, len(active_idx))
            oral_topn_idx = oral_topn_idx[(oral_topn_idx >= 0) & (oral_topn_idx < n_hypos)]
            compare_idx = np.union1d(active_idx, oral_topn_idx).astype(int)
        else:
            raise ValueError(f"Unsupported alignment_space: {alignment_space}")

        if compare_idx.size == 0:
            return (
                np.asarray([np.nan], dtype=float),
                np.asarray([np.nan], dtype=float),
                compare_idx,
            )

        return (
            OralModelAlignmentMixin._normalize_distribution(model_arr[compare_idx]),
            OralModelAlignmentMixin._normalize_distribution(oral_arr[compare_idx]),
            compare_idx,
        )

    @staticmethod
    def _target_probability_in_space(prob, compare_idx, target_hypo):
        """Return target probability after projection; absent target is zero."""
        p = np.asarray(prob, dtype=float).reshape(-1)
        idx = np.asarray(compare_idx, dtype=int).reshape(-1)
        if p.size == 0 or np.isnan(p).any() or idx.size == 0:
            return np.nan
        loc = np.flatnonzero(idx == int(target_hypo))
        if loc.size == 0:
            return 0.0
        return float(p[int(loc[0])])

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
    def _extract_model_distribution_log(info, model_distribution="prior"):
        """Return the model distribution time series used by distribution alignment."""
        state = str(model_distribution).strip().lower()
        if state == "prior":
            return OralModelAlignmentMixin._extract_prior_log(info)
        if state != "posterior":
            raise ValueError("model_distribution must be 'posterior' or 'prior'.")

        posterior_log = info.get("posterior_log") or []
        if posterior_log:
            return [np.asarray(x, dtype=float) for x in posterior_log]

        posteriors = []
        for step in info.get("best_step_results", []) or []:
            posterior = step.get("posterior")
            if posterior is None:
                posterior = step.get("post")
            if posterior is None:
                return []
            posteriors.append(np.asarray(posterior, dtype=float))
        return posteriors

    @staticmethod
    def _sem(values):
        arr = np.asarray(values, dtype=float).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size <= 1:
            return np.nan
        return float(np.std(arr, ddof=1) / np.sqrt(arr.size))

    @staticmethod
    def _rolling_mean(values, window_size=16):
        """Rolling mean for subject plots with a tolerant valid-sample rule."""
        window = max(1, int(window_size))
        min_periods = max(1, window // 4)
        return pd.Series(values, dtype=float).rolling(window=window, min_periods=min_periods).mean().to_numpy()

    @staticmethod
    def _format_p_value(p_value):
        try:
            p = float(p_value)
        except (TypeError, ValueError):
            return "n/a"
        if not np.isfinite(p):
            return "n/a"
        if p < 0.001:
            return "<.001"
        return f"={p:.3f}"

    @staticmethod
    def _safe_pearson(x, y):
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) < 2:
            return np.nan
        x = x[mask]
        y = y[mask]
        if np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
            return np.nan
        return float(stats.pearsonr(x, y).statistic)

    @staticmethod
    def _safe_spearman(x, y):
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) < 2:
            return np.nan
        x = x[mask]
        y = y[mask]
        if np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
            return np.nan
        return float(stats.spearmanr(x, y).statistic)

    @staticmethod
    def _safe_cosine_similarity(x, y):
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) < 1:
            return np.nan
        x = x[mask]
        y = y[mask]
        denom = float(np.linalg.norm(x) * np.linalg.norm(y))
        if denom <= 1e-12:
            return np.nan
        return float(np.clip(np.dot(x, y) / denom, -1.0, 1.0))

    @staticmethod
    def _safe_cohen_kappa(x, y):
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) < 1:
            return np.nan
        xb = x[mask] > 0.5
        yb = y[mask] > 0.5
        observed = float(np.mean(xb == yb))
        px = float(np.mean(xb))
        py = float(np.mean(yb))
        expected = px * py + (1.0 - px) * (1.0 - py)
        denom = 1.0 - expected
        if denom <= 1e-12:
            return np.nan
        return float((observed - expected) / denom)

    @staticmethod
    def _safe_binary_jaccard(x, y):
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) < 1:
            return np.nan
        xb = x[mask] > 0.5
        yb = y[mask] > 0.5
        union = int(np.sum(xb | yb))
        if union <= 0:
            return np.nan
        return float(np.sum(xb & yb) / union)

    @staticmethod
    def _holm_adjust_pvalues(p_values):
        """Holm-adjust a sequence of p-values while preserving NaNs."""
        p = np.asarray(p_values, dtype=float)
        adjusted = np.full(p.shape, np.nan, dtype=float)
        finite_idx = np.flatnonzero(np.isfinite(p))
        if finite_idx.size == 0:
            return adjusted

        ordered = finite_idx[np.argsort(p[finite_idx])]
        m = int(ordered.size)
        running_max = 0.0
        for rank, idx in enumerate(ordered):
            candidate = min(1.0, float(p[idx]) * float(m - rank))
            running_max = max(running_max, candidate)
            adjusted[idx] = running_max
        return adjusted

    @classmethod
    def _paired_distribution_space_stats(cls, subject_space_means, spaces):
        """Return compact paired statistics for the group-level bar panel."""
        pivot = subject_space_means.reindex(columns=list(spaces))
        complete = pivot.dropna()
        n_complete = int(len(complete))

        friedman_p = np.nan
        if n_complete >= 3 and len(spaces) >= 3:
            try:
                samples = [complete[space].to_numpy(dtype=float) for space in spaces]
                friedman_p = float(stats.friedmanchisquare(*samples).pvalue)
            except ValueError:
                friedman_p = np.nan

        pairs = []
        p_values = []
        for left_idx in range(len(spaces)):
            for right_idx in range(left_idx + 1, len(spaces)):
                left = spaces[left_idx]
                right = spaces[right_idx]
                pair = pivot[[left, right]].dropna()
                p_val = np.nan
                if len(pair) >= 2:
                    try:
                        p_val = float(stats.wilcoxon(pair[left], pair[right], zero_method="wilcox").pvalue)
                    except ValueError:
                        diff = pair[left].to_numpy(dtype=float) - pair[right].to_numpy(dtype=float)
                        p_val = 1.0 if np.allclose(diff, 0.0, equal_nan=False) else np.nan
                pairs.append((left, right))
                p_values.append(p_val)

        adjusted = cls._holm_adjust_pvalues(p_values)
        pair_text = []
        for (left, right), p_adj in zip(pairs, adjusted):
            left_label = cls.DISTRIBUTION_ALIGNMENT_SHORT_LABELS.get(left, left)
            right_label = cls.DISTRIBUTION_ALIGNMENT_SHORT_LABELS.get(right, right)
            pair_text.append(f"{left_label}-{right_label} p{cls._format_p_value(p_adj)}")

        return {
            "n": n_complete,
            "friedman_p": friedman_p,
            "pair_text": "; ".join(pair_text),
        }

    @classmethod
    def _distribution_space_time_stats(cls, distribution_results, spaces, bins=20):
        """Run a two-way repeated-measures ANOVA over space and normalized time bins."""
        if AnovaRM is None:
            return {"n": 0, "space_p": np.nan, "time_p": np.nan, "interaction_p": np.nan}

        df = distribution_results.copy()
        df = df[df["alignment_space"].isin(spaces)]
        df = df[np.isfinite(df["js_similarity"])]
        if df.empty:
            return {"n": 0, "space_p": np.nan, "time_p": np.nan, "interaction_p": np.nan}

        df["trial_bin"] = pd.cut(
            df["trial_pct"],
            bins=np.linspace(0, 1, int(bins) + 1),
            labels=np.arange(1, int(bins) + 1),
            include_lowest=True,
        ).astype(int)
        subject_bin = (
            df.groupby(["subject", "alignment_space", "trial_bin"], observed=True)["js_similarity"]
            .mean()
            .reset_index()
        )

        expected_cells = int(len(spaces) * int(bins))
        counts = subject_bin.groupby("subject").size()
        complete_subjects = counts[counts == expected_cells].index
        complete = subject_bin[subject_bin["subject"].isin(complete_subjects)].copy()
        if complete["subject"].nunique() < 3:
            return {"n": int(complete["subject"].nunique()), "space_p": np.nan, "time_p": np.nan, "interaction_p": np.nan}

        complete["alignment_space"] = complete["alignment_space"].astype(str)
        complete["trial_bin"] = complete["trial_bin"].astype(str)

        try:
            fit = AnovaRM(
                complete,
                depvar="js_similarity",
                subject="subject",
                within=["alignment_space", "trial_bin"],
            ).fit()
        except Exception:
            return {"n": int(complete["subject"].nunique()), "space_p": np.nan, "time_p": np.nan, "interaction_p": np.nan}

        table = fit.anova_table
        out = {
            "n": int(complete["subject"].nunique()),
            "space_p": np.nan,
            "time_p": np.nan,
            "interaction_p": np.nan,
        }
        for idx, row in table.iterrows():
            idx_str = str(idx)
            p_val = float(row.get("Pr > F", np.nan))
            if idx_str == "alignment_space":
                out["space_p"] = p_val
            elif idx_str == "trial_bin":
                out["time_p"] = p_val
            elif "alignment_space" in idx_str and "trial_bin" in idx_str:
                out["interaction_p"] = p_val
        return out

    @staticmethod
    def _center_oral_distribution(center, choice, partition):
        """Map one oral center report to a full hypothesis distribution."""
        center = np.asarray(center, dtype=float).reshape(-1)
        if center.size == 0 or np.isnan(center).any():
            return np.full(partition.length, np.nan, dtype=float)

        cat_idx = int(choice) - 1
        if cat_idx < 0 or cat_idx >= int(partition.n_cats):
            return np.full(partition.length, np.nan, dtype=float)
        distances = np.linalg.norm(partition.prototypes[:, 0, cat_idx, :] - center, axis=1)
        return OralModelAlignmentMixin._adaptive_softmax_from_distances(distances)

    @staticmethod
    def _get_region_overlap_scorer(
        partition,
        n_samples=1000,
        bounds=(0.0, 1.0),
        random_state=42,
        dist_tol=1e-9,
    ):
        """Return cached region scorer for fixed Monte Carlo points."""
        if random_state is None:
            return RegionOverlapScorer(
                partition=partition,
                n_samples=n_samples,
                bounds=bounds,
                random_state=random_state,
                dist_tol=dist_tol,
            )

        key = (
            partition.__class__.__name__,
            int(partition.n_dims),
            int(partition.n_cats),
            int(n_samples),
            float(bounds[0]),
            float(bounds[1]),
            int(random_state),
            float(dist_tol),
        )
        scorer = _REGION_SCORER_CACHE.get(key)
        if scorer is None:
            scorer = RegionOverlapScorer(
                partition=partition,
                n_samples=n_samples,
                bounds=bounds,
                random_state=random_state,
                dist_tol=dist_tol,
            )
            _REGION_SCORER_CACHE[key] = scorer
        return scorer

    @staticmethod
    def _region_distribution_cache_key(
        region,
        choice,
        partition,
        n_samples=1000,
        bounds=(0.0, 1.0),
        random_state=42,
        dist_tol=1e-9,
    ):
        """Build a stable cache key for one oral region distribution."""
        if random_state is None:
            return None
        A, b = Oral_region_mapping._parse_region(region)
        if A is None or b is None:
            return None
        A = np.ascontiguousarray(A, dtype=float)
        b = np.ascontiguousarray(b, dtype=float)
        return (
            partition.__class__.__name__,
            int(partition.n_dims),
            int(partition.n_cats),
            int(n_samples),
            float(bounds[0]),
            float(bounds[1]),
            int(random_state),
            float(dist_tol),
            int(choice),
            A.shape,
            A.tobytes(),
            b.shape,
            b.tobytes(),
        )

    @staticmethod
    def _region_oral_distribution(
        region,
        choice,
        partition,
        n_samples=1000,
        random_state=42,
    ):
        """Map one oral region report to a hypothesis distribution."""
        cache_key = OralModelAlignmentMixin._region_distribution_cache_key(
            region,
            choice,
            partition,
            n_samples=int(n_samples),
            bounds=(0.0, 1.0),
            random_state=random_state,
            dist_tol=1e-9,
        )
        if cache_key is not None and cache_key in _REGION_DISTRIBUTION_CACHE:
            return _REGION_DISTRIBUTION_CACHE[cache_key].copy()

        cat_idx = int(choice) - 1
        scorer = OralModelAlignmentMixin._get_region_overlap_scorer(
            partition=partition,
            n_samples=int(n_samples),
            bounds=(0.0, 1.0),
            random_state=random_state,
            dist_tol=1e-9,
        )
        scores = scorer.score_all(
            region,
            cat_idx=cat_idx,
            metric="iou",
        )
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        dist = OralModelAlignmentMixin._normalize_distribution(scores)
        if cache_key is not None and not np.isnan(dist).any():
            _REGION_DISTRIBUTION_CACHE[cache_key] = dist.copy()
        return dist

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
    def _stimulus_for_trial(info, subj_df, trial_idx):
        """Return perceived stimulus if logged, otherwise observed feature columns."""
        steps = info.get("best_step_results") or info.get("step_results") or []
        if trial_idx < len(steps):
            stimulus = steps[trial_idx].get("perceived_stimulus")
            if stimulus is not None:
                return np.asarray(stimulus, dtype=float)
        feature_cols = ["feature1", "feature2", "feature3", "feature4"]
        if all(col in subj_df.columns for col in feature_cols):
            return subj_df.loc[trial_idx, feature_cols].to_numpy(dtype=float)
        return np.full(4, np.nan, dtype=float)

    @staticmethod
    def _model_distribution_for_oral_alignment(
        info,
        subj_df,
        trial_idx,
        partition,
        choice,
        model_distribution="choice_conditioned_prior",
        beta=10.0,
    ):
        """Return the model belief state aligned to oral-report timing."""
        state = str(model_distribution).strip().lower().replace("-", "_")
        if state in {"choice_conditioned", "choice_conditioned_prior", "choice_conditional_prior"}:
            prior_log = OralModelAlignmentMixin._extract_prior_log(info)
            if trial_idx >= len(prior_log):
                return np.asarray([], dtype=float)
            stimulus = OralModelAlignmentMixin._stimulus_for_trial(info, subj_df, trial_idx)
            return OralModelAlignmentMixin._choice_conditioned_prior(
                partition=partition,
                prior=prior_log[trial_idx],
                stimulus=stimulus,
                choice=choice,
                beta=beta,
            )

        model_log = OralModelAlignmentMixin._extract_model_distribution_log(info, model_distribution=state)
        if trial_idx >= len(model_log):
            return np.asarray([], dtype=float)
        return OralModelAlignmentMixin._normalize_distribution(model_log[trial_idx])

    @staticmethod
    def _expected_center_similarity(partition, model_dist, oral_center, choice, hypo_indices=None):
        """Compare oral center with the model's choice-conditioned expected center."""
        model_dist = OralModelAlignmentMixin._normalize_distribution(model_dist)
        center = np.asarray(oral_center, dtype=float).reshape(-1)
        if np.isnan(model_dist).any() or center.size == 0 or np.isnan(center).any():
            return np.nan

        cat_idx = int(choice) - 1
        if cat_idx < 0 or cat_idx >= int(partition.n_cats):
            return np.nan
        centers = partition.prototypes[:, 0, cat_idx, :]
        if hypo_indices is not None:
            idx = np.asarray(hypo_indices, dtype=int).reshape(-1)
            idx = idx[(idx >= 0) & (idx < centers.shape[0])]
            if idx.size == 0 or idx.size != model_dist.size:
                return np.nan
            centers = centers[idx]
        expected_center = np.sum(model_dist[:, None] * centers, axis=0)
        dist = float(np.linalg.norm(center - expected_center))
        max_dist = float(np.sqrt(partition.n_dims))
        if max_dist <= 0:
            return np.nan
        return float(np.clip(1.0 - dist / max_dist, 0.0, 1.0))

    @staticmethod
    def _expected_center(partition, model_dist, choice, hypo_indices=None):
        """Return model belief projected into the oral-center representation."""
        model_dist = OralModelAlignmentMixin._normalize_distribution(model_dist)
        if np.isnan(model_dist).any():
            return np.full(partition.n_dims, np.nan, dtype=float)

        cat_idx = int(choice) - 1
        if cat_idx < 0 or cat_idx >= int(partition.n_cats):
            return np.full(partition.n_dims, np.nan, dtype=float)
        centers = partition.prototypes[:, 0, cat_idx, :]
        if hypo_indices is not None:
            idx = np.asarray(hypo_indices, dtype=int).reshape(-1)
            idx = idx[(idx >= 0) & (idx < centers.shape[0])]
            if idx.size == 0 or idx.size != model_dist.size:
                return np.full(partition.n_dims, np.nan, dtype=float)
            centers = centers[idx]
        return np.sum(model_dist[:, None] * centers, axis=0)

    @staticmethod
    def _fuzzy_region_alignment_metrics(
        partition,
        model_dist,
        oral_region,
        choice,
        n_samples=1000,
        random_state=42,
    ):
        """Compare a model fuzzy region with a reported oral region.

        For each Monte Carlo point x, the model fuzzy field is
        ``sum_h p(h) * 1[x in region_h(choice)]``. The oral report is a binary
        mask over the same points.
        """
        model_dist = OralModelAlignmentMixin._normalize_distribution(model_dist)
        if np.isnan(model_dist).any():
            return {
                "fuzzy_iou_similarity": np.nan,
                "fuzzy_cosine_similarity": np.nan,
                "model_mass_inside_oral": np.nan,
                "oral_region_covered_by_model": np.nan,
                "model_expected_volume": np.nan,
                "oral_volume": np.nan,
            }

        cat_idx = int(choice) - 1
        scorer = OralModelAlignmentMixin._get_region_overlap_scorer(
            partition=partition,
            n_samples=int(n_samples),
            bounds=(0.0, 1.0),
            random_state=random_state,
            dist_tol=1e-9,
        )
        if cat_idx < 0 or cat_idx >= len(scorer.hypothesis_masks):
            return {
                "fuzzy_iou_similarity": np.nan,
                "fuzzy_cosine_similarity": np.nan,
                "model_mass_inside_oral": np.nan,
                "oral_region_covered_by_model": np.nan,
                "model_expected_volume": np.nan,
                "oral_volume": np.nan,
            }

        A, b = Oral_region_mapping._parse_region(oral_region)
        if A is None or b is None:
            return {
                "fuzzy_iou_similarity": np.nan,
                "fuzzy_cosine_similarity": np.nan,
                "model_mass_inside_oral": np.nan,
                "oral_region_covered_by_model": np.nan,
                "model_expected_volume": np.nan,
                "oral_volume": np.nan,
            }

        n_hypos = min(model_dist.size, scorer.hypothesis_masks[cat_idx].shape[0])
        if n_hypos <= 0:
            return {
                "fuzzy_iou_similarity": np.nan,
                "fuzzy_cosine_similarity": np.nan,
                "model_mass_inside_oral": np.nan,
                "oral_region_covered_by_model": np.nan,
                "model_expected_volume": np.nan,
                "oral_volume": np.nan,
            }

        model_dist = OralModelAlignmentMixin._normalize_distribution(model_dist[:n_hypos])
        active_idx = np.flatnonzero(np.nan_to_num(model_dist, nan=0.0) > 1e-12)
        if active_idx.size == 0:
            return {
                "fuzzy_iou_similarity": np.nan,
                "fuzzy_cosine_similarity": np.nan,
                "model_mass_inside_oral": np.nan,
                "oral_region_covered_by_model": np.nan,
                "model_expected_volume": np.nan,
                "oral_volume": np.nan,
            }
        hypo_masks = scorer.hypothesis_masks[cat_idx][active_idx].astype(float)
        model_field = model_dist[active_idx] @ hypo_masks
        oral_field = Oral_region_mapping._points_in_region(
            scorer.points,
            A,
            b,
            dist_tol=1e-9,
        ).astype(float)

        fuzzy_intersection = float(np.sum(np.minimum(model_field, oral_field)))
        fuzzy_union = float(np.sum(np.maximum(model_field, oral_field)))
        fuzzy_iou = fuzzy_intersection / fuzzy_union if fuzzy_union > 0 else np.nan

        dot = float(np.sum(model_field * oral_field))
        model_norm = float(np.sqrt(np.sum(model_field ** 2)))
        oral_norm = float(np.sqrt(np.sum(oral_field ** 2)))
        fuzzy_cosine = dot / (model_norm * oral_norm) if model_norm > 0 and oral_norm > 0 else np.nan

        model_mass = float(np.sum(model_field))
        oral_mass = float(np.sum(oral_field))
        model_inside_oral = dot / model_mass if model_mass > 0 else np.nan
        oral_covered_by_model = dot / oral_mass if oral_mass > 0 else np.nan
        total_weight = float(scorer.n_samples)

        return {
            "fuzzy_iou_similarity": float(np.clip(fuzzy_iou, 0.0, 1.0)) if np.isfinite(fuzzy_iou) else np.nan,
            "fuzzy_cosine_similarity": (
                float(np.clip(fuzzy_cosine, 0.0, 1.0)) if np.isfinite(fuzzy_cosine) else np.nan
            ),
            "model_mass_inside_oral": (
                float(np.clip(model_inside_oral, 0.0, 1.0)) if np.isfinite(model_inside_oral) else np.nan
            ),
            "oral_region_covered_by_model": (
                float(np.clip(oral_covered_by_model, 0.0, 1.0)) if np.isfinite(oral_covered_by_model) else np.nan
            ),
            "model_expected_volume": float(model_mass / total_weight) if total_weight > 0 else np.nan,
            "oral_volume": float(oral_mass / total_weight) if total_weight > 0 else np.nan,
        }

    # -----------------------------------------------------------------------
    # Supporting utility: full oral mass display
    # -----------------------------------------------------------------------

    def compute_oral_mass_probabilities(
        self,
        oral_df,
        oral_mode="center",
        subjects=None,
        region_n_samples=1000,
        region_stimulus_sigma=None,
    ):
        """Compute full oral_t hypothesis distributions per subject."""
        df = oral_df.copy()
        if subjects is not None:
            subject_set = set(subjects)
            df = df[df["iSub"].isin(subject_set)]

        out = {}
        for iSub, subj_df in df.groupby("iSub"):
            subj_df = subj_df.reset_index(drop=True)
            if subj_df.empty:
                continue

            condition = int(subj_df["condition"].iloc[0])
            n_cats = 2 if condition == 1 else 4
            target_hypo = 0 if condition == 1 else 42
            partition = Partition(n_dims=4, n_cats=n_cats)
            n_trials = len(subj_df)
            oral_mass = np.full((n_trials, partition.length), np.nan, dtype=float)
            valid_oral = []

            for trial_idx in range(n_trials):
                choice = int(subj_df.loc[trial_idx, "choice"])
                if oral_mode == "center":
                    center = Oral_center_mapping._parse_center(subj_df.loc[trial_idx, "oral_center"])
                    oral_dist = self._center_oral_distribution(center, choice, partition)
                elif oral_mode == "region":
                    region = (subj_df.loc[trial_idx, "oral_A"], subj_df.loc[trial_idx, "oral_b"])
                    oral_dist = self._region_oral_distribution(
                        region,
                        choice,
                        partition,
                        n_samples=region_n_samples,
                        random_state=42,
                    )
                else:
                    raise ValueError(f"Unsupported oral_mode: {oral_mode}")

                valid = not np.isnan(oral_dist).any()
                valid_oral.append(bool(valid))
                if valid:
                    oral_mass[trial_idx, : len(oral_dist)] = oral_dist

            out[int(iSub)] = {
                "iSub": int(iSub),
                "condition": condition,
                "target_hypo": target_hypo,
                "oral_mode": oral_mode,
                "region_stimulus_sigma": np.nan,
                "oral_mass": oral_mass,
                "valid_oral": valid_oral,
            }

        return out

    @staticmethod
    def save_oral_mass_probabilities(oral_mass_results, save_path):
        """Save full oral_t hypothesis distributions to a compressed npz file."""
        results = {int(k): v for k, v in oral_mass_results.items()}
        if not results:
            raise RuntimeError("No oral mass results to save.")

        subjects = np.asarray(sorted(results), dtype=int)
        max_trials = max(np.asarray(results[sid]["oral_mass"]).shape[0] for sid in subjects)
        max_hypos = max(np.asarray(results[sid]["oral_mass"]).shape[1] for sid in subjects)
        oral_mass = np.full((len(subjects), max_trials, max_hypos), np.nan, dtype=float)
        valid_oral = np.zeros((len(subjects), max_trials), dtype=bool)
        n_trials = np.zeros(len(subjects), dtype=int)
        n_hypos = np.zeros(len(subjects), dtype=int)
        conditions = np.zeros(len(subjects), dtype=int)
        target_hypos = np.zeros(len(subjects), dtype=int)
        oral_modes = []
        region_stimulus_sigmas = np.full(len(subjects), np.nan, dtype=float)

        for row_idx, sid in enumerate(subjects):
            info = results[int(sid)]
            arr = np.asarray(info["oral_mass"], dtype=float)
            trials, hypos = arr.shape
            oral_mass[row_idx, :trials, :hypos] = arr
            valid = np.asarray(info.get("valid_oral", []), dtype=bool).reshape(-1)
            valid_oral[row_idx, : min(trials, valid.size)] = valid[:trials]
            n_trials[row_idx] = trials
            n_hypos[row_idx] = hypos
            conditions[row_idx] = int(info.get("condition"))
            target_hypos[row_idx] = int(info.get("target_hypo"))
            oral_modes.append(str(info.get("oral_mode", "")))
            try:
                region_stimulus_sigmas[row_idx] = float(info.get("region_stimulus_sigma", np.nan))
            except (TypeError, ValueError):
                region_stimulus_sigmas[row_idx] = np.nan

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_path,
            subjects=subjects,
            conditions=conditions,
            target_hypos=target_hypos,
            n_trials=n_trials,
            n_hypos=n_hypos,
            valid_oral=valid_oral,
            oral_mass=oral_mass,
            oral_modes=np.asarray(oral_modes, dtype=str),
            region_stimulus_sigmas=region_stimulus_sigmas,
        )
        logger.info("Oral mass probabilities saved to %s", save_path)
        return save_path

    @staticmethod
    def load_oral_mass_probabilities(path):
        """Load oral_t hypothesis distributions saved by ``save_oral_mass_probabilities``."""
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            subjects = data["subjects"].astype(int)
            conditions = data["conditions"].astype(int)
            target_hypos = data["target_hypos"].astype(int)
            n_trials = data["n_trials"].astype(int)
            n_hypos = data["n_hypos"].astype(int)
            valid_oral = data["valid_oral"].astype(bool)
            oral_mass = data["oral_mass"].astype(float)
            oral_modes = data["oral_modes"].astype(str) if "oral_modes" in data.files else np.asarray([""] * len(subjects))
            region_stimulus_sigmas = (
                data["region_stimulus_sigmas"].astype(float)
                if "region_stimulus_sigmas" in data.files
                else np.full(len(subjects), np.nan, dtype=float)
            )

            out = {}
            for row_idx, sid in enumerate(subjects):
                trials = int(n_trials[row_idx])
                hypos = int(n_hypos[row_idx])
                out[int(sid)] = {
                    "iSub": int(sid),
                    "condition": int(conditions[row_idx]),
                    "target_hypo": int(target_hypos[row_idx]),
                    "oral_mode": str(oral_modes[row_idx]),
                    "region_stimulus_sigma": float(region_stimulus_sigmas[row_idx]),
                    "oral_mass": oral_mass[row_idx, :trials, :hypos].copy(),
                    "valid_oral": valid_oral[row_idx, :trials].tolist(),
                }
        return out

    def compute_model_distribution_probabilities(
        self,
        model_results,
        subjects=None,
        model_distribution="prior",
        mass_key=None,
    ):
        """Collect model belief distributions in the same dict shape as oral mass."""
        model_res = self._filter_results(model_results, subjects)
        state = str(model_distribution).strip().lower()
        mass_key = mass_key or f"{state}_mass"
        out = {}

        for iSub, info in model_res.items():
            sid = int(iSub)
            condition = int(info.get("condition", 1))
            raw_target_hypo = info.get("target_hypothesis")
            target_hypo = int(raw_target_hypo) if raw_target_hypo is not None else (0 if condition == 1 else 42)
            model_log = self._extract_model_distribution_log(info, model_distribution=state)
            if not model_log:
                continue

            n_trials = len(model_log)
            max_hypos = max(np.asarray(x, dtype=float).reshape(-1).size for x in model_log)
            mass = np.full((n_trials, max_hypos), np.nan, dtype=float)
            valid = []
            for trial_idx, raw in enumerate(model_log):
                dist = self._normalize_distribution(np.asarray(raw, dtype=float).reshape(-1))
                is_valid = dist.size > 0 and not np.isnan(dist).any()
                valid.append(bool(is_valid))
                if is_valid:
                    mass[trial_idx, : dist.size] = dist

            out[sid] = {
                "iSub": sid,
                "condition": condition,
                "target_hypo": target_hypo,
                "model_distribution": state,
                mass_key: mass,
                "valid_model": valid,
            }

        return out

    def compute_combined_oral_model_probabilities(
        self,
        model_results,
        oral_df,
        oral_mode="center",
        subjects=None,
        region_n_samples=1000,
        region_stimulus_sigma=None,
        model_distribution="prior",
        oral_mass_results=None,
        active_threshold=1e-12,
    ):
        """Project oral mass and model belief into oral-equivalence groups.

        For each trial, hypotheses with the same current-choice oral
        representation are summed together. The first returned dict stores the
        combined oral mass under ``oral_mass``; the second stores the combined
        model distribution under ``prior_mass`` or ``posterior_mass``.
        """
        model_res = self._filter_results(model_results, subjects)
        oral_df = oral_df.copy()
        state = str(model_distribution).strip().lower()
        model_mass_key = f"{state}_mass"
        oral_out = {}
        model_out = {}

        for iSub, info in model_res.items():
            sid = int(iSub)
            subj_df = oral_df[oral_df["iSub"] == sid].reset_index(drop=True)
            if subj_df.empty:
                continue

            condition = int(info.get("condition", subj_df["condition"].iloc[0]))
            n_cats = 2 if condition == 1 else 4
            raw_target_hypo = info.get("target_hypothesis")
            target_hypo = int(raw_target_hypo) if raw_target_hypo is not None else (0 if condition == 1 else 42)
            partition = Partition(n_dims=4, n_cats=n_cats)
            model_log = self._extract_model_distribution_log(info, model_distribution=state)
            n_trials = min(len(subj_df), len(model_log))
            if n_trials <= 0:
                continue

            oral_rows: List[np.ndarray] = []
            model_rows: List[np.ndarray] = []
            valid_oral: List[bool] = []
            valid_model: List[bool] = []
            n_groups_per_trial: List[int] = []
            target_group_per_trial: List[int] = []
            active_group_count: List[int] = []

            for trial_idx in range(n_trials):
                choice = int(subj_df.loc[trial_idx, "choice"])
                raw_model = np.asarray(model_log[trial_idx], dtype=float).reshape(-1)
                model_dist = self._normalize_distribution(raw_model)

                precomputed_oral = self._oral_distribution_from_precomputed(oral_mass_results, sid, trial_idx)
                if precomputed_oral is not None:
                    oral_dist = precomputed_oral
                elif oral_mode == "center":
                    center = Oral_center_mapping._parse_center(subj_df.loc[trial_idx, "oral_center"])
                    oral_dist = self._center_oral_distribution(center, choice, partition)
                elif oral_mode == "region":
                    region = (subj_df.loc[trial_idx, "oral_A"], subj_df.loc[trial_idx, "oral_b"])
                    oral_dist = self._region_oral_distribution(
                        region,
                        choice,
                        partition,
                        n_samples=region_n_samples,
                        random_state=42,
                    )
                else:
                    raise ValueError(f"Unsupported oral_mode: {oral_mode}")

                group_ids, _ = self._oral_equivalence_groups(partition, choice, oral_mode=oral_mode)
                combined_oral = self._project_distribution_to_groups(oral_dist, group_ids, normalize=True)
                combined_model = self._project_distribution_to_groups(model_dist, group_ids, normalize=True)
                raw_group_mass = self._project_distribution_to_groups(raw_model, group_ids, normalize=False)

                n_groups = int(combined_oral.size) if combined_oral.size else 0
                oral_rows.append(combined_oral)
                model_rows.append(combined_model)
                valid_oral.append(bool(combined_oral.size > 0 and not np.isnan(combined_oral).any()))
                valid_model.append(bool(combined_model.size > 0 and not np.isnan(combined_model).any()))
                n_groups_per_trial.append(n_groups)
                if 0 <= target_hypo < len(group_ids):
                    target_group_per_trial.append(int(group_ids[target_hypo]))
                else:
                    target_group_per_trial.append(-1)
                if raw_group_mass.size > 0 and not np.isnan(raw_group_mass).all():
                    active_group_count.append(int(np.sum(raw_group_mass > float(active_threshold))))
                else:
                    active_group_count.append(0)

            max_groups = max((row.size for row in oral_rows + model_rows), default=0)
            oral_arr = np.full((n_trials, max_groups), np.nan, dtype=float)
            model_arr = np.full((n_trials, max_groups), np.nan, dtype=float)
            for trial_idx, (oral_row, model_row) in enumerate(zip(oral_rows, model_rows)):
                if oral_row.size and not np.isnan(oral_row).all():
                    oral_arr[trial_idx, : oral_row.size] = oral_row
                if model_row.size and not np.isnan(model_row).all():
                    model_arr[trial_idx, : model_row.size] = model_row

            common = {
                "iSub": sid,
                "condition": condition,
                "target_hypo": target_hypo,
                "oral_mode": oral_mode,
                "model_distribution": state,
                "distribution_projection": "oral_equivalence",
                "region_stimulus_sigma": np.nan,
                "n_groups_per_trial": n_groups_per_trial,
                "target_group_per_trial": target_group_per_trial,
                "active_group_count": active_group_count,
            }
            oral_out[sid] = {
                **common,
                "oral_mass": oral_arr,
                "valid_oral": valid_oral,
            }
            model_out[sid] = {
                **common,
                model_mass_key: model_arr,
                "valid_model": valid_model,
            }

        return oral_out, model_out

    @staticmethod
    def _oral_equivalence_representation_json(partition, oral_mode, choice, hypo_idx, decimals=6):
        """Return a compact JSON representation of one oral-equivalence key."""
        cat_idx = int(choice) - 1
        if cat_idx < 0 or cat_idx >= int(partition.n_cats):
            if oral_mode == "center":
                return json.dumps({"center": None}, ensure_ascii=False)
            return json.dumps({"region": None}, ensure_ascii=False)
        if oral_mode == "center":
            center = np.round(partition.prototypes[int(hypo_idx), 0, cat_idx, :], int(decimals))
            return json.dumps({"center": center.tolist()}, ensure_ascii=False)
        if oral_mode == "region":
            region = Oral_region_mapping._true_region(partition.regions, int(hypo_idx), cat_idx)
            A, b = Oral_region_mapping._parse_region(region)
            if A is None or b is None:
                return json.dumps({"region": None}, ensure_ascii=False)
            return json.dumps(
                {
                    "constraint": "A @ x <= b",
                    "A": np.round(A, int(decimals)).tolist(),
                    "b": np.round(b, int(decimals)).tolist(),
                },
                ensure_ascii=False,
            )
        raise ValueError(f"Unsupported oral_mode: {oral_mode}")

    def compute_oral_equivalence_group_tables(
        self,
        oral_df,
        oral_mode="center",
        subjects=None,
        target_hypotheses_by_condition=None,
    ):
        """Return lookup and trial tables describing oral-equivalence groups.

        The lookup table lists all hypothesis groups for each
        ``condition x oral_mode x choice``. The trial table is compact: each
        trial points to the relevant lookup key, because the full grouping only
        depends on the current choice and oral mode.
        """
        mode = str(oral_mode).strip().lower()
        if mode not in {"center", "region"}:
            raise ValueError(f"Unsupported oral_mode: {oral_mode}")

        df = oral_df.copy()
        if subjects is not None:
            subject_set = {int(s) for s in subjects}
            df = df[df["iSub"].astype(int).isin(subject_set)]
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()

        target_map = {int(k): int(v) for k, v in (target_hypotheses_by_condition or {}).items()}
        lookup_rows = []
        trial_rows = []
        lookup_cache: Dict[Tuple[int, str, int], Dict[str, Any]] = {}

        for condition in sorted(df["condition"].dropna().astype(int).unique()):
            n_cats = 2 if int(condition) == 1 else 4
            partition = Partition(n_dims=4, n_cats=n_cats)
            target_hypo = int(target_map.get(int(condition), 0 if int(condition) == 1 else 42))
            condition_df = df[df["condition"].astype(int) == int(condition)]
            choices = sorted(condition_df["choice"].dropna().astype(int).unique())

            for choice in choices:
                group_ids, _ = self._oral_equivalence_groups(partition, int(choice), oral_mode=mode)
                valid_groups = sorted(int(g) for g in np.unique(group_ids[group_ids >= 0]))
                key_prefix = f"cond{int(condition)}_{mode}_choice{int(choice)}"
                group_lookup: Dict[int, List[int]] = {}

                for group_id in valid_groups:
                    hypos = np.flatnonzero(group_ids == int(group_id)).astype(int).tolist()
                    group_lookup[int(group_id)] = hypos
                    rep_hypo = hypos[0] if hypos else -1
                    lookup_rows.append(
                        {
                            "condition": int(condition),
                            "oral_mode": mode,
                            "choice": int(choice),
                            "lookup_key": key_prefix,
                            "group_id": int(group_id),
                            "group_key": f"{key_prefix}_g{int(group_id):03d}",
                            "n_hypotheses": int(len(hypos)),
                            "hypotheses": json.dumps(hypos, ensure_ascii=False),
                            "representative_hypothesis": int(rep_hypo),
                            "target_hypo": int(target_hypo),
                            "target_in_group": bool(target_hypo in hypos),
                            "representation": self._oral_equivalence_representation_json(
                                partition,
                                mode,
                                int(choice),
                                int(rep_hypo),
                            ) if rep_hypo >= 0 else "{}",
                        }
                    )

                target_group_id = int(group_ids[target_hypo]) if 0 <= target_hypo < len(group_ids) else -1
                lookup_cache[(int(condition), mode, int(choice))] = {
                    "lookup_key": key_prefix,
                    "n_groups": int(len(valid_groups)),
                    "n_multi_hypothesis_groups": int(sum(len(v) > 1 for v in group_lookup.values())),
                    "max_group_size": int(max((len(v) for v in group_lookup.values()), default=0)),
                    "target_group_id": target_group_id,
                    "target_group_hypotheses": group_lookup.get(target_group_id, []),
                }

        for iSub, subj_df in df.groupby("iSub"):
            subj_df = subj_df.reset_index(drop=True)
            if subj_df.empty:
                continue
            condition = int(subj_df["condition"].iloc[0])
            target_hypo = int(target_map.get(condition, 0 if condition == 1 else 42))
            for trial_idx, row in subj_df.iterrows():
                choice = int(row["choice"])
                cached = lookup_cache.get((condition, mode, choice), {})
                target_group_hypos = cached.get("target_group_hypotheses", [])
                trial_rows.append(
                    {
                        "iSub": int(iSub),
                        "subject": int(iSub),
                        "condition": condition,
                        "trial": int(trial_idx + 1),
                        "choice": choice,
                        "oral_mode": mode,
                        "lookup_key": cached.get("lookup_key", f"cond{condition}_{mode}_choice{choice}"),
                        "target_hypo": target_hypo,
                        "target_group_id": int(cached.get("target_group_id", -1)),
                        "target_group_hypotheses": json.dumps(target_group_hypos, ensure_ascii=False),
                        "target_group_size": int(len(target_group_hypos)),
                        "n_groups": int(cached.get("n_groups", 0)),
                        "n_multi_hypothesis_groups": int(cached.get("n_multi_hypothesis_groups", 0)),
                        "max_group_size": int(cached.get("max_group_size", 0)),
                    }
                )

        lookup_df = pd.DataFrame(lookup_rows)
        trial_df = pd.DataFrame(trial_rows)
        if not lookup_df.empty:
            lookup_df = lookup_df.sort_values(["condition", "oral_mode", "choice", "group_id"]).reset_index(drop=True)
        if not trial_df.empty:
            trial_df = trial_df.sort_values(["condition", "iSub", "trial"]).reset_index(drop=True)
        return lookup_df, trial_df

    @staticmethod
    def save_oral_equivalence_group_outputs(
        lookup_df,
        trial_df,
        output_dir,
        prefix="oral_equivalence",
    ):
        """Save oral-equivalence lookup/trial tables and a readable report."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        lookup_path = output_dir / f"{prefix}_group_lookup.csv"
        multi_path = output_dir / f"{prefix}_multi_hypothesis_groups.csv"
        trial_path = output_dir / f"{prefix}_trial_groups.csv"
        report_path = output_dir / f"{prefix}_group_report.md"

        lookup = lookup_df.copy()
        trial = trial_df.copy()
        multi = lookup[lookup["n_hypotheses"].astype(int) > 1].copy() if not lookup.empty else pd.DataFrame()

        lookup.to_csv(lookup_path, index=False)
        multi.to_csv(multi_path, index=False)
        trial.to_csv(trial_path, index=False)

        lines = [
            "# Oral-Equivalence Hypothesis Groups",
            "",
            "Hypotheses are grouped when they have the same oral representation for the current choice.",
            "Use `oral_equivalence_trial_groups.csv` to map each trial to a choice-level lookup key,",
            "and `oral_equivalence_group_lookup.csv` to inspect every group under that key.",
            "",
        ]
        if lookup.empty:
            lines.append("No grouping rows were generated.")
        else:
            for (condition, oral_mode, choice), sub in lookup.groupby(
                ["condition", "oral_mode", "choice"],
                observed=True,
            ):
                n_groups = int(len(sub))
                n_multi = int(np.sum(sub["n_hypotheses"].astype(int) > 1))
                max_size = int(sub["n_hypotheses"].astype(int).max())
                lines.extend(
                    [
                        f"## Condition {int(condition)}, {oral_mode}, choice {int(choice)}",
                        "",
                        f"- groups: {n_groups}",
                        f"- multi-hypothesis groups: {n_multi}",
                        f"- max group size: {max_size}",
                        "",
                    ]
                )
                multi_sub = sub[sub["n_hypotheses"].astype(int) > 1]
                if multi_sub.empty:
                    lines.extend(["No multi-hypothesis groups for this choice.", ""])
                    continue
                lines.append("| group_id | n | hypotheses | target_in_group |")
                lines.append("| --- | ---: | --- | --- |")
                for _, row in multi_sub.iterrows():
                    lines.append(
                        "| "
                        f"{int(row['group_id'])} | "
                        f"{int(row['n_hypotheses'])} | "
                        f"`{row['hypotheses']}` | "
                        f"{bool(row['target_in_group'])} |"
                    )
                lines.append("")

        report_path.write_text("\n".join(lines), encoding="utf-8")
        return {
            "lookup": lookup_path,
            "multi_hypothesis_groups": multi_path,
            "trial_groups": trial_path,
            "report": report_path,
        }

    @staticmethod
    def _oral_distribution_from_precomputed(oral_mass_results, iSub, trial_idx):
        """Fetch one precomputed oral_t distribution by subject and trial."""
        if oral_mass_results is None:
            return None
        info = oral_mass_results.get(iSub)
        if info is None:
            info = oral_mass_results.get(int(iSub))
        if info is None:
            info = oral_mass_results.get(str(iSub))
        if info is None:
            return None

        arr = np.asarray(info.get("oral_mass"), dtype=float)
        if arr.ndim != 2 or trial_idx < 0 or trial_idx >= arr.shape[0]:
            return None
        dist = arr[trial_idx].reshape(-1)
        if dist.size == 0 or np.isnan(dist).all():
            return None
        return dist.copy()

    def plot_oral_mass_probabilities(
        self,
        oral_mass_results,
        subjects=None,
        save_path=None,
        limit=True,
        mass_key="oral_mass",
        title="Oral Mass for k by Subject",
        ylabel="Oral Mass",
        target_label="target",
        **kwargs,
    ):
        """Plot trial-by-hypothesis or trial-by-group mass, matching posterior.png layout."""
        results = self._filter_results(oral_mass_results, subjects)
        grouped = defaultdict(list)
        for iSub, info in results.items():
            grouped[info["condition"]].append((iSub, info))

        if not grouped:
            raise RuntimeError("No oral mass results to plot.")

        n_rows, n_cols, rows_by_condition = self._layout_by_condition(grouped, kwargs)
        fig = plt.figure(figsize=(n_cols * 8, n_rows * 5))
        fig.suptitle(
            title,
            fontsize=kwargs.get("fontsize", 16),
            y=kwargs.get("y", 0.99),
        )

        scatter_size = kwargs.get("scatter_size", 3)
        alpha = kwargs.get("alpha", 0.28)
        cmap = kwargs.get("cmap", "viridis")

        row_offset = 0
        for condition, subs in sorted(grouped.items()):
            for idx, (iSub, info) in enumerate(subs):
                local_row = idx // n_cols
                col = idx % n_cols
                ax = fig.add_subplot(n_rows, n_cols, (row_offset + local_row) * n_cols + col + 1)

                oral_mass = np.asarray(info.get(mass_key), dtype=float)
                if oral_mass.ndim != 2 or oral_mass.size == 0:
                    ax.text(0.5, 0.5, f"No {ylabel.lower()} data", ha="center", va="center", transform=ax.transAxes)
                    continue

                if limit:
                    max_k = 19 if int(condition) == 1 else 116
                    max_k = min(max_k, oral_mass.shape[1])
                else:
                    max_k = oral_mass.shape[1]

                mass = oral_mass[:, :max_k]
                n_trials, n_hypos = mass.shape
                x = np.repeat(np.arange(1, n_trials + 1), n_hypos)
                k = np.tile(np.arange(n_hypos), n_trials)
                y = mass.reshape(-1)
                finite = np.isfinite(y)

                if np.any(finite):
                    ax.scatter(
                        x[finite],
                        y[finite],
                        c=k[finite],
                        cmap=cmap,
                        vmin=0,
                        vmax=max(1, n_hypos - 1),
                        s=scatter_size,
                        alpha=alpha,
                        linewidths=0,
                        rasterized=True,
                    )

                    target_hypo = int(info.get("target_hypo", 0 if int(condition) == 1 else 42))
                    target_group = np.asarray(info.get("target_group_per_trial", []), dtype=float).reshape(-1)
                    if target_group.size >= n_trials:
                        target_x = []
                        target_y = []
                        for trial_idx in range(n_trials):
                            group_idx = int(target_group[trial_idx])
                            if 0 <= group_idx < n_hypos and np.isfinite(mass[trial_idx, group_idx]):
                                target_x.append(trial_idx + 1)
                                target_y.append(float(mass[trial_idx, group_idx]))
                        if target_x:
                            ax.scatter(
                                target_x,
                                target_y,
                                color="red",
                                s=max(12, scatter_size * 4),
                                alpha=0.85,
                                linewidths=0,
                                label=f"{target_label} group",
                            )
                    elif 0 <= target_hypo < n_hypos:
                        target_y = mass[:, target_hypo]
                        target_x = np.arange(1, n_trials + 1)
                        target_mask = np.isfinite(target_y)
                        ax.scatter(
                            target_x[target_mask],
                            target_y[target_mask],
                            color="red",
                            s=max(12, scatter_size * 4),
                            alpha=0.85,
                            linewidths=0,
                            label=f"{target_label} k={target_hypo}",
                        )

                    y_max = float(np.nanmax(y[finite]))
                    ax.set_ylim(0, min(1.0, max(0.02, y_max * 1.12)))
                else:
                    ax.text(0.5, 0.5, "No valid oral mass", ha="center", va="center", transform=ax.transAxes)

                ax.set(
                    title=f"Subject {iSub} (Condition {condition})",
                    xlabel="Trial",
                    ylabel=ylabel,
                )
                if ax.get_legend_handles_labels()[0]:
                    ax.legend()

            row_offset += rows_by_condition[condition]

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("%s plot saved to %s", title, save_path)
        return fig

    def plot_model_distribution_probabilities(
        self,
        model_mass_results,
        model_distribution="prior",
        subjects=None,
        save_path=None,
        limit=True,
        **kwargs,
    ):
        """Plot model belief distributions with the oral-mass plot style."""
        state = str(model_distribution).strip().lower()
        mass_key = f"{state}_mass"
        title = kwargs.pop("title", f"Model {state.capitalize()} for k by Subject")
        ylabel = kwargs.pop("ylabel", f"{state.capitalize()} Probability")
        return self.plot_oral_mass_probabilities(
            model_mass_results,
            subjects=subjects,
            save_path=save_path,
            limit=limit,
            mass_key=mass_key,
            title=title,
            ylabel=ylabel,
            target_label="target",
            **kwargs,
        )

    # -----------------------------------------------------------------------
    # Main family 1: distribution-based alignment
    # -----------------------------------------------------------------------

    def compute_distribution_alignment(
        self,
        model_results,
        oral_df,
        oral_mode="center",
        subjects=None,
        region_n_samples=1000,
        region_stimulus_sigma=None,
        model_distribution="prior",
        alignment_spaces=None,
        active_threshold=1e-12,
        oral_mass_results=None,
        combine_oral_equivalent=False,
    ):
        """Compute JS similarity for oral/model distributions in three spaces.

        By default this uses ``prior_t`` because oral reports are collected
        before feedback updates the model posterior for the current trial.
        ``model_distribution='posterior'`` is still available as a deliberately
        post-feedback diagnostic.

        The returned table is trial-level and long-format. Each trial appears
        once per comparison space:
        - ``full``: complete hypothesis space.
        - ``active``: model active hypothesis set.
        - ``union_topn``: union of the model active set and oral top-N set,
          where N is the active-set size.

        If ``combine_oral_equivalent`` is true, both distributions are first
        summed over trial-specific oral-equivalence classes. This makes the
        model comparison fairer when multiple hypotheses produce the same oral
        center or region for the current choice.
        """
        spaces = tuple(alignment_spaces or self.DISTRIBUTION_ALIGNMENT_SPACES)
        unsupported = set(spaces) - set(self.DISTRIBUTION_ALIGNMENT_SPACES)
        if unsupported:
            raise ValueError(f"Unsupported distribution alignment spaces: {sorted(unsupported)}")

        model_res = self._filter_results(model_results, subjects)
        oral_df = oral_df.copy()
        rows = []

        for iSub, info in model_res.items():
            sid = int(iSub)
            subj_df = oral_df[oral_df["iSub"] == sid].reset_index(drop=True)
            if subj_df.empty:
                continue

            condition = int(info.get("condition", subj_df["condition"].iloc[0]))
            n_cats = 2 if condition == 1 else 4
            partition = Partition(n_dims=4, n_cats=n_cats)
            model_log = self._extract_model_distribution_log(info, model_distribution=model_distribution)
            n_trials = min(len(subj_df), len(model_log))

            for trial_idx in range(n_trials):
                choice = int(subj_df.loc[trial_idx, "choice"])
                raw_model = np.asarray(model_log[trial_idx], dtype=float).reshape(-1)
                model_dist = self._normalize_distribution(raw_model)

                precomputed_oral = self._oral_distribution_from_precomputed(oral_mass_results, sid, trial_idx)
                if precomputed_oral is not None:
                    oral_dist = precomputed_oral
                elif oral_mode == "center":
                    center = Oral_center_mapping._parse_center(subj_df.loc[trial_idx, "oral_center"])
                    oral_dist = self._center_oral_distribution(center, choice, partition)
                elif oral_mode == "region":
                    region = (subj_df.loc[trial_idx, "oral_A"], subj_df.loc[trial_idx, "oral_b"])
                    oral_dist = self._region_oral_distribution(
                        region,
                        choice,
                        partition,
                        n_samples=region_n_samples,
                        random_state=42,
                    )
                else:
                    raise ValueError(f"Unsupported oral_mode: {oral_mode}")

                if combine_oral_equivalent:
                    group_ids, _ = self._oral_equivalence_groups(partition, choice, oral_mode=oral_mode)
                    model_for_compare = self._project_distribution_to_groups(model_dist, group_ids, normalize=True)
                    oral_for_compare = self._project_distribution_to_groups(oral_dist, group_ids, normalize=True)
                    raw_for_active = self._project_distribution_to_groups(raw_model, group_ids, normalize=False)
                    active_idx = self._active_hypothesis_indices(raw_for_active, active_threshold=active_threshold)
                    projection = "oral_equivalence"
                    n_projection_groups = int(model_for_compare.size) if model_for_compare.size else 0
                else:
                    model_for_compare = model_dist
                    oral_for_compare = oral_dist
                    raw_for_active = raw_model
                    active_idx = self._active_hypothesis_indices(raw_for_active, active_threshold=active_threshold)
                    projection = "hypothesis"
                    n_projection_groups = int(min(len(model_dist), len(oral_dist)))

                for space in spaces:
                    compare_model, compare_oral, compare_idx = self._comparison_space_distributions(
                        model_for_compare,
                        oral_for_compare,
                        alignment_space=space,
                        active_idx=active_idx,
                    )
                    valid = not (np.isnan(compare_model).any() or np.isnan(compare_oral).any())
                    js_similarity = self._js_similarity(compare_model, compare_oral) if valid else np.nan
                    if len(compare_idx) and not np.isnan(oral_for_compare).any():
                        oral_mass_in_space = float(np.sum(oral_for_compare[compare_idx]))
                    else:
                        oral_mass_in_space = np.nan

                    rows.append(
                        {
                            "iSub": sid,
                            "subject": sid,
                            "condition": condition,
                            "trial": trial_idx + 1,
                            "trial_pct": (trial_idx + 1) / float(n_trials) if n_trials else np.nan,
                            "oral_mode": oral_mode,
                            "model_distribution": str(model_distribution).strip().lower(),
                            "distribution_projection": projection,
                            "region_stimulus_sigma": np.nan,
                            "alignment_space": space,
                            "alignment_label": self.DISTRIBUTION_ALIGNMENT_LABELS.get(space, space),
                            "js_similarity": js_similarity,
                            "valid": bool(valid),
                            "n_hypo": int(min(len(model_dist), len(oral_dist))),
                            "n_projection_groups": n_projection_groups,
                            "active_set_size": int(len(active_idx)),
                            "comparison_set_size": int(len(compare_idx)),
                            "oral_mass_in_comparison_set": oral_mass_in_space,
                        }
                    )

        return pd.DataFrame(rows)

    def compute_distribution_based_alignment(self, *args, **kwargs):
        """Alias for the distribution-based alignment family."""
        return self.compute_distribution_alignment(*args, **kwargs)

    @staticmethod
    def summarize_distribution_alignment_by_bin(distribution_results, bins=20):
        """Return subject-balanced binned means and SEMs for JS similarity."""
        df = distribution_results.copy()
        if df.empty:
            return pd.DataFrame()

        df = df[np.isfinite(df["js_similarity"])]
        if df.empty:
            return pd.DataFrame()

        df["trial_bin"] = pd.cut(
            df["trial_pct"],
            bins=np.linspace(0, 1, int(bins) + 1),
            labels=np.arange(1, int(bins) + 1),
            include_lowest=True,
        ).astype(int)
        subject_bin = (
            df.groupby(["subject", "alignment_space", "alignment_label", "trial_bin"], observed=True)[
                "js_similarity"
            ]
            .mean()
            .reset_index()
        )

        rows = []
        for (space, label, trial_bin), group in subject_bin.groupby(
            ["alignment_space", "alignment_label", "trial_bin"],
            observed=True,
        ):
            values = group["js_similarity"].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            rows.append(
                {
                    "alignment_space": str(space),
                    "alignment_label": str(label),
                    "trial_bin": int(trial_bin),
                    "trial_pct": (int(trial_bin) - 0.5) / float(bins),
                    "js_similarity_mean": float(np.mean(values)) if values.size else np.nan,
                    "js_similarity_sem": OralModelAlignmentMixin._sem(values),
                    "n_subjects": int(values.size),
                }
            )
        return pd.DataFrame(rows)

    def plot_distribution_alignment_group(
        self,
        distribution_results,
        save_path=None,
        bins=20,
        title=None,
    ):
        """Plot group-level distribution alignment summary and time course."""
        df = distribution_results.copy()
        if df.empty:
            raise RuntimeError("No distribution alignment results to plot.")

        spaces = [space for space in self.DISTRIBUTION_ALIGNMENT_SPACES if space in set(df["alignment_space"])]
        if not spaces:
            raise RuntimeError("No supported distribution alignment spaces to plot.")

        subject_space_means = (
            df.groupby(["subject", "alignment_space"], observed=True)["js_similarity"]
            .mean()
            .unstack("alignment_space")
            .reindex(columns=spaces)
        )
        binned = self.summarize_distribution_alignment_by_bin(df, bins=bins)

        condition_label = ",".join(str(int(c)) for c in sorted(df["condition"].dropna().unique()))
        model_state = str(df["model_distribution"].dropna().iloc[0]) if "model_distribution" in df else "model"
        fig_title = title or f"Condition {condition_label}: oral vs model {model_state} distribution alignment"
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), dpi=170)
        fig.suptitle(fig_title, fontsize=15, y=0.99)

        ax = axes[0]
        x = np.arange(len(spaces), dtype=float)
        means = [float(np.nanmean(subject_space_means[space].to_numpy(dtype=float))) for space in spaces]
        sems = [self._sem(subject_space_means[space].to_numpy(dtype=float)) for space in spaces]
        colors = [self.DISTRIBUTION_ALIGNMENT_COLORS.get(space, "#555555") for space in spaces]
        ax.bar(x, means, yerr=sems, color=colors, alpha=0.82, capsize=4, edgecolor="white", linewidth=0.8)

        rng = np.random.default_rng(123)
        for subject, row in subject_space_means.iterrows():
            vals = row.to_numpy(dtype=float)
            finite = np.isfinite(vals)
            if np.sum(finite) >= 2:
                ax.plot(x[finite], vals[finite], color="#888888", alpha=0.22, lw=0.8, zorder=1)
            jitter = rng.normal(0.0, 0.035, size=len(spaces))
            ax.scatter(
                x[finite] + jitter[finite],
                vals[finite],
                s=18,
                color="#222222",
                alpha=0.65,
                linewidths=0,
                zorder=3,
            )

        labels = [
            "Full\nhypothesis\nspace" if space == "full" else
            "Model\nactive set" if space == "active" else
            "Active +\noral top-N\nunion"
            for space in spaces
        ]
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 1)
        ax.set_ylabel("JS similarity")
        ax.set_title("Subject means")
        ax.grid(axis="y", alpha=0.18, linewidth=0.7)

        stats_bar = self._paired_distribution_space_stats(subject_space_means, spaces)
        ax.text(
            0.02,
            0.98,
            (
                f"Friedman p{self._format_p_value(stats_bar['friedman_p'])}, "
                f"n={stats_bar['n']}\n{stats_bar['pair_text']}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82, "edgecolor": "#cccccc"},
        )

        ax = axes[1]
        for space in spaces:
            sub = binned[binned["alignment_space"] == space].sort_values("trial_bin")
            if sub.empty:
                continue
            line_x = sub["trial_pct"].to_numpy(dtype=float)
            mean = sub["js_similarity_mean"].to_numpy(dtype=float)
            sem = sub["js_similarity_sem"].to_numpy(dtype=float)
            self._line_with_sem(
                ax,
                line_x,
                mean,
                sem,
                self.DISTRIBUTION_ALIGNMENT_LABELS.get(space, space),
                self.DISTRIBUTION_ALIGNMENT_COLORS.get(space, "#555555"),
            )

        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Normalized trial")
        ax.set_ylabel("JS similarity")
        ax.set_title("Group time course")
        ax.grid(alpha=0.18, linewidth=0.7)
        ax.legend(frameon=False, loc="best")

        stats_time = self._distribution_space_time_stats(df, spaces, bins=bins)
        ax.text(
            0.02,
            0.02,
            (
                f"RM-ANOVA n={stats_time['n']}\n"
                f"space p{self._format_p_value(stats_time['space_p'])}; "
                f"time p{self._format_p_value(stats_time['time_p'])}; "
                f"space x time p{self._format_p_value(stats_time['interaction_p'])}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82, "edgecolor": "#cccccc"},
        )

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("Distribution alignment group plot saved to %s", save_path)
        return fig

    def plot_distribution_based_alignment_group(self, *args, **kwargs):
        """Alias for the distribution-based group plot."""
        return self.plot_distribution_alignment_group(*args, **kwargs)

    def plot_distribution_alignment_subjectwise(
        self,
        distribution_results,
        subjects=None,
        save_path=None,
        window_size=16,
        n_cols=8,
        title=None,
    ):
        """Plot rolling distribution-alignment traces in each subject panel."""
        df = distribution_results.copy()
        if subjects is not None:
            subject_set = {int(s) for s in subjects}
            df = df[df["subject"].isin(subject_set)]
        if df.empty:
            raise RuntimeError("No subject-level distribution alignment results to plot.")

        spaces = [space for space in self.DISTRIBUTION_ALIGNMENT_SPACES if space in set(df["alignment_space"])]
        subjects_sorted = sorted(df["subject"].dropna().astype(int).unique())
        n_rows, n_cols, figsize = self._subjectwise_grid_layout(subjects_sorted, n_cols)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=figsize,
            dpi=170,
            sharex=True,
            sharey=True,
        )
        axes = np.asarray(axes).reshape(n_rows, n_cols)
        condition_label = ",".join(str(int(c)) for c in sorted(df["condition"].dropna().unique()))
        model_state = str(df["model_distribution"].dropna().iloc[0]) if "model_distribution" in df else "model"
        fig_title = (
            title
            or f"Condition {condition_label}: subject-wise oral vs model {model_state} distribution alignment"
        )
        fig.suptitle(fig_title, fontsize=self.SUBJECTWISE_SUPTITLE_FONTSIZE, y=0.995)

        for ax, sid in zip(axes.flat, subjects_sorted):
            sub = df[df["subject"] == sid]
            title_bits = []
            for space in spaces:
                one = sub[sub["alignment_space"] == space].sort_values("trial")
                if one.empty:
                    continue
                x = one["trial_pct"].to_numpy(dtype=float)
                y = self._rolling_mean(one["js_similarity"].to_numpy(dtype=float), window_size=window_size)
                ax.plot(
                    x,
                    y,
                    lw=0.9,
                    alpha=0.82,
                    color=self.DISTRIBUTION_ALIGNMENT_COLORS.get(space, "#555555"),
                    label=self.DISTRIBUTION_ALIGNMENT_SHORT_LABELS.get(space, space),
                )
                title_bits.append(
                    f"{self.DISTRIBUTION_ALIGNMENT_SHORT_LABELS.get(space, space)}={np.nanmean(one['js_similarity']):.2f}"
                )
            ax.set_title(
                f"S{int(sid)}  " + ", ".join(title_bits),
                fontsize=self.SUBJECTWISE_TITLE_FONTSIZE,
            )
            ax.set_ylim(0, 1)
            ax.set_xlim(0, 1)
            ax.grid(alpha=0.18, linewidth=0.6)

        for ax in list(axes.flat)[len(subjects_sorted):]:
            ax.axis("off")
        self._style_subjectwise_grid_axes(axes, n_rows, n_cols, "JS similarity")

        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(spaces),
            frameon=False,
            bbox_to_anchor=(0.5, 0.965),
            fontsize=self.SUBJECTWISE_LEGEND_FONTSIZE,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("Distribution alignment subject-wise plot saved to %s", save_path)
        return fig

    def plot_distribution_based_alignment_subjectwise(self, *args, **kwargs):
        """Alias for the distribution-based subject-wise plot."""
        return self.plot_distribution_alignment_subjectwise(*args, **kwargs)

    def save_distribution_alignment_outputs(
        self,
        distribution_results,
        output_dir,
        prefix="distribution_based_alignment",
        group_plot_path=None,
        subjectwise_plot_path=None,
        window_size=16,
        bins=20,
        title_prefix=None,
    ):
        """Write distribution-alignment CSVs and the group/subject plots."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df = distribution_results.copy()
        if df.empty:
            raise RuntimeError("No distribution alignment results to save.")

        trial_csv = output_dir / f"{prefix}_trial_metrics.csv"
        subject_csv = output_dir / f"{prefix}_subject_means.csv"
        binned_csv = output_dir / f"{prefix}_binned.csv"
        group_plot = Path(group_plot_path) if group_plot_path else output_dir / f"{prefix}_group.png"
        subjectwise_plot = (
            Path(subjectwise_plot_path)
            if subjectwise_plot_path
            else output_dir / f"{prefix}_subject.png"
        )
        group_plot.parent.mkdir(parents=True, exist_ok=True)
        subjectwise_plot.parent.mkdir(parents=True, exist_ok=True)

        subject_means = (
            df.groupby(["subject", "alignment_space", "alignment_label"], observed=True)["js_similarity"]
            .mean()
            .reset_index()
        )
        binned = self.summarize_distribution_alignment_by_bin(df, bins=bins)
        df.to_csv(trial_csv, index=False)
        subject_means.to_csv(subject_csv, index=False)
        binned.to_csv(binned_csv, index=False)

        condition_label = ",".join(str(int(c)) for c in sorted(df["condition"].dropna().unique()))
        model_state = str(df["model_distribution"].dropna().iloc[0]) if "model_distribution" in df else "model"
        prefix_title = title_prefix or f"Condition {condition_label}"
        fig = self.plot_distribution_alignment_group(
            df,
            save_path=str(group_plot),
            bins=bins,
            title=f"{prefix_title}: oral vs model {model_state} distribution alignment",
        )
        plt.close(fig)
        fig = self.plot_distribution_alignment_subjectwise(
            df,
            save_path=str(subjectwise_plot),
            window_size=window_size,
            title=f"{prefix_title}: subject-wise oral vs model {model_state} distribution alignment",
        )
        plt.close(fig)

        return {
            "trial_metrics": trial_csv,
            "subject_means": subject_csv,
            "binned": binned_csv,
            "group_plot": group_plot,
            "subjectwise_plot": subjectwise_plot,
        }

    def save_distribution_based_alignment_outputs(self, *args, **kwargs):
        """Alias for writing distribution-based alignment outputs."""
        kwargs.setdefault("prefix", "distribution_based_alignment")
        return self.save_distribution_alignment_outputs(*args, **kwargs)

    # -----------------------------------------------------------------------
    # Main family 2: oral-based alignment
    # -----------------------------------------------------------------------

    def compute_oral_based_alignment(
        self,
        model_results,
        oral_df,
        oral_mode="center",
        subjects=None,
        region_n_samples=1000,
        region_stimulus_sigma=None,
        model_distribution="choice_conditioned_prior",
        beta=10.0,
    ):
        """Compute alignment after projecting model belief into oral space.

        Center mode compares the reported oral center with the model's expected
        center under the current model belief. Region mode compares the reported
        oral region with the model's fuzzy region field over Monte Carlo points.
        """
        if oral_mode not in {"center", "region"}:
            raise ValueError("oral_mode must be 'center' or 'region'.")

        model_res = self._filter_results(model_results, subjects)
        oral_df = oral_df.copy()
        rows = []

        for iSub, info in model_res.items():
            sid = int(iSub)
            subj_df = oral_df[oral_df["iSub"] == sid].reset_index(drop=True)
            if subj_df.empty:
                continue

            condition = int(info.get("condition", subj_df["condition"].iloc[0]))
            n_cats = 2 if condition == 1 else 4
            partition = Partition(n_dims=4, n_cats=n_cats)
            model_state = str(model_distribution).strip().lower().replace("-", "_")
            if model_state in {"choice_conditioned", "choice_conditioned_prior", "choice_conditional_prior"}:
                model_len = len(self._extract_prior_log(info))
            else:
                model_len = len(self._extract_model_distribution_log(info, model_distribution=model_state))
            n_trials = min(len(subj_df), model_len)

            for trial_idx in range(n_trials):
                choice = int(subj_df.loc[trial_idx, "choice"])
                model_dist = self._model_distribution_for_oral_alignment(
                    info=info,
                    subj_df=subj_df,
                    trial_idx=trial_idx,
                    partition=partition,
                    choice=choice,
                    model_distribution=model_distribution,
                    beta=beta,
                )
                valid_model = model_dist.size > 0 and not np.isnan(model_dist).any()

                base = {
                    "iSub": sid,
                    "subject": sid,
                    "condition": condition,
                    "trial": trial_idx + 1,
                    "trial_pct": (trial_idx + 1) / float(n_trials) if n_trials else np.nan,
                    "oral_mode": oral_mode,
                    "model_distribution": str(model_distribution).strip().lower(),
                    "region_stimulus_sigma": np.nan,
                    "primary_metric": self.ORAL_BASED_PRIMARY_METRIC[oral_mode],
                    "oral_based_similarity": np.nan,
                    "expected_center_similarity": np.nan,
                    "expected_center_distance": np.nan,
                    "fuzzy_iou_similarity": np.nan,
                    "fuzzy_cosine_similarity": np.nan,
                    "model_mass_inside_oral": np.nan,
                    "oral_region_covered_by_model": np.nan,
                    "model_expected_volume": np.nan,
                    "oral_volume": np.nan,
                    "valid": False,
                }

                if not valid_model:
                    rows.append(base)
                    continue

                if oral_mode == "center":
                    oral_center = Oral_center_mapping._parse_center(subj_df.loc[trial_idx, "oral_center"])
                    expected_center = self._expected_center(partition, model_dist, choice)
                    if oral_center.size == partition.n_dims and not np.isnan(oral_center).any():
                        distance = float(np.linalg.norm(oral_center - expected_center))
                        similarity = self._expected_center_similarity(
                            partition=partition,
                            model_dist=model_dist,
                            oral_center=oral_center,
                            choice=choice,
                        )
                        base.update(
                            {
                                "oral_based_similarity": similarity,
                                "expected_center_similarity": similarity,
                                "expected_center_distance": distance,
                                "valid": bool(np.isfinite(similarity)),
                            }
                        )
                else:
                    region = (subj_df.loc[trial_idx, "oral_A"], subj_df.loc[trial_idx, "oral_b"])
                    metrics = self._fuzzy_region_alignment_metrics(
                        partition=partition,
                        model_dist=model_dist,
                        oral_region=region,
                        choice=choice,
                        n_samples=region_n_samples,
                        random_state=42,
                    )
                    primary = metrics["fuzzy_iou_similarity"]
                    base.update(metrics)
                    base.update(
                        {
                            "oral_based_similarity": primary,
                            "valid": bool(np.isfinite(primary)),
                        }
                    )

                rows.append(base)

        return pd.DataFrame(rows)

    @staticmethod
    def summarize_oral_based_alignment_by_bin(oral_based_results, bins=20):
        """Return subject-balanced binned means and SEMs for oral-based similarity."""
        df = oral_based_results.copy()
        if df.empty:
            return pd.DataFrame()

        df = df[np.isfinite(df["oral_based_similarity"])]
        if df.empty:
            return pd.DataFrame()

        df["trial_bin"] = pd.cut(
            df["trial_pct"],
            bins=np.linspace(0, 1, int(bins) + 1),
            labels=np.arange(1, int(bins) + 1),
            include_lowest=True,
        ).astype(int)
        subject_bin = (
            df.groupby(["subject", "trial_bin"], observed=True)["oral_based_similarity"]
            .mean()
            .reset_index()
        )

        rows = []
        for trial_bin, group in subject_bin.groupby("trial_bin", observed=True):
            values = group["oral_based_similarity"].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            rows.append(
                {
                    "trial_bin": int(trial_bin),
                    "trial_pct": (int(trial_bin) - 0.5) / float(bins),
                    "oral_based_similarity_mean": float(np.mean(values)) if values.size else np.nan,
                    "oral_based_similarity_sem": OralModelAlignmentMixin._sem(values),
                    "n_subjects": int(values.size),
                }
            )
        return pd.DataFrame(rows)

    @classmethod
    def _oral_based_time_stats(cls, oral_based_results, bins=20):
        """Run one-way repeated-measures ANOVA for time bins."""
        if AnovaRM is None:
            return {"n": 0, "time_p": np.nan}

        df = oral_based_results.copy()
        df = df[np.isfinite(df["oral_based_similarity"])]
        if df.empty:
            return {"n": 0, "time_p": np.nan}

        df["trial_bin"] = pd.cut(
            df["trial_pct"],
            bins=np.linspace(0, 1, int(bins) + 1),
            labels=np.arange(1, int(bins) + 1),
            include_lowest=True,
        ).astype(int)
        subject_bin = (
            df.groupby(["subject", "trial_bin"], observed=True)["oral_based_similarity"]
            .mean()
            .reset_index()
        )

        counts = subject_bin.groupby("subject").size()
        complete_subjects = counts[counts == int(bins)].index
        complete = subject_bin[subject_bin["subject"].isin(complete_subjects)].copy()
        if complete["subject"].nunique() < 3:
            return {"n": int(complete["subject"].nunique()), "time_p": np.nan}

        complete["trial_bin"] = complete["trial_bin"].astype(str)
        try:
            fit = AnovaRM(
                complete,
                depvar="oral_based_similarity",
                subject="subject",
                within=["trial_bin"],
            ).fit()
        except Exception:
            return {"n": int(complete["subject"].nunique()), "time_p": np.nan}

        table = fit.anova_table
        p_val = float(table.loc["trial_bin", "Pr > F"]) if "trial_bin" in table.index else np.nan
        return {"n": int(complete["subject"].nunique()), "time_p": p_val}

    def plot_oral_based_alignment_group(
        self,
        oral_based_results,
        save_path=None,
        bins=20,
        title=None,
    ):
        """Plot group-level oral-based alignment summary and time course."""
        df = oral_based_results.copy()
        if df.empty:
            raise RuntimeError("No oral-based alignment results to plot.")

        primary_metric = str(df["primary_metric"].dropna().iloc[0])
        metric_label = self.ORAL_BASED_METRIC_LABELS.get(primary_metric, primary_metric)
        color = self.ORAL_BASED_METRIC_COLORS.get(primary_metric, "#4c78a8")
        subject_means = (
            df.groupby("subject", observed=True)["oral_based_similarity"]
            .mean()
            .reset_index()
        )
        binned = self.summarize_oral_based_alignment_by_bin(df, bins=bins)

        condition_label = ",".join(str(int(c)) for c in sorted(df["condition"].dropna().unique()))
        oral_mode = str(df["oral_mode"].dropna().iloc[0])
        model_state = str(df["model_distribution"].dropna().iloc[0])
        fig_title = title or f"Condition {condition_label}: {oral_mode} oral-based alignment ({model_state})"
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=170)
        fig.suptitle(fig_title, fontsize=15, y=0.99)

        ax = axes[0]
        values = subject_means["oral_based_similarity"].to_numpy(dtype=float)
        mean = float(np.nanmean(values)) if values.size else np.nan
        sem = self._sem(values)
        ax.bar([0], [mean], yerr=[sem], color=color, alpha=0.82, capsize=5, edgecolor="white", linewidth=0.8)
        rng = np.random.default_rng(123)
        finite = np.isfinite(values)
        ax.scatter(
            rng.normal(0.0, 0.035, size=int(np.sum(finite))),
            values[finite],
            s=20,
            color="#222222",
            alpha=0.68,
            linewidths=0,
            zorder=3,
        )
        ax.set_xticks([0], [metric_label])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Similarity")
        ax.set_title("Subject means")
        ax.grid(axis="y", alpha=0.18, linewidth=0.7)
        ax.text(
            0.02,
            0.98,
            f"mean={mean:.3f}\nSEM={sem:.3f}\nn={int(np.sum(finite))}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82, "edgecolor": "#cccccc"},
        )

        ax = axes[1]
        if not binned.empty:
            x = binned["trial_pct"].to_numpy(dtype=float)
            self._line_with_sem(
                ax,
                x,
                binned["oral_based_similarity_mean"].to_numpy(dtype=float),
                binned["oral_based_similarity_sem"].to_numpy(dtype=float),
                metric_label,
                color,
            )
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Normalized trial")
        ax.set_ylabel("Similarity")
        ax.set_title("Group time course")
        ax.grid(alpha=0.18, linewidth=0.7)
        ax.legend(frameon=False, loc="best")

        time_stats = self._oral_based_time_stats(df, bins=bins)
        ax.text(
            0.02,
            0.02,
            f"RM-ANOVA n={time_stats['n']}\ntime p{self._format_p_value(time_stats['time_p'])}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82, "edgecolor": "#cccccc"},
        )

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("Oral-based alignment group plot saved to %s", save_path)
        return fig

    def plot_oral_based_alignment_subjectwise(
        self,
        oral_based_results,
        subjects=None,
        save_path=None,
        window_size=16,
        n_cols=8,
        title=None,
    ):
        """Plot rolling oral-based alignment in each subject panel."""
        df = oral_based_results.copy()
        if subjects is not None:
            subject_set = {int(s) for s in subjects}
            df = df[df["subject"].isin(subject_set)]
        if df.empty:
            raise RuntimeError("No oral-based subject-level alignment results to plot.")

        primary_metric = str(df["primary_metric"].dropna().iloc[0])
        metric_label = self.ORAL_BASED_METRIC_LABELS.get(primary_metric, primary_metric)
        color = self.ORAL_BASED_METRIC_COLORS.get(primary_metric, "#4c78a8")
        subjects_sorted = sorted(df["subject"].dropna().astype(int).unique())
        n_rows, n_cols, figsize = self._subjectwise_grid_layout(subjects_sorted, n_cols)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=figsize,
            dpi=170,
            sharex=True,
            sharey=True,
        )
        axes = np.asarray(axes).reshape(n_rows, n_cols)

        condition_label = ",".join(str(int(c)) for c in sorted(df["condition"].dropna().unique()))
        oral_mode = str(df["oral_mode"].dropna().iloc[0])
        model_state = str(df["model_distribution"].dropna().iloc[0])
        fig_title = title or f"Condition {condition_label}: subject-wise {oral_mode} oral-based alignment"
        fig.suptitle(
            f"{fig_title} ({model_state})",
            fontsize=self.SUBJECTWISE_SUPTITLE_FONTSIZE,
            y=0.995,
        )

        for ax, sid in zip(axes.flat, subjects_sorted):
            sub = df[df["subject"] == sid].sort_values("trial")
            x = sub["trial_pct"].to_numpy(dtype=float)
            y = self._rolling_mean(sub["oral_based_similarity"].to_numpy(dtype=float), window_size=window_size)
            ax.plot(x, y, lw=0.95, alpha=0.84, color=color, label=metric_label)
            ax.set_title(
                f"S{int(sid)}  mean={np.nanmean(y):.2f}",
                fontsize=self.SUBJECTWISE_TITLE_FONTSIZE,
            )
            ax.set_ylim(0, 1)
            ax.set_xlim(0, 1)
            ax.grid(alpha=0.18, linewidth=0.6)

        for ax in list(axes.flat)[len(subjects_sorted):]:
            ax.axis("off")
        self._style_subjectwise_grid_axes(axes, n_rows, n_cols, "Similarity")

        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=1,
            frameon=False,
            bbox_to_anchor=(0.5, 0.965),
            fontsize=self.SUBJECTWISE_LEGEND_FONTSIZE,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("Oral-based alignment subject-wise plot saved to %s", save_path)
        return fig

    def save_oral_based_alignment_outputs(
        self,
        oral_based_results,
        output_dir,
        prefix="oral_based_alignment",
        group_plot_path=None,
        subjectwise_plot_path=None,
        window_size=16,
        bins=20,
        title_prefix=None,
    ):
        """Write oral-based alignment CSVs and group/subject plots."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df = oral_based_results.copy()
        if df.empty:
            raise RuntimeError("No oral-based alignment results to save.")

        trial_csv = output_dir / f"{prefix}_trial_metrics.csv"
        subject_csv = output_dir / f"{prefix}_subject_means.csv"
        binned_csv = output_dir / f"{prefix}_binned.csv"
        group_plot = Path(group_plot_path) if group_plot_path else output_dir / f"{prefix}_group.png"
        subjectwise_plot = (
            Path(subjectwise_plot_path)
            if subjectwise_plot_path
            else output_dir / f"{prefix}_subject.png"
        )
        group_plot.parent.mkdir(parents=True, exist_ok=True)
        subjectwise_plot.parent.mkdir(parents=True, exist_ok=True)

        subject_means = (
            df.groupby("subject", observed=True)[
                [
                    "oral_based_similarity",
                    "expected_center_similarity",
                    "fuzzy_iou_similarity",
                    "fuzzy_cosine_similarity",
                    "model_mass_inside_oral",
                    "oral_region_covered_by_model",
                    "model_expected_volume",
                    "oral_volume",
                ]
            ]
            .mean()
            .reset_index()
        )
        binned = self.summarize_oral_based_alignment_by_bin(df, bins=bins)
        df.to_csv(trial_csv, index=False)
        subject_means.to_csv(subject_csv, index=False)
        binned.to_csv(binned_csv, index=False)

        condition_label = ",".join(str(int(c)) for c in sorted(df["condition"].dropna().unique()))
        oral_mode = str(df["oral_mode"].dropna().iloc[0])
        model_state = str(df["model_distribution"].dropna().iloc[0])
        prefix_title = title_prefix or f"Condition {condition_label}"
        fig = self.plot_oral_based_alignment_group(
            df,
            save_path=str(group_plot),
            bins=bins,
            title=f"{prefix_title}: {oral_mode} oral-based alignment ({model_state})",
        )
        plt.close(fig)
        fig = self.plot_oral_based_alignment_subjectwise(
            df,
            save_path=str(subjectwise_plot),
            window_size=window_size,
            title=f"{prefix_title}: subject-wise {oral_mode} oral-based alignment",
        )
        plt.close(fig)

        return {
            "trial_metrics": trial_csv,
            "subject_means": subject_csv,
            "binned": binned_csv,
            "group_plot": group_plot,
            "subjectwise_plot": subjectwise_plot,
        }

    # -----------------------------------------------------------------------
    # Main family 3: target-based alignment
    # -----------------------------------------------------------------------

    def compute_target_based_alignment(
        self,
        model_results,
        oral_df,
        oral_mode="center",
        subjects=None,
        region_n_samples=1000,
        region_stimulus_sigma=None,
        oral_mass_results=None,
        alignment_spaces=None,
        active_threshold=1e-12,
    ):
        """Extract target-hypothesis mass on full, active, and union spaces.

        ``full`` uses the complete hypothesis space. ``active`` renormalizes
        model prior and oral mass inside the current model active set.
        ``union_topn`` renormalizes inside the union of the model active set and
        the oral top-N set, where N is the active-set size.
        """
        model_res = self._filter_results(model_results, subjects)
        oral_df = oral_df.copy()
        rows = []
        spaces = tuple(alignment_spaces or self.TARGET_ALIGNMENT_SPACES)
        unsupported = set(spaces) - set(self.TARGET_ALIGNMENT_SPACES)
        if unsupported:
            raise ValueError(f"Unsupported target alignment spaces: {sorted(unsupported)}")

        for iSub, info in model_res.items():
            sid = int(iSub)
            subj_df = oral_df[oral_df["iSub"] == sid].reset_index(drop=True)
            if subj_df.empty:
                continue

            condition = int(info.get("condition", subj_df["condition"].iloc[0]))
            n_cats = 2 if condition == 1 else 4
            target_hypo = 0 if condition == 1 else 42
            partition = Partition(n_dims=4, n_cats=n_cats)
            prior_log = self._extract_prior_log(info)
            n_trials = min(len(subj_df), len(prior_log))

            for trial_idx in range(n_trials):
                choice = int(subj_df.loc[trial_idx, "choice"])
                raw_prior = np.asarray(prior_log[trial_idx], dtype=float).reshape(-1)
                prior = self._normalize_distribution(raw_prior)
                active_idx = self._active_hypothesis_indices(raw_prior, active_threshold=active_threshold)

                precomputed_oral = self._oral_distribution_from_precomputed(oral_mass_results, sid, trial_idx)
                if precomputed_oral is not None:
                    oral_dist = precomputed_oral
                elif oral_mode == "center":
                    center = Oral_center_mapping._parse_center(subj_df.loc[trial_idx, "oral_center"])
                    oral_dist = self._center_oral_distribution(center, choice, partition)
                elif oral_mode == "region":
                    region = (subj_df.loc[trial_idx, "oral_A"], subj_df.loc[trial_idx, "oral_b"])
                    oral_dist = self._region_oral_distribution(
                        region,
                        choice,
                        partition,
                        n_samples=region_n_samples,
                        random_state=42,
                    )
                else:
                    raise ValueError(f"Unsupported oral_mode: {oral_mode}")

                for space in spaces:
                    compare_prior, compare_oral, compare_idx = self._comparison_space_distributions(
                        prior,
                        oral_dist,
                        alignment_space=space,
                        active_idx=active_idx,
                    )
                    model_target_prior = self._target_probability_in_space(compare_prior, compare_idx, target_hypo)
                    oral_target_mass = self._target_probability_in_space(compare_oral, compare_idx, target_hypo)
                    if len(compare_idx) and not np.isnan(oral_dist).any():
                        oral_mass_in_comparison = float(np.sum(np.asarray(oral_dist, dtype=float)[compare_idx]))
                    else:
                        oral_mass_in_comparison = np.nan

                    rows.append(
                        {
                            "iSub": sid,
                            "subject": sid,
                            "condition": condition,
                            "trial": trial_idx + 1,
                            "trial_pct": (trial_idx + 1) / float(n_trials) if n_trials else np.nan,
                            "oral_mode": oral_mode,
                            "model_distribution": "prior",
                            "alignment_space": space,
                            "alignment_label": self.TARGET_ALIGNMENT_LABELS.get(space, space),
                            "target_hypo": target_hypo,
                            "active_set_size": int(len(active_idx)),
                            "comparison_set_size": int(len(compare_idx)),
                            "oral_mass_in_comparison_set": oral_mass_in_comparison,
                            "model_target_prior": model_target_prior,
                            "oral_target_mass": oral_target_mass,
                            "valid": bool(np.isfinite(model_target_prior) and np.isfinite(oral_target_mass)),
                        }
                    )

        return pd.DataFrame(rows)

    def summarize_target_based_alignment(self, target_based_results):
        """Compute subject-level metrics between model/oral target trajectories."""
        df = target_based_results.copy()
        if df.empty:
            return pd.DataFrame()
        if "alignment_space" not in df.columns:
            df["alignment_space"] = "full"
        if "alignment_label" not in df.columns:
            df["alignment_label"] = df["alignment_space"].map(self.TARGET_ALIGNMENT_LABELS).fillna(df["alignment_space"])

        rows = []
        for (sid, space), sub in df.groupby(["subject", "alignment_space"], observed=True):
            model_vals = sub["model_target_prior"].to_numpy(dtype=float)
            oral_vals = sub["oral_target_mass"].to_numpy(dtype=float)
            valid = np.isfinite(model_vals) & np.isfinite(oral_vals)
            rows.append(
                {
                    "subject": int(sid),
                    "iSub": int(sid),
                    "condition": int(sub["condition"].dropna().iloc[0]),
                    "oral_mode": str(sub["oral_mode"].dropna().iloc[0]),
                    "alignment_space": str(space),
                    "alignment_label": str(sub["alignment_label"].dropna().iloc[0]),
                    "target_hypo": int(sub["target_hypo"].dropna().iloc[0]),
                    "n_trials": int(len(sub)),
                    "n_valid": int(np.sum(valid)),
                    "valid_rate": float(np.mean(valid)) if len(valid) else np.nan,
                    "active_set_size_mean": (
                        float(np.nanmean(sub["active_set_size"].to_numpy(dtype=float)))
                        if "active_set_size" in sub
                        else np.nan
                    ),
                    "comparison_set_size_mean": (
                        float(np.nanmean(sub["comparison_set_size"].to_numpy(dtype=float)))
                        if "comparison_set_size" in sub
                        else np.nan
                    ),
                    "oral_mass_in_comparison_set_mean": (
                        float(np.nanmean(sub["oral_mass_in_comparison_set"].to_numpy(dtype=float)))
                        if "oral_mass_in_comparison_set" in sub
                        else np.nan
                    ),
                    "model_target_prior_mean": (
                        float(np.nanmean(model_vals)) if np.any(np.isfinite(model_vals)) else np.nan
                    ),
                    "oral_target_mass_mean": (
                        float(np.nanmean(oral_vals)) if np.any(np.isfinite(oral_vals)) else np.nan
                    ),
                    "pearson_r": self._safe_pearson(model_vals, oral_vals),
                    "spearman_rho": self._safe_spearman(model_vals, oral_vals),
                    "cosine_similarity": self._safe_cosine_similarity(model_vals, oral_vals),
                }
            )
        return pd.DataFrame(rows)

    def plot_target_based_alignment_group(
        self,
        target_subject_metrics,
        save_path=None,
        title=None,
    ):
        """Plot group-level target trajectory metrics for each comparison space."""
        df = target_subject_metrics.copy()
        if df.empty:
            raise RuntimeError("No target-based subject metrics to plot.")
        if "alignment_space" not in df.columns:
            df["alignment_space"] = "full"
        if "alignment_label" not in df.columns:
            df["alignment_label"] = df["alignment_space"].map(self.TARGET_ALIGNMENT_LABELS).fillna(df["alignment_space"])

        metrics = list(self.TARGET_BASED_METRICS)
        condition_label = ",".join(str(int(c)) for c in sorted(df["condition"].dropna().unique()))
        oral_mode = str(df["oral_mode"].dropna().iloc[0])
        fig_title = title or f"Condition {condition_label}: target-based alignment ({oral_mode})"
        spaces = [space for space in self.TARGET_ALIGNMENT_SPACES if space in set(df["alignment_space"])]
        if not spaces:
            spaces = sorted(df["alignment_space"].dropna().unique())
        fig, axes = plt.subplots(1, len(spaces), figsize=(5.1 * len(spaces), 5.2), dpi=170, sharey=True)
        axes = np.asarray(axes).reshape(-1)
        fig.suptitle(fig_title, fontsize=15, y=0.98)

        rng = np.random.default_rng(123)
        for ax, space in zip(axes, spaces):
            sub = df[df["alignment_space"] == space]
            x = np.arange(len(metrics), dtype=float)
            means = []
            sems = []
            for metric in metrics:
                vals = sub[metric].to_numpy(dtype=float)
                finite = np.isfinite(vals)
                means.append(float(np.nanmean(vals)) if np.any(finite) else np.nan)
                sems.append(self._sem(vals))
            colors = [self.TARGET_BASED_METRIC_COLORS.get(metric, "#555555") for metric in metrics]
            ax.bar(x, means, yerr=sems, color=colors, alpha=0.82, capsize=4, edgecolor="white", linewidth=0.8)

            for idx, metric in enumerate(metrics):
                vals = sub[metric].to_numpy(dtype=float)
                finite = np.isfinite(vals)
                ax.scatter(
                    rng.normal(float(idx), 0.035, size=int(np.sum(finite))),
                    vals[finite],
                    s=18,
                    color="#222222",
                    alpha=0.62,
                    linewidths=0,
                    zorder=3,
                )

            ax.axhline(0, color="#333333", lw=0.8, alpha=0.5)
            ax.set_xticks(x, [self.TARGET_BASED_METRIC_LABELS.get(metric, metric) for metric in metrics])
            ax.tick_params(axis="x", labelrotation=12)
            ax.set_ylim(-1.0, 1.0)
            ax.set_title(self.TARGET_ALIGNMENT_LABELS.get(space, space))
            ax.grid(axis="y", alpha=0.18, linewidth=0.7)
            ax.text(
                0.02,
                0.02,
                f"n={sub['subject'].nunique()}\nvalid={np.nanmean(sub['valid_rate']):.2f}",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82, "edgecolor": "#cccccc"},
            )
        axes[0].set_ylabel("Subject-level metric")

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("Target-based alignment group plot saved to %s", save_path)
        return fig

    def plot_target_based_alignment_subjectwise(
        self,
        target_based_results,
        target_subject_metrics=None,
        subjects=None,
        save_path=None,
        window_size=16,
        n_cols=8,
        title=None,
        alignment_space="full",
    ):
        """Plot model target prior and oral target mass in each subject panel."""
        df = target_based_results.copy()
        if "alignment_space" not in df.columns:
            df["alignment_space"] = "full"
        if alignment_space is not None:
            df = df[df["alignment_space"] == alignment_space]
        if subjects is not None:
            subject_set = {int(s) for s in subjects}
            df = df[df["subject"].isin(subject_set)]
        if df.empty:
            raise RuntimeError("No target-based trial metrics to plot.")

        if target_subject_metrics is None:
            target_subject_metrics = self.summarize_target_based_alignment(df)
        else:
            target_subject_metrics = target_subject_metrics.copy()
            if "alignment_space" not in target_subject_metrics.columns:
                target_subject_metrics["alignment_space"] = "full"
            if alignment_space is not None:
                target_subject_metrics = target_subject_metrics[
                    target_subject_metrics["alignment_space"] == alignment_space
                ]
        metric_lookup = {
            int(row["subject"]): row
            for _, row in target_subject_metrics.iterrows()
        }

        subjects_sorted = sorted(df["subject"].dropna().astype(int).unique())
        n_rows, n_cols, figsize = self._subjectwise_grid_layout(subjects_sorted, n_cols)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=figsize,
            dpi=170,
            sharex=True,
            sharey=True,
        )
        axes = np.asarray(axes).reshape(n_rows, n_cols)
        condition_label = ",".join(str(int(c)) for c in sorted(df["condition"].dropna().unique()))
        oral_mode = str(df["oral_mode"].dropna().iloc[0])
        space = str(df["alignment_space"].dropna().iloc[0])
        space_label = self.TARGET_ALIGNMENT_LABELS.get(space, space)
        fig_title = title or f"Condition {condition_label}: target-based alignment ({oral_mode})"
        fig.suptitle(
            f"{fig_title} - {space_label}",
            fontsize=self.SUBJECTWISE_SUPTITLE_FONTSIZE,
            y=0.995,
        )

        for ax, sid in zip(axes.flat, subjects_sorted):
            sub = df[df["subject"] == sid].sort_values("trial")
            x = sub["trial_pct"].to_numpy(dtype=float)
            ax.plot(
                x,
                self._rolling_mean(sub["model_target_prior"].to_numpy(dtype=float), window_size=window_size),
                lw=1.0,
                alpha=0.86,
                color=self.TARGET_BASED_LINE_COLORS["model"],
                label="Model target prior",
            )
            ax.plot(
                x,
                self._rolling_mean(sub["oral_target_mass"].to_numpy(dtype=float), window_size=window_size),
                lw=1.0,
                alpha=0.86,
                color=self.TARGET_BASED_LINE_COLORS["oral"],
                label="Oral target mass",
            )
            metrics = metric_lookup.get(int(sid))
            if metrics is not None:
                ax.set_title(
                    f"S{int(sid)}  r={metrics.get('pearson_r', np.nan):.2f}, "
                    f"cos={metrics.get('cosine_similarity', np.nan):.2f}",
                    fontsize=self.SUBJECTWISE_TITLE_FONTSIZE,
                )
            else:
                ax.set_title(f"S{int(sid)}", fontsize=self.SUBJECTWISE_TITLE_FONTSIZE)
            ax.set_ylim(0, 1)
            ax.set_xlim(0, 1)
            ax.grid(alpha=0.18, linewidth=0.6)

        for ax in list(axes.flat)[len(subjects_sorted):]:
            ax.axis("off")
        self._style_subjectwise_grid_axes(axes, n_rows, n_cols, "Target probability/mass")

        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.965),
            fontsize=self.SUBJECTWISE_LEGEND_FONTSIZE,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("Target-based alignment subject-wise plot saved to %s", save_path)
        return fig

    def save_target_based_alignment_outputs(
        self,
        target_based_results,
        output_dir,
        prefix="target_based_alignment",
        group_plot_path=None,
        subjectwise_plot_path=None,
        window_size=16,
        title_prefix=None,
    ):
        """Write target-based alignment CSVs and group/subject plots."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df = target_based_results.copy()
        if df.empty:
            raise RuntimeError("No target-based alignment results to save.")

        trial_csv = output_dir / f"{prefix}_trial_metrics.csv"
        subject_csv = output_dir / f"{prefix}_subject_metrics.csv"
        group_plot = Path(group_plot_path) if group_plot_path else output_dir / f"{prefix}_group.png"
        group_plot.parent.mkdir(parents=True, exist_ok=True)

        subject_metrics = self.summarize_target_based_alignment(df)
        df.to_csv(trial_csv, index=False)
        subject_metrics.to_csv(subject_csv, index=False)
        if "alignment_space" not in df.columns:
            df["alignment_space"] = "full"
        spaces = [space for space in self.TARGET_ALIGNMENT_SPACES if space in set(df["alignment_space"])]
        if not spaces:
            spaces = sorted(df["alignment_space"].dropna().unique())

        subjectwise_plots = {}
        for space in spaces:
            suffix = self.TARGET_ALIGNMENT_SUFFIXES.get(space, str(space))
            subjectwise_plot = output_dir / f"{prefix}_{suffix}_subject.png"
            if len(spaces) == 1 and subjectwise_plot_path:
                subjectwise_plot = Path(subjectwise_plot_path)
            subjectwise_plot.parent.mkdir(parents=True, exist_ok=True)
            subjectwise_plots[space] = subjectwise_plot

        condition_label = ",".join(str(int(c)) for c in sorted(df["condition"].dropna().unique()))
        oral_mode = str(df["oral_mode"].dropna().iloc[0])
        prefix_title = title_prefix or f"Condition {condition_label}"
        fig = self.plot_target_based_alignment_group(
            subject_metrics,
            save_path=str(group_plot),
            title=f"{prefix_title}: target-based alignment ({oral_mode})",
        )
        plt.close(fig)
        for space, subjectwise_plot in subjectwise_plots.items():
            fig = self.plot_target_based_alignment_subjectwise(
                df,
                target_subject_metrics=subject_metrics,
                save_path=str(subjectwise_plot),
                window_size=window_size,
                title=f"{prefix_title}: target-based alignment ({oral_mode})",
                alignment_space=space,
            )
            plt.close(fig)

        return {
            "trial_metrics": trial_csv,
            "subject_metrics": subject_csv,
            "group_plot": group_plot,
            "subjectwise_plot": subjectwise_plots.get("full") or next(iter(subjectwise_plots.values()), None),
            "subjectwise_plots": subjectwise_plots,
        }

    # -----------------------------------------------------------------------
    # Main family 4: hit-based alignment
    # -----------------------------------------------------------------------

    def compute_hit_based_alignment(
        self,
        model_results,
        oral_df,
        oral_mode="center",
        subjects=None,
        region_n_samples=1000,
        region_stimulus_sigma=None,
        oral_mass_results=None,
        active_threshold=1e-12,
        rank_top_k=None,
    ):
        """Binarize target alignment for model active sets and oral top-N sets.

        For each trial:
        - default rule: model hit = target in active set; oral hit = target in
          oral top-N, where N is the model active-set size.
        - rank_top_k rule: model/oral hit = target is ranked in the top K for
          that condition. Use {1: 2, 2: 4, 3: 4} for cond1 top2 and cond2/3
          top4.
        """
        model_res = self._filter_results(model_results, subjects)
        oral_df = oral_df.copy()
        rows = []

        for iSub, info in model_res.items():
            sid = int(iSub)
            subj_df = oral_df[oral_df["iSub"] == sid].reset_index(drop=True)
            if subj_df.empty:
                continue

            condition = int(info.get("condition", subj_df["condition"].iloc[0]))
            n_cats = 2 if condition == 1 else 4
            target_hypo = 0 if condition == 1 else 42
            partition = Partition(n_dims=4, n_cats=n_cats)
            prior_log = self._extract_prior_log(info)
            n_trials = min(len(subj_df), len(prior_log))
            resolved_rank_top_k = self._resolve_rank_top_k(rank_top_k, condition)
            hit_rule = "rank_topk" if resolved_rank_top_k is not None else "active_set_topn"
            hit_rule_label = (
                f"top{resolved_rank_top_k}"
                if resolved_rank_top_k is not None
                else "active_set_topN"
            )

            for trial_idx in range(n_trials):
                choice = int(subj_df.loc[trial_idx, "choice"])
                raw_prior = np.asarray(prior_log[trial_idx], dtype=float).reshape(-1)
                active_idx = self._active_hypothesis_indices(raw_prior, active_threshold=active_threshold)
                model_valid = raw_prior.size > 0 and not np.isnan(raw_prior).all() and len(active_idx) > 0
                active_set = set(active_idx.tolist())
                model_target_rank = self._target_rank(raw_prior, target_hypo, min_value=active_threshold)
                if model_valid and resolved_rank_top_k is None:
                    model_target_hit = float(target_hypo in active_set)
                elif model_valid:
                    model_target_hit = float(
                        target_hypo in active_set
                        and np.isfinite(model_target_rank)
                        and model_target_rank <= resolved_rank_top_k
                    )
                else:
                    model_target_hit = np.nan

                precomputed_oral = self._oral_distribution_from_precomputed(oral_mass_results, sid, trial_idx)
                if precomputed_oral is not None:
                    oral_dist = precomputed_oral
                elif oral_mode == "center":
                    center = Oral_center_mapping._parse_center(subj_df.loc[trial_idx, "oral_center"])
                    oral_dist = self._center_oral_distribution(center, choice, partition)
                elif oral_mode == "region":
                    region = (subj_df.loc[trial_idx, "oral_A"], subj_df.loc[trial_idx, "oral_b"])
                    oral_dist = self._region_oral_distribution(
                        region,
                        choice,
                        partition,
                        n_samples=region_n_samples,
                        random_state=42,
                    )
                else:
                    raise ValueError(f"Unsupported oral_mode: {oral_mode}")

                oral_valid = np.asarray(oral_dist, dtype=float).size > 0 and not np.isnan(oral_dist).any()
                if oral_valid and model_valid:
                    comparison_top_n = resolved_rank_top_k if resolved_rank_top_k is not None else len(active_idx)
                    oral_topn_idx = self._oral_topn_indices(oral_dist, comparison_top_n)
                    oral_topn_set = set(oral_topn_idx.tolist())
                    oral_target_rank = self._target_rank(oral_dist, target_hypo, min_value=0.0)
                    if resolved_rank_top_k is None:
                        oral_target_hit = float(target_hypo in oral_topn_set)
                    else:
                        oral_target_hit = float(
                            np.isfinite(oral_target_rank)
                            and oral_target_rank <= resolved_rank_top_k
                        )
                    oral_topn_mass = float(np.sum(np.asarray(oral_dist, dtype=float)[oral_topn_idx]))
                    active_oral_mass = float(
                        np.sum(np.asarray(oral_dist, dtype=float)[active_idx[active_idx < len(oral_dist)]])
                    )
                else:
                    oral_topn_idx = np.asarray([], dtype=int)
                    oral_target_hit = np.nan
                    oral_target_rank = np.nan
                    oral_topn_mass = np.nan
                    active_oral_mass = np.nan

                rows.append(
                    {
                        "iSub": sid,
                        "subject": sid,
                        "condition": condition,
                        "trial": trial_idx + 1,
                        "trial_pct": (trial_idx + 1) / float(n_trials) if n_trials else np.nan,
                        "oral_mode": oral_mode,
                        "model_distribution": "prior",
                        "hit_rule": hit_rule,
                        "hit_rule_label": hit_rule_label,
                        "rank_top_k": int(resolved_rank_top_k) if resolved_rank_top_k is not None else np.nan,
                        "target_hypo": target_hypo,
                        "active_set_size": int(len(active_idx)) if model_valid else 0,
                        "oral_topn_size": int(len(oral_topn_idx)),
                        "active_fraction": (
                            float(len(active_idx) / raw_prior.size) if model_valid and raw_prior.size else np.nan
                        ),
                        "model_target_rank": model_target_rank,
                        "oral_target_rank": oral_target_rank,
                        "model_target_hit": model_target_hit,
                        "oral_target_hit": oral_target_hit,
                        "hit_agreement": (
                            float(model_target_hit == oral_target_hit)
                            if np.isfinite(model_target_hit) and np.isfinite(oral_target_hit)
                            else np.nan
                        ),
                        "both_target_hit": (
                            float(model_target_hit == 1.0 and oral_target_hit == 1.0)
                            if np.isfinite(model_target_hit) and np.isfinite(oral_target_hit)
                            else np.nan
                        ),
                        "oral_topn_mass": oral_topn_mass,
                        "active_oral_mass": active_oral_mass,
                        "valid": bool(np.isfinite(model_target_hit) and np.isfinite(oral_target_hit)),
                    }
                )

        return pd.DataFrame(rows)

    def summarize_hit_based_alignment(self, hit_based_results):
        """Compute subject-level association metrics between binary hit traces."""
        df = hit_based_results.copy()
        if df.empty:
            return pd.DataFrame()

        def finite_mean(values):
            arr = np.asarray(values, dtype=float).reshape(-1)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return np.nan
            return float(np.mean(arr))

        rows = []
        for sid, sub in df.groupby("subject", observed=True):
            model_hits = sub["model_target_hit"].to_numpy(dtype=float)
            oral_hits = sub["oral_target_hit"].to_numpy(dtype=float)
            valid = np.isfinite(model_hits) & np.isfinite(oral_hits)
            if np.any(valid):
                mh = model_hits[valid]
                oh = oral_hits[valid]
                agreement = float(np.mean(mh == oh))
                joint_hit = float(np.mean((mh > 0.5) & (oh > 0.5)))
                model_hit_rate = float(np.mean(mh > 0.5))
                oral_hit_rate = float(np.mean(oh > 0.5))
            else:
                agreement = np.nan
                joint_hit = np.nan
                model_hit_rate = np.nan
                oral_hit_rate = np.nan

            rows.append(
                {
                    "subject": int(sid),
                    "iSub": int(sid),
                    "condition": int(sub["condition"].dropna().iloc[0]),
                    "oral_mode": str(sub["oral_mode"].dropna().iloc[0]),
                    "hit_rule": str(sub["hit_rule"].dropna().iloc[0]) if "hit_rule" in sub else "active_set_topn",
                    "hit_rule_label": (
                        str(sub["hit_rule_label"].dropna().iloc[0])
                        if "hit_rule_label" in sub
                        else "active_set_topN"
                    ),
                    "rank_top_k": (
                        float(sub["rank_top_k"].dropna().iloc[0])
                        if "rank_top_k" in sub and not sub["rank_top_k"].dropna().empty
                        else np.nan
                    ),
                    "target_hypo": int(sub["target_hypo"].dropna().iloc[0]),
                    "n_trials": int(len(sub)),
                    "n_valid": int(np.sum(valid)),
                    "valid_rate": float(np.mean(valid)) if len(valid) else np.nan,
                    "model_hit_rate": model_hit_rate,
                    "oral_hit_rate": oral_hit_rate,
                    "joint_hit_rate": joint_hit,
                    "active_set_size_mean": finite_mean(sub["active_set_size"]),
                    "oral_topn_mass_mean": finite_mean(sub["oral_topn_mass"]),
                    "active_oral_mass_mean": finite_mean(sub["active_oral_mass"]),
                    "model_target_rank_mean": finite_mean(sub["model_target_rank"]),
                    "oral_target_rank_mean": finite_mean(sub["oral_target_rank"]),
                    "phi_correlation": self._safe_pearson(model_hits, oral_hits),
                    "cohen_kappa": self._safe_cohen_kappa(model_hits, oral_hits),
                    "hit_agreement_rate": agreement,
                    "positive_hit_jaccard": self._safe_binary_jaccard(model_hits, oral_hits),
                }
            )
        return pd.DataFrame(rows)

    def plot_hit_based_alignment_group(
        self,
        hit_subject_metrics,
        save_path=None,
        title=None,
    ):
        """Plot group-level metrics for binary hit-based alignment."""
        df = hit_subject_metrics.copy()
        if df.empty:
            raise RuntimeError("No hit-based subject metrics to plot.")

        metrics = list(self.HIT_BASED_METRICS)
        condition_label = ",".join(str(int(c)) for c in sorted(df["condition"].dropna().unique()))
        oral_mode = str(df["oral_mode"].dropna().iloc[0])
        hit_rule_label = (
            str(df["hit_rule_label"].dropna().iloc[0])
            if "hit_rule_label" in df and not df["hit_rule_label"].dropna().empty
            else "active_set_topN"
        )
        fig_title = title or f"Condition {condition_label}: hit-based alignment ({oral_mode})"
        fig, ax = plt.subplots(1, 1, figsize=(8.8, 5.3), dpi=170)
        fig.suptitle(fig_title, fontsize=15, y=0.98)

        x = np.arange(len(metrics), dtype=float)
        means = []
        sems = []
        for metric in metrics:
            vals = df[metric].to_numpy(dtype=float)
            finite = np.isfinite(vals)
            means.append(float(np.nanmean(vals)) if np.any(finite) else np.nan)
            sems.append(self._sem(vals))
        colors = [self.HIT_BASED_METRIC_COLORS.get(metric, "#555555") for metric in metrics]
        ax.bar(x, means, yerr=sems, color=colors, alpha=0.84, capsize=4, edgecolor="white", linewidth=0.8)

        rng = np.random.default_rng(123)
        for idx, metric in enumerate(metrics):
            vals = df[metric].to_numpy(dtype=float)
            finite = np.isfinite(vals)
            ax.scatter(
                rng.normal(float(idx), 0.035, size=int(np.sum(finite))),
                vals[finite],
                s=20,
                color="#222222",
                alpha=0.65,
                linewidths=0,
                zorder=3,
            )

        ax.axhline(0, color="#333333", lw=0.8, alpha=0.5)
        ax.set_xticks(x, [self.HIT_BASED_METRIC_LABELS.get(metric, metric) for metric in metrics])
        ax.set_ylim(-1.0, 1.0)
        ax.set_ylabel("Subject-level metric")
        if hit_rule_label.startswith("top"):
            ax.set_title(f"Model {hit_rule_label} target hit vs oral {hit_rule_label} target hit")
        else:
            ax.set_title("Model active-set target hit vs oral top-N target hit")
        ax.grid(axis="y", alpha=0.18, linewidth=0.7)
        ax.text(
            0.02,
            0.02,
            (
                f"n={df['subject'].nunique()}\n"
                f"valid rate={np.nanmean(df['valid_rate']):.2f}\n"
                f"model hit={np.nanmean(df['model_hit_rate']):.2f}, oral hit={np.nanmean(df['oral_hit_rate']):.2f}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82, "edgecolor": "#cccccc"},
        )

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("Hit-based alignment group plot saved to %s", save_path)
        return fig

    def plot_hit_based_alignment_subjectwise(
        self,
        hit_based_results,
        hit_subject_metrics=None,
        subjects=None,
        save_path=None,
        window_size=16,
        n_cols=8,
        title=None,
    ):
        """Plot rolling binary target-hit rates in each subject panel."""
        df = hit_based_results.copy()
        if subjects is not None:
            subject_set = {int(s) for s in subjects}
            df = df[df["subject"].isin(subject_set)]
        if df.empty:
            raise RuntimeError("No hit-based trial metrics to plot.")

        if hit_subject_metrics is None:
            hit_subject_metrics = self.summarize_hit_based_alignment(df)
        metric_lookup = {
            int(row["subject"]): row
            for _, row in hit_subject_metrics.iterrows()
        }

        subjects_sorted = sorted(df["subject"].dropna().astype(int).unique())
        n_rows, n_cols, figsize = self._subjectwise_grid_layout(subjects_sorted, n_cols)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=figsize,
            dpi=170,
            sharex=True,
            sharey=True,
        )
        axes = np.asarray(axes).reshape(n_rows, n_cols)
        condition_label = ",".join(str(int(c)) for c in sorted(df["condition"].dropna().unique()))
        oral_mode = str(df["oral_mode"].dropna().iloc[0])
        hit_rule_label = (
            str(df["hit_rule_label"].dropna().iloc[0])
            if "hit_rule_label" in df and not df["hit_rule_label"].dropna().empty
            else "active_set_topN"
        )
        if hit_rule_label.startswith("top"):
            model_line_label = f"Model {hit_rule_label} target hit"
            oral_line_label = f"Oral {hit_rule_label} target hit"
        else:
            model_line_label = "Model active-set target hit"
            oral_line_label = "Oral top-N target hit"
        fig_title = title or f"Condition {condition_label}: hit-based alignment ({oral_mode})"
        fig.suptitle(fig_title, fontsize=self.SUBJECTWISE_SUPTITLE_FONTSIZE, y=0.995)

        for ax, sid in zip(axes.flat, subjects_sorted):
            sub = df[df["subject"] == sid].sort_values("trial")
            x = sub["trial_pct"].to_numpy(dtype=float)
            ax.plot(
                x,
                self._rolling_mean(sub["model_target_hit"].to_numpy(dtype=float), window_size=window_size),
                lw=1.05,
                alpha=0.88,
                color=self.HIT_BASED_LINE_COLORS["model"],
                label=model_line_label,
            )
            ax.plot(
                x,
                self._rolling_mean(sub["oral_target_hit"].to_numpy(dtype=float), window_size=window_size),
                lw=1.05,
                alpha=0.88,
                color=self.HIT_BASED_LINE_COLORS["oral"],
                label=oral_line_label,
            )
            metrics = metric_lookup.get(int(sid))
            if metrics is not None:
                ax.set_title(
                    f"S{int(sid)}  phi={metrics.get('phi_correlation', np.nan):.2f}, "
                    f"agr={metrics.get('hit_agreement_rate', np.nan):.2f}",
                    fontsize=self.SUBJECTWISE_TITLE_FONTSIZE,
                )
            else:
                ax.set_title(f"S{int(sid)}", fontsize=self.SUBJECTWISE_TITLE_FONTSIZE)
            ax.set_ylim(0, 1)
            ax.set_xlim(0, 1)
            ax.grid(alpha=0.18, linewidth=0.6)

        for ax in list(axes.flat)[len(subjects_sorted):]:
            ax.axis("off")
        self._style_subjectwise_grid_axes(axes, n_rows, n_cols, "Rolling hit rate")

        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.965),
            fontsize=self.SUBJECTWISE_LEGEND_FONTSIZE,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("Hit-based alignment subject-wise plot saved to %s", save_path)
        return fig

    def save_hit_based_alignment_outputs(
        self,
        hit_based_results,
        output_dir,
        prefix="hit_based_alignment",
        group_plot_path=None,
        subjectwise_plot_path=None,
        window_size=16,
        title_prefix=None,
    ):
        """Write hit-based alignment CSVs and group/subject plots."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df = hit_based_results.copy()
        if df.empty:
            raise RuntimeError("No hit-based alignment results to save.")

        trial_csv = output_dir / f"{prefix}_trial_metrics.csv"
        subject_csv = output_dir / f"{prefix}_subject_metrics.csv"
        group_plot = Path(group_plot_path) if group_plot_path else output_dir / f"{prefix}_group.png"
        subjectwise_plot = (
            Path(subjectwise_plot_path)
            if subjectwise_plot_path
            else output_dir / f"{prefix}_subject.png"
        )
        group_plot.parent.mkdir(parents=True, exist_ok=True)
        subjectwise_plot.parent.mkdir(parents=True, exist_ok=True)

        subject_metrics = self.summarize_hit_based_alignment(df)
        df.to_csv(trial_csv, index=False)
        subject_metrics.to_csv(subject_csv, index=False)

        condition_label = ",".join(str(int(c)) for c in sorted(df["condition"].dropna().unique()))
        oral_mode = str(df["oral_mode"].dropna().iloc[0])
        prefix_title = title_prefix or f"Condition {condition_label}"
        fig = self.plot_hit_based_alignment_group(
            subject_metrics,
            save_path=str(group_plot),
            title=f"{prefix_title}: hit-based alignment ({oral_mode})",
        )
        plt.close(fig)
        fig = self.plot_hit_based_alignment_subjectwise(
            df,
            hit_subject_metrics=subject_metrics,
            save_path=str(subjectwise_plot),
            window_size=window_size,
            title=f"{prefix_title}: hit-based alignment ({oral_mode})",
        )
        plt.close(fig)

        return {
            "trial_metrics": trial_csv,
            "subject_metrics": subject_csv,
            "group_plot": group_plot,
            "subjectwise_plot": subjectwise_plot,
        }

    # -----------------------------------------------------------------------
    # Main family 5: coverage-based alignment
    # -----------------------------------------------------------------------

    def compute_coverage_based_alignment(
        self,
        model_results,
        oral_df,
        oral_mode="center",
        subjects=None,
        region_n_samples=1000,
        region_stimulus_sigma=None,
        active_threshold=1e-12,
        oral_mass_results=None,
    ):
        """Compute how much oral top-N mass is captured by model active sets.

        Per trial, ``N`` is the number of hypotheses with non-zero model prior.
        The metric compares oral mass in the model active set against the oral
        top-N oracle under the same hypothesis-count budget.
        """
        model_res = self._filter_results(model_results, subjects)
        oral_df = oral_df.copy()
        rows = []

        for iSub, info in model_res.items():
            sid = int(iSub)
            subj_df = oral_df[oral_df["iSub"] == sid].reset_index(drop=True)
            if subj_df.empty:
                continue

            condition = int(info.get("condition", subj_df["condition"].iloc[0]))
            n_cats = 2 if condition == 1 else 4
            partition = Partition(n_dims=4, n_cats=n_cats)
            prior_log = self._extract_prior_log(info)
            n_trials = min(len(subj_df), len(prior_log))

            for trial_idx in range(n_trials):
                raw_prior = np.asarray(prior_log[trial_idx], dtype=float).reshape(-1)
                if raw_prior.size == 0 or np.isnan(raw_prior).all():
                    continue

                active_idx = np.flatnonzero(np.nan_to_num(raw_prior, nan=0.0) > float(active_threshold))
                n_active = int(len(active_idx))
                if n_active <= 0:
                    continue

                choice = int(subj_df.loc[trial_idx, "choice"])
                precomputed_oral = self._oral_distribution_from_precomputed(oral_mass_results, sid, trial_idx)
                if precomputed_oral is not None:
                    oral_dist = precomputed_oral
                elif oral_mode == "center":
                    center = Oral_center_mapping._parse_center(subj_df.loc[trial_idx, "oral_center"])
                    oral_dist = self._center_oral_distribution(center, choice, partition)
                elif oral_mode == "region":
                    region = (subj_df.loc[trial_idx, "oral_A"], subj_df.loc[trial_idx, "oral_b"])
                    oral_dist = self._region_oral_distribution(
                        region,
                        choice,
                        partition,
                        n_samples=region_n_samples,
                        random_state=42,
                    )
                else:
                    raise ValueError(f"Unsupported oral_mode: {oral_mode}")

                if np.isnan(oral_dist).any():
                    continue

                n_hypo = int(len(oral_dist))
                top_n = min(n_active, n_hypo)
                oral_top_idx = np.argsort(oral_dist)[::-1][:top_n]
                active_idx = active_idx[active_idx < n_hypo]
                if active_idx.size == 0:
                    continue

                active_oral_mass = float(np.sum(oral_dist[active_idx]))
                oracle_topn_oral_mass = float(np.sum(oral_dist[oral_top_idx]))
                random_expected_mass = float(top_n / n_hypo) if n_hypo else np.nan
                overlap_count = len(set(active_idx.tolist()) & set(oral_top_idx.tolist()))
                active_capture_ratio = (
                    active_oral_mass / oracle_topn_oral_mass
                    if oracle_topn_oral_mass > 0
                    else np.nan
                )

                rows.append(
                    {
                        "iSub": sid,
                        "subject": sid,
                        "condition": condition,
                        "trial": trial_idx + 1,
                        "trial_pct": (trial_idx + 1) / float(n_trials),
                        "oral_mode": oral_mode,
                        "n_hypo": n_hypo,
                        "n_active": n_active,
                        "active_fraction": n_active / float(n_hypo) if n_hypo else np.nan,
                        "active_oral_mass": active_oral_mass,
                        "oracle_topn_oral_mass": oracle_topn_oral_mass,
                        "random_expected_mass": random_expected_mass,
                        "active_capture_ratio": active_capture_ratio,
                        "active_topn_overlap": overlap_count / float(top_n) if top_n else np.nan,
                        "active_topn_overlap_count": int(overlap_count),
                        "oral_topn_mean_mass": oracle_topn_oral_mass / float(top_n) if top_n else np.nan,
                        "active_mean_oral_mass": active_oral_mass / float(n_active) if n_active else np.nan,
                    }
                )

        return pd.DataFrame(rows)

    def summarize_coverage_based_alignment(self, coverage_results):
        """Return subject means for the two coverage-based alignment metrics."""
        df = coverage_results.copy()
        if df.empty:
            return pd.DataFrame()

        metrics = list(self.COVERAGE_BASED_METRICS) + [
            "active_oral_mass",
            "oracle_topn_oral_mass",
            "random_expected_mass",
            "n_active",
            "active_fraction",
        ]
        present_metrics = [metric for metric in metrics if metric in df.columns]
        subject_means = (
            df.groupby("subject", observed=True)[present_metrics]
            .mean()
            .reset_index()
        )
        meta = (
            df.groupby("subject", observed=True)[["iSub", "condition", "oral_mode"]]
            .first()
            .reset_index()
        )
        return meta.merge(subject_means, on="subject", how="left")

    @staticmethod
    def summarize_coverage_based_alignment_by_bin(coverage_results, bins=20):
        """Return subject-balanced binned means and SEMs for coverage alignment."""
        df = coverage_results.copy()
        if df.empty:
            return pd.DataFrame()

        df["trial_bin"] = pd.cut(
            df["trial_pct"],
            bins=np.linspace(0, 1, int(bins) + 1),
            labels=np.arange(1, int(bins) + 1),
            include_lowest=True,
        ).astype(int)
        metrics = [
            "active_capture_ratio",
            "active_topn_overlap",
            "active_oral_mass",
            "oracle_topn_oral_mass",
            "random_expected_mass",
            "n_active",
            "active_fraction",
        ]
        subject_bin = df.groupby(["subject", "trial_bin"], observed=True)[metrics].mean().reset_index()

        rows = []
        for trial_bin, group in subject_bin.groupby("trial_bin", observed=True):
            item = {"trial_bin": int(trial_bin), "trial_pct": (int(trial_bin) - 0.5) / int(bins)}
            for metric in metrics:
                vals = group[metric].to_numpy(dtype=float)
                valid = vals[~np.isnan(vals)]
                item[f"{metric}_mean"] = float(np.mean(valid)) if valid.size else np.nan
                item[f"{metric}_sem"] = (
                    float(np.std(valid, ddof=1) / np.sqrt(valid.size))
                    if valid.size > 1
                    else np.nan
                )
            rows.append(item)
        return pd.DataFrame(rows)

    @staticmethod
    def _line_with_sem(ax, x, mean, sem, label, color):
        ax.plot(x, mean, lw=2.2, label=label, color=color)
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.18, linewidth=0)

    def plot_coverage_based_alignment_group(
        self,
        coverage_results,
        save_path=None,
        bins=20,
        title=None,
    ):
        """Plot group-level coverage alignment: subject bars and time course."""
        df = coverage_results.copy()
        if df.empty:
            raise RuntimeError("No coverage-based alignment results to plot.")

        binned = self.summarize_coverage_based_alignment_by_bin(df, bins=bins)
        subject_means = self.summarize_coverage_based_alignment(df)

        condition_label = ",".join(str(int(c)) for c in sorted(df["condition"].dropna().unique()))
        oral_mode = str(df["oral_mode"].dropna().iloc[0])
        fig_title = title or f"Condition {condition_label}: coverage-based alignment ({oral_mode})"
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=170)
        fig.suptitle(fig_title, fontsize=15, y=0.99)

        ax = axes[0]
        metric_names = list(self.COVERAGE_BASED_METRICS)
        x_bar = np.arange(len(metric_names), dtype=float)
        means = []
        sems = []
        for metric in metric_names:
            vals = subject_means[metric].to_numpy(dtype=float)
            finite = np.isfinite(vals)
            means.append(float(np.nanmean(vals)) if np.any(finite) else np.nan)
            sems.append(self._sem(vals))
        colors = [self.COVERAGE_BASED_COLORS[metric] for metric in metric_names]
        ax.bar(x_bar, means, yerr=sems, color=colors, alpha=0.84, capsize=4, edgecolor="white", linewidth=0.8)
        rng = np.random.default_rng(123)
        for idx, metric in enumerate(metric_names):
            vals = subject_means[metric].to_numpy(dtype=float)
            finite = np.isfinite(vals)
            ax.scatter(
                rng.normal(float(idx), 0.035, size=int(np.sum(finite))),
                vals[finite],
                s=20,
                color="#222222",
                alpha=0.65,
                linewidths=0,
                zorder=3,
            )
        ax.set_xticks(x_bar, [self.COVERAGE_BASED_LABELS[metric] for metric in metric_names])
        ax.tick_params(axis="x", labelrotation=10)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Subject mean")
        ax.set_title("Group mean")
        ax.grid(axis="y", alpha=0.18, linewidth=0.7)

        ax = axes[1]
        x = binned["trial_pct"].to_numpy(dtype=float)
        for metric in metric_names:
            self._line_with_sem(
                ax,
                x,
                binned[f"{metric}_mean"].to_numpy(dtype=float),
                binned[f"{metric}_sem"].to_numpy(dtype=float),
                self.COVERAGE_BASED_LABELS[metric],
                self.COVERAGE_BASED_COLORS[metric],
            )
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Normalized trial")
        ax.set_ylabel("Coverage")
        ax.set_title("Group time course")
        ax.grid(alpha=0.18, linewidth=0.7)
        ax.legend(frameon=False, loc="best")

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
            logger.info("Coverage-based alignment group plot saved to %s", save_path)
        return fig

    def plot_coverage_based_alignment_subjectwise(
        self,
        coverage_results,
        save_path=None,
        window_size=16,
        n_cols=8,
        title=None,
    ):
        """Plot rolling coverage-based alignment traces in each subject panel."""
        df = coverage_results.copy()
        if df.empty:
            raise RuntimeError("No coverage-based alignment results to plot.")

        subjects = sorted(df["subject"].unique())
        n_rows, n_cols, figsize = self._subjectwise_grid_layout(subjects, n_cols)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=figsize,
            dpi=170,
            sharey=True,
        )
        axes = np.asarray(axes).reshape(n_rows, n_cols)
        condition_label = ",".join(str(int(c)) for c in sorted(df["condition"].dropna().unique()))
        oral_mode = str(df["oral_mode"].dropna().iloc[0])
        fig_title = title or f"Condition {condition_label}: subject-wise coverage-based alignment ({oral_mode})"
        fig.suptitle(fig_title, fontsize=self.SUBJECTWISE_SUPTITLE_FONTSIZE, y=0.995)

        for ax, sid in zip(axes.flat, subjects):
            sub = df[df["subject"] == sid].sort_values("trial")
            x = sub["trial_pct"].to_numpy(dtype=float)
            for metric in self.COVERAGE_BASED_METRICS:
                ax.plot(
                    x,
                    self._rolling_mean(sub[metric], window_size),
                    lw=1.05,
                    alpha=0.88,
                    color=self.COVERAGE_BASED_COLORS[metric],
                    label=self.COVERAGE_BASED_LABELS[metric],
                )
            ax.set_title(
                (
                    f"S{int(sid)}  "
                    f"cap={np.nanmean(sub['active_capture_ratio']):.2f}, "
                    f"ov={np.nanmean(sub['active_topn_overlap']):.2f}"
                ),
                fontsize=self.SUBJECTWISE_TITLE_FONTSIZE,
            )
            ax.set_ylim(0, 1)
            ax.set_xlim(0, 1)
            ax.grid(alpha=0.18, linewidth=0.6)

        for ax in list(axes.flat)[len(subjects):]:
            ax.axis("off")
        self._style_subjectwise_grid_axes(axes, n_rows, n_cols, "Coverage")

        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.965),
            fontsize=self.SUBJECTWISE_LEGEND_FONTSIZE,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("Coverage-based alignment subject-wise plot saved to %s", save_path)
        return fig

    def save_coverage_based_alignment_outputs(
        self,
        coverage_results,
        output_dir,
        prefix="coverage_based_alignment",
        group_plot_path=None,
        subjectwise_plot_path=None,
        window_size=16,
        bins=20,
        title_prefix=None,
    ):
        """Write coverage-based alignment CSVs and group/subject plots."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df = coverage_results.copy()
        if df.empty:
            raise RuntimeError("No coverage-based alignment results to save.")

        trial_csv = output_dir / f"{prefix}_trial_metrics.csv"
        subject_csv = output_dir / f"{prefix}_subject_means.csv"
        binned_csv = output_dir / f"{prefix}_binned.csv"
        group_plot = Path(group_plot_path) if group_plot_path else output_dir / f"{prefix}_group.png"
        subjectwise_plot = (
            Path(subjectwise_plot_path)
            if subjectwise_plot_path
            else output_dir / f"{prefix}_subject.png"
        )
        group_plot.parent.mkdir(parents=True, exist_ok=True)
        subjectwise_plot.parent.mkdir(parents=True, exist_ok=True)

        subject_means = self.summarize_coverage_based_alignment(df)
        binned = self.summarize_coverage_based_alignment_by_bin(df, bins=bins)

        df.to_csv(trial_csv, index=False)
        subject_means.to_csv(subject_csv, index=False)
        binned.to_csv(binned_csv, index=False)

        condition_label = ",".join(str(int(c)) for c in sorted(df["condition"].dropna().unique()))
        oral_mode = str(df["oral_mode"].dropna().iloc[0])
        prefix_title = title_prefix or f"Condition {condition_label}"
        fig = self.plot_coverage_based_alignment_group(
            df,
            save_path=str(group_plot),
            bins=bins,
            title=f"{prefix_title}: coverage-based alignment ({oral_mode})",
        )
        plt.close(fig)
        fig = self.plot_coverage_based_alignment_subjectwise(
            df,
            save_path=str(subjectwise_plot),
            window_size=window_size,
            title=f"{prefix_title}: subject-wise coverage-based alignment ({oral_mode})",
        )
        plt.close(fig)

        return {
            "trial_metrics": trial_csv,
            "subject_means": subject_csv,
            "binned": binned_csv,
            "group_plot": group_plot,
            "subjectwise_plot": subjectwise_plot,
        }

"""口述报告和模型状态的映射、概率构造与对齐计算。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
from ...hypothesis_space import ContinuousPartition
from .mapping import OralCenterMapper, OralRegionMapper, RegionOverlapScorer

_REGION_SCORER_CACHE: Dict[Tuple[Any, ...], RegionOverlapScorer] = {}
_REGION_DISTRIBUTION_CACHE: Dict[Tuple[Any, ...], Tuple[np.ndarray, Dict[str, Any]]] = {}
_ORAL_EQUIVALENCE_GROUP_CACHE: Dict[Tuple[Any, ...], Tuple[np.ndarray, Tuple[str, ...]]] = {}


class OralAlignmentScoringMixin:
    """不负责写盘和绘图的口述—模型对齐计算。"""

    ORAL_ENCODER_VERSION = "fixed_likelihood_category_state_v2"
    ORAL_HYPOTHESIS_PRIOR = "uniform_hypothesis"
    CENTER_ORAL_DISTRIBUTION_METHOD = "gaussian_component_mixture"
    REGION_ORAL_DISTRIBUTION_METHOD = "fixed_iou_energy"
    DEFAULT_ORAL_STATE_MODE = "latest_by_category"
    VALID_ORAL_STATE_MODES = ("latest_by_category", "instantaneous")
    ORAL_AGGREGATION_METHODS = {
        "latest_by_category": "latest_by_category_likelihood_product",
        "instantaneous": "current_report_only",
    }
    DEFAULT_ORAL_CENTER_SIGMA = 0.10
    DEFAULT_ORAL_REGION_TEMPERATURE = 0.10
    DEFAULT_TARGET_BAND_DRAWS = 5000
    DEFAULT_TARGET_BAND_SEED = 20260810
    TARGET_BAND_TYPE = "observed_history_conditional_latent_target_occupancy"
    TRAJECTORY_TARGET_BAND_TYPE = (
        "observed_history_conditional_trajectory_repeat_target_mass"
    )
    ORAL_ENCODER_METADATA_FIELDS = (
        "oral_encoder_version",
        "oral_distribution_method",
        "oral_state_mode",
        "oral_aggregation_method",
        "oral_hypothesis_prior",
        "oral_center_sigma",
        "oral_region_temperature",
        "hypothesis_space_version",
        "hypothesis_space_signature",
    )
    ORAL_TRIAL_DIAGNOSTIC_FIELDS = (
        "oral_min_distance",
        "oral_log_evidence",
        "oral_distribution_entropy",
        "oral_effective_hypotheses",
        "oral_max_probability",
        "oral_fit_score",
        "oral_state_observed_categories",
        "oral_state_category_mask",
        "oral_state_update_category",
        "oral_state_update_valid",
        "instantaneous_oral_min_distance",
        "instantaneous_oral_log_evidence",
        "instantaneous_oral_distribution_entropy",
        "instantaneous_oral_effective_hypotheses",
        "instantaneous_oral_max_probability",
        "instantaneous_oral_fit_score",
    )

    SUBJECTWISE_SUPTITLE_FONTSIZE = 16
    SUBJECTWISE_TITLE_FONTSIZE = 12
    SUBJECTWISE_LABEL_FONTSIZE = 10
    SUBJECTWISE_TICK_FONTSIZE = 10
    SUBJECTWISE_LEGEND_FONTSIZE = 10
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

    @staticmethod
    def _partition_for_model_result(info, expected_n_cats=None):
        """Rebuild the continuous hypothesis space saved with a model result."""
        provenance = info.get("model_provenance") or {}
        resolved = provenance.get("resolved") if isinstance(provenance, Mapping) else None
        partition_config = (
            resolved.get("partition") if isinstance(resolved, Mapping) else None
        )
        likelihood_config = (
            resolved.get("likelihood") if isinstance(resolved, Mapping) else None
        )
        if not isinstance(partition_config, Mapping):
            raise ValueError(
                "Oral/model alignment requires saved partition provenance."
            )
        class_path = str(partition_config.get("class", ""))
        if not class_path.endswith("ContinuousPartition"):
            raise ValueError(
                "Oral/model alignment currently requires ContinuousPartition "
                f"provenance, got {class_path!r}."
            )
        kwargs = partition_config.get("kwargs", {})
        if not isinstance(kwargs, Mapping):
            raise ValueError("Saved partition kwargs must be a mapping.")
        if not isinstance(likelihood_config, Mapping):
            raise ValueError(
                "Oral/model alignment requires saved likelihood provenance."
            )
        distance_mode = likelihood_config.get("distance_mode")
        if distance_mode is None:
            raise ValueError(
                "Oral/model alignment requires saved likelihood.distance_mode."
            )
        partition = ContinuousPartition(**dict(kwargs))
        if expected_n_cats is not None and partition.n_cats != int(expected_n_cats):
            raise ValueError(
                "Saved partition category count does not match result condition: "
                f"{partition.n_cats} vs {int(expected_n_cats)}."
            )
        return partition, str(distance_mode)

    @classmethod
    def _partitions_for_model_results(cls, model_results):
        partitions = {}
        for subject, info in model_results.items():
            condition = int(info.get("condition", 1))
            n_cats = 2 if condition == 1 else 4
            partitions[int(subject)] = cls._partition_for_model_result(
                info,
                expected_n_cats=n_cats,
            )[0]
        return partitions
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
        # Match the canonical particle-filter accuracy-band palette:
        # model expectation is orange and the observed comparator is black.
        "model": "#E69F00",
        "oral": "#111111",
    }
    TARGET_BASED_BAND_COLORS = {
        "q05_q95": "#9DB9D8",
        "q25_q75": "#4F81B8",
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
    def _normalize_distribution_rows(values):
        """Vectorized row-wise counterpart of ``_normalize_distribution``."""
        arr = np.asarray(values, dtype=float)
        if arr.ndim != 2:
            raise ValueError("Probability rows must form a 2-D matrix.")
        clean = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        clean = np.clip(clean, 0.0, None)
        totals = clean.sum(axis=1, keepdims=True)
        out = np.full(clean.shape, np.nan, dtype=float)
        valid = totals[:, 0] > 0.0
        out[valid] = clean[valid] / totals[valid]
        return out

    @staticmethod
    def _validate_positive_scale(value, name):
        """Return a finite positive observation scale."""
        scale = float(value)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"{name} must be finite and positive, got {value!r}.")
        return scale

    @classmethod
    def _validate_oral_state_mode(cls, value):
        """Return a supported oral evidence aggregation mode."""
        mode = str(value).strip().lower()
        if mode not in cls.VALID_ORAL_STATE_MODES:
            raise ValueError(
                f"oral_state_mode must be one of {cls.VALID_ORAL_STATE_MODES}, "
                f"got {value!r}."
            )
        return mode

    @staticmethod
    def _logsumexp(values):
        """Stable scalar log-sum-exp for a one-dimensional array."""
        arr = np.asarray(values, dtype=float).reshape(-1)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return np.nan
        maximum = float(np.max(finite))
        return float(maximum + np.log(np.sum(np.exp(finite - maximum))))

    @staticmethod
    def _normalize_log_weights(log_weights):
        """Normalize finite log weights into a probability vector."""
        values = np.asarray(log_weights, dtype=float).reshape(-1)
        normalizer = OralAlignmentScoringMixin._logsumexp(values)
        if values.size == 0 or not np.isfinite(normalizer):
            return np.full(values.shape, np.nan, dtype=float), np.nan
        probability = np.exp(values - normalizer)
        probability = OralAlignmentScoringMixin._normalize_distribution(probability)
        return probability, float(normalizer)

    @classmethod
    def _oral_encoder_metadata(
        cls,
        partition,
        oral_mode,
        *,
        center_sigma=None,
        region_temperature=None,
        oral_state_mode="instantaneous",
    ):
        """Return reproducibility metadata for one oral encoder."""
        mode = str(oral_mode).strip().lower()
        if mode == "center":
            method = cls.CENTER_ORAL_DISTRIBUTION_METHOD
            center_value = cls._validate_positive_scale(
                cls.DEFAULT_ORAL_CENTER_SIGMA if center_sigma is None else center_sigma,
                "oral_center_sigma",
            )
            region_value = np.nan
        elif mode == "region":
            method = cls.REGION_ORAL_DISTRIBUTION_METHOD
            center_value = np.nan
            region_value = cls._validate_positive_scale(
                cls.DEFAULT_ORAL_REGION_TEMPERATURE if region_temperature is None else region_temperature,
                "oral_region_temperature",
            )
        else:
            raise ValueError(f"Unsupported oral_mode: {oral_mode}")

        state_mode = cls._validate_oral_state_mode(oral_state_mode)
        space = partition.hypothesis_space
        return {
            "oral_encoder_version": cls.ORAL_ENCODER_VERSION,
            "oral_distribution_method": method,
            "oral_state_mode": state_mode,
            "oral_aggregation_method": cls.ORAL_AGGREGATION_METHODS[state_mode],
            "oral_hypothesis_prior": cls.ORAL_HYPOTHESIS_PRIOR,
            "oral_center_sigma": float(center_value),
            "oral_region_temperature": float(region_value),
            "hypothesis_space_version": str(space.version),
            "hypothesis_space_signature": json.dumps(space.signature, ensure_ascii=False),
        }

    @classmethod
    def _empty_oral_diagnostics(
        cls,
        partition,
        oral_mode,
        *,
        center_sigma=None,
        region_temperature=None,
        oral_state_mode="instantaneous",
    ):
        """Return encoder metadata plus missing trial-level diagnostics."""
        return {
            **cls._oral_encoder_metadata(
                partition,
                oral_mode,
                center_sigma=center_sigma,
                region_temperature=region_temperature,
                oral_state_mode=oral_state_mode,
            ),
            **{field: np.nan for field in cls.ORAL_TRIAL_DIAGNOSTIC_FIELDS},
        }

    @classmethod
    def _complete_oral_diagnostics(
        cls,
        probability,
        *,
        metadata,
        min_distance,
        log_evidence,
        fit_score,
    ):
        """Attach comparable concentration and absolute-fit diagnostics."""
        prob = cls._normalize_distribution(probability)
        if np.isnan(prob).any():
            return {
                **metadata,
                **{field: np.nan for field in cls.ORAL_TRIAL_DIAGNOSTIC_FIELDS},
            }
        positive = prob > 0.0
        entropy = -float(np.sum(prob[positive] * np.log(prob[positive])))
        return {
            **metadata,
            "oral_min_distance": float(min_distance),
            "oral_log_evidence": float(log_evidence),
            "oral_distribution_entropy": entropy,
            "oral_effective_hypotheses": cls._effective_sample_size(prob),
            "oral_max_probability": float(np.max(prob)),
            "oral_fit_score": float(fit_score),
        }

    @classmethod
    def _category_state_distribution(
        cls,
        latest_by_category,
        partition,
        oral_mode,
        *,
        center_sigma=None,
        region_temperature=None,
    ):
        """Combine the latest valid likelihood from every observed category.

        Each category contributes at most once. Updating a category replaces
        its previous likelihood, so repeated reports are not treated as
        independent evidence. The uniform hypothesis prior is applied once
        after the category log likelihoods are summed.
        """
        mode = str(oral_mode).strip().lower()
        metadata = cls._oral_encoder_metadata(
            partition,
            mode,
            center_sigma=center_sigma,
            region_temperature=region_temperature,
            oral_state_mode="latest_by_category",
        )
        if not latest_by_category:
            probability = np.full(int(partition.length), np.nan, dtype=float)
            return probability, {
                **metadata,
                **{field: np.nan for field in cls.ORAL_TRIAL_DIAGNOSTIC_FIELDS},
            }

        n_hypotheses = int(partition.length)
        log_prior = -np.log(float(n_hypotheses))
        joint_log_likelihood = np.zeros(n_hypotheses, dtype=float)
        for entry in latest_by_category.values():
            distribution = np.asarray(entry["distribution"], dtype=float).reshape(-1)
            log_evidence = float(entry["diagnostics"].get("oral_log_evidence", np.nan))
            if (
                distribution.size != n_hypotheses
                or np.isnan(distribution).any()
                or not np.isfinite(log_evidence)
            ):
                probability = np.full(n_hypotheses, np.nan, dtype=float)
                return probability, {
                    **metadata,
                    **{field: np.nan for field in cls.ORAL_TRIAL_DIAGNOSTIC_FIELDS},
                }
            with np.errstate(divide="ignore"):
                # q_k(h) = L_k(h) * pi(h) / Z_k. Recover log L_k so
                # the uniform hypothesis prior is applied only once jointly.
                joint_log_likelihood += np.log(distribution) + log_evidence - log_prior

        probability, log_evidence = cls._normalize_log_weights(
            joint_log_likelihood + log_prior
        )
        n_observed = int(len(latest_by_category))
        maximum_log_likelihood = float(np.max(joint_log_likelihood))
        if mode == "center":
            sigma = cls._validate_positive_scale(
                cls.DEFAULT_ORAL_CENTER_SIGMA if center_sigma is None else center_sigma,
                "oral_center_sigma",
            )
            gaussian_log_normalizer = -int(partition.n_dims) * np.log(
                sigma * np.sqrt(2.0 * np.pi)
            )
            ideal_log_likelihood = n_observed * gaussian_log_normalizer
            log_fit = min(0.0, maximum_log_likelihood - ideal_log_likelihood)
            effective_distance = np.sqrt(max(0.0, -2.0 * sigma ** 2 * log_fit))
        elif mode == "region":
            temperature = cls._validate_positive_scale(
                cls.DEFAULT_ORAL_REGION_TEMPERATURE
                if region_temperature is None
                else region_temperature,
                "oral_region_temperature",
            )
            log_fit = min(0.0, maximum_log_likelihood)
            effective_distance = -temperature * maximum_log_likelihood
        else:
            raise ValueError(f"Unsupported oral_mode: {oral_mode}")

        diagnostics = cls._complete_oral_diagnostics(
            probability,
            metadata=metadata,
            min_distance=effective_distance,
            log_evidence=log_evidence,
            fit_score=np.exp(log_fit),
        )
        return probability, diagnostics

    @staticmethod
    def _js_similarity(p, q):
        """Return 1 - normalized Jensen-Shannon divergence."""
        p = OralAlignmentScoringMixin._normalize_distribution(p)
        q = OralAlignmentScoringMixin._normalize_distribution(q)
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
        p = OralAlignmentScoringMixin._normalize_distribution(prob)
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
    def _category_prototypes(partition, hypo_idx, cat_idx):
        """Return all valid prototypes for one hypothesis/category."""
        return np.asarray(
            partition.prototype_geometry.get_category_prototypes(
                int(hypo_idx),
                int(cat_idx),
            ),
            dtype=float,
        )

    @staticmethod
    def _category_representative_center(partition, hypo_idx, cat_idx):
        """Summarize multiple components only where a single center is required."""
        prototypes = OralAlignmentScoringMixin._category_prototypes(
            partition,
            hypo_idx,
            cat_idx,
        )
        return np.mean(prototypes, axis=0)

    @staticmethod
    def _region_signature(region, decimals=12):
        """Return a stable signature for a convex or union category region."""
        signatures = []
        for component in OralRegionMapper._region_components(region):
            A, b = OralRegionMapper._parse_region(component)
            if A is None or b is None:
                return ("invalid",)
            A = np.round(np.asarray(A, dtype=float), int(decimals))
            b = np.round(np.asarray(b, dtype=float), int(decimals))
            signatures.append(
                (
                    A.shape,
                    tuple(A.reshape(-1).tolist()),
                    b.shape,
                    tuple(b.reshape(-1).tolist()),
                )
            )
        return ("union", tuple(signatures)) if len(signatures) > 1 else signatures[0]

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
            getattr(getattr(partition, "hypothesis_space", None), "version", None),
            getattr(partition, "prototype_method", None),
            getattr(partition, "pairwise_similarity_tolerance", None),
            getattr(partition, "center_band_tolerance", None),
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
                prototypes = OralAlignmentScoringMixin._category_prototypes(
                    partition,
                    hypo_idx,
                    cat_idx,
                )
                signature = (
                    "prototypes",
                    tuple(
                        OralAlignmentScoringMixin._rounded_signature(
                            prototype,
                            decimals=decimals,
                        )
                        for prototype in prototypes
                    ),
                )
            elif mode == "region":
                region = OralRegionMapper._true_region(
                    partition.hypothesis_space,
                    hypo_idx,
                    cat_idx,
                )
                signature = OralAlignmentScoringMixin._region_signature(region, decimals=decimals)
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
        return OralAlignmentScoringMixin._normalize_distribution(out)

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
                active_idx = OralAlignmentScoringMixin._active_hypothesis_indices(model_arr)
            compare_idx = np.asarray(active_idx, dtype=int).reshape(-1)
            compare_idx = compare_idx[(compare_idx >= 0) & (compare_idx < n_hypos)]
        elif alignment_space == "union_topn":
            if active_idx is None:
                active_idx = OralAlignmentScoringMixin._active_hypothesis_indices(model_arr)
            active_idx = np.asarray(active_idx, dtype=int).reshape(-1)
            active_idx = active_idx[(active_idx >= 0) & (active_idx < n_hypos)]
            oral_topn_idx = OralAlignmentScoringMixin._oral_topn_indices(oral_arr, len(active_idx))
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
            OralAlignmentScoringMixin._normalize_distribution(model_arr[compare_idx]),
            OralAlignmentScoringMixin._normalize_distribution(oral_arr[compare_idx]),
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
    def _repeat_target_probabilities_in_space(
        repeat_priors,
        compare_idx,
        target_hypo,
    ):
        """Project all repeat priors into one comparison space at once."""
        priors = np.asarray(repeat_priors, dtype=float)
        idx = np.asarray(compare_idx, dtype=int).reshape(-1)
        if priors.ndim != 2:
            raise ValueError("repeat_priors must be a 2-D matrix.")
        if idx.size == 0:
            return np.full(priors.shape[0], np.nan, dtype=float)
        idx = idx[(idx >= 0) & (idx < priors.shape[1])]
        if idx.size == 0:
            return np.full(priors.shape[0], np.nan, dtype=float)
        denominator = np.sum(priors[:, idx], axis=1)
        out = np.full(priors.shape[0], np.nan, dtype=float)
        valid = np.isfinite(denominator) & (denominator > 0.0)
        if int(target_hypo) in set(idx.tolist()):
            out[valid] = priors[valid, int(target_hypo)] / denominator[valid]
        else:
            out[valid] = 0.0
        return out

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

    def _extract_prior_repeat_logs(self, info):
        """Return persisted repeat priors or a transparent single-run fallback.

        Particle-filter repeats target the same observed-history-conditional
        marginal distribution.  Averaging their saved ``marginal_prior``
        arrays reduces finite-particle Monte-Carlo noise before latent-state
        sampling, matching the aggregation used by the behavioral accuracy
        band.  Trajectory repeats retain distinct realized latent paths and
        are used as an ensemble, matching the trajectory behavioral band.
        Lightweight/unit-test inputs without a persisted run stream retain
        the existing single-prior behavior.
        """
        fallback = self._extract_prior_log(info)
        fallback_arrays = [np.asarray(fallback, dtype=float)] if fallback else []
        state_kind = str(info.get("state_distribution_kind", "")).lower()
        is_particle = state_kind == "particle_marginal"
        fallback_source = (
            "representative_pf_marginal_prior"
            if is_particle
            else "representative_trajectory_prior"
        )

        subject_json_path = info.get("_subject_json_path")
        raw_ref = info.get("raw_runs_ref") or {}
        if not subject_json_path or not raw_ref.get("path"):
            return fallback_arrays, fallback_source

        loader = getattr(self, "_load_run_stream", None)
        if loader is None:
            return fallback_arrays, fallback_source

        stream = loader(info, Path(subject_json_path))
        repeat_priors = []
        for run_obj in stream:
            if not isinstance(run_obj, Mapping):
                continue
            state_log = run_obj.get("state_log") or {}
            raw = state_log.get("marginal_prior" if is_particle else "prior")
            if raw is None:
                continue
            try:
                arr = np.asarray(raw, dtype=float)
            except (TypeError, ValueError):
                continue
            if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
                continue
            normalized = self._normalize_distribution_rows(arr)
            if np.isnan(normalized).any():
                continue
            repeat_priors.append(normalized)

        if not repeat_priors:
            return fallback_arrays, fallback_source

        n_hypotheses = {arr.shape[1] for arr in repeat_priors}
        if len(n_hypotheses) != 1:
            raise ValueError("Repeat priors disagree on hypothesis-space size.")
        min_trials = min(arr.shape[0] for arr in repeat_priors)
        return (
            [arr[:min_trials].copy() for arr in repeat_priors],
            (
                "pf_repeat_mean_marginal_prior"
                if is_particle
                else "trajectory_repeat_prior_ensemble"
            ),
        )

    @classmethod
    def compute_target_sampling_band(
        cls,
        probabilities,
        *,
        window_size=16,
        n_draws=DEFAULT_TARGET_BAND_DRAWS,
        seed=DEFAULT_TARGET_BAND_SEED,
    ):
        """Sample a pointwise rolling target-occupancy interval.

        ``probabilities[t]`` is the PF marginal probability that the latent
        hypothesis at trial ``t`` is the target.  Each draw samples the
        corresponding Bernoulli target indicator and applies a complete
        rolling window.  This is the latent-state analogue of the behavioral
        accuracy band, not a confidence interval for the probability itself.
        """
        values = np.asarray(probabilities, dtype=float).reshape(-1)
        window = int(window_size)
        draws = int(n_draws)
        if window <= 0 or values.size < window:
            raise ValueError(
                "Not enough target probabilities for the requested rolling interval: "
                f"n_trials={values.size}, window_size={window}."
            )
        if draws < 2:
            raise ValueError("n_draws must be at least 2.")
        finite = np.isfinite(values)
        if np.any(values[finite] < 0.0) or np.any(values[finite] > 1.0):
            raise ValueError("Target probabilities must lie in [0, 1].")

        n_trials = values.size
        valid_window = np.convolve(
            finite.astype(int),
            np.ones(window, dtype=int),
            mode="valid",
        ) == window
        expected = np.full(n_trials, np.nan, dtype=float)
        if np.any(valid_window):
            rolling_expected = np.convolve(
                np.where(finite, values, 0.0),
                np.ones(window, dtype=float) / float(window),
                mode="valid",
            )
            expected[window - 1 :][valid_window] = rolling_expected[valid_window]

        rng = np.random.default_rng(int(seed))
        uniforms = rng.random((draws, n_trials))
        sampled = uniforms < np.where(finite, values, 0.0)[None, :]
        cumulative = np.concatenate(
            [
                np.zeros((draws, 1), dtype=np.int32),
                np.cumsum(sampled, axis=1, dtype=np.int32),
            ],
            axis=1,
        )
        rolling_samples = (
            cumulative[:, window:] - cumulative[:, :-window]
        ) / float(window)

        quantile_arrays = {
            key: np.full(n_trials, np.nan, dtype=float)
            for key in ("q05", "q25", "q50", "q75", "q95")
        }
        if np.any(valid_window):
            quantiles = np.quantile(
                rolling_samples[:, valid_window],
                [0.05, 0.25, 0.50, 0.75, 0.95],
                axis=0,
            )
            for row, key in enumerate(("q05", "q25", "q50", "q75", "q95")):
                target = quantile_arrays[key][window - 1 :]
                target[valid_window] = quantiles[row]

        return {
            "band_type": cls.TARGET_BAND_TYPE,
            "n_draws": draws,
            "seed": int(seed),
            "window_size": window,
            "expected": expected,
            **quantile_arrays,
        }

    @classmethod
    def compute_trajectory_target_band(
        cls,
        probability_runs,
        *,
        window_size=16,
    ):
        """Summarize rolling target mass across realized trajectory repeats."""
        values = np.asarray(probability_runs, dtype=float)
        if values.ndim != 2 or values.shape[0] < 1:
            raise ValueError(
                "probability_runs must be a non-empty (runs, trials) matrix."
            )
        window = int(window_size)
        if window <= 0 or values.shape[1] < window:
            raise ValueError(
                "Not enough target probabilities for the requested rolling interval: "
                f"n_trials={values.shape[1]}, window_size={window}."
            )
        finite = np.isfinite(values)
        if np.any(values[finite] < 0.0) or np.any(values[finite] > 1.0):
            raise ValueError("Target probabilities must lie in [0, 1].")

        n_runs, n_trials = values.shape
        rolling = np.full((n_runs, n_trials), np.nan, dtype=float)
        valid_cumulative = np.concatenate(
            [
                np.zeros((n_runs, 1), dtype=np.int32),
                np.cumsum(finite, axis=1, dtype=np.int32),
            ],
            axis=1,
        )
        value_cumulative = np.concatenate(
            [
                np.zeros((n_runs, 1), dtype=float),
                np.cumsum(np.where(finite, values, 0.0), axis=1),
            ],
            axis=1,
        )
        valid_window = (
            valid_cumulative[:, window:] - valid_cumulative[:, :-window]
        ) == window
        rolling_values = (
            value_cumulative[:, window:] - value_cumulative[:, :-window]
        ) / float(window)
        rolling[:, window - 1 :] = np.where(
            valid_window,
            rolling_values,
            np.nan,
        )

        complete = np.all(np.isfinite(rolling), axis=0)
        expected = np.full(n_trials, np.nan, dtype=float)
        quantile_arrays = {
            key: np.full(n_trials, np.nan, dtype=float)
            for key in ("q05", "q25", "q50", "q75", "q95")
        }
        if np.any(complete):
            expected[complete] = np.mean(rolling[:, complete], axis=0)
            quantiles = np.quantile(
                rolling[:, complete],
                [0.05, 0.25, 0.50, 0.75, 0.95],
                axis=0,
            )
            for row, key in enumerate(("q05", "q25", "q50", "q75", "q95")):
                quantile_arrays[key][complete] = quantiles[row]

        return {
            "band_type": cls.TRAJECTORY_TARGET_BAND_TYPE,
            "n_runs": int(n_runs),
            "window_size": window,
            "expected": expected,
            **quantile_arrays,
        }

    @staticmethod
    def _extract_model_distribution_log(info, model_distribution="prior"):
        """Return the model distribution time series used by distribution alignment."""
        state = str(model_distribution).strip().lower()
        if state == "prior":
            return OralAlignmentScoringMixin._extract_prior_log(info)
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

    @classmethod
    def _center_oral_distribution(
        cls,
        center,
        choice,
        partition,
        center_sigma=None,
        return_diagnostics=False,
    ):
        """Map one oral center to a normalized hypothesis posterior.

        Each category component contributes one isotropic Gaussian report
        likelihood. Component likelihoods are averaged within a hypothesis,
        the hypothesis prior is uniform, and the resulting 29/116 weights are
        normalized. ``center_sigma`` is fixed across trials so absolute
        distances retain a common scale.
        """
        sigma = cls._validate_positive_scale(
            cls.DEFAULT_ORAL_CENTER_SIGMA if center_sigma is None else center_sigma,
            "oral_center_sigma",
        )
        metadata = cls._oral_encoder_metadata(
            partition,
            "center",
            center_sigma=sigma,
        )
        center = np.asarray(center, dtype=float).reshape(-1)
        if center.size != int(partition.n_dims) or not np.isfinite(center).all():
            probability = np.full(partition.length, np.nan, dtype=float)
            diagnostics = cls._empty_oral_diagnostics(
                partition,
                "center",
                center_sigma=sigma,
            )
            return (probability, diagnostics) if return_diagnostics else probability

        cat_idx = int(choice) - 1
        if cat_idx < 0 or cat_idx >= int(partition.n_cats):
            probability = np.full(partition.length, np.nan, dtype=float)
            diagnostics = cls._empty_oral_diagnostics(
                partition,
                "center",
                center_sigma=sigma,
            )
            return (probability, diagnostics) if return_diagnostics else probability

        log_likelihood = np.full(int(partition.length), np.nan, dtype=float)
        min_distances = np.full(int(partition.length), np.nan, dtype=float)
        gaussian_log_normalizer = -int(partition.n_dims) * np.log(
            sigma * np.sqrt(2.0 * np.pi)
        )
        for hypo_idx in range(int(partition.length)):
            prototypes = cls._category_prototypes(partition, hypo_idx, cat_idx)
            squared_distances = np.sum((prototypes - center) ** 2, axis=1)
            component_log_likelihood = (
                gaussian_log_normalizer - squared_distances / (2.0 * sigma ** 2)
            )
            log_likelihood[hypo_idx] = (
                cls._logsumexp(component_log_likelihood) - np.log(len(prototypes))
            )
            min_distances[hypo_idx] = float(np.sqrt(np.min(squared_distances)))

        log_prior = -np.log(float(partition.length))
        probability, log_evidence = cls._normalize_log_weights(log_likelihood + log_prior)
        min_distance = float(np.nanmin(min_distances))
        diagnostics = cls._complete_oral_diagnostics(
            probability,
            metadata=metadata,
            min_distance=min_distance,
            log_evidence=log_evidence,
            fit_score=np.exp(-(min_distance ** 2) / (2.0 * sigma ** 2)),
        )
        return (probability, diagnostics) if return_diagnostics else probability

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
            getattr(partition, "pairwise_similarity_tolerance", None),
            getattr(partition, "center_band_tolerance", None),
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
        region_temperature=0.10,
        bounds=(0.0, 1.0),
        random_state=42,
        dist_tol=1e-9,
    ):
        """Build a stable cache key for one oral region distribution."""
        if random_state is None:
            return None
        A, b = OralRegionMapper._parse_region(region)
        if A is None or b is None:
            return None
        A = np.ascontiguousarray(A, dtype=float)
        b = np.ascontiguousarray(b, dtype=float)
        return (
            partition.__class__.__name__,
            int(partition.n_dims),
            int(partition.n_cats),
            getattr(partition, "pairwise_similarity_tolerance", None),
            getattr(partition, "center_band_tolerance", None),
            getattr(getattr(partition, "hypothesis_space", None), "version", None),
            int(n_samples),
            float(region_temperature),
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

    @classmethod
    def _region_oral_distribution(
        cls,
        region,
        choice,
        partition,
        n_samples=1000,
        random_state=42,
        region_temperature=None,
        return_diagnostics=False,
    ):
        """Map one oral region to a fixed-scale hypothesis distribution."""
        temperature = cls._validate_positive_scale(
            cls.DEFAULT_ORAL_REGION_TEMPERATURE
            if region_temperature is None
            else region_temperature,
            "oral_region_temperature",
        )
        metadata = cls._oral_encoder_metadata(
            partition,
            "region",
            region_temperature=temperature,
        )
        cache_key = cls._region_distribution_cache_key(
            region,
            choice,
            partition,
            n_samples=int(n_samples),
            region_temperature=temperature,
            bounds=(0.0, 1.0),
            random_state=random_state,
            dist_tol=1e-9,
        )
        if cache_key is not None and cache_key in _REGION_DISTRIBUTION_CACHE:
            cached_dist, cached_diagnostics = _REGION_DISTRIBUTION_CACHE[cache_key]
            probability = cached_dist.copy()
            diagnostics = dict(cached_diagnostics)
            return (probability, diagnostics) if return_diagnostics else probability

        cat_idx = int(choice) - 1
        scorer = cls._get_region_overlap_scorer(
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
        if scores.size == 0 or np.isnan(scores).all():
            probability = np.full(int(partition.length), np.nan, dtype=float)
            diagnostics = cls._empty_oral_diagnostics(
                partition,
                "region",
                region_temperature=temperature,
            )
            return (probability, diagnostics) if return_diagnostics else probability

        scores = np.clip(
            np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0),
            0.0,
            1.0,
        )
        mismatch = 1.0 - scores
        log_likelihood = -mismatch / temperature
        log_prior = -np.log(float(partition.length))
        probability, log_evidence = cls._normalize_log_weights(log_likelihood + log_prior)
        min_mismatch = float(np.min(mismatch))
        diagnostics = cls._complete_oral_diagnostics(
            probability,
            metadata=metadata,
            min_distance=min_mismatch,
            log_evidence=log_evidence,
            fit_score=float(np.max(scores)),
        )
        if cache_key is not None and not np.isnan(probability).any():
            _REGION_DISTRIBUTION_CACHE[cache_key] = (probability.copy(), dict(diagnostics))
        return (probability, diagnostics) if return_diagnostics else probability

    @classmethod
    def _oral_distribution_for_trial(
        cls,
        trial_row,
        partition,
        oral_mode,
        *,
        oral_center_sigma=None,
        oral_region_temperature=None,
        region_n_samples=1000,
        return_diagnostics=False,
    ):
        """Encode one trial with the configured fixed-scale oral likelihood."""
        mode = str(oral_mode).strip().lower()
        choice = int(trial_row["choice"])
        if mode == "center":
            center = OralCenterMapper._parse_center(trial_row["oral_center"])
            return cls._center_oral_distribution(
                center,
                choice,
                partition,
                center_sigma=oral_center_sigma,
                return_diagnostics=return_diagnostics,
            )
        if mode == "region":
            region = (trial_row["oral_A"], trial_row["oral_b"])
            return cls._region_oral_distribution(
                region,
                choice,
                partition,
                n_samples=region_n_samples,
                random_state=42,
                region_temperature=oral_region_temperature,
                return_diagnostics=return_diagnostics,
            )
        raise ValueError(f"Unsupported oral_mode: {oral_mode}")

    @staticmethod
    def _choice_conditioned_prior(
        partition,
        prior,
        stimulus,
        choice,
        *,
        distance_mode,
        beta=10.0,
    ):
        """Condition prior_t on the category choice made before oral report."""
        prior = OralAlignmentScoringMixin._normalize_distribution(prior)
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
                distance_mode=distance_mode,
            )[:, 0]
            if 0 <= choice_idx < len(prob):
                likelihood[hypo_idx] = float(prob[choice_idx])

        conditioned = prior * likelihood
        return OralAlignmentScoringMixin._normalize_distribution(conditioned)

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
        distance_mode=None,
        beta=10.0,
    ):
        """Return the model belief state aligned to oral-report timing."""
        state = str(model_distribution).strip().lower().replace("-", "_")
        if state in {"choice_conditioned", "choice_conditioned_prior", "choice_conditional_prior"}:
            prior_log = OralAlignmentScoringMixin._extract_prior_log(info)
            if trial_idx >= len(prior_log):
                return np.asarray([], dtype=float)
            stimulus = OralAlignmentScoringMixin._stimulus_for_trial(info, subj_df, trial_idx)
            return OralAlignmentScoringMixin._choice_conditioned_prior(
                partition=partition,
                prior=prior_log[trial_idx],
                stimulus=stimulus,
                choice=choice,
                distance_mode=distance_mode,
                beta=beta,
            )

        model_log = OralAlignmentScoringMixin._extract_model_distribution_log(info, model_distribution=state)
        if trial_idx >= len(model_log):
            return np.asarray([], dtype=float)
        return OralAlignmentScoringMixin._normalize_distribution(model_log[trial_idx])

    @staticmethod
    def _expected_center_similarity(partition, model_dist, oral_center, choice, hypo_indices=None):
        """Compare oral center with the model's choice-conditioned expected center."""
        model_dist = OralAlignmentScoringMixin._normalize_distribution(model_dist)
        center = np.asarray(oral_center, dtype=float).reshape(-1)
        if np.isnan(model_dist).any() or center.size == 0 or np.isnan(center).any():
            return np.nan

        cat_idx = int(choice) - 1
        if cat_idx < 0 or cat_idx >= int(partition.n_cats):
            return np.nan
        centers = np.asarray(
            [
                OralAlignmentScoringMixin._category_representative_center(
                    partition,
                    hypo_idx,
                    cat_idx,
                )
                for hypo_idx in range(int(partition.length))
            ],
            dtype=float,
        )
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
        model_dist = OralAlignmentScoringMixin._normalize_distribution(model_dist)
        if np.isnan(model_dist).any():
            return np.full(partition.n_dims, np.nan, dtype=float)

        cat_idx = int(choice) - 1
        if cat_idx < 0 or cat_idx >= int(partition.n_cats):
            return np.full(partition.n_dims, np.nan, dtype=float)
        centers = np.asarray(
            [
                OralAlignmentScoringMixin._category_representative_center(
                    partition,
                    hypo_idx,
                    cat_idx,
                )
                for hypo_idx in range(int(partition.length))
            ],
            dtype=float,
        )
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
        model_dist = OralAlignmentScoringMixin._normalize_distribution(model_dist)
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
        scorer = OralAlignmentScoringMixin._get_region_overlap_scorer(
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

        A, b = OralRegionMapper._parse_region(oral_region)
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

        model_dist = OralAlignmentScoringMixin._normalize_distribution(model_dist[:n_hypos])
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
        oral_field = OralRegionMapper._points_in_region(
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

    def compute_oral_mass_probabilities(
        self,
        oral_df,
        oral_mode="center",
        subjects=None,
        oral_state_mode=DEFAULT_ORAL_STATE_MODE,
        oral_center_sigma=DEFAULT_ORAL_CENTER_SIGMA,
        oral_region_temperature=DEFAULT_ORAL_REGION_TEMPERATURE,
        region_n_samples=1000,
        region_stimulus_sigma=None,
        partitions_by_subject=None,
    ):
        """Compute fixed-likelihood oral hypothesis distributions per subject.

        The default primary ``oral_mass`` keeps the latest valid report for
        every category and jointly normalizes their likelihood product. The
        current report's one-category distribution is retained separately as
        ``instantaneous_oral_mass`` for audit and legacy comparisons.
        """
        mode = str(oral_mode).strip().lower()
        state_mode = self._validate_oral_state_mode(oral_state_mode)
        if mode == "center":
            oral_center_sigma = self._validate_positive_scale(
                oral_center_sigma,
                "oral_center_sigma",
            )
        elif mode == "region":
            oral_region_temperature = self._validate_positive_scale(
                oral_region_temperature,
                "oral_region_temperature",
            )
        else:
            raise ValueError(f"Unsupported oral_mode: {oral_mode}")

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
            partition = None
            if partitions_by_subject is not None:
                partition = partitions_by_subject.get(int(iSub))
            if partition is None:
                partition = ContinuousPartition(n_dims=4, n_cats=n_cats)
            if partition.n_cats != n_cats:
                raise ValueError(
                    "Oral partition category count does not match subject condition: "
                    f"{partition.n_cats} vs {n_cats}."
                )
            n_trials = len(subj_df)
            oral_mass = np.full((n_trials, partition.length), np.nan, dtype=float)
            instantaneous_oral_mass = np.full(
                (n_trials, partition.length),
                np.nan,
                dtype=float,
            )
            valid_oral = []
            valid_oral_report = []
            diagnostic_arrays = {
                field: np.full(n_trials, np.nan, dtype=float)
                for field in self.ORAL_TRIAL_DIAGNOSTIC_FIELDS
            }
            encoder_metadata = self._oral_encoder_metadata(
                partition,
                mode,
                center_sigma=oral_center_sigma,
                region_temperature=oral_region_temperature,
                oral_state_mode=state_mode,
            )
            latest_by_category = {}

            for trial_idx in range(n_trials):
                report_dist, report_diagnostics = self._oral_distribution_for_trial(
                    subj_df.loc[trial_idx],
                    partition,
                    mode,
                    oral_center_sigma=oral_center_sigma,
                    oral_region_temperature=oral_region_temperature,
                    region_n_samples=region_n_samples,
                    return_diagnostics=True,
                )

                report_valid = not np.isnan(report_dist).any()
                valid_oral_report.append(bool(report_valid))
                if report_valid:
                    instantaneous_oral_mass[trial_idx, : len(report_dist)] = report_dist
                    category = int(subj_df.loc[trial_idx, "choice"])
                    latest_by_category[category] = {
                        "distribution": report_dist.copy(),
                        "diagnostics": dict(report_diagnostics),
                    }

                if state_mode == "latest_by_category":
                    state_dist, state_diagnostics = self._category_state_distribution(
                        latest_by_category,
                        partition,
                        mode,
                        center_sigma=oral_center_sigma,
                        region_temperature=oral_region_temperature,
                    )
                    observed_categories = len(latest_by_category)
                    category_mask = sum(
                        1 << (int(category) - 1)
                        for category in latest_by_category
                        if 1 <= int(category) <= n_cats
                    )
                else:
                    state_dist = report_dist
                    state_diagnostics = {
                        **report_diagnostics,
                        **encoder_metadata,
                    }
                    observed_categories = int(report_valid)
                    category = int(subj_df.loc[trial_idx, "choice"])
                    category_mask = (
                        1 << (category - 1)
                        if report_valid and 1 <= category <= n_cats
                        else 0
                    )

                state_valid = not np.isnan(state_dist).any()
                valid_oral.append(bool(state_valid))
                if state_valid:
                    oral_mass[trial_idx, : len(state_dist)] = state_dist

                for field in self.ORAL_TRIAL_DIAGNOSTIC_FIELDS:
                    if field.startswith("instantaneous_"):
                        source_field = field.removeprefix("instantaneous_")
                        value = report_diagnostics.get(source_field, np.nan)
                    else:
                        value = state_diagnostics.get(field, np.nan)
                    diagnostic_arrays[field][trial_idx] = float(value)
                diagnostic_arrays["oral_state_observed_categories"][trial_idx] = float(
                    observed_categories
                )
                diagnostic_arrays["oral_state_category_mask"][trial_idx] = float(
                    category_mask
                )
                diagnostic_arrays["oral_state_update_category"][trial_idx] = float(
                    int(subj_df.loc[trial_idx, "choice"])
                )
                diagnostic_arrays["oral_state_update_valid"][trial_idx] = float(
                    report_valid
                )

            out[int(iSub)] = {
                "iSub": int(iSub),
                "condition": condition,
                "target_hypo": target_hypo,
                "oral_mode": mode,
                "region_stimulus_sigma": np.nan,
                **encoder_metadata,
                **diagnostic_arrays,
                "oral_mass": oral_mass,
                "valid_oral": valid_oral,
                "instantaneous_oral_mass": instantaneous_oral_mass,
                "valid_oral_report": valid_oral_report,
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
        oral_state_mode=DEFAULT_ORAL_STATE_MODE,
        oral_center_sigma=DEFAULT_ORAL_CENTER_SIGMA,
        oral_region_temperature=DEFAULT_ORAL_REGION_TEMPERATURE,
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
        if oral_mass_results is None:
            partitions_by_subject = self._partitions_for_model_results(model_res)
            oral_mass_results = self.compute_oral_mass_probabilities(
                oral_df,
                oral_mode=oral_mode,
                subjects=sorted(int(sid) for sid in model_res),
                oral_state_mode=oral_state_mode,
                oral_center_sigma=oral_center_sigma,
                oral_region_temperature=oral_region_temperature,
                region_n_samples=region_n_samples,
                region_stimulus_sigma=region_stimulus_sigma,
                partitions_by_subject=partitions_by_subject,
            )
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
            partition, _ = self._partition_for_model_result(
                info,
                expected_n_cats=n_cats,
            )
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

                oral_dist, _ = self._resolve_oral_distribution(
                    oral_mass_results,
                    sid,
                    trial_idx,
                    subj_df.loc[trial_idx],
                    partition,
                    oral_mode,
                    oral_center_sigma=oral_center_sigma,
                    oral_region_temperature=oral_region_temperature,
                    region_n_samples=region_n_samples,
                )

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
                **self._oral_encoder_metadata(
                    partition,
                    oral_mode,
                    center_sigma=oral_center_sigma,
                    region_temperature=oral_region_temperature,
                    oral_state_mode=oral_state_mode,
                ),
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
            prototypes = np.round(
                OralAlignmentScoringMixin._category_prototypes(
                    partition,
                    int(hypo_idx),
                    cat_idx,
                ),
                int(decimals),
            )
            key = "center" if len(prototypes) == 1 else "component_centers"
            value = prototypes[0].tolist() if len(prototypes) == 1 else prototypes.tolist()
            return json.dumps({key: value}, ensure_ascii=False)
        if oral_mode == "region":
            region = OralRegionMapper._true_region(
                partition.hypothesis_space,
                int(hypo_idx),
                cat_idx,
            )
            components = []
            for component in OralRegionMapper._region_components(region):
                A, b = OralRegionMapper._parse_region(component)
                if A is None or b is None:
                    return json.dumps({"region": None}, ensure_ascii=False)
                components.append(
                    {
                        "constraint": "A @ x <= b",
                        "A": np.round(A, int(decimals)).tolist(),
                        "b": np.round(b, int(decimals)).tolist(),
                    }
                )
            payload = components[0] if len(components) == 1 else {"union": components}
            return json.dumps(payload, ensure_ascii=False)
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
            partition = ContinuousPartition(n_dims=4, n_cats=n_cats)
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

    @classmethod
    def _oral_diagnostics_from_precomputed(cls, oral_mass_results, iSub, trial_idx):
        """Fetch encoder provenance and trial diagnostics from oral mass results."""
        if oral_mass_results is None:
            return {}
        info = oral_mass_results.get(iSub)
        if info is None:
            info = oral_mass_results.get(int(iSub))
        if info is None:
            info = oral_mass_results.get(str(iSub))
        if info is None:
            return {}

        out = {
            field: info.get(field, np.nan)
            for field in cls.ORAL_ENCODER_METADATA_FIELDS
        }
        for field in cls.ORAL_TRIAL_DIAGNOSTIC_FIELDS:
            values = np.asarray(info.get(field, []), dtype=float).reshape(-1)
            out[field] = (
                float(values[int(trial_idx)])
                if 0 <= int(trial_idx) < values.size
                else np.nan
            )
        return out

    @classmethod
    def _resolve_oral_distribution(
        cls,
        oral_mass_results,
        iSub,
        trial_idx,
        trial_row,
        partition,
        oral_mode,
        *,
        oral_center_sigma=None,
        oral_region_temperature=None,
        region_n_samples=1000,
    ):
        """Use precomputed oral mass or reproduce it with the same encoder."""
        precomputed = cls._oral_distribution_from_precomputed(
            oral_mass_results,
            iSub,
            trial_idx,
        )
        if precomputed is not None:
            if precomputed.size != int(partition.length):
                raise ValueError(
                    "Precomputed oral distribution does not match the model "
                    "hypothesis space: "
                    f"{precomputed.size} vs {partition.length}."
                )
            return (
                precomputed,
                cls._oral_diagnostics_from_precomputed(
                    oral_mass_results,
                    iSub,
                    trial_idx,
                ),
            )
        return cls._oral_distribution_for_trial(
            trial_row,
            partition,
            oral_mode,
            oral_center_sigma=oral_center_sigma,
            oral_region_temperature=oral_region_temperature,
            region_n_samples=region_n_samples,
            return_diagnostics=True,
        )

    def compute_distribution_alignment(
        self,
        model_results,
        oral_df,
        oral_mode="center",
        subjects=None,
        oral_state_mode=DEFAULT_ORAL_STATE_MODE,
        oral_center_sigma=DEFAULT_ORAL_CENTER_SIGMA,
        oral_region_temperature=DEFAULT_ORAL_REGION_TEMPERATURE,
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
        if oral_mass_results is None:
            partitions_by_subject = self._partitions_for_model_results(model_res)
            oral_mass_results = self.compute_oral_mass_probabilities(
                oral_df,
                oral_mode=oral_mode,
                subjects=sorted(int(sid) for sid in model_res),
                oral_state_mode=oral_state_mode,
                oral_center_sigma=oral_center_sigma,
                oral_region_temperature=oral_region_temperature,
                region_n_samples=region_n_samples,
                region_stimulus_sigma=region_stimulus_sigma,
                partitions_by_subject=partitions_by_subject,
            )
        rows = []

        for iSub, info in model_res.items():
            sid = int(iSub)
            subj_df = oral_df[oral_df["iSub"] == sid].reset_index(drop=True)
            if subj_df.empty:
                continue

            condition = int(info.get("condition", subj_df["condition"].iloc[0]))
            n_cats = 2 if condition == 1 else 4
            partition, _ = self._partition_for_model_result(
                info,
                expected_n_cats=n_cats,
            )
            model_log = self._extract_model_distribution_log(info, model_distribution=model_distribution)
            n_trials = min(len(subj_df), len(model_log))

            for trial_idx in range(n_trials):
                choice = int(subj_df.loc[trial_idx, "choice"])
                raw_model = np.asarray(model_log[trial_idx], dtype=float).reshape(-1)
                model_dist = self._normalize_distribution(raw_model)

                oral_dist, oral_diagnostics = self._resolve_oral_distribution(
                    oral_mass_results,
                    sid,
                    trial_idx,
                    subj_df.loc[trial_idx],
                    partition,
                    oral_mode,
                    oral_center_sigma=oral_center_sigma,
                    oral_region_temperature=oral_region_temperature,
                    region_n_samples=region_n_samples,
                )

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
                            **oral_diagnostics,
                        }
                    )

        return pd.DataFrame(rows)

    def compute_distribution_based_alignment(self, *args, **kwargs):
        """Alias for the distribution-based alignment family."""
        return self.compute_distribution_alignment(*args, **kwargs)

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
            partition, distance_mode = self._partition_for_model_result(
                info,
                expected_n_cats=n_cats,
            )
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
                    distance_mode=distance_mode,
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
                    oral_center = OralCenterMapper._parse_center(subj_df.loc[trial_idx, "oral_center"])
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

    def compute_target_based_alignment(
        self,
        model_results,
        oral_df,
        oral_mode="center",
        subjects=None,
        oral_state_mode=DEFAULT_ORAL_STATE_MODE,
        oral_center_sigma=DEFAULT_ORAL_CENTER_SIGMA,
        oral_region_temperature=DEFAULT_ORAL_REGION_TEMPERATURE,
        region_n_samples=1000,
        region_stimulus_sigma=None,
        oral_mass_results=None,
        alignment_spaces=None,
        active_threshold=1e-12,
        trajectory_band_window_size=16,
    ):
        """Extract target-hypothesis mass on full, active, and union spaces.

        ``full`` uses the complete hypothesis space. ``active`` renormalizes
        model prior and oral mass inside the current model active set.
        ``union_topn`` renormalizes inside the union of the model active set and
        the oral top-N set, where N is the active-set size.
        """
        model_res = self._filter_results(model_results, subjects)
        oral_df = oral_df.copy()
        if oral_mass_results is None:
            partitions_by_subject = self._partitions_for_model_results(model_res)
            oral_mass_results = self.compute_oral_mass_probabilities(
                oral_df,
                oral_mode=oral_mode,
                subjects=sorted(int(sid) for sid in model_res),
                oral_state_mode=oral_state_mode,
                oral_center_sigma=oral_center_sigma,
                oral_region_temperature=oral_region_temperature,
                region_n_samples=region_n_samples,
                region_stimulus_sigma=region_stimulus_sigma,
                partitions_by_subject=partitions_by_subject,
            )
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
            raw_target_hypo = info.get("target_hypothesis")
            target_hypo = (
                int(raw_target_hypo)
                if raw_target_hypo is not None
                else (0 if condition == 1 else 42)
            )
            partition, _ = self._partition_for_model_result(
                info,
                expected_n_cats=n_cats,
            )
            prior_repeat_logs, model_state_source = self._extract_prior_repeat_logs(info)
            if not prior_repeat_logs:
                continue
            model_inference_backend = (
                "particle_filter"
                if str(info.get("state_distribution_kind", "")).lower()
                == "particle_marginal"
                else "trajectory"
            )
            n_trials = min(
                len(subj_df),
                *(arr.shape[0] for arr in prior_repeat_logs),
            )
            n_model_runs = int(len(prior_repeat_logs))
            trajectory_probability_runs = (
                {
                    space: np.full((n_model_runs, n_trials), np.nan, dtype=float)
                    for space in spaces
                }
                if model_inference_backend == "trajectory"
                else None
            )
            subject_rows_by_space = {space: [] for space in spaces}

            for trial_idx in range(n_trials):
                choice = int(subj_df.loc[trial_idx, "choice"])
                repeat_priors = self._normalize_distribution_rows(
                    np.vstack([run[trial_idx] for run in prior_repeat_logs])
                )
                prior = self._normalize_distribution(np.mean(repeat_priors, axis=0))
                active_idx = self._active_hypothesis_indices(
                    prior,
                    active_threshold=active_threshold,
                )

                oral_dist, oral_diagnostics = self._resolve_oral_distribution(
                    oral_mass_results,
                    sid,
                    trial_idx,
                    subj_df.loc[trial_idx],
                    partition,
                    oral_mode,
                    oral_center_sigma=oral_center_sigma,
                    oral_region_temperature=oral_region_temperature,
                    region_n_samples=region_n_samples,
                )

                for space in spaces:
                    compare_prior, compare_oral, compare_idx = self._comparison_space_distributions(
                        prior,
                        oral_dist,
                        alignment_space=space,
                        active_idx=active_idx,
                    )
                    repeat_target_probabilities = (
                        self._repeat_target_probabilities_in_space(
                            repeat_priors,
                            compare_idx,
                            target_hypo,
                        )
                    )
                    if trajectory_probability_runs is not None:
                        trajectory_probability_runs[space][:, trial_idx] = (
                            repeat_target_probabilities
                        )
                    model_target_prior = (
                        float(np.nanmean(repeat_target_probabilities))
                        if np.any(np.isfinite(repeat_target_probabilities))
                        else np.nan
                    )
                    model_target_repeat_sd = (
                        float(np.nanstd(repeat_target_probabilities, ddof=1))
                        if np.sum(np.isfinite(repeat_target_probabilities)) > 1
                        else 0.0
                    )
                    oral_target_mass = self._target_probability_in_space(compare_oral, compare_idx, target_hypo)
                    if len(compare_idx) and not np.isnan(oral_dist).any():
                        oral_mass_in_comparison = float(np.sum(np.asarray(oral_dist, dtype=float)[compare_idx]))
                    else:
                        oral_mass_in_comparison = np.nan

                    row = {
                            "iSub": sid,
                            "subject": sid,
                            "condition": condition,
                            "trial": trial_idx + 1,
                            "trial_pct": (trial_idx + 1) / float(n_trials) if n_trials else np.nan,
                            "oral_mode": oral_mode,
                            "model_distribution": "prior",
                            "model_state_source": model_state_source,
                            "model_inference_backend": model_inference_backend,
                            "model_target_n_runs": n_model_runs,
                            "model_target_n_pf_runs": (
                                n_model_runs
                                if model_inference_backend == "particle_filter"
                                else np.nan
                            ),
                            "model_target_repeat_sd": model_target_repeat_sd,
                            "alignment_space": space,
                            "alignment_label": self.TARGET_ALIGNMENT_LABELS.get(space, space),
                            "target_hypo": target_hypo,
                            "active_set_size": int(len(active_idx)),
                            "comparison_set_size": int(len(compare_idx)),
                            "oral_mass_in_comparison_set": oral_mass_in_comparison,
                            "model_target_prior": model_target_prior,
                            "oral_target_mass": oral_target_mass,
                            "valid": bool(np.isfinite(model_target_prior) and np.isfinite(oral_target_mass)),
                            **oral_diagnostics,
                        }
                    rows.append(row)
                    subject_rows_by_space[space].append(row)

            if (
                trajectory_probability_runs is not None
                and n_trials >= int(trajectory_band_window_size)
            ):
                for space in spaces:
                    band = self.compute_trajectory_target_band(
                        trajectory_probability_runs[space],
                        window_size=trajectory_band_window_size,
                    )
                    for trial_idx, row in enumerate(subject_rows_by_space[space]):
                        row.update(
                            {
                                "model_target_expected_rolling": band["expected"][trial_idx],
                                "model_target_q05_rolling": band["q05"][trial_idx],
                                "model_target_q25_rolling": band["q25"][trial_idx],
                                "model_target_q50_rolling": band["q50"][trial_idx],
                                "model_target_q75_rolling": band["q75"][trial_idx],
                                "model_target_q95_rolling": band["q95"][trial_idx],
                                "model_target_band_type": band["band_type"],
                                "model_target_band_n_draws": np.nan,
                                "model_target_band_n_runs": band["n_runs"],
                                "model_target_band_base_seed": np.nan,
                                "model_target_band_subject_seed": np.nan,
                                "model_target_band_window_size": band["window_size"],
                            }
                        )

        return pd.DataFrame(rows)

    def compute_hit_based_alignment(
        self,
        model_results,
        oral_df,
        oral_mode="center",
        subjects=None,
        oral_state_mode=DEFAULT_ORAL_STATE_MODE,
        oral_center_sigma=DEFAULT_ORAL_CENTER_SIGMA,
        oral_region_temperature=DEFAULT_ORAL_REGION_TEMPERATURE,
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
        if oral_mass_results is None:
            partitions_by_subject = self._partitions_for_model_results(model_res)
            oral_mass_results = self.compute_oral_mass_probabilities(
                oral_df,
                oral_mode=oral_mode,
                subjects=sorted(int(sid) for sid in model_res),
                oral_state_mode=oral_state_mode,
                oral_center_sigma=oral_center_sigma,
                oral_region_temperature=oral_region_temperature,
                region_n_samples=region_n_samples,
                region_stimulus_sigma=region_stimulus_sigma,
                partitions_by_subject=partitions_by_subject,
            )
        rows = []

        for iSub, info in model_res.items():
            sid = int(iSub)
            subj_df = oral_df[oral_df["iSub"] == sid].reset_index(drop=True)
            if subj_df.empty:
                continue

            condition = int(info.get("condition", subj_df["condition"].iloc[0]))
            n_cats = 2 if condition == 1 else 4
            raw_target_hypo = info.get("target_hypothesis")
            target_hypo = (
                int(raw_target_hypo)
                if raw_target_hypo is not None
                else (0 if condition == 1 else 42)
            )
            partition, _ = self._partition_for_model_result(
                info,
                expected_n_cats=n_cats,
            )
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

                oral_dist, oral_diagnostics = self._resolve_oral_distribution(
                    oral_mass_results,
                    sid,
                    trial_idx,
                    subj_df.loc[trial_idx],
                    partition,
                    oral_mode,
                    oral_center_sigma=oral_center_sigma,
                    oral_region_temperature=oral_region_temperature,
                    region_n_samples=region_n_samples,
                )

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
                        **oral_diagnostics,
                    }
                )

        return pd.DataFrame(rows)

    def compute_coverage_based_alignment(
        self,
        model_results,
        oral_df,
        oral_mode="center",
        subjects=None,
        oral_state_mode=DEFAULT_ORAL_STATE_MODE,
        oral_center_sigma=DEFAULT_ORAL_CENTER_SIGMA,
        oral_region_temperature=DEFAULT_ORAL_REGION_TEMPERATURE,
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
        if oral_mass_results is None:
            partitions_by_subject = self._partitions_for_model_results(model_res)
            oral_mass_results = self.compute_oral_mass_probabilities(
                oral_df,
                oral_mode=oral_mode,
                subjects=sorted(int(sid) for sid in model_res),
                oral_state_mode=oral_state_mode,
                oral_center_sigma=oral_center_sigma,
                oral_region_temperature=oral_region_temperature,
                region_n_samples=region_n_samples,
                region_stimulus_sigma=region_stimulus_sigma,
                partitions_by_subject=partitions_by_subject,
            )
        rows = []

        for iSub, info in model_res.items():
            sid = int(iSub)
            subj_df = oral_df[oral_df["iSub"] == sid].reset_index(drop=True)
            if subj_df.empty:
                continue

            condition = int(info.get("condition", subj_df["condition"].iloc[0]))
            n_cats = 2 if condition == 1 else 4
            partition, _ = self._partition_for_model_result(
                info,
                expected_n_cats=n_cats,
            )
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
                oral_dist, oral_diagnostics = self._resolve_oral_distribution(
                    oral_mass_results,
                    sid,
                    trial_idx,
                    subj_df.loc[trial_idx],
                    partition,
                    oral_mode,
                    oral_center_sigma=oral_center_sigma,
                    oral_region_temperature=oral_region_temperature,
                    region_n_samples=region_n_samples,
                )

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
                        **oral_diagnostics,
                    }
                )

        return pd.DataFrame(rows)


__all__ = ["OralAlignmentScoringMixin"]

"""Normalized hypothesis-to-report observation kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .shared import normalize_probability_vector


@dataclass(frozen=True)
class OralReportReadoutResult:
    probabilities: np.ndarray
    reliability: float


def read_oral_report(
    hypothesis_weights: Sequence[float] | np.ndarray,
    report_mapping: Sequence[Sequence[float]] | np.ndarray,
    *,
    reliability: float = 1.0,
    baseline: Sequence[float] | np.ndarray | None = None,
) -> OralReportReadoutResult:
    """Map rule weights through a normalized hypothesis-to-report matrix."""

    weights = normalize_probability_vector(hypothesis_weights, strict=True)
    mapping = np.asarray(report_mapping, dtype=float)
    if mapping.ndim != 2 or mapping.shape[0] != weights.size or mapping.shape[1] <= 0:
        raise ValueError(
            "report_mapping must have shape [hypotheses, report_codes], "
            f"got {mapping.shape} for {weights.size} hypotheses."
        )
    if not np.all(np.isfinite(mapping)) or np.any(mapping < 0.0):
        raise ValueError("report_mapping must be finite and non-negative.")
    row_totals = np.sum(mapping, axis=1, keepdims=True)
    if np.any(row_totals <= 0.0):
        raise ValueError("every report_mapping row must have positive mass.")
    mapping = mapping / row_totals
    reliability_value = float(reliability)
    if not np.isfinite(reliability_value) or not 0.0 <= reliability_value <= 1.0:
        raise ValueError("oral-report reliability must lie in [0, 1].")
    report_count = int(mapping.shape[1])
    baseline_probabilities = (
        np.full(report_count, 1.0 / report_count, dtype=float)
        if baseline is None
        else normalize_probability_vector(baseline, report_count, strict=True)
    )
    core = normalize_probability_vector(weights @ mapping, report_count, strict=True)
    observed = normalize_probability_vector(
        reliability_value * core + (1.0 - reliability_value) * baseline_probabilities,
        report_count,
        strict=True,
    )
    return OralReportReadoutResult(observed, reliability_value)


__all__ = ["OralReportReadoutResult", "read_oral_report"]

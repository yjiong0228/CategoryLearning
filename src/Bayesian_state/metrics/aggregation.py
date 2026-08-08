"""Pure aggregation of repeated-run metric values."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .numeric import finite_array


def behavior_ppc_group_metrics(values: Mapping[str, Any]) -> dict[str, float]:
    """Aggregate the established behavior-PPC columns for one subject."""
    choice_error = finite_array(values.get("choice_error", []))
    accuracy_mae = finite_array(values.get("acc_mae", []))
    volatility_ratio = finite_array(values.get("vol_ratio", []))
    history_correlation = finite_array(values.get("history_corr", []))
    switch_score = finite_array(values.get("switch_score", []))

    def mean(array: np.ndarray) -> float:
        return float(np.mean(array)) if array.size else float("nan")

    def median(array: np.ndarray) -> float:
        return float(np.median(array)) if array.size else float("nan")

    def quantile(array: np.ndarray, probability: float) -> float:
        return float(np.quantile(array, probability)) if array.size else float("nan")

    return {
        "choice_error_mean": mean(choice_error),
        "acc_mae_mean": mean(accuracy_mae),
        "acc_mae_median": median(accuracy_mae),
        "vol_ratio_median": median(volatility_ratio),
        "vol_ratio_q10": quantile(volatility_ratio, 0.10),
        "vol_ratio_q90": quantile(volatility_ratio, 0.90),
        "history_corr_mean": mean(history_correlation),
        "history_corr_median": median(history_correlation),
        "switch_score_mean": mean(switch_score),
        "switch_score_median": median(switch_score),
    }


__all__ = ["behavior_ppc_group_metrics"]

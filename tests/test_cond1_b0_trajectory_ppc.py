from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_cond1_b0_trajectory_ppc import (
    acquisition_hazard_from_half_life,
    benjamini_hochberg,
    block_bounded_rolling_accuracy,
    detect_events,
    empirical_crps,
    longest_true_run,
    rolling_accuracy,
    split_for_subject,
    trajectory_metrics,
)


def test_trajectory_metrics_capture_drop_and_error_run() -> None:
    feedback = np.asarray([1] * 12 + [0] * 12 + [1] * 12, dtype=float)
    choices = np.asarray([1, 2] * 18, dtype=int)
    metrics = trajectory_metrics(feedback, choices, window=12)
    assert metrics["max_adjacent_drop"] == 1.0
    assert metrics["event_count"] == 1.0
    assert metrics["max_event_duration"] == 12.0
    assert metrics["longest_error_streak"] == 12.0


def test_rolling_accuracy_handles_vector_and_matrix() -> None:
    values = np.asarray([1.0, 0.0, 1.0, 0.0])
    vector = rolling_accuracy(values, 2)
    matrix = rolling_accuracy(np.vstack([values, values]), 2)
    assert np.allclose(vector, [0.5, 0.5, 0.5])
    assert np.allclose(matrix[0], vector)
    assert matrix.shape == (2, 3)


def test_empirical_crps_is_zero_for_degenerate_exact_prediction() -> None:
    observed = np.asarray([0.25, 0.75])
    simulations = np.tile(observed, (20, 1))
    assert empirical_crps(observed, simulations) == 0.0


def test_helpers_are_deterministic_and_nan_preserving() -> None:
    assert longest_true_run([False, True, True, False, True]) == 2
    events = detect_events(
        np.asarray([1] * 12 + [0] * 12 + [1] * 12),
        12,
    )
    assert events == [{"onset": 12, "end": 23, "duration": 12}]
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03, np.nan])
    assert np.allclose(adjusted[:3], [0.03, 0.04, 0.04])
    assert np.isnan(adjusted[3])


def test_split_modes_define_short_and_long_horizons() -> None:
    frame = pd.DataFrame(
        {
            "iSession": [1] * 8,
            "iBlock": [1] * 4 + [2] * 4,
        }
    )
    assert split_for_subject(frame, mode="last_block") == (4, "last_block")
    assert split_for_subject(frame, mode="early_anchor") == (
        4,
        "after_first_block",
    )


def test_block_bounded_rolling_never_crosses_a_boundary() -> None:
    feedback = np.asarray([1, 1, 0, 0, 1, 1, 1, 1], dtype=float)
    blocks = np.asarray([1, 1, 1, 1, 2, 2, 2, 2])
    rolling, end_indices = block_bounded_rolling_accuracy(
        feedback,
        blocks,
        3,
    )
    assert np.allclose(rolling, [2 / 3, 1 / 3, 1.0, 1.0])
    assert np.array_equal(end_indices, [2, 3, 6, 7])


def test_acquisition_half_life_maps_to_geometric_median() -> None:
    hazard = acquisition_hazard_from_half_life(64.0)
    assert np.isclose(np.power(1.0 - hazard, 64), 0.5)

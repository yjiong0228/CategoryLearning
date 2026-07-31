from __future__ import annotations

import numpy as np

from scripts.analyze_cond1_phase_coverage import (
    gradual_change_count,
    phase_metrics,
    phase_vector,
)


def block_ids(n: int) -> np.ndarray:
    return np.asarray(["1:1"] * n)


def chunk(*correct_counts: int, window: int = 12) -> np.ndarray:
    pieces = []
    for count in correct_counts:
        pieces.append(
            np.asarray(
                [1.0] * int(count)
                + [0.0] * (int(window) - int(count)),
                dtype=float,
            )
        )
    return np.concatenate(pieces)


def test_phase_metrics_detect_abrupt_rise() -> None:
    feedback = chunk(4, 4, 10, 11)
    metrics = phase_metrics(
        feedback,
        block_ids(len(feedback)),
        window=12,
    )
    assert metrics["abrupt_rise_count"] == 1.0
    assert metrics["abrupt_drop_count"] == 0.0
    assert metrics["stable_high_chunk_fraction"] == 0.5
    assert metrics["first_high_latency_fraction"] == 2.0 / 3.0


def test_phase_metrics_detect_drop_and_recovery() -> None:
    feedback = chunk(10, 10, 4, 4, 10, 10)
    metrics = phase_metrics(
        feedback,
        block_ids(len(feedback)),
        window=12,
    )
    assert metrics["abrupt_drop_count"] == 1.0
    assert metrics["abrupt_rise_count"] == 1.0
    assert metrics["recovery_count"] == 1.0
    assert metrics["direction_reversal_count"] == 1.0


def test_gradual_change_requires_two_small_consistent_steps() -> None:
    values = np.asarray([0.25, 5 / 12, 7 / 12, 0.75])
    assert gradual_change_count(values, 12) == 1
    assert gradual_change_count(np.asarray([0.25, 0.5, 0.75]), 12) == 0


def test_phase_metrics_identify_chance_and_high_fractions() -> None:
    feedback = chunk(5, 6, 7, 8, 10, 12)
    metrics = phase_metrics(
        feedback,
        block_ids(len(feedback)),
        window=12,
    )
    assert metrics["chance_chunk_fraction"] == 4.0 / 6.0
    assert metrics["stable_high_chunk_fraction"] == 2.0 / 6.0
    assert metrics["phase_diversity"] == 3.0


def test_phase_vector_is_deterministic_and_window_complete() -> None:
    feedback = chunk(4, 6, 8, 10, 5, 11)
    blocks = block_ids(len(feedback))
    first, keys = phase_vector(feedback, blocks, (8, 12, 16))
    second, second_keys = phase_vector(feedback, blocks, (8, 12, 16))
    assert keys == second_keys
    assert len(keys) == 30
    np.testing.assert_allclose(first, second)

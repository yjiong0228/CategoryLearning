from __future__ import annotations

import numpy as np

from src.Bayesian_state.evaluation.autonomous_trajectories import (
    AutonomousEnsemble,
    rolling_binary_ensemble,
    summarize_trajectory_shapes,
    sustained_mastery_onsets,
)
from src.Bayesian_state.metrics.trial import sliding_binary_metrics


def test_rolling_ensemble_matches_established_pipeline_alignment():
    values = np.asarray(
        [
            [0, 1, 1, 0, 1, 1, 1, 0],
            [1, 0, 0, 1, 0, 0, 0, 1],
        ],
        dtype=float,
    )
    trial, curves = rolling_binary_ensemble(values, window_size=3)

    assert np.array_equal(trial, np.arange(4, 9))
    for row_index, row in enumerate(values):
        expected, _, _ = sliding_binary_metrics(row, row, window_size=3)
        assert np.allclose(curves[row_index], expected)


def test_sustained_mastery_requires_consecutive_rolling_windows():
    trial = np.arange(5, 13)
    curves = np.asarray(
        [
            [0.5, 0.8, 0.9, 0.7, 0.9, 0.9, 1.0, 1.0],
            [0.5, 0.8, 0.9, 0.9, 0.7, 0.9, 1.0, 1.0],
        ]
    )
    onset = sustained_mastery_onsets(
        curves,
        trial,
        threshold=0.8,
        sustain_windows=3,
    )

    assert onset[0] == 9
    assert onset[1] == 6


def test_shape_summary_returns_medoid_and_complete_partition():
    feedback = np.asarray(
        [
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int8,
    )
    ensemble = AutonomousEnsemble(
        choices=np.where(feedback > 0, 1, 2).astype(np.int16),
        feedback=feedback,
        expected_correct_probability=np.full(
            feedback.shape,
            0.5,
            dtype=np.float32,
        ),
        trajectory_seeds=np.arange(feedback.shape[0], dtype=np.uint32),
    )
    summary = summarize_trajectory_shapes(
        ensemble,
        observed_feedback=feedback[1],
        window_size=3,
        mastery_threshold=0.75,
        mastery_sustain_windows=2,
        final_block_trials=4,
        max_clusters=3,
        cluster_seed=7,
    )

    assert 0 <= summary.medoid_index < feedback.shape[0]
    assert summary.rolling_accuracy.shape == (feedback.shape[0], 9)
    assert summary.central_50_indices.size == 3
    assert summary.central_90_indices.size == 6
    assert summary.cluster_labels.shape == (feedback.shape[0],)
    assert np.isclose(np.sum(summary.cluster_shares), 1.0)
    assert set(np.unique(summary.cluster_labels)) == set(
        range(summary.cluster_count)
    )
    central = summary.rolling_accuracy[summary.central_50_indices]
    assert np.all(central >= summary.central_50_lower)
    assert np.all(central <= summary.central_50_upper)

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_cond1_behavior_event_atlas import (
    DetectorSpec,
    benjamini_hochberg,
    detect_abrupt_events,
    robust_scale,
    stay_metrics,
)


def test_detect_abrupt_event_and_recovery_within_one_block() -> None:
    feedback = np.asarray(
        [1] * 12 + [0] * 12 + [1] * 12,
        dtype=float,
    )
    events = detect_abrupt_events(feedback, DetectorSpec(window=12))
    assert len(events) == 1
    event = events[0]
    assert event["onset"] == 12
    assert event["recovered"] is True
    assert event["recovery_onset"] == 24
    assert event["end"] == 23
    assert event["duration"] == 12


def test_detector_does_not_create_event_without_learned_baseline() -> None:
    feedback = np.asarray([0, 1] * 24, dtype=float)
    events = detect_abrupt_events(feedback, DetectorSpec(window=8))
    assert events == []


def test_robust_scale_is_finite_for_constant_values() -> None:
    center, scale = robust_scale(np.ones(12, dtype=float))
    assert center == 1.0
    assert scale == 1.0


def test_stay_metrics_separate_win_and_loss() -> None:
    phase = pd.DataFrame(
        {
            "choice": [1, 1, 2, 2, 2],
            "feedback": [1, 0, 1, 0, 1],
        }
    )
    result = stay_metrics(phase)
    assert np.isclose(result["win_stay_rate"], 1.0)
    assert np.isclose(result["lose_stay_rate"], 0.5)


def test_benjamini_hochberg_is_monotone_in_rank() -> None:
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03, np.nan])
    assert np.allclose(adjusted[:3], [0.03, 0.04, 0.04])
    assert np.isnan(adjusted[3])

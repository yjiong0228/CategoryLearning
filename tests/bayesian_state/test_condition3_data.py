"""Response-key metadata must not confuse choices with the correct category."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.Bayesian_state.simulation.data import (
    SubjectTrialDataLoader,
    _coerce_trial_arrays,
    prepare_trial_sequence,
)


def _subject_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "iSub": [301] * 4,
            "condition": [3] * 4,
            "session": [1, 1, 2, 2],
            "feature1": [0.1, 0.2, 0.3, 0.4],
            "choice": [1, 2, 3, 4],
            # Deliberately all errors: category/presskey would infer a wrong map.
            "category": [2, 3, 4, 1],
            "presskey": [4, 2, 3, 1],
            "feedback": [0.5, 0.0, 0.5, 0.0],
        }
    )


def _loader(frame: pd.DataFrame) -> SubjectTrialDataLoader:
    loader = SubjectTrialDataLoader({"data": {"feature_columns": ["feature1"]}})
    loader.learning_data = frame
    return loader


def test_response_mapping_uses_full_subject_choices_before_both_truncations():
    frame = _subject_frame()
    original = frame.copy(deep=True)
    loader = _loader(frame)
    arrays = loader._extract_arrays(loader._get_subject_frame(301, 0.5), 1)

    assert arrays.choice_to_presskey == {1: 4, 2: 2, 3: 3, 4: 1}
    assert arrays.presskey_to_choice == {4: 1, 2: 2, 3: 3, 1: 4}
    np.testing.assert_array_equal(arrays.presskeys, [4])
    np.testing.assert_array_equal(arrays.choices, [1])
    np.testing.assert_array_equal(arrays.categories, [2])
    assert arrays.probability_coordinate == "choice"
    pd.testing.assert_frame_equal(frame, original)
    # The learner's observation remains just stimulus, opaque choice, feedback.
    assert len(prepare_trial_sequence(arrays.stimulus, arrays.choices, arrays.feedback)[0]) == 3


@pytest.mark.parametrize("choices, presskeys", [([1, 1], [4, 2]), ([1, 2], [4, 4])])
def test_response_mapping_rejects_non_bijections_across_sessions_and_stop_at(choices, presskeys):
    frame = _subject_frame().iloc[:2].copy()
    frame["choice"] = choices
    frame["presskey"] = presskeys
    frame["session"] = [1, 2]
    with pytest.raises(ValueError, match="one-to-one"):
        _loader(frame)._get_subject_frame(301, 0.5)


@pytest.mark.parametrize("invalid_key", [np.nan, np.inf, 1.5])
def test_present_response_keys_must_be_finite_integers(invalid_key):
    frame = _subject_frame()
    frame["presskey"] = frame["presskey"].astype(float)
    frame.loc[3, "presskey"] = invalid_key
    with pytest.raises(ValueError, match="presskey.*finite integer"):
        _loader(frame)._get_subject_frame(301, 0.25)


def test_missing_response_keys_preserve_legacy_data_and_tuple_inputs():
    arrays = _loader(_subject_frame().drop(columns="presskey"))._extract_arrays(
        _subject_frame().drop(columns="presskey"), None
    )
    assert arrays.presskeys is None
    assert arrays.choice_to_presskey is None
    assert arrays.presskey_to_choice is None
    legacy = _coerce_trial_arrays(([[0.1]], [1], [0.5], [2], [[0.1, 0.2, 0.3, 0.4]]))
    np.testing.assert_allclose(legacy.target_probs, [[0.1, 0.2, 0.3, 0.4]])
    assert legacy.presskeys is None


def test_mapping_is_scoped_to_subject_and_keeps_partial_observed_coverage():
    first = _subject_frame().iloc[:2]
    second = _subject_frame().copy()
    second["iSub"] = 302
    second["presskey"] = [1, 2, 3, 4]
    loader = _loader(pd.concat([first, second], ignore_index=True))
    arrays = loader._extract_arrays(loader._get_subject_frame(301, 1.0), None)
    assert arrays.choice_to_presskey == {1: 4, 2: 2}
    assert arrays.presskey_to_choice == {4: 1, 2: 2}

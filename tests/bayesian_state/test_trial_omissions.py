"""Omitted trials are absent from the model history and keep source alignment."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.Bayesian_state.simulation.data import SubjectTrialDataLoader
from src.Bayesian_state.simulation.runner import StateModelSimulationRunner


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "iSub": [337] * 4,
        "condition": [3] * 4,
        "iSession": [1, 1, 2, 2],
        "iRun": [1] * 4,
        "iTrial": [1, 2, 3, 4],
        "feature1": [.1, .2, .3, .4],
        "category": [1, 2, 3, 4],
        "choice": [1, np.nan, 3, 4],
        "presskey": [4, np.nan, 3, 1],
        "feedback": [1, np.nan, 1, 1],
        "miss": [0, 1, 0, 0],
    })


def _loader(frame: pd.DataFrame) -> SubjectTrialDataLoader:
    loader = SubjectTrialDataLoader({"data": {"feature_columns": ["feature1"]}})
    loader.learning_data = frame
    return loader


def test_skip_omissions_preserves_source_rows_and_response_mapping() -> None:
    frame = _frame()
    original = frame.copy(deep=True)
    loader = _loader(frame)
    selected = loader._get_subject_frame(337, 1.)
    arrays = loader._extract_arrays(selected, None)
    np.testing.assert_allclose(arrays.stimulus[:, 0], [.1, .3, .4])
    np.testing.assert_array_equal(arrays.choices, [1, 3, 4])
    np.testing.assert_array_equal(arrays.feedback, [1, 1, 1])
    assert arrays.choice_to_presskey == {1: 4, 3: 3, 4: 1}
    assert selected.iTrial.tolist() == [1, 3, 4]
    assert selected.index.tolist() == [0, 2, 3]
    selection = selected.attrs["trial_selection"]
    assert selection["source_trial_count"] == 4
    assert selection["source_missed_trial_count"] == 1
    assert selection["trial_identifier_columns"] == ["iSession", "iRun", "iTrial"]
    assert selection["retained_trial_identifiers"] == [[1, 1, 1], [2, 1, 3], [2, 1, 4]]
    assert selection["missed_trial_identifiers"] == [[1, 1, 2]]
    pd.testing.assert_frame_equal(frame, original)


def test_skip_omissions_preserves_the_original_stop_at_boundary() -> None:
    loader = _loader(_frame())
    selected = loader._get_subject_frame(337, .5)
    assert selected.iTrial.tolist() == [1]
    arrays = loader._extract_arrays(selected, 1)
    assert arrays.choice_to_presskey == {1: 4, 3: 3, 4: 1}
    np.testing.assert_array_equal(arrays.choices, [1])


def test_max_trials_counts_answered_records() -> None:
    loader = _loader(_frame())
    arrays = loader._extract_arrays(loader._get_subject_frame(337, 1.), 2)
    np.testing.assert_array_equal(arrays.choices, [1, 3])


@pytest.mark.parametrize("miss", [0, None])
def test_unmarked_missing_choices_still_fail(miss) -> None:
    frame = _frame()
    if miss is None:
        frame = frame.drop(columns="miss")
    else:
        frame.loc[1, "miss"] = miss
    with pytest.raises(ValueError, match="choices must be finite integers"):
        _loader(frame)._get_subject_frame(337, 1.)


@pytest.mark.parametrize("flag", [np.nan, -1, 2, .5])
def test_invalid_omission_flags_are_rejected(flag) -> None:
    frame = _frame().astype({"miss": float})
    frame.loc[1, "miss"] = flag
    with pytest.raises(ValueError, match="miss flags"):
        _loader(frame)._get_subject_frame(337, 1.)


def test_all_omitted_sequence_cannot_be_fitted() -> None:
    frame = _frame()
    frame["miss"] = 1
    with pytest.raises(ValueError, match="no answered trials"):
        _loader(frame)._get_subject_frame(337, 1.)


def test_full_v3_skipping_matches_the_answered_only_history(monkeypatch, tmp_path) -> None:
    from src.Bayesian_state.hypothesis_space.similarity import ContinuousSimilarity

    monkeypatch.setattr(ContinuousSimilarity, "RUNTIME_CACHE_DIR", tmp_path / "similarity")
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((
        root / "configs/exp123/model_struct/model_0923_cond3_full_v3.yaml"
    ).read_text())
    config["modules"]["perception_mod"]["kwargs"] = {
        "mean": [0.] * 4, "std": [0.] * 4,
    }
    config["modules"]["hypo_transitions_mod"]["kwargs"]["relation_samples"] = 512
    config["provenance"]["hypothesis_similarity"]["sample_count"] = 512
    config["inference"]["particle_count"] = 4
    frame = _frame()
    for index in range(2, 5):
        frame[f"feature{index}"] = [.2, .3, .4, .5]
    runs = []
    for history in (frame, frame.loc[frame["miss"].eq(0)].drop(columns="miss")):
        runner = StateModelSimulationRunner(config, n_jobs=1)
        runner.learning_data = history
        runs.append(runner.simulate_subject(
            subject_id=337, simulation_repeats=1, window_size=1,
            max_trials=2, keep_logs=True, prediction_mode="prior_t",
            selection_prediction_mode="prior_t", loss_metric="choice_nll",
            hyper_candidate_seed=123, compute_statistics=False,
        ))
    for field in ("pred_category_probs", "observed_choice", "observed_feedback"):
        np.testing.assert_array_equal(
            runs[0]["best"].metrics_by_mode["prior_t"][field],
            runs[1]["best"].metrics_by_mode["prior_t"][field],
        )
    assert runs[0]["best"].mean_error == runs[1]["best"].mean_error
    selection = runs[0]["selection_meta"]["trial_selection"]
    assert selection["modeled_trial_count"] == 2
    assert selection["retained_trial_identifiers"] == [[1, 1, 1], [2, 1, 3]]
    assert selection["missed_trial_identifiers"] == [[1, 1, 2]]
    assert "trial_selection" not in runs[1]["selection_meta"]

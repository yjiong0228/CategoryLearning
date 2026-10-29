"""Exp5 identities, category/key semantics and declared session handling."""
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest
import yaml

from src.Bayesian_state.simulation.exp5_data import load_exp5_subject, FEATURES
from src.Bayesian_state.workflows.runs.prepare_model_1024_exp5 import prepare

CONFIG = Path("configs/exp5/model1024_inputs.yaml")
RAW = Path("data/exp5/raw/Rule1")


def test_full_cohort_preserves_all_behavior_and_only_defers_overnight_rows():
    cases = [load_exp5_subject(CONFIG, subject) for subject in range(501, 520)]
    assert sum(len(case.frame) for case in cases) == 3296
    assert sum(len(case.arrays.choices) for case in cases) == 3232
    for case in cases:
        frame = case.analysis_frame
        np.testing.assert_array_equal(case.arrays.categories, np.where(frame.feature2 == frame.feature4, 1, 2))
        np.testing.assert_array_equal(case.arrays.feedback, case.arrays.choices == case.arrays.categories)
        assert not case.frame.duplicated(["iSession", "iTrial"]).any()
        assert case.arrays.stimulus.shape[1] == 5
        assert case.frame.analysis_included.all() == (case.subject_id != 508)
    case508 = cases[7]
    assert len(case508.frame) == 224 and len(case508.arrays.choices) == 160
    assert case508.frame.iloc[160].session_boundary == "overnight"
    for subject, length in ((515, 192), (517, 448), (518, 192)):
        case = cases[subject-501]
        assert len(case.arrays.choices) == length
        assert case.analysis_frame.iSession.unique().tolist() == [1, 2]
        assert case.frame.loc[case.frame.iSession == 2, "session_boundary"].eq("same_day_continuation").all()


def test_response_mapping_and_sampling_change_are_preserved():
    first, second, eighth = [load_exp5_subject(CONFIG, s) for s in (501, 502, 508)]
    assert first.arrays.choice_to_presskey == {1: 1, 2: 2}
    assert second.arrays.choice_to_presskey == {1: 2, 2: 1}
    assert not any(block["balanced_twice_each"] for block in first.block_audit)
    assert all(block["balanced_twice_each"] for block in eighth.block_audit if block["trials"] == 64)
    assert eighth.block_audit[2]["trials"] == 32


@pytest.fixture
def copied_input(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("Rule1_stimulus.csv", "503_1_bhv.csv"):
        shutil.copyfile(RAW / name, raw / name)
    settings = yaml.safe_load(CONFIG.read_text())
    settings.update(raw_dir="raw", subjects=[503], continued_subjects=[], overnight_subjects=[], oral_extended_subjects=[])
    config = tmp_path / "inputs.yaml"
    config.write_text(yaml.safe_dump(settings))
    return config, raw


@pytest.mark.parametrize("field,value", [("iTrial", 2), ("iSession", 2), ("choice", 1.5),
                                          ("feedback", .5), ("stiID", 99), ("category", 3)])
def test_invalid_identity_response_or_stimulus_is_rejected(copied_input, field, value):
    config, raw = copied_input
    path = raw / "503_1_bhv.csv"
    frame = pd.read_csv(path).astype({field: float})
    frame.loc[0, field] = value
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_exp5_subject(config, 503)


def test_wrong_stimulus_rule_and_key_mapping_are_rejected(copied_input):
    config, raw = copied_input
    path = raw / "Rule1_stimulus.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "category"] = 3 - frame.loc[0, "category"]
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="32 distinct patterns"):
        load_exp5_subject(config, 503)
    shutil.copyfile(RAW / path.name, path)
    behavior = raw / "503_1_bhv.csv"
    frame = pd.read_csv(behavior)
    frame.loc[0, "presskey"] = 3 - frame.loc[0, "presskey"]
    frame.to_csv(behavior, index=False)
    with pytest.raises(ValueError, match="one-to-one"):
        load_exp5_subject(config, 503)


def test_preparation_separates_observations_from_task_targets_and_protects_outputs(copied_input, tmp_path):
    config, _ = copied_input
    output = tmp_path / "prepared"
    result = prepare(config, output)
    assert result["recorded_trials"] == result["analysis_trials"] == 128
    with np.load(output / "503/observations.npz", allow_pickle=False) as observed:
        assert set(observed.files) == {"stimulus", "choices", "feedback"}
        assert observed["stimulus"].shape == (128, 5)
    assert set(FEATURES) <= set(pd.read_csv(output / "behavior.csv"))
    with pytest.raises(FileExistsError):
        prepare(config, output)

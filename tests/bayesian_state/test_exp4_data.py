"""Exp4 response coordinates, stochastic feedback and continuous histories."""
from copy import deepcopy
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest
import yaml

from src.Bayesian_state.simulation.exp4_data import load_exp4_subject
from src.Bayesian_state.workflows.runs.prepare_model_1024_exp4 import prepare

CONFIG = Path("configs/exp4/model1024_inputs.yaml")


@pytest.mark.parametrize("subject,count,sessions,names", [
    (401, 960, [0, 320, 640], ("green", "yellow", "pink", "blue")),
    (402, 320, [0], ("green", "pink", "yellow", "blue")),
    (403, 512, [0, 320], ("yellow", "green", "pink", "blue")),
])
def test_real_inputs_preserve_feedback_feature_coordinates_and_all_sessions(subject, count, sessions, names):
    case = load_exp4_subject(CONFIG, subject)
    assert len(case.observations()) == count
    assert case.feature_names == names
    assert case.arrays.categories is None
    assert [s["start_index"] for s in case.metadata()["sessions"]] == sessions
    raw = pd.read_csv(f"data/exp4/raw/Task2/Task2_{subject}_bhv.csv")
    np.testing.assert_array_equal(case.arrays.choices, raw.choice)
    np.testing.assert_array_equal(case.arrays.feedback, raw.feedback)
    np.testing.assert_array_equal(case.arrays.presskeys, raw.presskey)
    # A higher-probability choice can receive either feedback; no deterministic relabeling.
    optimal = case.arrays.target_probs.argmax(1) + 1
    assert np.any((case.arrays.choices == optimal) & (case.arrays.feedback == 0))
    assert np.any((case.arrays.choices != optimal) & (case.arrays.feedback == 1))
    assert all(len(row) == 3 for row in case.observations())
    assert case.metadata()["processed_probability_max_difference"] <= .0005
    summary = pd.read_csv("data/exp4/processed/Task1b_errorsummary.csv")
    expected = summary[summary.iSub == subject].groupby("feature_name").threshold_mean_mean.mean()
    np.testing.assert_allclose(case.perception_half_range, expected.loc[list(names)], atol=1e-14, rtol=0)
    assert len(case.metadata()["ambiguity_disagreement_trial_ids"]) == (54 if subject == 402 else 0)


@pytest.fixture
def copied_inputs(tmp_path):
    processed, raw = tmp_path / "processed", tmp_path / "raw"
    processed.mkdir(); raw.mkdir()
    for name in ("Task2_processed.csv", "Task1b_errorsummary.csv"):
        shutil.copyfile(Path("data/exp4/processed") / name, processed / name)
    for suffix in ("bhv", "sti"):
        name = f"Task2_401_{suffix}.csv"
        shutil.copyfile(Path("data/exp4/raw/Task2") / name, raw / name)
    config = yaml.safe_load(CONFIG.read_text())
    config["subjects"] = [401]
    config["dataset"]["processed_dir"] = "processed"
    config["raw_task_dir"] = "raw"
    path = tmp_path / "inputs.yaml"
    path.write_text(yaml.safe_dump(config))
    return path, processed, raw


@pytest.mark.parametrize("change,match", [
    ("order", "chronological"), ("duplicate", "Duplicate"),
    ("feedback", "feedback must be binary"), ("probability", "normalized"),
    ("feature_order", "fixed permutation"), ("raw_feedback", "mismatch in feedback"),
    ("keys", "fixed one-to-one"), ("threshold", "complete, finite"),
])
def test_rejects_invalid_or_misaligned_inputs(copied_inputs, change, match):
    config, processed, raw = copied_inputs
    path = processed / "Task2_processed.csv"
    frame = pd.read_csv(path)
    if change == "order":
        frame.iloc[[0, 1]] = frame.iloc[[1, 0]].to_numpy()
    elif change == "duplicate":
        frame.loc[1, "iTrial"] = frame.loc[0, "iTrial"]
    elif change == "feedback":
        frame["feedback"] = frame.feedback.astype(float)
        frame.loc[0, "feedback"] = .5
    elif change == "probability":
        frame.loc[0, "probCat1"] = .9
    elif change == "feature_order":
        frame.loc[320, ["feature1_name", "feature2_name"]] = ["yellow", "green"]
    else:
        path = raw / "Task2_401_bhv.csv" if change != "threshold" else processed / "Task1b_errorsummary.csv"
        frame = pd.read_csv(path)
        if change == "raw_feedback":
            frame.loc[0, "feedback"] = 1 - frame.loc[0, "feedback"]
        elif change == "keys":
            frame.loc[0, "presskey"] = 3 - frame.loc[0, "presskey"]
        else:
            frame.loc[0, "threshold_mean_mean"] = np.nan
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=match):
        load_exp4_subject(config, 401)


def test_environment_probabilities_do_not_enter_learner_observations():
    case = load_exp4_subject(CONFIG, 401)
    changed = deepcopy(case)
    changed.arrays.target_probs = changed.arrays.target_probs[:, ::-1].copy()
    for original, other in zip(case.observations(), changed.observations()):
        np.testing.assert_array_equal(original[0], other[0])
        assert original[1:] == other[1:]


def test_bundle_separates_environment_metadata_and_refuses_overwrite(tmp_path):
    output = tmp_path / "bundle"
    manifest = prepare(CONFIG, output, [401])
    assert manifest["status"] == "inputs_validated"
    assert manifest["model_execution"] == "not_run"
    with np.load(output / "401/observations.npz") as saved:
        assert set(saved.files) == {"stimulus", "choices", "feedback", "presskeys"}
        assert saved["stimulus"].shape == (960, 4)
    with np.load(output / "401/task_metadata.npz") as saved:
        np.testing.assert_array_equal(saved["trial_ids"][[319, 320]], [[1, 5, 320], [2, 1, 1]])
    before = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        prepare(CONFIG, output, [401])
    assert (output / "manifest.json").read_bytes() == before


def test_model1024_continuous_and_session_chunked_histories_agree(monkeypatch, tmp_path):
    from src.Bayesian_state.hypothesis_space.similarity import ContinuousSimilarity
    from src.Bayesian_state.model import StateModel
    from src.Bayesian_state.model.modules.base_module import ModuleRole
    from src.Bayesian_state.model.modules.beta import BetaModule
    from src.Bayesian_state.model.modules.precision_interventions import precision_intervention
    from src.Bayesian_state.model.modules.readout.choice import read_choice_probabilities_from_model
    from tests.bayesian_state.test_model_1023_belief_execution import installed, compact_config

    monkeypatch.setattr(ContinuousSimilarity, "RUNTIME_CACHE_DIR", tmp_path)
    case = load_exp4_subject(CONFIG, 401)
    config = compact_config(1)
    config["modules"]["perception_mod"]["kwargs"] = dict(noise_mode="uniform", module_seed=20261028)
    transition = config["modules"]["hypo_transitions_mod"]["kwargs"]
    transition["execution_inertia"] = 2.
    transition["search_controller"]["baseline_probability"] = .05
    observations = case.observations()
    with installed("STICKY"), precision_intervention(BetaModule, mode="fast", upper=25.):
        model = StateModel(config, context=case.model_context())
        perception = model.engine.get_module(ModuleRole.PERCEPTION)
        np.testing.assert_array_equal(perception.uniform_half_range, case.perception_half_range)
        for stimulus, choice, feedback in observations[:320]:
            model.begin_trial(stimulus)
            model.complete_trial(choice, feedback)
        # A second fixed-seed run consumes session 1 as a chunk and continues
        # on the same instance, retaining both cognition and perception RNG.
        clone = StateModel(config, context=case.model_context())
        clone.fit_step_by_step(observations[:320])
        for stimulus, choice, feedback in observations[320:324]:
            left, right = [m.begin_trial(stimulus) for m in (model, clone)]
            np.testing.assert_array_equal(left.prior, right.prior)
            np.testing.assert_array_equal(left.beta, right.beta)
            np.testing.assert_array_equal(left.perceived_stimulus, right.perceived_stimulus)
            first, second = [read_choice_probabilities_from_model(m, pending.perceived_stimulus)
                             for m, pending in ((model, left), (clone, right))]
            np.testing.assert_array_equal(first, second)
            assert np.isfinite(first).all() and np.all(first >= 0)
            np.testing.assert_allclose(first.sum(), 1., atol=1e-12)
            for m in (model, clone):
                m.complete_trial(choice, feedback)
        for m in (model, clone):
            h = m.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
            assert h.trial_index == 324

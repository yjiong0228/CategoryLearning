"""Task encoding, feedback information and fitting/evaluation boundaries."""

from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from src.SUSTAIN import Parameters
from src.SUSTAIN.__main__ import run
from src.SUSTAIN.benchmark import run_six_problems, six_problems
from src.SUSTAIN.data import encode_features, load_csv_history
from src.SUSTAIN.fitting import fit_parameters


@pytest.fixture
def data_config(tmp_path):
    path = tmp_path / "input.csv"
    pd.DataFrame(dict(iSub=[7]*8, condition=[1]*8, iSession=[1]*8, iTrial=list(range(1,9)),
                      feature=[-1,1]*4, choice=[1,1,2,1,1,2,1,2],
                      feedback=[1,0,0,0,1,1,1,1], category=[1,2]*4)).to_csv(path,index=False)
    return dict(path=str(path), filters={"iSub":7,"condition":1},
                group_columns=["iSub","condition"], session_policy="single",
                order_columns=["iSession","iTrial"],
                features=[dict(column="feature",kind="nominal",values=[-1,1])],
                category_values=[1,2], choice_column="choice",
                feedback=dict(mode="binary_correctness",column="feedback"),
                audit_category_column="category")


def test_binary_feedback_derives_correct_category_without_recoding(data_config):
    data = load_csv_history(data_config)
    np.testing.assert_array_equal(data.targets, [0,1]*4)
    np.testing.assert_array_equal(data.stimuli[:2], [[1,0],[0,1]])
    assert data.dimensions == (2,)
    assert data.frame.sustain_source_row.tolist() == list(range(1,9))


def test_nominal_and_continuous_are_distinct_encodings():
    frame = pd.DataFrame({"x":[-1,1],"y":[-2,2]})
    x, widths = encode_features(frame,[dict(column="x",kind="nominal",values=[-1,1]),
                                       dict(column="y",kind="continuous",bounds=[-2,2])])
    np.testing.assert_array_equal(x, [[1,0,0],[0,1,1]])
    assert widths == (2,1)


def test_hidden_category_column_is_not_a_feedback_fallback(data_config):
    config = deepcopy(data_config)
    config.pop("audit_category_column")
    expected = load_csv_history(config).targets.copy()
    frame = pd.read_csv(config["path"])
    frame["category"] = 999
    frame.to_csv(config["path"],index=False)
    np.testing.assert_array_equal(load_csv_history(config).targets, expected)
    with pytest.raises(ValueError,match="differs"):
        load_csv_history(data_config)


def test_four_category_correctness_is_rejected(data_config):
    data_config["category_values"] = [1,2,3,4]
    with pytest.raises(ValueError,match="two-category"):
        load_csv_history(data_config)


def test_full_label_feedback_supports_four_categories(data_config):
    data_config["category_values"] = [1,2,3,4]
    data_config["feedback"] = dict(mode="category_label",column="category")
    frame = pd.read_csv(data_config["path"])
    frame["category"] = [1,2,3,4]*2
    frame.to_csv(data_config["path"],index=False)
    np.testing.assert_array_equal(load_csv_history(data_config).targets, [0,1,2,3]*2)


@pytest.mark.parametrize("column,value,message", [
    ("choice",np.nan,"Missing or undeclared"),
    ("feedback",.5,"0/1"),
    ("feature",0,"undeclared nominal"),
    ("iTrial",2,"ordered"),
])
def test_bad_rows_are_rejected_without_silent_filtering(data_config,column,value,message):
    frame = pd.read_csv(data_config["path"])
    frame[column] = frame[column].astype(float)
    frame.loc[0,column] = value
    frame.to_csv(data_config["path"],index=False)
    with pytest.raises(ValueError,match=message):
        load_csv_history(data_config)


def test_mixed_subjects_and_unapproved_session_carry_are_rejected(data_config):
    frame = pd.read_csv(data_config["path"])
    frame.loc[7,"iSub"] = 9
    frame.to_csv(data_config["path"],index=False)
    data_config["filters"].pop("iSub")
    with pytest.raises(ValueError,match="single iSub"):
        load_csv_history(data_config)
    frame["iSub"] = 7
    frame.loc[7,"iSession"] = 2
    frame.to_csv(data_config["path"],index=False)
    with pytest.raises(ValueError,match="Multiple sessions"):
        load_csv_history(data_config)
    data_config["session_policy"] = "carry"
    assert len(load_csv_history(data_config).targets) == 8


def test_choices_cannot_be_stimulus_features(data_config):
    data_config["features"] = [dict(column="choice",kind="nominal",values=[1,2])]
    with pytest.raises(ValueError,match="separate"):
        load_csv_history(data_config)


def test_fit_uses_only_training_prefix_and_is_reproducible(data_config):
    data = load_csv_history(data_config)
    arguments = dict(dimensions=data.dimensions,n_categories=2,base=Parameters(),
                     bounds=dict(r=[0,10],beta=[0,10],d=[0,30],eta=[0,1]),
                     train_trials=4,maxiter=1,popsize=2,seed=1)
    a, log_a = fit_parameters(data.stimuli,data.targets,data.choices,**arguments)
    changed_choices, changed_targets = data.choices.copy(), data.targets.copy()
    changed_choices[4:] = 1 - changed_choices[4:]
    changed_targets[4:] = 1 - changed_targets[4:]
    b, log_b = fit_parameters(data.stimuli,changed_targets,changed_choices,**arguments)
    assert a == b and log_a == log_b
    assert log_a["best_nll"] <= log_a["initial_nll"]
    with pytest.raises(ValueError,match="not marginalized"):
        fit_parameters(data.stimuli,data.targets,data.choices,model_options={"ties":"random"},**arguments)


def test_cli_score_generate_fit_and_existing_output_guard(data_config,tmp_path):
    config = dict(schema="sustain_v1",seed=42,trace=True,
                  parameters=dict(r=2,beta=1,d=3,eta=.2,tau=0),
                  data=data_config, fit=dict(train_trials=4,maxiter=0,popsize=2,
                  bounds=dict(r=[0,10],beta=[0,10],d=[0,30],eta=[0,1])))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    score = run(path,tmp_path / "score")
    assert score["evaluation_nll"] is not None
    np.testing.assert_allclose(score["negative_log_likelihood"],score["training_nll"] + score["evaluation_nll"])
    before = (tmp_path / "score/summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        run(path,tmp_path / "score")
    assert before == (tmp_path / "score/summary.json").read_bytes()
    generated = run(path,tmp_path / "generate",simulate=True)
    assert generated["negative_log_likelihood"] is None
    np.testing.assert_allclose(pd.read_csv(tmp_path / "score/trials.csv")[["p_category_1","p_category_2"]],
                               pd.read_csv(tmp_path / "generate/trials.csv")[["p_category_1","p_category_2"]])
    fitted = run(path,tmp_path / "fit",fit=True)
    assert fitted["optimization"]["evaluations"] >= 5
    provenance = json.loads((tmp_path / "fit/provenance.json").read_text())
    assert provenance["data"]["sha256"] and provenance["source_sha256"]


def test_six_types_have_correct_structure_and_reproducible_paths():
    features, labels = six_problems()
    assert features.shape == (8,6) and labels.shape == (6,8)
    assert np.all(labels.sum(axis=1) == 4)
    a = run_six_problems(Parameters(),repetitions=1,blocks=2,seed=9)
    b = run_six_problems(Parameters(),repetitions=1,blocks=2,seed=9)
    assert a == b
    assert len(a["paths"]) == 6
    assert all(len(p["trials"]) == 16 for p in a["paths"])
    assert np.isfinite(a["mean_error_by_block"]).all()

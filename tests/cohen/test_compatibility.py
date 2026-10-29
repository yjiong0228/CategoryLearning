"""Both stimulus domains and their input encodings stay available together."""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from src.Cohen import create_model
from src.Cohen.binary import FeatureMixtureStrict
from src.Cohen.continuous import ContinuousFeatureMixture
from src.Cohen.data import load_csv, prepare_dataframe
from src.Cohen.features import all_binary_patterns
from src.Cohen.feedback import task_feedback


@pytest.mark.parametrize("n_bits", [4, 5])
@pytest.mark.parametrize("feature_encoding", ["pm1", "zero_one"])
@pytest.mark.parametrize("label_encoding,negative", [("category12", 2), ("pm1", -1), ("zero_one", 0)])
def test_equivalent_binary_encodings_preserve_trajectories(tmp_path, n_bits, feature_encoding, label_encoding, negative):
    X = all_binary_patterns(n_bits)
    X = np.concatenate([X, X[:2]])  # Repeated patterns keep the same stimulus ID.
    y = X[:, 2]
    choices = np.roll(y, 1)
    features = X if feature_encoding == "pm1" else (X + 1) / 2
    source = pd.DataFrame(features, columns=[f"feature{i+1}" for i in range(n_bits)])
    source["category"] = np.where(y == 1, 1, negative)
    source["choice"] = np.where(choices == 1, 1, negative)
    path = tmp_path / "binary.csv"
    source.to_csv(path, index=False)
    frame, actual_X, actual_y, actual_r = load_csv(path, n_bits, True,
        feature_encoding=feature_encoding, label_encoding=label_encoding)
    np.testing.assert_array_equal(actual_X, X)
    np.testing.assert_array_equal(actual_y, y)
    np.testing.assert_array_equal(actual_r, choices)
    pd.testing.assert_frame_equal(frame[source.columns], source)
    assert frame.stiID.nunique() == 2 ** n_bits
    assert frame.stiID.iloc[-2:].tolist() == frame.stiID.iloc[:2].tolist()
    expected = FeatureMixtureStrict(n_bits, seed=11).run_session(X, y, choices)
    result = create_model("binary", n_dimensions=n_bits, seed=11).run_session(actual_X, actual_y, actual_r)
    for name in ("p_pos", "log_likelihood", "alpha_pos_before_choice"):
        np.testing.assert_array_equal(result[name], expected[name])


def test_encoding_is_explicit_and_legacy_defaults_remain():
    frame = pd.DataFrame({"feature1": [0], "feature2": [1], "category": [-1], "choice": [1]})
    with pytest.raises(ValueError, match="feature columns"):
        prepare_dataframe(frame, 2, True, label_encoding="pm1")
    with pytest.raises(ValueError, match="categories 1 and 2"):
        prepare_dataframe(frame, 2, True, feature_encoding="zero_one")
    with pytest.raises(ValueError, match="zero_one"):
        prepare_dataframe(frame.assign(feature1=.2), 2, True, feature_encoding="zero_one", label_encoding="pm1")
    # Legacy stiID/category12 inputs still produce the same coding.
    _, X, y, r = prepare_dataframe(pd.DataFrame({"stiID": [1, 4], "category": [1, 2], "choice": [2, 1]}), 2, True)
    np.testing.assert_array_equal(X, [[-1, -1], [1, 1]])
    np.testing.assert_array_equal(y, [1, -1])
    np.testing.assert_array_equal(r, [-1, 1])


@pytest.mark.parametrize("condition", [1, 2, 3])
def test_common_model_factory_and_probability_interface(condition):
    model = create_model("exp123", condition=condition, seed=6)
    k = 2 if condition == 1 else 4
    reference = ContinuousFeatureMixture(n_categories=k, seed=6)
    X = np.array([[.2, .4, .6, .8]])
    np.testing.assert_array_equal(model.predict_proba(X), reference.predict_proba(X))
    binary = create_model("binary", n_dimensions=4, seed=6)
    P = binary.predict_proba([[-1, 1, -1, 1]])
    assert P.shape == (1, 2)
    assert P[0, 0] == binary.prob_pos([[-1, 1, -1, 1]])[0]
    np.testing.assert_allclose(P.sum(axis=1), 1)
    assert binary.update_rule == "si_ratio"
    assert model.update_rule == "exact_gradient"
    from src.Cohen.cohen_schneidman import create_model as compatibility_factory
    assert compatibility_factory is create_model


@pytest.mark.parametrize("task,kwargs", [
    ("binary", {"condition": 1}), ("binary", {"quadrature_order": 4}),
    ("exp123", {}), ("exp123", {"condition": 2, "n_dimensions": 5}),
    ("exp123", {"condition": 1, "memory_window": 2}), ("typo", {}),
])
def test_factory_rejects_incompatible_task_options(task, kwargs):
    with pytest.raises(ValueError):
        create_model(task, **kwargs)


@pytest.mark.parametrize("feature_encoding,label_encoding", [("pm1", "pm1"), ("zero_one", "zero_one")])
def test_unified_binary_cli_reports_original_label_codes(tmp_path, feature_encoding, label_encoding):
    X = all_binary_patterns(4)[:8]
    y = X[:, 2]
    features = X if feature_encoding == "pm1" else (X + 1) / 2
    source = pd.DataFrame(features, columns=[f"feature{i}" for i in range(1, 5)])
    source["category"] = y if label_encoding == "pm1" else (y + 1) / 2
    source["choice"] = source.category
    path, output = tmp_path / "paper.csv", tmp_path / "output"
    source.to_csv(path, index=False)
    proc = subprocess.run([sys.executable, "-m", "src.Cohen", "--task", "binary", "--csv", str(path),
        "--outdir", str(output), "--n-bits", "4", "--max-order", "1",
        "--feature-encoding", feature_encoding, "--label-encoding", label_encoding], text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr
    predictions = pd.read_csv(output / "trial_predictions.csv")
    expected = predictions.model_pred_pm1 if label_encoding == "pm1" else (predictions.model_pred_pm1 + 1) // 2
    np.testing.assert_array_equal(predictions.model_pred_category, expected)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["runs"][0]["label_encoding"] == label_encoding
    assert summary["runs"][0]["task_relevant_feature"] is None


@pytest.mark.parametrize("entry", ["src.Cohen", "src.Cohen.cohen_schneidman", "src.Cohen.exp123"])
def test_both_shared_and_existing_continuous_entries(tmp_path, entry):
    rows = []
    for condition, sid in ((1, 101), (2, 201), (3, 301)):
        for t in range(1, 4):
            category, choice = 1, 2
            rows.append(dict(iSub=sid, condition=condition, iSession=1, iTrial=t, category=category,
                choice=choice, feedback=task_feedback(choice, category, condition),
                feature1=.25, feature2=.35, feature3=.55, feature4=.75))
    data, output = tmp_path / "continuous.csv", tmp_path / "output"
    pd.DataFrame(rows).to_csv(data, index=False)
    config = dict(data=str(data), output_dir=str(output), session_policy="carry", mode="replay",
                  seed=5, model={"quadrature_order": 3})
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    selector = [] if entry == "src.Cohen.exp123" else ["--task", "exp123"]
    proc = subprocess.run([sys.executable, "-m", entry, *selector, "--config", str(config_path)], text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "computationally_complete"
    assert [r["n_categories"] for r in summary["runs"]] == [2, 4, 4]
    assert [r["n_trials"] for r in summary["runs"]] == [3, 3, 3]

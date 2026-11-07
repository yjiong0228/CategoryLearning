"""Standalone paper exports: synthetic bundles, no model execution."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "evaluation/export_paper_tables.py"
KEYS = ("prior", "filtered_prior", "posterior", "predictive_execution_probability",
        "execution_probability", "active_probability")


def exporter():
    assert SCRIPT.exists(), "Standalone exporter has not been implemented"
    spec = importlib.util.spec_from_file_location("paper_tables", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bundle(tmp_path, *, features=True, fourcat=True):
    source = tmp_path / "source"
    source.mkdir()
    ids = np.array([6, 0, 42, 4] if fourcat else [6, 0, 4], dtype=np.int64)
    arrays = dict(valid=np.array([1, 1, 0, 1], dtype=np.uint8),
                  trial_ids=np.array([[2, 9, t] for t in (11, 12, 13, 14)]),
                  session_ids=np.array([9]*4), catalogue_rule_ids=ids)
    for k, key in enumerate(KEYS):
        values = np.full((4, len(ids)), (k+1)/100)
        values[2] = np.nan
        arrays["conditional_catalogue_" + key] = values
    if features:
        sets = np.zeros((4, 16))
        sets[:, 1] = .25
        sets[:, 3] = .75
        sets[2] = np.nan
        arrays["conditional_feature_set_probability"] = sets
        arrays["conditional_feature_probability"] = sets @ ((np.arange(16)[:, None] >> np.arange(4)) & 1)
    manifest = dict(schema_version="gm1031-evaluation-inputs-v1", subject_id=101,
                    condition=2 if fourcat else 1, n_categories=4 if fourcat else 2,
                    trials=4, protocol=dict(feature_order=["tail", "head", "neck", "leg"]),
                    catalogue_metadata=[dict(index=int(i), label=f"H{i}", family="test", program={"id": int(i)}) for i in ids],
                    oral_reports=[dict(text="Tail used", feature1_use="1", feature2_use="0", feature3_use="0", feature4_use="0"),
                                  {}, dict(text="Tail used", feature1_use="1", feature2_use="0", feature3_use="0", feature4_use="0"),
                                  dict(text="Unclear coding", feature1_use="bad", feature2_use="0", feature3_use="0", feature4_use="0")])
    save(source, arrays, manifest)
    return source, arrays, manifest


def save(source, arrays, manifest):
    file = source / "evaluation_inputs.npz"
    np.savez(file, **arrays)
    manifest["inputs_sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
    manifest["arrays"] = {k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in arrays.items()}
    (source / "evaluation_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_tables_preserve_rule_mass_ids_windows_and_oral_missingness(tmp_path):
    source, _, _ = bundle(tmp_path)
    output = tmp_path / "paper"
    exporter().export_bundle(source, output, window=3)
    trials = rows(output / "rule_trials.csv")
    assert len(trials) == 16
    h6 = trials[0]
    assert h6["rule_id"] == "6" and h6["rule_label"] == "H6"
    assert json.loads(h6["trial_id"]) == [2, 9, 11]
    assert float(h6["catalogue_execution_probability"]) == .05
    assert trials[8]["catalogue_execution_probability"] == ""
    windows = rows(output / "rule_windows.csv")
    assert len(windows) == 12
    assert windows[0]["scope"] == "whole_run"
    assert windows[0]["n_valid"] == "3"
    assert windows[4]["n_valid"] == "2"
    assert windows[8]["start_index"] == "3" and windows[8]["stop_index"] == "4"
    features = rows(output / "feature_trials.csv")
    assert float(features[0]["tail_probability"]) == 1
    assert float(features[0]["head_probability"]) == .75
    assert float(features[0]["oral_exact_set_probability"]) == .25
    assert float(features[0]["oral_expected_discrete_dimension_set_jaccard"]) == .625
    for row in features[1:]:
        assert row["oral_exact_set_probability"] == ""
    summary = json.loads((output / "summary.json").read_text())
    assert summary["plotted_rule_ids"] == [0, 4, 6, 42]
    assert summary["oral"]["scored_trials"] == 1
    assert "not geometric region IoU" in summary["oral"]["jaccard_definition"]
    import matplotlib.image as mpimg
    for name in ("rule_probabilities.png", "feature_usage.png"):
        pixels = mpimg.imread(output / name)
        assert pixels.shape[0] > 200 and pixels.std() > .01


@pytest.mark.parametrize("change,match", [
    ("sha", "SHA"), ("spec", "manifest"), ("shape", "shape"),
    ("mass", "sum"), ("marginal", "marginal"), ("ids", "unique"),
    ("order", "feature_order"), ("oral_length", "oral_reports"),
    ("nan", "finite"), ("missing_rule", "catalogue"),
    ("partial_feature", "feature"), ("mask", "valid"),
])
def test_rejects_corrupt_inputs_before_creating_output(tmp_path, change, match):
    source, a, m = bundle(tmp_path)
    if change == "shape": a["conditional_catalogue_prior"] = np.zeros((3, 4))
    if change == "mass": a["conditional_feature_set_probability"][0] *= .5
    if change == "marginal": a["conditional_feature_probability"][0, 0] = 0
    if change == "ids": a["catalogue_rule_ids"][1] = 6
    if change == "order": m["protocol"]["feature_order"] = ["head"]*4
    if change == "oral_length": m["oral_reports"] = [{}]
    if change == "nan": a["conditional_catalogue_prior"][0, 0] = np.nan
    if change == "missing_rule": del a["conditional_catalogue_prior"]
    if change == "partial_feature": del a["conditional_feature_probability"]
    if change == "mask": a["valid"][0] = 2
    save(source, a, m)
    if change == "sha": m["inputs_sha256"] = "0"*64
    if change == "spec": m["arrays"]["valid"]["shape"] = [99]
    (source / "evaluation_manifest.json").write_text(json.dumps(m), encoding="utf-8")
    output = tmp_path / "paper"
    with pytest.raises(ValueError, match=match):
        exporter().export_bundle(source, output)
    assert not output.exists()


def test_no_overwrites_or_nested_outputs(tmp_path):
    source, _, _ = bundle(tmp_path)
    module = exporter()
    with pytest.raises(ValueError, match="separate"):
        module.export_bundle(source, source / "paper")
    out = tmp_path / "existing"
    out.mkdir()
    (out / "keep").write_text("untouched")
    with pytest.raises(FileExistsError):
        module.export_bundle(source, out)
    assert (out / "keep").read_text() == "untouched"
    with pytest.raises(ValueError, match="positive integer"):
        module.export_bundle(source, tmp_path / "bad", window=0)


def test_missing_feature_extension_is_explicit_and_unscored(tmp_path):
    source, _, _ = bundle(tmp_path, features=False, fourcat=False)
    output = tmp_path / "paper"
    exporter().export_bundle(source, output)
    report = json.loads((output / "summary.json").read_text())
    assert report["feature_status"] == "not_available"
    assert report["oral"]["scored_trials"] == 0
    assert report["plotted_rule_ids"] == [0, 4, 6]
    assert rows(output / "feature_trials.csv")[0]["tail_probability"] == ""


def test_empty_set_jaccard_and_cli(tmp_path):
    source, a, m = bundle(tmp_path)
    a["conditional_feature_set_probability"][0] = 0
    a["conditional_feature_set_probability"][0, 0] = 1
    a["conditional_feature_probability"][0] = 0
    m["oral_reports"][0] = {f"feature{i}_use": "0" for i in range(1, 5)}
    m["oral_reports"][0]["text"] = "No dimensions used"
    save(source, a, m)
    assert SCRIPT.exists(), "Standalone exporter has not been implemented"
    output = tmp_path / "cli"
    result = subprocess.run([sys.executable, "-B", str(SCRIPT), "--input", str(source),
                             "--output", str(output), "--window", "2"],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    first = rows(output / "feature_trials.csv")[0]
    assert float(first["oral_exact_set_probability"]) == 1
    assert float(first["oral_expected_discrete_dimension_set_jaccard"]) == 1


@pytest.mark.parametrize("text", [None, "", " \t\n", "missing_key"])
@pytest.mark.parametrize("flag", ["0", "1"])
def test_missing_report_text_excludes_flags_from_scores_and_overlay(tmp_path, text, flag):
    _, a, m = bundle(tmp_path)
    report = {f"feature{i}_use": flag for i in range(1, 5)}
    if text != "missing_key":
        report["text"] = text
    m["oral_reports"][0] = report
    oral, exact, jaccard = exporter()._oral_scores(a, m)
    assert np.isnan(oral[0]).all()
    assert np.isnan(exact[0]) and np.isnan(jaccard[0])


def test_missing_text_zero_flags_export_blank_but_explicit_empty_set_is_scored(tmp_path):
    source, a, m = bundle(tmp_path)
    for t in (0, 1):
        a["conditional_feature_set_probability"][t] = 0
        a["conditional_feature_set_probability"][t, 0] = 1
        a["conditional_feature_probability"][t] = 0
        m["oral_reports"][t] = {f"feature{i}_use": "0" for i in range(1, 5)}
    m["oral_reports"][0]["text"] = None
    m["oral_reports"][1]["text"] = " No dimensions used "
    save(source, a, m)
    output = tmp_path / "paper"
    exporter().export_bundle(source, output)
    missing, explicit = rows(output / "feature_trials.csv")[:2]
    assert missing["oral_report_valid"] == missing["oral_scored"] == "0"
    assert missing["oral_exact_set_probability"] == ""
    assert missing["oral_expected_discrete_dimension_set_jaccard"] == ""
    assert all(missing[f"{part}_oral_use"] == "" for part in m["protocol"]["feature_order"])
    assert explicit["oral_report_valid"] == explicit["oral_scored"] == "1"
    assert float(explicit["oral_exact_set_probability"]) == 1
    assert float(explicit["oral_expected_discrete_dimension_set_jaccard"]) == 1
    summary = json.loads((output / "summary.json").read_text())
    assert summary["oral"]["scored_trials"] == 1


def test_h42_legend_uses_conventional_reference_label(tmp_path, monkeypatch):
    _, a, m = bundle(tmp_path)
    import matplotlib.axes
    labels = []
    original = matplotlib.axes.Axes.plot

    def capture_plot(self, *args, **kwargs):
        labels.append(kwargs.get("label", ""))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", capture_plot)
    output = tmp_path / "plots"
    output.mkdir()
    exporter()._plots(output, a, m["protocol"]["feature_order"], np.full((4, 4), np.nan), [0, 4, 6, 42], True)
    assert "H42 (conventional reference)" in labels
    assert not any("ground truth" in label.lower() for label in labels)


def test_default_windows_missing_reports_and_all_invalid_window(tmp_path):
    source, a, m = bundle(tmp_path)
    m["oral_reports"] = []
    a["valid"][:] = [0, 0, 0, 1]
    for key in a:
        if key.startswith("conditional_"):
            a[key][:3] = np.nan
    save(source, a, m)
    output = tmp_path / "paper"
    exporter().export_bundle(source, output, window=2)
    windows = rows(output / "rule_windows.csv")
    assert windows[4]["n_valid"] == "0"
    assert windows[4]["catalogue_prior"] == ""
    assert all(r["oral_exact_set_probability"] == "" for r in rows(output / "feature_trials.csv"))


def test_default_window_is_32_and_scalar_trial_labels_preserved(tmp_path):
    source, a, m = bundle(tmp_path)
    a["trial_ids"] = np.array([17, 19, 23, 31])
    save(source, a, m)
    output = tmp_path / "paper"
    exporter().export_bundle(source, output)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["window"] == 32
    assert rows(output / "rule_trials.csv")[0]["trial_id"] == "17"
    assert len(rows(output / "rule_windows.csv")) == 8

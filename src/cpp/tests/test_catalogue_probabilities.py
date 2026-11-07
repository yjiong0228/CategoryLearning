"""Offline catalogue validation uses synthetic data only; no native runtime."""
import json

import numpy as np
import pytest

from evaluation.plot_results import evaluate_bundle, validate_inputs
from test_behavior_plots import synthetic_bundle


KEYS = (
    "catalogue_prior", "catalogue_filtered_prior", "catalogue_posterior",
    "catalogue_predictive_execution_probability", "catalogue_execution_probability",
    "catalogue_active_probability",
)
PREFIXES = ("conditional_", "autonomous_diagnostic_")
MATCHES = dict(zip(KEYS[:4], ("prior", "filtered_prior", "posterior", "dominant_probability")))
MATCHES["catalogue_active_probability"] = "active_probability"


def catalogue_bundle(h=7):
    arrays, manifest = synthetic_bundle(1)
    manifest.update(
        parameters={"capacity": 4},
        catalogue_metadata=[dict(index=i, label=f"H{i}", family="test", program={"test_rule": i}) for i in range(h)],
        catalogue_axis=dict(rule_count=h, belief_timing="pre/post feedback", execution_timing="current choice"),
        catalogue_state_keys=list(KEYS),
    )
    manifest["catalogue_axis"].update({key: "saved timing and conditioning" for key in KEYS})
    arrays["catalogue_rule_ids"] = np.arange(h, dtype=np.int64)
    for prefix in PREFIXES:
        for j, key in enumerate(KEYS):
            value = np.tile(np.arange(1, h + 1, dtype=float) / (h * (h + 1)), (4, 1))
            value *= (j + 1) / 6
            if key == "catalogue_active_probability":
                value *= min(3, h)  # Presence can sum above one, each column stays <= 1.
            if prefix == "autonomous_diagnostic_":
                value[:] = 0
                if key in ("catalogue_prior", "catalogue_filtered_prior"):
                    value[:, 0] = .2
                    if h > 1: value[:, 1] = .3
                elif key == "catalogue_posterior":
                    value[:, 0] = .1
                    if h > 1: value[:, 1] = .35
                elif key in ("catalogue_predictive_execution_probability", "catalogue_execution_probability"):
                    value[:, 0] = 1
                else:
                    value[:, :min(2, h)] = 1
            arrays[prefix + key] = value
            if key in MATCHES:
                semantic = np.zeros((4, 4))
                semantic[:, 3] = value.sum(axis=1)
                semantic[:, 1] = 1 if key == "catalogue_active_probability" else 1 - semantic[:, 3]
                arrays[prefix + MATCHES[key]] = semantic
    return arrays, manifest


def save_bundle(path, arrays, manifest):
    path.mkdir()
    np.savez_compressed(path / "evaluation_inputs.npz", **arrays)
    (path / "evaluation_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize("h", [1, 7, 11])
def test_dynamic_axis_preserved_without_normalization(tmp_path, h):
    arrays, manifest = catalogue_bundle(h)
    arrays["unrecognized_future_array"] = np.array([42.])
    source = tmp_path / "source"
    save_bundle(source, arrays, manifest)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    summary = evaluate_bundle(source, tmp_path / "output", plots=False)
    catalogue = summary["catalogue_probabilities"]
    assert catalogue["status"] == "available"
    assert catalogue["rule_count"] == h
    assert catalogue["metadata"] == manifest["catalogue_metadata"]
    assert catalogue["axis"] == manifest["catalogue_axis"]
    assert set(catalogue["definitions"]) == set(KEYS)
    assert "observed" in catalogue["sources"]["conditional_"]
    assert "generated" in catalogue["sources"]["autonomous_diagnostic_"]
    assert "current choice" in catalogue["definitions"]["catalogue_execution_probability"]
    assert "post-feedback" in catalogue["definitions"]["catalogue_posterior"].lower()
    with np.load(tmp_path / "output/arrays.npz", allow_pickle=False) as saved:
        for key in ["catalogue_rule_ids", "unrecognized_future_array", *[p + k for p in PREFIXES for k in KEYS]]:
            np.testing.assert_array_equal(saved[key], arrays[key])
        assert saved["conditional_prior"].shape == (4, 4)
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}
    assert summary["artifacts"] == ["summary.json", "arrays.npz"]


def test_legacy_is_explicitly_unavailable_without_fabricated_arrays(tmp_path):
    arrays, manifest = synthetic_bundle(1)
    save_bundle(tmp_path / "source", arrays, manifest)
    summary = evaluate_bundle(tmp_path / "source", tmp_path / "output", plots=False)
    assert summary["catalogue_probabilities"]["status"] == "not_available"
    assert summary["catalogue_probabilities"]["metadata"] is None
    with np.load(tmp_path / "output/arrays.npz") as saved:
        assert not any("catalogue" in key for key in saved.files)


@pytest.mark.parametrize("missing", ["catalogue_rule_ids", *[p + k for p in PREFIXES for k in KEYS]])
def test_partial_arrays_rejected(missing):
    arrays, manifest = catalogue_bundle()
    del arrays[missing]
    with pytest.raises(ValueError, match="catalogue"):
        validate_inputs(arrays, manifest)


@pytest.mark.parametrize("missing", ["catalogue_metadata", "catalogue_axis", "catalogue_state_keys"])
def test_partial_metadata_rejected(missing):
    arrays, manifest = catalogue_bundle()
    del manifest[missing]
    with pytest.raises(ValueError, match="catalogue"):
        validate_inputs(arrays, manifest)


@pytest.mark.parametrize("bad", ["shape", "dtype", "nan", "inf", "negative", "above_one", "sum", "semantic"])
@pytest.mark.parametrize("prefix", PREFIXES)
@pytest.mark.parametrize("key", KEYS)
def test_invalid_probabilities_rejected(prefix, key, bad):
    arrays, manifest = catalogue_bundle()
    name = prefix + key
    if bad == "shape":
        arrays[name] = arrays[name][:, :4]
    elif bad == "dtype":
        arrays[name] = arrays[name].astype(str)
    elif bad == "sum":
        arrays[name][:] = .9
    elif bad == "semantic":
        if key not in MATCHES:
            pytest.skip("No matching semantic array for this conditioning/quantity")
        arrays[prefix + MATCHES[key]][:, 3] = 0
    else:
        arrays[name][0, 0] = dict(nan=np.nan, inf=np.inf, negative=-.1, above_one=1.1)[bad]
    with pytest.raises(ValueError, match="catalogue"):
        validate_inputs(arrays, manifest)


@pytest.mark.parametrize("bad", ["ids_order", "ids_dtype", "ids_shape", "metadata_count", "index", "label", "family", "program", "axis_count", "keys", "capacity", "empty_family", "bad_program", "missing_timing", "bad_timing", "axis_number"])
def test_invalid_mapping_rejected(bad):
    arrays, manifest = catalogue_bundle()
    if bad == "ids_order":
        arrays["catalogue_rule_ids"] = np.arange(7, dtype=np.int64)[::-1]
    elif bad == "ids_dtype":
        arrays["catalogue_rule_ids"] = np.arange(7, dtype=float)
    elif bad == "ids_shape":
        arrays["catalogue_rule_ids"] = np.arange(7, dtype=np.int64)[None]
    elif bad == "metadata_count":
        manifest["catalogue_metadata"].pop()
    elif bad in ("index", "label"):
        manifest["catalogue_metadata"][0][bad] = 1 if bad == "index" else "H4"
    elif bad in ("family", "program"):
        del manifest["catalogue_metadata"][0][bad]
    elif bad == "axis_count":
        manifest["catalogue_axis"]["rule_count"] = 4
    elif bad == "keys":
        manifest["catalogue_state_keys"] = list(KEYS[:-1])
    elif bad == "empty_family":
        manifest["catalogue_metadata"][0]["family"] = ""
    elif bad == "bad_program":
        manifest["catalogue_metadata"][0]["program"] = "unknown"
    elif bad == "missing_timing":
        del manifest["catalogue_axis"][KEYS[0]]
    elif bad == "bad_timing":
        manifest["catalogue_axis"][KEYS[0]] = None
    elif bad == "axis_number":
        manifest["catalogue_axis"]["axis"] = 0
    else:
        manifest["parameters"]["capacity"] = 0
    with pytest.raises(ValueError, match="catalogue"):
        validate_inputs(arrays, manifest)


def test_skipped_rows_must_be_nan_and_valid_unscored_rows_checked():
    arrays, manifest = catalogue_bundle()
    arrays["valid"][1] = 0
    arrays["autonomous_valid"][:, 1] = 0
    for prefix in PREFIXES:
        for key in KEYS:
            arrays[prefix + key][1] = np.nan
    validate_inputs(arrays, manifest)
    arrays["conditional_catalogue_prior"][1] = 0
    with pytest.raises(ValueError, match="catalogue.*NaN"):
        validate_inputs(arrays, manifest)
    arrays["conditional_catalogue_prior"][1] = np.nan
    arrays["conditional_catalogue_prior"][0, 0] = np.nan  # score_mask[0] is false, valid is true
    with pytest.raises(ValueError, match="catalogue.*finite"):
        validate_inputs(arrays, manifest)


def test_filtered_execution_is_not_compared_with_predictive_semantics():
    arrays, manifest = catalogue_bundle()
    arrays["conditional_catalogue_execution_probability"][:] = 0
    validate_inputs(arrays, manifest)


def test_invalid_bundle_creates_no_output(tmp_path):
    arrays, manifest = catalogue_bundle()
    arrays["conditional_catalogue_prior"][0, 0] = np.inf
    save_bundle(tmp_path / "source", arrays, manifest)
    with pytest.raises(ValueError, match="catalogue"):
        evaluate_bundle(tmp_path / "source", tmp_path / "output", plots=False)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("part", ["metadata", "ids", "array"])
def test_isolated_extension_field_does_not_silently_become_legacy(part):
    arrays, manifest = synthetic_bundle(1)
    if part == "metadata":
        manifest["catalogue_metadata"] = []
    elif part == "ids":
        arrays["catalogue_rule_ids"] = np.arange(7, dtype=np.int64)
    else:
        arrays["conditional_catalogue_prior"] = np.zeros((4, 7))
    with pytest.raises(ValueError, match="catalogue"):
        validate_inputs(arrays, manifest)


def test_fixed_semantic_axis_is_still_four_columns():
    arrays, manifest = catalogue_bundle()
    arrays["conditional_prior"] = np.zeros((4, 7))
    with pytest.raises(ValueError, match="conditional_prior.*shape"):
        validate_inputs(arrays, manifest)


def test_missing_semantic_match_rejected():
    arrays, manifest = catalogue_bundle()
    del arrays["conditional_prior"]
    with pytest.raises(ValueError, match="catalogue_prior.*conditional_prior"):
        validate_inputs(arrays, manifest)


@pytest.mark.parametrize("key", ["catalogue_prior", "catalogue_predictive_execution_probability"])
def test_predictive_mass_cannot_exist_without_presence(key):
    arrays, manifest = catalogue_bundle()
    prefix = "conditional_"
    for name in ("catalogue_prior", "catalogue_predictive_execution_probability", "catalogue_active_probability"):
        arrays[prefix+name][:] = 0
        arrays[prefix+MATCHES[name]][:, 3] = 0
    arrays[prefix+key][:, 0] = .1
    arrays[prefix+MATCHES[key]][:, 3] = .1
    with pytest.raises(ValueError, match="catalogue.*presence"):
        validate_inputs(arrays, manifest)


@pytest.mark.parametrize("pair", ["belief", "execution"])
def test_single_path_predictive_and_filtered_quantities_are_identical(pair):
    arrays, manifest = catalogue_bundle()
    key = "catalogue_filtered_prior" if pair == "belief" else "catalogue_execution_probability"
    arrays["autonomous_diagnostic_"+key][:, 0] += -.1 if pair == "execution" else .01
    if key in MATCHES:
        arrays["autonomous_diagnostic_"+MATCHES[key]][:, 3] = arrays["autonomous_diagnostic_"+key].sum(axis=1)
    with pytest.raises(ValueError, match="catalogue.*single.path"):
        validate_inputs(arrays, manifest)


@pytest.mark.parametrize("key", ["catalogue_active_probability", "catalogue_execution_probability"])
def test_single_path_indicators_are_binary(key):
    arrays, manifest = catalogue_bundle()
    prefix = "autonomous_diagnostic_"
    # Keep predictive/filtered equality and every matching semantic sum intact.
    if key == "catalogue_execution_probability":
        arrays[prefix+"catalogue_predictive_execution_probability"][:, 0] = .5
        arrays[prefix+"dominant_probability"][:, 3] = .5
    else:
        for name in ("catalogue_predictive_execution_probability", "catalogue_execution_probability"):
            arrays[prefix+name][:] = 0
        arrays[prefix+"dominant_probability"][:, 3] = 0
    arrays[prefix+key][:, 0] = .5
    if key in MATCHES:
        arrays[prefix+MATCHES[key]][:, 3] = arrays[prefix+key].sum(axis=1)
    with pytest.raises(ValueError, match="catalogue.*binary"):
        validate_inputs(arrays, manifest)

"""Tiny deterministic tests for scientific result packaging, not model runs."""
import importlib.util
import gzip
from concurrent.futures import Future
from pathlib import Path
import zipfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module():
    path = ROOT / "tools/package_results_1031.py"
    assert path.exists(), "Result packager is not implemented"
    spec = importlib.util.spec_from_file_location("package_results_1031", path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def fixture():
    valid = np.array([1, 1, 0, 1], dtype=np.uint8)
    probabilities = np.tile([.7, .3], (4, 1))
    rules = np.tile([.2, .1, .05], (4, 1))
    sets = np.zeros((4, 16))
    sets[:, [0, 1, 3, 15]] = [.1, .2, .35, .35]
    predictive = np.tile([.05, .5, .15, .3], (4, 1))
    for a in (probabilities, rules, sets, predictive):
        a[2] = np.nan
    arrays = dict(valid=valid, score_mask=np.array([0, 1, 0, 1], np.uint8),
                  observed_choices=np.array([0, 1, -1, 0], np.int64),
                  true_categories=np.array([0, 1, 0, 0], np.int64),
                  observed_feedback=np.array([1., 1., np.nan, 1.]),
                  stimuli=np.full((4, 4), .4), trial_ids=np.arange(12).reshape(4, 3),
                  session_ids=np.ones(4, dtype=np.int64),
                  conditional_category_probabilities=probabilities,
                  conditional_category_probability_repeats=np.stack([probabilities, probabilities]),
                  conditional_catalogue_execution_probability=rules,
                  conditional_feature_set_probability=sets,
                  conditional_choice_weights=predictive,
                  conditional_prior=predictive, conditional_filtered_prior=predictive,
                  pf_seeds=np.array([10, 20], np.int64))
    return arrays


def test_repacked_npz_has_identical_member_bytes_and_preserves_all_dtypes(tmp_path):
    pack = module()
    original, target = tmp_path / "source.npz", tmp_path / "compressed.npz"
    values = dict(x=np.full((100, 100), np.nan), mask=np.array([0, 1], np.uint8),
                  ids=np.array([-1, 6], np.int64), text=np.array("unchanged metadata"))
    np.savez(original, **values)
    result = pack.repack_npz(original, target)
    assert result["member_count"] == 4
    assert target.stat().st_size < original.stat().st_size
    with zipfile.ZipFile(original) as a, zipfile.ZipFile(target) as b:
        assert a.namelist() == b.namelist()
        for name in a.namelist():
            assert a.read(name) == b.read(name)
    with np.load(target, allow_pickle=False) as a:
        for name, reference in values.items():
            np.testing.assert_array_equal(a[name], reference)
    with pytest.raises(FileExistsError):
        pack.repack_npz(original, target)


def test_repacker_rejects_ambiguous_and_non_numpy_members(tmp_path):
    pack = module()
    for name in ("../x.npy", "/x.npy", "x.txt"):
        source = tmp_path / "source.zip"
        with zipfile.ZipFile(source, "w") as z:
            z.writestr(name, b"not a numpy array")
        target = tmp_path / "not_created.npz"
        with pytest.raises(ValueError):
            pack.repack_npz(source, target)
        assert not target.exists()


def test_filtered_execution_is_reconstructed_not_renamed_predictive_execution():
    pack = module()
    arrays = fixture()
    groups = pack.filtered_execution(arrays, np.array([1, 15, 3]))
    np.testing.assert_allclose(groups[[0, 1, 3]], np.tile([.1, .25, .3, .35], (3, 1)), atol=1e-12)
    assert np.isnan(groups[2]).all()
    assert not np.allclose(groups[0], arrays["conditional_choice_weights"][0])
    fit = pack.python_fit_arrays(arrays, np.array([1, 15, 3]))
    np.testing.assert_array_equal(fit["representation_execution"], arrays["conditional_choice_weights"])
    np.testing.assert_allclose(fit["filtered_representation_execution"], groups, equal_nan=True)
    np.testing.assert_array_equal(fit["choices"], [1, 2, 0, 1])
    np.testing.assert_array_equal(fit["categories"], [1, 2, 1, 1])
    assert "ess" not in fit and "resampled" not in fit
    assert fit["choice_probabilities_by_repeat"].shape == (2, 4, 2)


def test_inconsistent_feature_and_rule_mass_is_not_silently_repaired():
    pack = module()
    arrays = fixture()
    arrays["conditional_feature_set_probability"][0, 15] = 0
    with pytest.raises(ValueError):
        pack.filtered_execution(arrays, np.array([1, 15, 3]))


def test_generation_keeps_graded_feedback_separate_from_accuracy():
    pack = module()
    arrays = fixture()
    arrays.update(autonomous_choices=np.array([[0, 0, -1, 0]], np.int64),
                  autonomous_feedback=np.array([[1., .5, np.nan, 1.]]),
                  autonomous_valid=np.array([[1, 1, 0, 1]], np.uint8),
                  autonomous_category_probabilities=np.full((1, 4, 2), .5),
                  simulation_seeds=np.array([30], np.int64))
    generated = pack.python_generation_arrays(arrays)
    np.testing.assert_array_equal(generated["choices"], [[1, 1, 0, 1]])
    assert generated["feedback"][0, 1] == .5
    assert generated["choice_correct"][0, 1] == 0
    assert np.isnan(generated["choice_correct"][0, 2])
    assert "representation_group" not in generated


def test_original_source_csv_without_optional_miss_column():
    pack = module()
    row = dict(iSession="1", iBlock="2", iTrial="3", choice="1", category="1", feedback="1",
               feature1="0.1", feature2="0.2", feature3="0.3", feature4="0.4")
    arrays = dict(trial_ids=np.array([[1, 2, 3]]), stimuli=np.array([[.1, .2, .3, .4]]),
                  valid=np.array([1], np.uint8), observed_choices=np.array([0]),
                  observed_feedback=np.array([1.]), true_categories=np.array([0]))
    assert pack.subject_rows([row], arrays) == [row]


def test_large_tables_are_losslessly_gzipped_not_reformatted(tmp_path):
    pack = module()
    source, target = tmp_path / "table.csv", tmp_path / "table.csv.gz"
    payload = b"\xef\xbb\xbftrial,text\r\n1,exact source bytes\r\n" * 100
    source.write_bytes(payload)
    result = pack.compress_table(source, target)
    assert gzip.decompress(target.read_bytes()) == payload
    assert result["source_bytes"] == len(payload)
    with pytest.raises(FileExistsError):
        pack.compress_table(source, target)


def test_failed_worker_cancels_queued_subjects():
    pack = module()
    failed = Future()
    failed.set_exception(ValueError("failed payload integrity"))
    pending = [Future(), Future()]
    with pytest.raises(ValueError, match="payload integrity"):
        list(pack.completed_results([failed, *pending]))
    assert all(f.cancelled() for f in pending)

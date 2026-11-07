"""Metadata-only transfer checks; no model imports or native execution."""
import csv
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/export_evaluation_configs.py"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def exporter():
    return runpy.run_path(str(SCRIPT))["export_evaluation_configs"]


def save_inputs(evalpath, arrays):
    path = evalpath.with_name("evaluation_inputs.npz")
    np.savez(path, **arrays)
    value = json.loads(evalpath.read_text())
    value["inputs_sha256"] = digest(path)
    value["arrays"] = {k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in arrays.items()}
    write(evalpath, value)


def fixture_inputs(repo, trials=5):
    with (repo.parents[1] / "data/exp123/processed/Task2_processed.csv").open(newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["iSub"] == "101"][:trials]
    valid = np.array([int(r["miss"]) == 0 for r in rows], dtype="uint8")
    score = valid.copy()
    score[np.flatnonzero(score)[0]] = 0
    return dict(trial_ids=np.array([[int(r[k]) for k in ("iSession", "iBlock", "iTrial")] for r in rows], dtype="int64"),
                session_ids=np.array([int(r["iSession"]) for r in rows], dtype="int64"), valid=valid, score_mask=score,
                stimuli=np.array([[float(r[f"feature{k}"]) for k in range(1, 5)] for r in rows], dtype="float64"),
                observed_choices=np.array([int(r["choice"]) - 1 if v else -1 for r, v in zip(rows, valid)], dtype="int64"),
                observed_feedback=np.array([float(r["feedback"]) for r in rows], dtype="float64"),
                true_categories=np.array([int(r["category"]) - 1 for r in rows], dtype="int64"),
                category_to_response=np.tile(np.arange(2, dtype="int64"), (trials, 1)))


def old(tmp_path):
    repo = tmp_path / "repo/src/cpp"
    data = repo.parents[1] / "data/exp123/processed/Task2_processed.csv"
    data.parent.mkdir(parents=True)
    data.write_text("iSub,condition,miss,iSession,iBlock,iTrial,feature1,feature2,feature3,feature4,choice,feedback,category\n"
                    + "".join(f"101,1,{miss},1,1,{t},0.1,0.2,0.3,0.4,1,{feedback},1\n"
                              for t, (miss, feedback) in enumerate([(0, 1), (1, "nan"), (0, 1), (0, 1), (1, "nan")], 1))
                    + "201,2,0,1,1,1,0.1,0.2,0.3,0.4,1,1,1\n", encoding="utf-8")
    write(repo / "resources/data_sources.json", {"files": [{"path": data.relative_to(repo.parents[1]).as_posix(), "sha256": digest(data)}]})
    write(repo / "configs/parameter_space.yaml", {})
    write(repo / "resources/catalogue_c1.json", {})
    resource = repo / "resources/catalogue_c1.npz"
    resource.write_bytes(b"metadata test only")
    fingerprint = hashlib.sha256((digest(resource.with_suffix(".json")) + digest(resource)).encode()).hexdigest()
    params = dict(capacity=3, similarity_width=.25, baseline_probability=.5, error_gain=0.,
                  failure_decay=0., global_search=.5, retention=1., execution_inertia=6., beta_init=12., learning_rate=.01)
    provenance = dict(model_version="1031-unified-v2:cpp-v1", scientific_fingerprint="a" * 64,
                      catalogue_fingerprint=fingerprint, source_data_scope="exp123", source_manifest="data/source_1031.json")
    common = dict(subject_id=101, condition=1, data_fingerprint="b" * 64,
                  resource_fingerprint=fingerprint, provenance=provenance)
    fit = dict(common, best_parameters=params, model_version="a" * 64)
    evaluation = dict(common, parameters=params, scientific_model_contract="1031-unified-v2", trials=5, n_categories=2,
                      protocol={"dataset": "exp123"}, config=dict(seed=1234, jobs=12, particle_count=8,
                      pf_repeats=2, simulation_repeats=4, band_draws=32, window=8, plots=True, keep_states=True))
    batch = tmp_path / "batch"
    fitpath = batch / "cond1/subject_101/fit/subject_101/best_parameters.json"
    evalpath = batch / "cond1/subject_101/evaluate/subject_101/evaluation_manifest.json"
    write(fitpath, fit)
    write(evalpath, evaluation)
    save_inputs(evalpath, fixture_inputs(repo))
    return repo, batch, fitpath, evalpath, tmp_path / "export"


def check_preserves_params_budget_paths_and_transfer_record(old):
    repo, batch, fitpath, evalpath, output = old
    exporter()(batch, output, jobs=2, repo=repo)
    cfg = json.loads((output / "subject_101.json").read_text())
    saved = json.loads(evalpath.read_text())
    assert cfg["parameters"] == json.loads(fitpath.read_text())["best_parameters"]
    assert cfg["seed"] == saved["config"]["seed"] and cfg["jobs"] == 2
    assert cfg["evaluation"] == {k: v for k, v in saved["config"].items() if k not in ("seed", "jobs")}
    assert cfg["subjects"] == [101] and cfg["dataset"] == "exp123"
    assert "max_trials" not in cfg and "fit" not in cfg
    for key, target in [("data_root", "../../data"), ("resource_root", "resources"), ("parameter_space", "configs/parameter_space.yaml")]:
        assert (output / cfg[key]).resolve() == (repo / target).resolve()
    source = json.loads((output / "source.json").read_text())
    entry = source["exports"][0]
    assert entry["fit_sha256"] == digest(fitpath) and entry["evaluation_sha256"] == digest(evalpath)
    assert "--fit-dir" not in entry["command"]
    assert "NEWRESULTROOT/subject_101" in entry["command"]


def check_subset_uses_valid_prefix_not_row_count(old):
    repo, batch, _, evalpath, output = old
    value = json.loads(evalpath.read_text())
    value["trials"] = 3
    write(evalpath, value)
    save_inputs(evalpath, fixture_inputs(repo, 3))
    exporter()(batch, output, repo=repo)
    assert json.loads((output / "subject_101.json").read_text())["max_trials"] == 2


def check_invalid_inputs_fail_before_write(old, change):
    repo, batch, fitpath, evalpath, output = old
    value = json.loads(evalpath.read_text())
    mutations = {"subject": ("subject_id", 201), "condition": ("condition", 2),
                 "parameters": ("parameters", {}), "resources": ("resource_fingerprint", "wrong"),
                 "data": ("data_fingerprint", "wrong"), "contract": ("scientific_model_contract", "foreign"),
                 "version": ("provenance", {"model_version": "foreign"}), "budget": ("config", {"seed": 1, "unknown": 5}),
                 "trials": ("trials", 2)}
    if change in mutations:
        key, replacement = mutations[change]
        value[key] = replacement
        write(evalpath, value)
    elif change == "missing_fit":
        fitpath.unlink()
    elif change == "missing_eval":
        evalpath.unlink()
    elif change == "source_hash":
        (repo.parents[1] / "data/exp123/processed/Task2_processed.csv").write_text("changed")
    elif change == "current_resource":
        (repo / "resources/catalogue_c1.npz").write_bytes(b"changed")
    elif change == "existing":
        output.mkdir()
    with unittest.TestCase().assertRaises((ValueError, OSError, KeyError)):
        exporter()(batch, output, subjects=[999] if change == "unknown_subject" else None,
                   jobs=13 if change == "jobs" else 12, repo=repo)
    assert not output.exists() or (change == "existing" and not list(output.iterdir()))


def check_one_actual_subject_readonly(tmp_path):
    batch = ROOT / "results/local_cond123_20261103_121458_843"
    if not batch.exists():
        raise unittest.SkipTest("completed local batch unavailable")
    output = tmp_path / "actual"
    result = subprocess.run([sys.executable, str(SCRIPT), "--input", str(batch), "--output", str(output),
                             "--subjects", "101", "--jobs", "2"], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    cfg = json.loads((output / "subject_101.json").read_text())
    assert cfg["subjects"] == [101] and "max_trials" not in cfg


class ExportTests(unittest.TestCase):
    def test_git_line_endings_are_explicitly_allowed(self):
        repo, batch, _, _, output = old(self.path)
        manifest_path = repo / "resources/data_sources.json"
        manifest = json.loads(manifest_path.read_text())
        data = repo.parents[1] / "data/exp123/processed/Task2_processed.csv"
        data.write_bytes(data.read_bytes().replace(b"\r\n", b"\n"))
        manifest["files"][0]["git_sha256"] = digest(data)
        write(manifest_path, manifest)
        exporter()(batch, output, repo=repo)
        self.assertTrue((output / "subject_101.json").is_file())

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)

    def test_preserves_params_budget_paths_and_transfer_record(self):
        check_preserves_params_budget_paths_and_transfer_record(old(self.path))

    def test_preserves_optional_particle_export_from_server_evaluation(self):
        repo, batch, _, evalpath, output = old(self.path)
        value = json.loads(evalpath.read_text())
        value["config"].update(export_particles=True, jobs=64)
        write(evalpath, value)
        exporter()(batch, output, jobs=2, repo=repo)
        config = json.loads((output / "subject_101.json").read_text())
        self.assertIs(config["evaluation"]["export_particles"], True)

    def test_rejects_nonboolean_particle_export(self):
        repo, batch, _, evalpath, output = old(self.path)
        value = json.loads(evalpath.read_text())
        value["config"]["export_particles"] = 1
        write(evalpath, value)
        with self.assertRaisesRegex(ValueError, "budget"):
            exporter()(batch, output, repo=repo)
        self.assertFalse(output.exists())

    def test_subset_uses_valid_prefix_not_row_count(self):
        check_subset_uses_valid_prefix_not_row_count(old(self.path))

    def test_one_actual_subject_readonly(self):
        check_one_actual_subject_readonly(self.path)

    def test_later_missing_subject_writes_nothing(self):
        repo, batch, _, _, output = old(self.path)
        (batch / "cond2/subject_201").mkdir(parents=True)
        with self.assertRaises(OSError):
            exporter()(batch, output, repo=repo)
        self.assertFalse(output.exists())

    def test_selected_subject_ignores_unselected_missing_metadata(self):
        repo, batch, _, _, output = old(self.path)
        (batch / "cond2/subject_201").mkdir(parents=True)
        exporter()(batch, output, subjects=[101], repo=repo)
        self.assertEqual(sorted(p.name for p in output.iterdir()), ["source.json", "subject_101.json"])

    def test_tampered_valid_trial_count_fails_before_write(self):
        repo, batch, _, evalpath, output = old(self.path)
        value = json.loads(evalpath.read_text())
        value["trials"] = 1
        write(evalpath, value)
        with self.assertRaisesRegex(ValueError, "shape|trial"):
            exporter()(batch, output, repo=repo)
        self.assertFalse(output.exists())

    def test_missing_inputs_fails_before_write(self):
        repo, batch, _, evalpath, output = old(self.path)
        evalpath.with_name("evaluation_inputs.npz").unlink()
        with self.assertRaises(OSError):
            exporter()(batch, output, repo=repo)
        self.assertFalse(output.exists())

    def test_inputs_checksum_fails_before_write(self):
        repo, batch, _, evalpath, output = old(self.path)
        with evalpath.with_name("evaluation_inputs.npz").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "hash|checksum"):
            exporter()(batch, output, repo=repo)
        self.assertFalse(output.exists())

    def test_input_values_schema_and_metadata_fail_before_write(self):
        for change in ("trial_ids", "session_ids", "valid", "score_mask", "stimuli", "observed_choices",
                       "observed_feedback", "true_categories", "category_to_response", "dtype", "shape", "metadata"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                repo, batch, _, evalpath, output = old(Path(temp))
                arrays = fixture_inputs(repo)
                if change == "dtype":
                    arrays["stimuli"] = arrays["stimuli"].astype("float32")
                elif change == "shape":
                    arrays["stimuli"] = arrays["stimuli"][:, :3]
                elif change != "metadata":
                    arrays[change].flat[0] += 1
                save_inputs(evalpath, arrays)  # Updated checksum must not hide changed scientific inputs.
                if change == "metadata":
                    value = json.loads(evalpath.read_text())
                    value["arrays"]["stimuli"]["dtype"] = "float32"
                    write(evalpath, value)
                with self.assertRaisesRegex(ValueError, "input|shape|dtype|CSV"):
                    exporter()(batch, output, repo=repo)
                self.assertFalse(output.exists())


def invalid_test(change):
    def test(self):
        check_invalid_inputs_fail_before_write(old(self.path), change)
    return test


for change in ("subject", "condition", "parameters", "resources", "data", "contract", "version", "budget",
               "trials", "missing_fit", "missing_eval", "source_hash", "current_resource", "unknown_subject", "existing", "jobs"):
    setattr(ExportTests, "test_reject_" + change, invalid_test(change))


if __name__ == "__main__":
    unittest.main()

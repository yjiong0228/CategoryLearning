"""Bounded integration checks against the actual native executable."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / ("build/Release/genmodel_cpp.exe" if os.name == "nt" else "build/genmodel_cpp")


@pytest.fixture
def cli():
    if not EXE.is_file():
        pytest.skip("Build build/Release/genmodel_cpp.exe to run native integration checks")

    def run(*args, ok=True):
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        result = subprocess.run([str(EXE), *map(str, args)], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=180)
        if ok:
            assert result.returncode == 0, result.stdout + result.stderr
        else:
            assert result.returncode != 0
        return result
    return run


@pytest.fixture
def config(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs/cond1.yaml").read_text(encoding="utf-8"))
    cfg.update(subjects=[122], max_trials=8, jobs=1,
               data_root=str(ROOT.parents[1] / "data"), resource_root=str(ROOT / "resources"),
               parameter_space=str(ROOT / "configs/parameter_space.yaml"))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_help(cli):
    assert "evaluate" in cli("--help").stdout


def test_fit_evaluate_plot_and_raw_filter(cli, config, tmp_path):
    fit, evaluation = tmp_path / "fit", tmp_path / "evaluate"
    cli("fit", "--config", config, "--output", fit)
    cli("evaluate", "--config", config, "--fit-dir", fit, "--output", evaluation)
    subject = evaluation / "subject_122"
    manifest = json.loads((subject / "evaluation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "gm1031-evaluation-inputs-v1"
    assert manifest["scientific_model_contract"] == "1031-unified-v2"
    assert manifest["semantic_axis"]["kinds"] == ["uncommitted", "example", "generalization", "catalogue_rule"]
    assert len(manifest["state_metadata"]) == 4
    assert len(manifest["catalogue_metadata"]) == 29
    assert manifest["catalogue_axis"]["rule_count"] == 29
    assert manifest["catalogue_metadata"][6]["label"] == "H6"
    content = json.loads((subject / "content_path.json").read_text(encoding="utf-8"))
    assert isinstance(content, list) and content
    assert {"trial_index", "executed_uid", "contents_pre", "contents_post"} <= content[0].keys()
    result = subprocess.run([sys.executable, "-B", "evaluation/plot_results.py", "--input", str(subject)],
                            cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((subject / "evaluation/summary.json").read_text(encoding="utf-8"))
    assert summary["oral"]["status"] == "not_implemented_1031"
    for name in ("behavior", "conditional_states", "autonomous_states"):
        assert (subject / f"evaluation/{name}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    raw = tmp_path / "raw"
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    cfg["seed"] = manifest["pf_seeds"][0]
    cfg["evaluation"]["keep_states"] = True
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    cli("filter", "--config", config, "--fit-dir", fit, "--output", raw)
    with np.load(raw / "subject_122/filter.npz") as values, np.load(subject / "evaluation_inputs.npz") as evaluated:
        assert values["category_probabilities"].shape == (8, 2)
        np.testing.assert_allclose(values["category_probabilities"].sum(axis=1), 1, atol=1e-7)
        np.testing.assert_array_equal(values["category_probabilities"],
                                      evaluated["conditional_category_probability_repeats"][0])
        assert values["prior"].shape == (8, 4)
        assert np.all(values["active_probability"] >= 0)
        for key in manifest["catalogue_state_keys"]:
            assert values[key].shape == (8, 29)
        score = evaluated["conditional_score_mask"].astype(bool)
        expected_nll = -np.log(evaluated["conditional_choice_probabilities"][score]).mean()
        assert summary["conditional"]["mean_choice_nll"] == pytest.approx(expected_nll)
        first_beta = values["beta"].copy()
        first_count = values["active_probability"].copy()
        first_rules = {key: values[key].copy() for key in manifest["catalogue_state_keys"]}
    cfg["seed"] = manifest["pf_seeds"][1]
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    other = tmp_path / "other_raw"
    cli("filter", "--config", config, "--fit-dir", fit, "--output", other)
    with np.load(other / "subject_122/filter.npz") as values, np.load(subject / "evaluation_inputs.npz") as evaluated:
        denominator = first_count + values["active_probability"]
        numerator = first_beta * first_count + values["beta"] * values["active_probability"]
        expected_beta = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
        np.testing.assert_allclose(evaluated["conditional_beta"], expected_beta, atol=1e-12)
        for key in manifest["catalogue_state_keys"]:
            np.testing.assert_allclose(evaluated["conditional_" + key],
                (first_rules[key]+values[key])/2, atol=1e-12)
        np.testing.assert_array_equal(evaluated["catalogue_rule_ids"], np.arange(29))
    # Offline evaluation must preserve the new rule axis rather than treating it
    # as four semantic-kind columns or discarding it from arrays.npz.
    with np.load(subject / "evaluation/arrays.npz") as plotted, np.load(subject / "evaluation_inputs.npz") as raw_values:
        for key in manifest["catalogue_state_keys"]:
            np.testing.assert_array_equal(plotted["conditional_"+key], raw_values["conditional_"+key])
    # Task order and worker count cannot change seeded forecasts or marginals.
    cfg["seed"] = manifest["config"]["seed"]
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    parallel = tmp_path / "parallel"
    cli("evaluate", "--config", config, "--fit-dir", fit, "--output", parallel, "--jobs", 2)
    with np.load(subject / "evaluation_inputs.npz") as serial, np.load(parallel / "subject_122/evaluation_inputs.npz") as multi:
        assert set(serial.files) == set(multi.files)
        for key in serial.files:
            np.testing.assert_array_equal(serial[key], multi[key])
    # Existing results must never be overwritten.
    cli("evaluate", "--config", config, "--fit-dir", fit, "--output", evaluation, ok=False)


@pytest.mark.parametrize("condition,subject", [(2, 206), (3, 314)])
def test_other_condition_semantic_states(cli, config, tmp_path, condition, subject):
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    cfg.update(subjects=[subject], max_trials=4)
    cfg["evaluation"].update(particle_count=4, pf_repeats=1, simulation_repeats=2)
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    output = tmp_path / "evaluate"
    cli("evaluate", "--config", config, "--output", output)
    bundle = output / f"subject_{subject}"
    with np.load(bundle / "evaluation_inputs.npz") as values:
        assert values["conditional_prior"].shape == (4, 4)
        assert values["conditional_catalogue_execution_probability"].shape == (4, 152)
        assert values["autonomous_diagnostic_catalogue_prior"].shape == (4, 152)
        np.testing.assert_allclose(values["conditional_catalogue_prior"].sum(axis=1),
                                  values["conditional_prior"][:, 3], atol=1e-12)
        np.testing.assert_allclose(values["conditional_prior"].sum(axis=1), 1, atol=1e-7)
        if condition == 3:
            assert values["conditional_pairing"].shape == (4, 4, 3)
            np.testing.assert_allclose(values["conditional_pairing"].sum(axis=(1, 2)), 1, atol=1e-7)
    result = subprocess.run([sys.executable, "-B", "evaluation/plot_results.py", "--input", str(bundle)],
                            cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("stale", [{"manual_eps": 0.1}, {"parameters": {"epsilon": 0.1}},
                                  {"model": "1031-prototype"}])
def test_reject_stale_config(cli, config, tmp_path, stale):
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    cfg.update(stale)
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    cli("filter", "--config", config, "--output", tmp_path / "bad", ok=False)

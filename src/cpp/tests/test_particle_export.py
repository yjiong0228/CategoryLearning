"""Conditional content archives must describe observed history before feedback."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / ("build/Release/genmodel_cpp.exe" if os.name == "nt" else "build/genmodel_cpp")
pytestmark = pytest.mark.skipif(not EXE.is_file(), reason="Build the C++ executable first")


@pytest.mark.parametrize("subject,categories,groups", [(122, 2, 1), (206, 4, 1), (314, 4, 3)])
def test_observed_history_particle_archive(tmp_path, subject, categories, groups):
    cfg = yaml.safe_load((ROOT / "configs/cond1.yaml").read_text(encoding="utf-8"))
    cfg.update(subjects=[subject], max_trials=12, jobs=1,
               data_root=str(ROOT.parents[1] / "data"), resource_root=str(ROOT / "resources"),
               parameter_space=str(ROOT / "configs/parameter_space.yaml"))
    cfg["evaluation"].update(particle_count=8, pf_repeats=1,
                             simulation_repeats=2, export_particles=True)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    output = tmp_path / "results"
    result = subprocess.run([str(EXE), "evaluate", "--config", str(path),
                             "--output", str(output)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    bundle = output / f"subject_{subject}"
    manifest = json.loads((bundle / "evaluation_manifest.json").read_text(encoding="utf-8"))
    packet = manifest["conditional_particle_contents"]
    assert packet["phase"] == "current_choice_filtered_before_feedback_before_resampling"
    assert packet["seed"] == manifest["pf_seeds"][0]
    file = bundle / packet["file"]
    assert hashlib.sha256(file.read_bytes()).hexdigest() == packet["sha256"]
    with np.load(file) as p, np.load(bundle / "evaluation_inputs.npz") as a:
        k = manifest["parameters"]["capacity"]
        assert p["particle_weights"].shape == (12, 8)
        assert p["particle_template_anchor"].shape == (12, 8, k, 4)
        assert p["particle_template_category_belief"].shape == (12, 8, k, groups, categories)
        assert p["particle_pairing"].shape == (12, 8, k, groups)
        active = p["particle_active"].astype(bool)
        assert active.shape == (12, 8, k)
        np.testing.assert_allclose(p["particle_weights"].sum(axis=1), 1)
        np.testing.assert_allclose(p["particle_predictive_weights"].sum(axis=1), 1)
        np.testing.assert_allclose(np.where(active, p["particle_belief"], 0).sum(axis=2), 1)
        np.testing.assert_allclose(p["particle_selection_probabilities"].sum(axis=2), 1)
        np.testing.assert_allclose(p["particle_choice_weights"].sum(axis=2), 1)
        np.testing.assert_array_equal(p["observed_choices"], a["observed_choices"])
        np.testing.assert_array_equal(p["trial_ids"], a["trial_ids"])
        np.testing.assert_array_equal(p["stimuli"], a["stimuli"])
        # Bayes weighting uses this choice, but not this trial's feedback.
        emissions = p["particle_rule_choice_probabilities"]
        per_particle = np.sum(np.where(active[..., None], emissions, 0)
                              * p["particle_choice_weights"][..., None], axis=2)
        forecast = np.einsum("tp,tpc->tc", p["particle_predictive_weights"], per_particle)
        np.testing.assert_allclose(forecast, a["conditional_category_probabilities"], atol=1e-12)
        likelihood = np.take_along_axis(per_particle, a["observed_choices"][:, None, None], axis=2)[..., 0]
        filtered = p["particle_predictive_weights"] * likelihood
        filtered /= filtered.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(filtered, p["particle_weights"], atol=1e-12)
        features = np.zeros((12, 16))
        for t in range(12):
            for particle in range(8):
                for slot in range(k):
                    if active[t, particle, slot]:
                        code = p["particle_feature_mask"][t, particle, slot]
                        features[t, code] += p["particle_weights"][t, particle] * p["particle_choice_weights"][t, particle, slot]
        np.testing.assert_allclose(features, a["conditional_feature_set_probability"], atol=1e-12)
        np.testing.assert_allclose(features.sum(axis=1), 1)
        bits = ((np.arange(16)[:, None] >> np.arange(4)) & 1)
        np.testing.assert_allclose(features @ bits, a["conditional_feature_probability"], atol=1e-12)
        for kind in range(4):
            reconstructed = (p["particle_weights"][..., None] * np.where(active, p["particle_belief"], 0)
                             * (p["particle_semantic_kind"] == kind)).sum(axis=(1, 2))
            np.testing.assert_allclose(reconstructed, a["conditional_filtered_prior"][:, kind], atol=1e-12)
        for h in (0, 4, 6):
            mass = (p["particle_weights"][..., None] * p["particle_choice_weights"]
                    * active * (p["particle_catalogue_rule"] == h)).sum(axis=(1, 2))
            np.testing.assert_allclose(mass, a["conditional_catalogue_execution_probability"][:, h], atol=1e-12)

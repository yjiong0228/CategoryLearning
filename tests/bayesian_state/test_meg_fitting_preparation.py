"""Scientific checks for the MEG fit plan and preparation artifact validator."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from src.Bayesian_state.model.config import ModelConfig
from src.Bayesian_state.optimization.search.coordinate_descent import HyperCDOptimizer
from src.Bayesian_state.simulation.config import load_yaml, resolve_engine_config
from src.Bayesian_state.workflows.runs.prepare_model_0923_meg import audit_simulation, _smoke_space


ROOT = Path(__file__).resolve().parents[2]


def test_meg_search_preserves_full_model_and_varying_behavior_coordinates(tmp_path):
    path = ROOT / "configs/meg/hyper_cd_cfg/model0923_task3b.yaml"
    config = load_yaml(path)
    config["output_dir"] = str(tmp_path / "search")
    optimizer = HyperCDOptimizer(config, path)
    base = optimizer.base_sim_config
    engine = resolve_engine_config(base, optimizer.base_sim_config_path.parent)
    space, anchor = _smoke_space(config, engine)
    assert len(anchor) == 13
    assert engine["provenance"]["specification_version"] == "0923-full-v3"
    assert engine["modules"]["perception_mod"]["kwargs"]["noise_mode"] == "uniform"
    assert not any("perception_mod" in key or "oral" in key or "rating" in key for key in space)
    for stage, particle_count in (("coarse", 32), ("fine", 64)):
        stage_config = optimizer._prepare_stage_config(stage)
        stage_engine = resolve_engine_config(stage_config, optimizer.base_sim_config_path.parent)
        assert stage_engine["inference"]["particle_count"] == particle_count
        assert stage_config["max_trials"] is None
        assert stage_config["evaluation_protocol"]["mode"] == "all"
        assert stage_config["repeat_aggregation"] == "mean_probability"
        specs = optimizer._param_specs_for_stage(stage)
        assert set(specs) == set(space)
        # All lower and all upper supports must keep the same scientific model.
        for index in (0, -1):
            point = {key: spec["values"][index] for key, spec in specs.items()}
            _, candidate = optimizer._apply_hyperparams(point, stage_config, stage_engine)
            ModelConfig.from_mapping(candidate)
            assert candidate["likelihood"]["feedback_likelihood_mode"] == "hierarchical_pairing"
            assert candidate["modules"]["memory_mod"]["kwargs"]["forgetting_mode"] == "factorized"


def _payload():
    identifiers = [[1, 1, 1], [1, 1, 3], [2, 1, 4]]
    metrics = {
        "pred_category_probs": [[.25] * 4, [.6, .1, .1, .2], [.1, .2, .3, .4]],
        "valid_trial_mask": [False, True, True],
        "observed_choice": [2, 1, 4],
    }
    state = {key: [[.4, .6]] * 3 for key in (
        "marginal_prior", "choice_weights", "dominant_probability",
        "filtered_choice_weights", "filtered_dominant_probability",
        "pairing_prior", "pairing_posterior")}
    return {
        "representative_run": {"metrics_by_mode": {"prior_t": metrics}, "state_log": state},
        "simulation": {"mean_error": -np.log(.6 * .4) / 2},
        "selection": {"selection_meta": {"trial_selection": {
            "retained_trial_identifiers": identifiers}}},
    }, identifiers


def test_preparation_audit_checks_choice_nll_and_omission_alignment():
    payload, identifiers = _payload()
    result = audit_simulation(payload, identifiers)
    assert result["modeled_trials"] == 3
    assert result["scored_trials"] == 2
    np.testing.assert_allclose(result["total_choice_nll"], -np.log(.6) - np.log(.4))


@pytest.mark.parametrize("corruption", ["nll", "initialization_mask", "prediction", "pairing", "alignment"])
def test_preparation_audit_rejects_invalid_results(corruption):
    payload, identifiers = _payload()
    corrupted = deepcopy(payload)
    if corruption == "nll":
        corrupted["simulation"]["mean_error"] += .1
    elif corruption == "initialization_mask":
        corrupted["representative_run"]["metrics_by_mode"]["prior_t"]["valid_trial_mask"][0] = True
    elif corruption == "prediction":
        corrupted["representative_run"]["metrics_by_mode"]["prior_t"]["pred_category_probs"][1][0] = .9
    elif corruption == "pairing":
        corrupted["representative_run"]["state_log"]["pairing_posterior"][1] = [float("nan"), .6]
    else:
        corrupted["selection"]["selection_meta"]["trial_selection"]["retained_trial_identifiers"][1][2] = 2
    with pytest.raises((ValueError, AssertionError)):
        audit_simulation(corrupted, identifiers)

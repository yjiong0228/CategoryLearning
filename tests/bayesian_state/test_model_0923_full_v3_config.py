"""Keep the revised 0923 scientific contract explicit and legacy seeds stable."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.Bayesian_state.hypothesis_space.similarity import ContinuousSimilarity
from src.Bayesian_state.model import ModelContext, StateModel
from src.Bayesian_state.model.config import ModelConfig
from src.Bayesian_state.model.modules.base_module import ModuleRole
from src.Bayesian_state.simulation.parameters import (
    apply_fixed_hyperparams_to_engine_config,
    infer_fixed_hyperparams_from_engine_config,
)
from src.Bayesian_state.simulation.provenance import build_model_provenance


ROOT = Path(__file__).resolve().parents[2]
NEW_PARAMETER_PATHS = (
    "engine.modules.beta_mod.kwargs.learning_rate",
    "engine.modules.memory_mod.kwargs.forgetting_mode",
    "engine.modules.hypo_transitions_mod.kwargs.mechanism_version",
)


def _config(condition=1, *, revised=True):
    suffix = "full_v3" if revised else "full"
    return yaml.safe_load((ROOT / f"configs/exp123/model_struct/model_0923_cond{condition}_{suffix}.yaml").read_text())


@pytest.mark.parametrize("condition", [1, 2, 3])
def test_full_v3_profiles_validate_without_changing_their_declared_settings(condition):
    config = _config(condition)
    before = deepcopy(config)
    ModelConfig.from_mapping(config)
    assert config == before
    assert config["provenance"]["specification_version"] == "0923-full-v3"
    assert config["modules"]["hypo_transitions_mod"]["kwargs"]["mechanism_version"] == "0923-full-v3"
    assert config["modules"]["beta_mod"]["kwargs"]["beta_update_mode"] == "feedback_gradient"
    assert config["modules"]["memory_mod"]["kwargs"]["forgetting_mode"] == (
        "factorized" if condition == 3 else "exact_power"
    )


@pytest.mark.parametrize("condition", [1, 2, 3])
@pytest.mark.parametrize("legacy_rate", ["increase_rate", "decrease_rate", "correct_additive"])
def test_full_v3_rejects_legacy_precision_controls_even_when_zero(condition, legacy_rate):
    config = _config(condition)
    config["modules"]["beta_mod"]["kwargs"][legacy_rate] = 0.
    with pytest.raises(ValueError):
        ModelConfig.from_mapping(config)


@pytest.mark.parametrize("rate", [None, True, -.01, 1.01, float("nan"), float("inf")])
def test_full_v3_requires_an_explicit_valid_learning_rate(rate):
    config = _config()
    beta = config["modules"]["beta_mod"]["kwargs"]
    if rate is None:
        del beta["learning_rate"]
    else:
        beta["learning_rate"] = rate
    with pytest.raises(ValueError, match="learning_rate"):
        ModelConfig.from_mapping(config)


@pytest.mark.parametrize("change", ["legacy_transition", "missing_transition_version", "old_specification", "old_beta"])
def test_full_v3_rejects_a_partially_upgraded_model(change):
    config = _config()
    transition = config["modules"]["hypo_transitions_mod"]["kwargs"]
    if change == "legacy_transition":
        transition["mechanism_version"] = "legacy"
    elif change == "missing_transition_version":
        del transition["mechanism_version"]
    elif change == "old_specification":
        config["provenance"]["specification_version"] = "0923-full-v2"
    else:
        config["modules"]["beta_mod"]["kwargs"]["beta_update_mode"] = "probabilistic_feedback"
    with pytest.raises(ValueError):
        ModelConfig.from_mapping(config)


@pytest.mark.parametrize("mode", [None, "joint"])
def test_full_v3_condition3_rejects_legacy_joint_forgetting(mode):
    config = _config(3)
    memory = config["modules"]["memory_mod"]["kwargs"]
    if mode is None:
        del memory["forgetting_mode"]
    else:
        memory["forgetting_mode"] = mode
    with pytest.raises(ValueError, match="forgetting_mode"):
        ModelConfig.from_mapping(config)


@pytest.mark.parametrize("condition", [1, 2])
@pytest.mark.parametrize("mode", [None, "dual_trace"])
def test_full_v3_binary_feedback_rejects_legacy_floored_memory(condition, mode):
    config = _config(condition)
    memory = config["modules"]["memory_mod"]["kwargs"]
    if mode is None:
        del memory["forgetting_mode"]
    else:
        memory["forgetting_mode"] = mode
    with pytest.raises(ValueError, match="forgetting_mode"):
        ModelConfig.from_mapping(config)


@pytest.mark.parametrize("condition", [1, 2, 3])
def test_saved_full_v2_requires_an_explicit_version_change_for_gradient_learning(condition):
    config = _config(condition, revised=False)
    beta = config["modules"]["beta_mod"]["kwargs"]
    beta.pop("increase_rate")
    beta.pop("decrease_rate")
    beta.update(beta_update_mode="feedback_gradient", learning_rate=.1)
    with pytest.raises(ValueError, match="specification_version"):
        ModelConfig.from_mapping(config)


@pytest.mark.parametrize("condition", [1, 2, 3])
def test_full_v3_provenance_records_the_actual_updated_mechanisms(condition):
    config = _config(condition)
    before = deepcopy(config)
    parameters = infer_fixed_hyperparams_from_engine_config(config)
    provenance = build_model_provenance(config, repeat_aggregation="mean_probability")
    resolved = provenance["resolved"]
    assert config == before
    assert parameters[NEW_PARAMETER_PATHS[0]] == .1
    assert parameters[NEW_PARAMETER_PATHS[2]] == "0923-full-v3"
    assert resolved["action_beta"]["learning_rate"] == .1
    assert resolved["action_beta"]["beta_update_mode"] == "feedback_gradient"
    assert resolved["candidate_management"]["mechanism_version"] == "0923-full-v3"
    assert resolved["candidate_management"]["capacity_policy"] == "remove_lowest_belief_before_single_transfer"
    assert resolved["choice_readout"]["rule_weights"] == "belief_times_previous_dominant_inertia_normalized"
    assert resolved["choice_readout"]["dominant_rule"] == "maximum_weight_previous_dominant_tie_preference"
    assert resolved["transition_event_semantics"]["swap_probability"] == "search_probability"
    memory_mode = "factorized" if condition == 3 else "exact_power"
    assert parameters[NEW_PARAMETER_PATHS[1]] == memory_mode
    assert resolved["evidence_memory"]["forgetting_mode"] == memory_mode
    changed = deepcopy(config)
    changed["modules"]["beta_mod"]["kwargs"]["learning_rate"] = .2
    alternate = build_model_provenance(changed, repeat_aggregation="mean_probability")
    assert alternate["model_config_sha256"] != provenance["model_config_sha256"]


@pytest.mark.parametrize("condition", [1, 2, 3])
def test_saved_full_v2_fixed_parameters_and_provenance_keep_legacy_semantics(condition):
    config = _config(condition, revised=False)
    ModelConfig.from_mapping(config)
    parameters = infer_fixed_hyperparams_from_engine_config(config)
    assert all(path not in parameters for path in NEW_PARAMETER_PATHS)
    assert parameters["engine.modules.beta_mod.kwargs.increase_rate"] == .04
    assert parameters["engine.modules.beta_mod.kwargs.decrease_rate"] == .15
    resolved = build_model_provenance(config, repeat_aggregation="mean_probability")["resolved"]
    assert "evidence_memory" not in resolved
    assert "rule_weights" not in resolved["choice_readout"]
    assert "mechanism_version" not in resolved["candidate_management"]
    assert resolved["transition_event_semantics"]["swap_probability"] == "search_probability_times_conditional_admission_probability"


@pytest.mark.parametrize("condition", [1, 2, 3])
def test_full_v3_generic_parameter_overrides_reach_the_engine_and_precision_update(condition, monkeypatch, tmp_path):
    monkeypatch.setattr(ContinuousSimilarity, "RUNTIME_CACHE_DIR", tmp_path / "similarity")
    config = _config(condition)
    config["modules"]["perception_mod"]["kwargs"] = {
        "features": 4, "mean": [0.] * 4, "std": [0.] * 4, "module_seed": 32,
    }
    config["modules"]["hypo_transitions_mod"]["kwargs"].update(relation_samples=512, module_seed=23)
    config["provenance"]["hypothesis_similarity"]["sample_count"] = 512
    before = deepcopy(config)
    short_paths = {
        "memory_mod.kwargs.gamma": .73,
        "beta_mod.kwargs.beta_init": 3.2,
        "beta_mod.kwargs.learning_rate": .37,
        "hypo_transitions_mod.kwargs.max_capacity": 4,
        "hypo_transitions_mod.kwargs.exit_rate": .2,
        "hypo_transitions_mod.kwargs.local_structure_weight": .7,
        "hypo_transitions_mod.kwargs.transfer_scale": .35,
        "hypo_transitions_mod.kwargs.execution_inertia": .8,
        "hypo_transitions_mod.kwargs.search_controller.baseline_probability": .12,
        "hypo_transitions_mod.kwargs.search_controller.error_gain": 1.3,
        "hypo_transitions_mod.kwargs.search_controller.failure_decay": .45,
        "hypo_transitions_mod.kwargs.search_controller.global_search": .3,
        "hypo_transitions_mod.kwargs.search_controller.global_search_failure_gain": .7,
    }
    assert len(short_paths) == 13
    parameters = {f"engine.modules.{path}": value for path, value in short_paths.items()}
    resolved = apply_fixed_hyperparams_to_engine_config(config, parameters)
    assert config == before
    ModelConfig.from_mapping(resolved)
    for path, expected in short_paths.items():
        actual = resolved["modules"]
        for part in path.split("."):
            actual = actual[part]
        assert actual == expected

    model = StateModel(resolved, context=ModelContext(condition=condition))
    stimulus = np.array([.8, .3, .2, .4])
    model.begin_trial(stimulus)
    engine = model.engine
    beta = engine.get_module(ModuleRole.BETA)
    memory = engine.get_module(ModuleRole.MEMORY)
    active = np.flatnonzero(engine.hypotheses_mask)
    initial_beta = engine.beta[active].copy()
    np.testing.assert_allclose(initial_beta, 3.2)
    assert beta.learning_rate == .37
    assert memory.gamma == .73
    pairing = memory.conditional_pairing()[active] if condition == 3 else None
    gradient = engine.observation_likelihood.feedback_precision_gradient(
        (stimulus, 1, 1.), active, initial_beta, pairing_weights=pairing,
    )
    assert np.any(np.abs(gradient) > 1e-8)
    model.complete_trial(1, 1.)
    np.testing.assert_allclose(engine.beta[active], np.clip(initial_beta + .37 * gradient, .1, 25.))

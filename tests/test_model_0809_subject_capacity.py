from __future__ import annotations

from pathlib import Path

import yaml

from src.Bayesian_state.optimization.hyper_utils import values_product
from src.Bayesian_state.run_hyper_then_simulation import (
    _split_hyperparams_for_simulation_override,
)
from src.Bayesian_state.simulation.simulation_config import (
    EVALUATION_ROLE_OPTIMIZATION,
    resolve_evaluation_score_mask,
)


HYPER_CONFIG = Path(
    "configs/hyper_cd_cfg/model0809_cond1_dynamic_continuous_selected8.yaml"
)
SIMULATION_CONFIG = Path(
    "configs/simulation_cfg/model0809_cond1_dynamic_continuous_full_data.yaml"
)
CAPACITY_PATH = "engine.modules.hypo_transitions_mod.kwargs.capacity"
V2_PROBE_CONFIG = Path(
    "configs/simulation_cfg/generated_from_hyper/"
    "model0809_controller_v2a_selected3_probe.yaml"
)
V2B_PROBE_CONFIG = Path(
    "configs/simulation_cfg/generated_from_hyper/"
    "model0809_controller_v2b_selected3_probe.yaml"
)


def test_model0809_pilot_searches_subject_capacity_on_the_full_sequence():
    hyper = yaml.safe_load(HYPER_CONFIG.read_text(encoding="utf-8"))
    simulation = yaml.safe_load(SIMULATION_CONFIG.read_text(encoding="utf-8"))

    selected_subjects = [103, 104, 105, 108, 111, 120, 124, 132]
    assert hyper["subjects"] == selected_subjects
    assert simulation["subjects"] == selected_subjects
    assert hyper["output_dir"] == "../../results/model_dynamic_continuous/0809_v1/hyper_cd"
    assert simulation["output_dir"] == "../../results/model_dynamic_continuous/0809_v1/simulation"
    assert hyper["loss_metric"] == "choice_nll"
    assert simulation["loss_metric"] == "choice_nll"
    assert hyper["objective_order"][0]["path"] == "simulation.mean_error"
    assert hyper["hyperparam_space"][CAPACITY_PATH]["values"] == [3, 5, 7]
    assert len(hyper["hyperparam_space"]) == 7

    memory_space = hyper["hyperparam_space"]["engine.modules.memory_mod.kwargs"]
    assert memory_space["values_product"] == {
        "gamma": [0.50, 0.60, 0.70, 0.80],
        "w0": [0.03, 0.06, 0.10, 0.15],
        "feedback_gain": [1.0],
    }
    memory_candidates = values_product(memory_space)
    assert len(memory_candidates) == 16
    assert len({(item["gamma"], item["w0"]) for item in memory_candidates}) == 16
    assert {item["feedback_gain"] for item in memory_candidates} == {1.0}

    score_mask, context = resolve_evaluation_score_mask(
        192,
        simulation["evaluation_protocol"],
        role=EVALUATION_ROLE_OPTIMIZATION,
    )
    assert score_mask is None
    assert context["mode"] == "all"
    assert context["score_trial_count"] == 192


def test_selected_capacity_is_materialized_as_a_subject_engine_override():
    override = _split_hyperparams_for_simulation_override({CAPACITY_PATH: 5})

    assert (
        override["engine_config"]["modules"]["hypo_transitions_mod"]["kwargs"][
            "capacity"
        ]
        == 5
    )
    assert override["fixed_hyperparams"][CAPACITY_PATH] == 5


def test_controller_v2a_probe_is_isolated_and_keeps_prior_assignment_fixed():
    probe = yaml.safe_load(V2_PROBE_CONFIG.read_text(encoding="utf-8"))
    assert probe["subjects"] == [103, 120, 124]
    assert "0810_controller_v2a_probe" in probe["output_dir"]
    transition = probe["engine_config"]["modules"]["hypo_transitions_mod"]
    kwargs = transition["kwargs"]
    controller = kwargs["continuous_controller"]
    assert controller["mode"] == "failure_accumulator_v2"
    assert controller["exploration"]["failure_threshold"] < controller["range"][
        "failure_threshold"
    ]
    assert controller["exploration"]["rise_rate"] > controller["exploration"][
        "recovery_rate"
    ]
    assert "rate_controller" not in kwargs
    assert "range_controller" not in kwargs
    assert "prior_assignment" not in kwargs


def test_controller_v2b_probe_changes_only_global_prior_reset():
    v2a = yaml.safe_load(V2_PROBE_CONFIG.read_text(encoding="utf-8"))
    v2b = yaml.safe_load(V2B_PROBE_CONFIG.read_text(encoding="utf-8"))
    assert v2b["subjects"] == v2a["subjects"]
    assert "0810_controller_v2b_probe" in v2b["output_dir"]

    transition_a = v2a["engine_config"]["modules"]["hypo_transitions_mod"]
    transition_b = v2b["engine_config"]["modules"]["hypo_transitions_mod"]
    controller_a = transition_a["kwargs"]["continuous_controller"]
    controller_b = transition_b["kwargs"]["continuous_controller"]
    reset = controller_b.pop("prior_reset")
    assert reset == {"max_strength": 0.35}
    assert controller_b == controller_a

    v2a["output_dir"] = v2b["output_dir"]
    transition_a["kwargs"]["continuous_controller"] = controller_b
    v2a["engine_config"]["modules"]["hypo_transitions_mod"] = transition_a
    v2a.pop("statistics_config", None)
    v2b.pop("statistics_config", None)
    assert v2b == v2a

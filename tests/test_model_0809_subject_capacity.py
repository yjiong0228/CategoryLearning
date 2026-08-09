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
    "configs/hyper_cd_cfg/model0809_cond1_dynamic_continuous_subject103.yaml"
)
SIMULATION_CONFIG = Path(
    "configs/simulation_cfg/model0809_cond1_dynamic_continuous_full_data.yaml"
)
CAPACITY_PATH = "engine.modules.hypo_transitions_mod.kwargs.capacity"


def test_model0809_pilot_searches_subject_capacity_on_the_full_sequence():
    hyper = yaml.safe_load(HYPER_CONFIG.read_text(encoding="utf-8"))
    simulation = yaml.safe_load(SIMULATION_CONFIG.read_text(encoding="utf-8"))

    assert hyper["subjects"] == [103]
    assert simulation["subjects"] == [103]
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

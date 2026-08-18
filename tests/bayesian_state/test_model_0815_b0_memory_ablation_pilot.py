from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.run_model_0815_b0_memory_ablation_pilot import (
    BAYES_ONLY_CLASS,
    DUAL_MEMORY_CLASS,
    build_candidate_bank,
    build_memory_engine,
    select_family_candidate,
)
from src.Bayesian_state.model import ModelContext, StateModel
from src.Bayesian_state.model.modules.base_module import ModulePhase


ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG = (
    ROOT / "configs/model_struct/pmh_model_cond1_0815_b0_adaptive_controller.yaml"
)
AUDIT_CONFIG = (
    ROOT / "configs/specific_models/model_0815_b0_memory_ablation_pilot.yaml"
)


def _base_engine() -> dict:
    return yaml.safe_load(MODEL_CONFIG.read_text(encoding="utf-8"))


def _audit_config() -> dict:
    return yaml.safe_load(AUDIT_CONFIG.read_text(encoding="utf-8"))


def test_candidate_bank_is_nested_and_smoke_preserves_all_three_levels() -> None:
    formal = build_candidate_bank(_audit_config())
    smoke = build_candidate_bank(_audit_config(), smoke=True)

    assert len(formal) == 10
    assert {value["family"] for value in formal} == {
        "m_off",
        "m_leaky",
        "m_dual",
    }
    assert len([value for value in formal if value["family"] == "m_off"]) == 1
    assert len([value for value in formal if value["family"] == "m_leaky"]) == 3
    assert len([value for value in formal if value["family"] == "m_dual"]) == 6
    assert len(smoke) == 3
    assert {value["family"] for value in smoke} == {
        "m_off",
        "m_leaky",
        "m_dual",
    }
    assert all(
        value["w0"] == 0.0
        for value in formal
        if value["family"] == "m_leaky"
    )
    assert all(
        0.0 < value["w0"] < 1.0
        for value in formal
        if value["family"] == "m_dual"
    )


def test_memory_candidate_changes_only_the_optional_M_block() -> None:
    base = _base_engine()
    leaky = build_memory_engine(
        base,
        {
            "family": "m_leaky",
            "gamma": 0.8,
            "w0": 0.0,
            "feedback_gain": 1.0,
        },
    )
    dual = build_memory_engine(
        base,
        {
            "family": "m_dual",
            "gamma": 0.8,
            "w0": 0.15,
            "feedback_gain": 1.0,
        },
    )

    assert base["modules"]["memory_mod"]["class"] == BAYES_ONLY_CLASS
    assert leaky["modules"]["memory_mod"] == {
        "class": DUAL_MEMORY_CLASS,
        "kwargs": {"gamma": 0.8, "w0": 0.0, "feedback_gain": 1.0},
    }
    assert dual["modules"]["memory_mod"]["kwargs"]["w0"] == 0.15
    for key in (
        "perception_mod",
        "hypo_transitions_mod",
        "beta_mod",
    ):
        assert leaky["modules"][key] == base["modules"][key]
        assert dual["modules"][key] == base["modules"][key]
    for key in ("likelihood", "choice_readout", "output_noise", "agenda"):
        assert leaky[key] == base[key]
        assert dual[key] == base[key]


def _memory_only_config(memory_config: dict) -> dict:
    config = deepcopy(_base_engine())
    config["modules"] = {"memory_mod": deepcopy(memory_config)}
    config["agenda"] = ["memory_mod"]
    return config


def _posterior_sequence(memory_config: dict) -> np.ndarray:
    model = StateModel(
        _memory_only_config(memory_config),
        context=ModelContext(condition=1, subject_id=103),
    )
    engine = model.engine
    stimuli = np.asarray(
        [
            [0.20, 0.80, 0.30, 0.70],
            [0.75, 0.25, 0.60, 0.40],
            [0.35, 0.65, 0.85, 0.15],
            [0.90, 0.10, 0.45, 0.55],
        ],
        dtype=float,
    )
    choices = [1, 2, 1, 2]
    feedback = [1.0, 0.0, 1.0, 1.0]
    output = []
    for stimulus, choice, outcome in zip(stimuli, choices, feedback):
        engine.begin_trial(stimulus)
        engine.observation = (stimulus.copy(), int(choice), float(outcome))
        engine.compute_likelihood()
        engine.run_phase(ModulePhase.POST_CHOICE)
        output.append(np.asarray(engine.posterior, dtype=float).copy())
    return np.stack(output)


def test_gamma_one_single_channel_is_the_M_off_nested_boundary() -> None:
    bayes_only = _posterior_sequence({"class": BAYES_ONLY_CLASS, "kwargs": {}})
    gamma_one = _posterior_sequence(
        {
            "class": DUAL_MEMORY_CLASS,
            "kwargs": {"gamma": 1.0, "w0": 0.0, "feedback_gain": 1.0},
        }
    )
    forgetting = _posterior_sequence(
        {
            "class": DUAL_MEMORY_CLASS,
            "kwargs": {"gamma": 0.5, "w0": 0.0, "feedback_gain": 1.0},
        }
    )

    assert gamma_one == pytest.approx(bayes_only, abs=1e-12)
    assert not np.allclose(forgetting, bayes_only, atol=1e-6, rtol=0.0)


def test_family_selection_uses_training_nll_then_stable_id_tie_break() -> None:
    frame = pd.DataFrame(
        [
            {"family": "m_leaky", "candidate_id": "m_leaky_g080", "ensemble_nll_train": 0.42},
            {"family": "m_leaky", "candidate_id": "m_leaky_g050", "ensemble_nll_train": 0.42},
            {"family": "m_leaky", "candidate_id": "m_leaky_g095", "ensemble_nll_train": 0.44},
        ]
    )
    selected = select_family_candidate(frame, "m_leaky")
    assert selected["candidate_id"] == "m_leaky_g050"

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from scripts.run_model_0815_h4_nested_subject_screen import (
    _canonical_parameters,
    _resolved_inputs,
    build_nested_engine,
)


ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG = (
    ROOT
    / "configs/model_struct/pmh_model_cond1_0815_h4_nested_feedback_accumulator.yaml"
)
SCREEN_CONFIG = (
    ROOT
    / "configs/specific_models/model_0815_h4_nested_subject_screen.yaml"
)


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_parameterization_uses_nonnegative_immediate_logit_gain() -> None:
    parameters = _canonical_parameters(
        {
            "event_after_correct": 0.20,
            "immediate_error_logit_gain": 1.20,
            "global_search": 0.30,
            "accumulator_decay": 0.60,
            "accumulator_logit_gain": 0.0,
            "initial_failure": 0.0,
        }
    )

    assert parameters["event_after_error"] > parameters["event_after_correct"]
    assert 0.0 < parameters["event_after_error"] < 1.0
    with pytest.raises(ValueError, match="non-negative"):
        _canonical_parameters(
            {
                **parameters,
                "immediate_error_logit_gain": -0.01,
            }
        )


def test_engine_builder_changes_only_nested_controller_and_not_template() -> None:
    template = _yaml(MODEL_CONFIG)
    original = deepcopy(template)
    engine = build_nested_engine(
        template,
        {
            "event_after_correct": 0.18,
            "immediate_error_logit_gain": 0.60,
            "global_search": 0.40,
            "accumulator_decay": 0.85,
            "accumulator_logit_gain": 1.50,
            "initial_failure": 0.0,
        },
    )

    assert template == original
    template_h = template["modules"]["hypo_transitions_mod"]
    engine_h = engine["modules"]["hypo_transitions_mod"]
    template_without_h = deepcopy(template)
    engine_without_h = deepcopy(engine)
    template_without_h["modules"].pop("hypo_transitions_mod")
    engine_without_h["modules"].pop("hypo_transitions_mod")
    assert template_without_h == engine_without_h
    assert template_h["class"] == engine_h["class"]
    controller = engine_h["kwargs"]["nested_feedback_accumulator_controller"]
    assert controller["event_after_correct"] == pytest.approx(0.18)
    assert controller["event_after_error"] > 0.18
    assert controller["global_search"] == pytest.approx(0.40)
    assert controller["accumulator_decay"] == pytest.approx(0.85)
    assert controller["accumulator_logit_gain"] == pytest.approx(1.50)


def test_screen_design_has_zero_spike_and_disjoint_seed_roles() -> None:
    config = _yaml(SCREEN_CONFIG)
    design, reactive, accumulator = _resolved_inputs(config, smoke=False)

    assert design["train_trials"] == 32
    assert design["trials_per_subject"] == 64
    assert design["training_particle_count"] == 16
    assert design["evaluation_particle_count"] == 32
    assert design["training_seed_role"] != design["evaluation_seed_role"]
    assert accumulator["subject_gain_candidates"][0] == pytest.approx(0.0)
    assert accumulator["common_decay_anchor"] in accumulator[
        "common_decay_candidates"
    ]
    assert reactive["passes"] == 1


def test_smoke_design_satisfies_shared_metric_window() -> None:
    config = _yaml(SCREEN_CONFIG)
    design, reactive, accumulator = _resolved_inputs(config, smoke=True)

    assert design["subjects"] == [103]
    assert design["train_trials"] == 18
    assert design["trials_per_subject"] == 24
    assert design["training_particle_count"] == 8
    assert design["evaluation_particle_count"] == 8
    assert reactive["passes"] == 1
    assert accumulator["common_decay_candidates"] == [0.60]
    assert accumulator["subject_gain_candidates"] == [0.0, 1.5]

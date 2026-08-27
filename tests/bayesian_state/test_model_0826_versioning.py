from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _yaml(relative_path: str) -> dict:
    return yaml.safe_load((ROOT / relative_path).read_text(encoding="utf-8"))


def _sha256(relative_path: str) -> str:
    return sha256((ROOT / relative_path).read_bytes()).hexdigest()


def test_model_0818_remains_the_original_frozen_version() -> None:
    manuscript_path = "manuscript/model_0818.tex"
    model = _yaml("configs/model_struct/pmh_model_cond1_0818.yaml")
    parameter_space = _yaml(
        "configs/specific_models/model_0818_cond1_parameter_space.yaml"
    )
    controller = model["modules"]["hypo_transitions_mod"]["kwargs"][
        "nested_feedback_accumulator_controller"
    ]

    assert _sha256(manuscript_path) == (
        "0d745cd3e758d5aa03904945f2f786b54766ee18f54501e2713eab6713e3f584"
    )
    assert model["provenance"]["model_id"] == "model_0818"
    assert model["provenance"]["manuscript_path"] == manuscript_path
    assert model["provenance"]["manuscript_sha256"] == _sha256(manuscript_path)
    assert parameter_space["provenance"]["manuscript_sha256"] == _sha256(
        manuscript_path
    )
    assert "event_history_excludes_latest_error" not in controller


def test_model_0826_is_the_revised_self_consistent_version() -> None:
    manuscript_path = "manuscript/model_0826.tex"
    model = _yaml("configs/model_struct/pmh_model_cond1_0826.yaml")
    parameter_space = _yaml(
        "configs/specific_models/model_0826_cond1_parameter_space.yaml"
    )
    manuscript = (ROOT / manuscript_path).read_text(encoding="utf-8")
    transition = model["modules"]["hypo_transitions_mod"]["kwargs"]

    assert model["provenance"]["model_id"] == "model_0826"
    assert model["provenance"]["manuscript_path"] == manuscript_path
    assert model["provenance"]["manuscript_sha256"] == _sha256(manuscript_path)
    assert parameter_space["provenance"]["model_id"] == "model_0826"
    assert parameter_space["provenance"]["manuscript_path"] == manuscript_path
    assert parameter_space["provenance"]["manuscript_sha256"] == _sha256(
        manuscript_path
    )
    assert transition["nested_feedback_accumulator_controller"][
        "event_history_excludes_latest_error"
    ] is True
    assert transition["prior_assignment"]["method"] == "similarity_transport"
    assert r"+c_{A,s}F_{t-1}" in manuscript
    assert r"\label{eq:mass-preserving-transport}" in manuscript
    assert "动态规则精度" in manuscript
    assert "动作确信度" not in manuscript
    assert "model_0826_framework.png" in manuscript


def test_model_0826_counterfactual_uses_only_0826_entrypoints() -> None:
    audit = _yaml(
        "configs/specific_models/model_0826_belief_transport_counterfactual.yaml"
    )
    simulation = _yaml(
        "configs/simulation_cfg/model0826_cond1_exploratory_observed_fit.yaml"
    )

    assert audit["base_simulation_config"].endswith(
        "model0826_cond1_exploratory_observed_fit.yaml"
    )
    assert audit["output_dir"].startswith("results/model_0826/")
    assert simulation["engine_config_path"].endswith(
        "pmh_model_cond1_0826.yaml"
    )
    assert "results/model_0826/" in simulation["output_dir"]

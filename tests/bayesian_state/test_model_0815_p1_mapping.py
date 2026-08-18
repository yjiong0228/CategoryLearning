from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.Bayesian_state.inference.backends.particle_filter import (
    run_state_model_particle_filter,
)
from src.Bayesian_state.model import ModelContext, StateModel
from src.Bayesian_state.model.modules import ModuleRole
from src.Bayesian_state.model.readout import read_choice_probabilities_from_model
from src.Bayesian_state.simulation.autonomous import (
    run_autonomous_category_learning,
)


ROOT = Path(__file__).resolve().parents[2]
M0_PATH = ROOT / "configs/model_struct/pmh_model_cond1_0815_p0.yaml"
M1_PATH = (
    ROOT
    / "configs/model_struct/pmh_model_cond1_0815_p1_m1_orientation.yaml"
)


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _model() -> StateModel:
    return StateModel(
        _load(M1_PATH),
        context=ModelContext(condition=1, subject_id=103),
    )


def test_m1_differs_from_m0_only_by_mapping_module_and_provenance() -> None:
    m0 = _load(M0_PATH)
    m1 = _load(M1_PATH)
    assert m1["provenance"]["label_mapping"]["free_parameters_added"] == 0
    assert m1["modules"]["mapping_mod"]["kwargs"]["initial_probability"] == 0.5
    assert m1["agenda"].index("mapping_mod") < m1["agenda"].index("memory_mod")

    stripped_m0 = deepcopy(m0)
    stripped_m1 = deepcopy(m1)
    stripped_m0.pop("provenance")
    stripped_m1.pop("provenance")
    stripped_m1["modules"].pop("mapping_mod")
    stripped_m1["agenda"].remove("mapping_mod")
    assert stripped_m1 == stripped_m0


def test_neutral_orientation_makes_first_choice_uniform_and_updates_bayes() -> None:
    model = _model()
    stimulus = np.asarray([0.15, 0.85, 0.25, 0.75], dtype=float)
    prepared = model.begin_trial(stimulus)
    mapping = model.engine.get_module(ModuleRole.MAPPING, required=True)
    active = np.flatnonzero(model.engine.hypotheses_mask > 0.0)
    prediction = read_choice_probabilities_from_model(
        model,
        prepared.perceived_stimulus,
        power=1.0,
        lapse=0.0,
    )
    assert np.allclose(prediction, [0.5, 0.5], atol=1e-12)

    fixed_likelihood = model.observation_likelihood.process(
        (prepared.perceived_stimulus, 1, 1.0),
        tuple(range(model.engine.set_size)),
        beta=model.engine.beta,
    )
    model.complete_trial(1, 1.0)

    assert np.allclose(model.engine.likelihood[active], 0.5, atol=1e-12)
    assert np.allclose(
        mapping.orientation_probability[active],
        fixed_likelihood[active],
        atol=1e-12,
    )


def test_orientation_marginalization_and_newcomer_reset_are_exact() -> None:
    model = _model()
    mapping = model.engine.get_module(ModuleRole.MAPPING, required=True)
    hypothesis = 3
    mapping.orientation_probability[hypothesis] = 0.8
    fixed = np.asarray([0.9, 0.1], dtype=float)
    expected = 0.8 * fixed + 0.2 * fixed[::-1]
    assert np.allclose(
        mapping.orient_category_probabilities(hypothesis, fixed),
        expected,
        atol=1e-12,
    )

    mapping.initialize_orientation_for_hypotheses([hypothesis])
    assert mapping.orientation_probability[hypothesis] == pytest.approx(0.5)


def test_orientation_state_survives_particle_snapshot_roundtrip() -> None:
    model = _model()
    mapping = model.engine.get_module(ModuleRole.MAPPING, required=True)
    mapping.orientation_probability[5] = 0.73
    snapshot = model.engine.state_dict()
    mapping.orientation_probability[5] = 0.21
    model.engine.load_state_dict(snapshot)
    restored = model.engine.get_module(ModuleRole.MAPPING, required=True)
    assert restored.orientation_probability[5] == pytest.approx(0.73)


def test_orientation_oracle_conditioning_sets_both_pre_feedback_states() -> None:
    model = _model()
    mapping = model.engine.get_module(ModuleRole.MAPPING, required=True)
    truth = np.linspace(0.1, 0.9, model.engine.set_size)
    mapping.condition_on_orientation_probability(truth)
    np.testing.assert_allclose(mapping.orientation_probability, truth)
    np.testing.assert_allclose(mapping.predictive_orientation_probability, truth)

    with pytest.raises(ValueError, match="wrong shape"):
        mapping.condition_on_orientation_probability([0.5, 0.5])


def test_particle_filter_exposes_normalized_geometry_orientation_joint() -> None:
    config = _load(M1_PATH)
    stimulus = np.asarray(
        [
            [0.2, 0.8, 0.3, 0.7],
            [0.8, 0.2, 0.7, 0.3],
            [0.1, 0.4, 0.9, 0.6],
            [0.9, 0.6, 0.1, 0.4],
        ],
        dtype=float,
    )
    categories = np.asarray([1, 2, 1, 2], dtype=int)
    generated = run_autonomous_category_learning(
        engine_config=config,
        subject_id=103,
        condition=1,
        stimulus=stimulus,
        categories=categories,
        trajectory_seed=42,
    )
    filtered = run_state_model_particle_filter(
        engine_config=config,
        subject_id=103,
        stimulus=stimulus,
        choices=generated.trajectory.choices,
        feedback=generated.trajectory.feedback,
        particle_count=4,
        choice_readout_power=1.0,
        output_lapse=0.02,
        filter_seed=43,
        choice_transmission_audit=True,
    )
    joint = np.asarray(
        filtered.state_probabilities["executed_orientation_joint"], dtype=float
    )
    filtered_joint = np.asarray(
        filtered.state_probabilities["filtered_executed_orientation_joint"],
        dtype=float,
    )
    assert joint.shape == (stimulus.shape[0], 29, 2)
    assert np.allclose(np.sum(joint, axis=(1, 2)), 1.0, atol=1e-12)
    assert np.allclose(np.sum(filtered_joint, axis=(1, 2)), 1.0, atol=1e-12)
    assert np.allclose(
        np.sum(joint, axis=2),
        filtered.state_probabilities["executed_probability"],
        atol=1e-12,
    )
    assert np.allclose(
        np.sum(filtered_joint, axis=2),
        filtered.state_probabilities["filtered_executed_probability"],
        atol=1e-12,
    )
    assert np.allclose(filtered.marginal_probabilities.sum(axis=1), 1.0)
    ancestral = filtered.artifacts["audit_ancestral_paths"]
    observed = np.asarray(ancestral["observed_choice_probability"], dtype=float)
    correct = np.asarray(ancestral["correct_probability"], dtype=float)
    expected = np.where(
        generated.trajectory.feedback[None, :] >= 0.5,
        correct,
        1.0 - correct,
    )
    np.testing.assert_allclose(observed, expected)

    oracle = np.stack(
        [
            np.asarray(step["orientation_probability"], dtype=float)
            for step in generated.trajectory.step_log
        ]
    )
    conditioned = run_state_model_particle_filter(
        engine_config=config,
        subject_id=103,
        stimulus=stimulus,
        choices=generated.trajectory.choices,
        feedback=generated.trajectory.feedback,
        particle_count=4,
        choice_readout_power=1.0,
        output_lapse=0.02,
        filter_seed=43,
        orientation_oracle_schedule=oracle,
    )
    assert conditioned.metadata["orientation_oracle_conditioned"] is True

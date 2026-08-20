from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.Bayesian_state.inference import run_inference_backend
from src.Bayesian_state.model import ModelContext, StateModel
from src.Bayesian_state.model.readout import (
    read_choice_probabilities_from_model,
    resolve_executed_hypothesis,
)
from src.Bayesian_state.model.modules.hypothesis_transition.contracts import (
    HypothesisSelection,
)
from src.Bayesian_state.model.modules.hypothesis_transition.feedback_reactive import (
    FeedbackReactiveHypothesisTransitionModule,
)
from src.Bayesian_state.simulation.parameters import (
    apply_fixed_hyperparams_to_engine_config,
    infer_fixed_hyperparams_from_engine_config,
)
from src.Bayesian_state.simulation.provenance import build_model_provenance


ROOT = Path(__file__).resolve().parents[2]
H5_CONFIG = (
    ROOT
    / "configs/model_struct/pmh_model_cond1_0815_h5_similarity_transport.yaml"
)
EXECUTION_ENABLED_PATH = (
    "engine.modules.hypo_transitions_mod.kwargs."
    "persistent_execution.enabled"
)


def _deterministic_h5_config() -> dict:
    config = yaml.safe_load(H5_CONFIG.read_text(encoding="utf-8"))
    config["modules"]["perception_mod"]["kwargs"] = {
        "features": 4,
        "mean": [0.0, 0.0, 0.0, 0.0],
        "std": [0.0, 0.0, 0.0, 0.0],
        "module_seed": 11,
    }
    config["modules"]["hypo_transitions_mod"]["kwargs"][
        "module_seed"
    ] = 29
    return config


class _TinyPartition:
    def __init__(self, size: int = 8):
        positions = np.arange(size, dtype=float)
        self._similarity = np.exp(
            -np.abs(positions[:, None] - positions[None, :])
        )

    @property
    def similarity_matrix(self) -> np.ndarray:
        return self._similarity

    def get_similarity_matrix(self, *, kind, distance_mode, **kwargs):
        del kwargs
        assert kind == "assignment_agreement"
        assert distance_mode == "boundary"
        return self._similarity


class _TinyEngine:
    def __init__(self, size: int = 8):
        self.set_size = int(size)
        self.prior = np.full(size, 1.0 / size, dtype=float)
        self.posterior = None
        self.hypotheses_mask = None
        self.partition = _TinyPartition(size)
        self.distance_mode = "boundary"

    def get_module(self, role, *, required=False):
        del role
        if required:
            raise ValueError("tiny test engine has no auxiliary modules")
        return None


def _module(method: str, *, global_search: float = 0.30):
    engine = _TinyEngine()
    module = FeedbackReactiveHypothesisTransitionModule(
        engine,
        capacity=3,
        init_hypotheses=[0, 1, 2],
        feedback_reactive_controller={
            "event_after_correct": 0.20,
            "event_after_error": 0.60,
            "initial_event_probability": 0.20,
            "global_search": global_search,
        },
        prior_assignment={"method": method},
        module_seed=17,
    )
    return engine, module


def _one_replacement() -> HypothesisSelection:
    return HypothesisSelection.from_active_sets(
        [0, 1, 2],
        [0, 1, 3],
        replacement_pairs=((2, 3),),
    )


def test_similarity_transport_is_exact_carryover_without_replacement() -> None:
    _, module = _module("similarity_transport")
    posterior = np.asarray([0.80, 0.15, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
    module._pending_transition = {"posterior": posterior}
    selection = HypothesisSelection.from_active_sets([0, 1, 2], [0, 1, 2])

    prior = module.assign_prior(None, selection)

    np.testing.assert_allclose(prior, posterior)
    assert module._pending_transition["prior_transport_fraction"] == 0.0
    assert module._pending_transition["newcomer_prior_mass"] == 0.0


def test_similarity_transport_uses_replacement_fraction_and_semantic_kernel() -> None:
    _, module = _module("similarity_transport", global_search=0.30)
    posterior = np.asarray([0.80, 0.15, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
    selection = _one_replacement()
    module._pending_transition = {"posterior": posterior}

    prior = module.assign_prior(None, selection)

    module._ensure_geometry()
    local_full = posterior @ module._local_kernel
    local = local_full[selection.active_after]
    local /= local.sum()
    global_weights = module.base_prior[selection.active_after]
    global_weights /= global_weights.sum()
    semantic = 0.70 * local + 0.30 * global_weights
    carryover = posterior[selection.survivors]
    carryover /= carryover.sum()
    expected = np.zeros(module.total_hypo, dtype=float)
    expected[selection.survivors] = (2.0 / 3.0) * carryover
    expected[selection.active_after] += (1.0 / 3.0) * semantic

    np.testing.assert_allclose(prior, expected)
    assert module._pending_transition["prior_transport_fraction"] == pytest.approx(
        1.0 / 3.0
    )
    assert prior[3] != pytest.approx(posterior[2])
    assert module._pending_transition["newcomer_prior_mass"] == pytest.approx(
        prior[3]
    )


def test_global_boundary_flattens_semantic_component_over_new_workspace() -> None:
    _, module = _module("similarity_transport", global_search=1.0)
    posterior = np.asarray([0.80, 0.15, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
    module._pending_transition = {"posterior": posterior}

    prior = module.assign_prior(None, _one_replacement())

    # One third of the prior is transported, and a uniform base prior makes
    # that semantic component uniform over the three realized active rules.
    assert prior[3] == pytest.approx(1.0 / 9.0)
    assert module._pending_transition["semantic_newcomer_mass"] == pytest.approx(
        1.0 / 3.0
    )


def test_full_workspace_replacement_is_full_semantic_projection() -> None:
    _, module = _module("similarity_transport", global_search=0.0)
    posterior = np.asarray([0.80, 0.15, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
    selection = HypothesisSelection.from_active_sets(
        [0, 1, 2],
        [3, 4, 5],
        replacement_pairs=((0, 3), (1, 4), (2, 5)),
    )
    module._pending_transition = {"posterior": posterior}

    prior = module.assign_prior(None, selection)

    assert module._pending_transition["prior_transport_fraction"] == 1.0
    assert np.all(prior[[3, 4, 5]] > 0.0)
    assert np.sum(prior[[3, 4, 5]]) == pytest.approx(1.0)
    assert np.all(prior[[0, 1, 2, 6, 7]] == 0.0)


def test_pairwise_method_remains_an_exact_compatibility_path() -> None:
    _, module = _module("pairwise_mass_transfer")
    posterior = np.asarray([0.80, 0.15, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
    module._pending_transition = {"posterior": posterior}

    prior = module.assign_prior(None, _one_replacement())

    np.testing.assert_allclose(
        prior,
        [0.80, 0.15, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0],
    )
    assert module._pending_transition["prior_assignment_method"] == (
        "pairwise_mass_transfer"
    )


def test_realized_transition_logs_similarity_transport_diagnostics() -> None:
    engine = _TinyEngine()
    module = FeedbackReactiveHypothesisTransitionModule(
        engine,
        capacity=3,
        init_hypotheses=[0, 1, 2],
        feedback_reactive_controller={
            "event_after_correct": 1.0,
            "event_after_error": 1.0,
            "initial_event_probability": 1.0,
            "global_search": 0.30,
        },
        prior_assignment={"method": "similarity_transport"},
        module_seed=23,
    )
    module.process()  # Trial 0 only initializes the fixed-capacity workspace.
    engine.posterior = np.asarray(engine.prior, dtype=float).copy()
    module.record_outcome((np.asarray([0.5]), 1, 0.0))

    module.process()

    event = module.transition_log[-1]
    assert event["replacement_count"] == 3
    assert event["prior_assignment_method"] == "similarity_transport"
    assert event["prior_transport_fraction"] == 1.0
    assert event["newcomer_prior_mass"] == pytest.approx(1.0)
    assert event["prior_sum"] == pytest.approx(1.0)


def test_similarity_transport_has_no_method_specific_hyperparameters() -> None:
    with pytest.raises(ValueError, match="supports only the method key"):
        FeedbackReactiveHypothesisTransitionModule(
            _TinyEngine(),
            capacity=3,
            init_hypotheses=[0, 1, 2],
            feedback_reactive_controller={
                "event_after_correct": 0.20,
                "event_after_error": 0.60,
                "initial_event_probability": 0.20,
                "global_search": 0.30,
            },
            prior_assignment={
                "method": "similarity_transport",
                "transport_strength": 0.50,
            },
        )


def test_h5_config_assembles_and_records_prior_assignment_provenance() -> None:
    config = yaml.safe_load(H5_CONFIG.read_text(encoding="utf-8"))
    model = StateModel(config, context=ModelContext(condition=1, subject_id=103))
    transition = model.engine.modules["hypo_transitions_mod"]

    assert transition.prior_assignment_method == "similarity_transport"
    assert transition.persistent_execution_enabled is False
    assert transition.tau_local == pytest.approx(0.10)
    beta = model.engine.modules["beta_mod"]
    assert beta.increase_rate == pytest.approx(0.04)
    assert beta.correct_additive == pytest.approx(1.0)
    assert beta.increase_parameterization == "increase_rate"
    fixed = infer_fixed_hyperparams_from_engine_config(config)
    assert fixed[
        "engine.modules.hypo_transitions_mod.kwargs.prior_assignment"
    ] == {"method": "similarity_transport"}
    assert fixed[
        "engine.modules.hypo_transitions_mod.kwargs.persistent_execution"
    ] == {"enabled": False, "switch_scale": pytest.approx(0.20)}
    assert fixed[
        "engine.modules.hypo_transitions_mod.kwargs.tau_local"
    ] == pytest.approx(0.10)
    assert fixed[
        "engine.modules.beta_mod.kwargs.increase_rate"
    ] == pytest.approx(0.04)
    assert "engine.modules.beta_mod.kwargs.correct_additive" not in fixed

    declared = config["provenance"]["hypothesis_similarity"]
    resource = (
        ROOT
        / "src/Bayesian_state/hypothesis_space/resources/similarity"
        / declared["resource_filename"]
    )
    observed_hash = hashlib.sha256(resource.read_bytes()).hexdigest()
    assert observed_hash == declared["resource_sha256"]

    provenance = build_model_provenance(
        config,
        repeat_aggregation="mean_probability",
    )
    similarity = provenance["resolved"]["hypothesis_similarity"]
    assert similarity["basis"] == "boundary_fixed_labels"
    assert similarity["version"] == "shared_hypothesis_space_v1"
    assert similarity["resource_generation_seed"] == "not_recorded"
    assert similarity["fallback_computation_seed"] == 0
    assert similarity["resource_sha256"] == observed_hash
    assert similarity["tau_local"] == pytest.approx(0.10)
    assert similarity["tau_scope"] == "common_fixed_architecture"


def test_h5_execution_off_is_exactly_the_legacy_h5_path() -> None:
    explicit_off = _deterministic_h5_config()
    implicit_off = _deterministic_h5_config()
    implicit_off["modules"]["hypo_transitions_mod"]["kwargs"].pop(
        "persistent_execution"
    )
    explicit_model = StateModel(explicit_off, context=ModelContext(condition=1))
    implicit_model = StateModel(implicit_off, context=ModelContext(condition=1))
    stimuli = np.asarray(
        [
            [0.20, 0.30, 0.40, 0.50],
            [0.70, 0.20, 0.60, 0.10],
            [0.45, 0.55, 0.25, 0.75],
            [0.80, 0.65, 0.35, 0.15],
        ],
        dtype=float,
    )

    for stimulus, choice, feedback in zip(
        stimuli,
        [1, 2, 1, 1],
        [1.0, 0.0, 0.0, 1.0],
    ):
        explicit_trial = explicit_model.begin_trial(stimulus)
        implicit_trial = implicit_model.begin_trial(stimulus)
        np.testing.assert_allclose(explicit_trial.prior, implicit_trial.prior)
        np.testing.assert_allclose(
            read_choice_probabilities_from_model(
                explicit_model,
                explicit_trial.perceived_stimulus,
            ),
            read_choice_probabilities_from_model(
                implicit_model,
                implicit_trial.perceived_stimulus,
            ),
        )
        explicit_posterior, _, _ = explicit_model.complete_trial(choice, feedback)
        implicit_posterior, _, _ = implicit_model.complete_trial(choice, feedback)
        np.testing.assert_allclose(explicit_posterior, implicit_posterior)


def test_h5_subject_level_execution_on_uses_one_persistent_rule() -> None:
    config = apply_fixed_hyperparams_to_engine_config(
        _deterministic_h5_config(),
        {EXECUTION_ENABLED_PATH: True},
    )
    model = StateModel(config, context=ModelContext(condition=1))
    transition = model.engine.modules["hypo_transitions_mod"]

    assert transition.persistent_execution_enabled is True
    assert transition.execution_switch_scale == pytest.approx(0.20)
    assert transition.prior_assignment_method == "similarity_transport"
    assert resolve_executed_hypothesis(model.engine) in set(
        transition.active.tolist()
    )

    trial = model.begin_trial(np.asarray([0.20, 0.30, 0.40, 0.50]))
    executed = resolve_executed_hypothesis(model.engine)
    assert executed is not None
    observed = read_choice_probabilities_from_model(
        model,
        trial.perceived_stimulus,
    )
    expected = model.partition_model.get_category_probabilities(
        hypo=int(executed),
        data=([trial.perceived_stimulus], [1], [1.0]),
        beta=float(model.engine.beta[int(executed)]),
        distance_mode=model.engine.distance_mode,
    )[:, 0]
    np.testing.assert_allclose(observed, expected / np.sum(expected))

    saved = transition.state_dict()
    saved_executed = int(executed)
    alternative = next(
        int(value)
        for value in transition.active
        if int(value) != saved_executed
    )
    transition.executed_hypothesis = alternative
    transition.load_state_dict(saved)
    assert int(transition.executed_hypothesis) == saved_executed


def test_h5_execution_state_is_marginalized_by_particle_filter() -> None:
    config = apply_fixed_hyperparams_to_engine_config(
        _deterministic_h5_config(),
        {EXECUTION_ENABLED_PATH: True},
    )
    config["inference"].update(
        {
            "particle_count": 8,
            "resample_threshold_fraction": 0.95,
        }
    )
    stimuli = np.asarray(
        [
            [0.20, 0.30, 0.40, 0.50],
            [0.70, 0.20, 0.60, 0.10],
            [0.45, 0.55, 0.25, 0.75],
            [0.80, 0.65, 0.35, 0.15],
            [0.10, 0.85, 0.50, 0.40],
        ],
        dtype=float,
    )
    result = run_inference_backend(
        engine_config=config,
        subject_id=103,
        condition=1,
        stimulus=stimuli,
        choices=np.asarray([1, 2, 1, 1, 2]),
        feedback=np.asarray([1.0, 0.0, 0.0, 1.0, 1.0]),
        inference_seed=20260817,
        processed_data_dir=Path("."),
    )

    predictive = np.asarray(
        result.state_probabilities["executed_probability"], dtype=float
    )
    filtered = np.asarray(
        result.state_probabilities["filtered_executed_probability"], dtype=float
    )
    assert predictive.shape == filtered.shape == (stimuli.shape[0], 29)
    np.testing.assert_allclose(predictive.sum(axis=1), 1.0)
    np.testing.assert_allclose(filtered.sum(axis=1), 1.0)
    probabilities = np.asarray(
        result.observation_probabilities["prior_t"], dtype=float
    )
    assert probabilities.shape == (stimuli.shape[0], 2)
    assert np.all(np.isfinite(probabilities))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.Bayesian_state.model import ModelContext, ModuleRole, StateModel
from src.Bayesian_state.model.modules.hypothesis_transition.feedback_reactive import (
    FeedbackReactiveHypothesisTransitionModule,
)
from src.Bayesian_state.model.modules.hypothesis_transition.nested_feedback_accumulator import (
    NestedFeedbackAccumulatorHypothesisTransitionModule,
)
from src.Bayesian_state.model.readout import read_choice_probabilities_from_model
from src.Bayesian_state.simulation.parameters import (
    infer_fixed_hyperparams_from_engine_config,
)


ROOT = Path(__file__).resolve().parents[2]
NESTED_CONFIG = (
    ROOT
    / "configs/model_struct/pmh_model_cond1_0815_h4_nested_feedback_accumulator.yaml"
)
REACTIVE_CONFIG = (
    ROOT / "configs/model_struct/pmh_model_cond1_0815_h3_feedback_reactive.yaml"
)


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


def _reactive(seed: int = 17):
    engine = _TinyEngine()
    module = FeedbackReactiveHypothesisTransitionModule(
        engine,
        capacity=3,
        init_hypotheses=[0, 1, 2],
        feedback_reactive_controller={
            "event_after_correct": 0.20,
            "event_after_error": 0.60,
            "initial_event_probability": 0.20,
            "global_search": 0.30,
        },
        module_seed=seed,
    )
    return engine, module


def _nested(
    *,
    accumulator_logit_gain: float,
    global_search_failure_gain: float = 0.0,
    accumulator_decay: float = 0.80,
    event_after_correct: float = 0.20,
    event_after_error: float = 0.60,
    seed: int = 17,
):
    engine = _TinyEngine()
    module = NestedFeedbackAccumulatorHypothesisTransitionModule(
        engine,
        capacity=3,
        init_hypotheses=[0, 1, 2],
        nested_feedback_accumulator_controller={
            "event_after_correct": event_after_correct,
            "event_after_error": event_after_error,
            "initial_event_probability": event_after_correct,
            "global_search": 0.30,
            "accumulator_decay": accumulator_decay,
            "accumulator_logit_gain": accumulator_logit_gain,
            "global_search_failure_gain": global_search_failure_gain,
            "initial_failure": 0.0,
        },
        module_seed=seed,
    )
    return engine, module


def _advance(engine, module, feedback: float) -> None:
    engine.posterior = np.asarray(engine.prior, dtype=float).copy()
    module.record_outcome((np.asarray([0.5]), 1, feedback))
    module.process()


def test_zero_accumulator_gain_is_exact_reactive_transition_boundary() -> None:
    reactive_engine, reactive = _reactive()
    nested_engine, nested = _nested(accumulator_logit_gain=0.0)

    reactive.process()
    nested.process()
    feedback_sequence = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0]

    for feedback in feedback_sequence:
        _advance(reactive_engine, reactive, feedback)
        _advance(nested_engine, nested, feedback)

        reactive_event = reactive.transition_log[-1]
        nested_event = nested.transition_log[-1]
        assert np.array_equal(reactive.active, nested.active)
        assert np.allclose(reactive_engine.prior, nested_engine.prior)
        assert nested.current_event_probability == pytest.approx(
            reactive.current_event_probability
        )
        assert nested.current_m == pytest.approx(reactive.current_m)
        assert nested.current_g == pytest.approx(reactive.current_g)
        assert nested.failure_pressure == pytest.approx(
            reactive.failure_pressure
        )
        assert nested.mastery_evidence == pytest.approx(
            reactive.mastery_evidence
        )
        for key in (
            "replacement_count",
            "dropped_hypotheses",
            "new_hypotheses",
            "active_after",
            "swap_probability",
            "predictive_m",
            "predictive_g",
        ):
            assert nested_event[key] == pytest.approx(reactive_event[key])


def test_zero_gain_matches_reactive_end_to_end_choice_predictions() -> None:
    reactive_config = yaml.safe_load(REACTIVE_CONFIG.read_text(encoding="utf-8"))
    nested_config = yaml.safe_load(NESTED_CONFIG.read_text(encoding="utf-8"))
    for config in (reactive_config, nested_config):
        config["modules"]["perception_mod"]["kwargs"] = {
            "features": 4,
            "mean": [0.0, 0.0, 0.0, 0.0],
            "std": [0.0, 0.0, 0.0, 0.0],
            "module_seed": 11,
        }
        config["modules"]["hypo_transitions_mod"]["kwargs"][
            "module_seed"
        ] = 29

    context = ModelContext(condition=1)
    reactive_model = StateModel(reactive_config, context=context)
    nested_model = StateModel(nested_config, context=context)
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
    choices = [1, 2, 1, 1, 2]
    feedback = [1.0, 0.0, 0.0, 1.0, 1.0]

    for stimulus, choice, outcome in zip(stimuli, choices, feedback):
        reactive_trial = reactive_model.begin_trial(stimulus)
        nested_trial = nested_model.begin_trial(stimulus)
        reactive_probability = read_choice_probabilities_from_model(
            reactive_model,
            reactive_trial.perceived_stimulus,
        )
        nested_probability = read_choice_probabilities_from_model(
            nested_model,
            nested_trial.perceived_stimulus,
        )

        assert np.allclose(reactive_trial.prior, nested_trial.prior)
        assert np.allclose(reactive_trial.beta, nested_trial.beta)
        assert np.allclose(reactive_probability, nested_probability)

        reactive_posterior, _, _ = reactive_model.complete_trial(choice, outcome)
        nested_posterior, _, _ = nested_model.complete_trial(choice, outcome)
        assert np.allclose(reactive_posterior, nested_posterior)


def test_equal_reactive_events_and_zero_gain_are_constant_boundary() -> None:
    engine, module = _nested(
        accumulator_logit_gain=0.0,
        event_after_correct=0.30,
        event_after_error=0.30,
    )
    module.process()

    probabilities = []
    for feedback in (1.0, 0.0, 0.0, 1.0, 0.0):
        _advance(engine, module, feedback)
        probabilities.append(module.transition_log[-1]["swap_probability"])

    assert probabilities == pytest.approx([0.30] * len(probabilities))


def test_positive_accumulator_gain_builds_and_retains_failure_pressure() -> None:
    engine, module = _nested(
        accumulator_logit_gain=2.0,
        accumulator_decay=0.80,
        event_after_correct=0.10,
        event_after_error=0.50,
    )
    module.process()

    probabilities = []
    failures = []
    for feedback in (0.0, 0.0, 1.0, 1.0):
        _advance(engine, module, feedback)
        probabilities.append(module.current_event_probability)
        failures.append(module.failure_pressure)
        assert module.current_g == pytest.approx(0.30)

    assert failures == pytest.approx([0.20, 0.36, 0.288, 0.2304])
    assert probabilities[0] > 0.50
    assert probabilities[1] > probabilities[0]
    # Accumulated failure persists after a correct response, then decays.
    assert probabilities[2] > 0.10
    assert 0.10 < probabilities[3] < probabilities[2]


def test_global_search_gain_reuses_failure_without_changing_event_boundary() -> None:
    engine, module = _nested(
        accumulator_logit_gain=0.0,
        global_search_failure_gain=0.50,
        accumulator_decay=0.80,
        event_after_correct=0.20,
        event_after_error=0.60,
    )
    module.process()

    events = []
    global_search = []
    failures = []
    for feedback in (0.0, 0.0, 1.0):
        _advance(engine, module, feedback)
        events.append(module.current_event_probability)
        global_search.append(module.current_g)
        failures.append(module.failure_pressure)

    assert failures == pytest.approx([0.20, 0.36, 0.288])
    # c_acc=0 leaves the one-step reactive event probability unchanged.
    assert events == pytest.approx([0.60, 0.60, 0.20])
    assert global_search == pytest.approx(
        [0.30 + 0.70 * 0.50 * failure for failure in failures]
    )
    assert global_search[1] > global_search[0]
    # The shared failure trace persists after a correct response.
    assert 0.30 < global_search[2] < global_search[1]
    signals = module._transition_signals()
    assert signals["accumulator_logit_gain"] == pytest.approx(0.0)
    assert signals["global_search_failure_gain"] == pytest.approx(0.50)
    assert signals["event_accumulator_active"] is False
    assert signals["global_range_accumulator_active"] is True


def test_nested_accumulator_state_restores_pending_outcome_and_future_rng() -> None:
    engine, module = _nested(
        accumulator_logit_gain=1.5,
        global_search_failure_gain=0.70,
        seed=31,
    )
    module.process()
    engine.posterior = np.asarray(engine.prior, dtype=float).copy()
    module.record_outcome((np.asarray([0.5]), 1, 0.0))

    saved_module = module.state_dict()
    saved_prior = np.asarray(engine.prior, dtype=float).copy()
    saved_posterior = np.asarray(engine.posterior, dtype=float).copy()
    module.process()
    first_event = dict(module.transition_log[-1])
    first_prior = np.asarray(engine.prior, dtype=float).copy()

    engine.prior = saved_prior.copy()
    engine.posterior = saved_posterior.copy()
    module.load_state_dict(saved_module)
    module.process()
    second_event = module.transition_log[-1]

    assert module.controller_mode == "nested_feedback_accumulator_v1"
    assert np.allclose(engine.prior, first_prior)
    assert second_event["swap_probability"] == pytest.approx(
        first_event["swap_probability"]
    )
    assert second_event["predictive_g"] == pytest.approx(
        first_event["predictive_g"]
    )
    assert second_event["replacement_count"] == first_event["replacement_count"]
    assert second_event["active_after"] == first_event["active_after"]


def test_nested_accumulator_config_assembles_on_reactive_boundary() -> None:
    config = yaml.safe_load(NESTED_CONFIG.read_text(encoding="utf-8"))
    model = StateModel(
        config,
        context=ModelContext(condition=1, subject_id=103),
    )
    transition = model.engine.get_module(
        ModuleRole.HYPOTHESIS_TRANSITION,
        required=True,
    )

    assert isinstance(
        transition,
        NestedFeedbackAccumulatorHypothesisTransitionModule,
    )
    assert transition.accumulator_logit_gain == pytest.approx(0.0)
    assert transition.global_search_failure_gain == pytest.approx(0.0)
    assert transition.accumulator_active is False
    assert transition.current_g == pytest.approx(0.20)
    assert transition.capacity == 3
    fixed = infer_fixed_hyperparams_from_engine_config(config)
    controller_path = (
        "engine.modules.hypo_transitions_mod.kwargs."
        "nested_feedback_accumulator_controller"
    )
    assert fixed[controller_path]["global_search_failure_gain"] == pytest.approx(
        0.0
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"accumulator_logit_gain": -0.1}, "non-negative"),
        ({"accumulator_decay": 1.0}, "smaller than 1"),
        ({"global_search_failure_gain": -0.1}, "finite probability"),
        ({"global_search_failure_gain": 1.1}, "finite probability"),
    ],
)
def test_nested_accumulator_rejects_invalid_accumulation_parameters(
    overrides: dict[str, float],
    message: str,
) -> None:
    controller = {
        "event_after_correct": 0.20,
        "event_after_error": 0.60,
        "global_search": 0.30,
        **overrides,
    }
    with pytest.raises(ValueError, match=message):
        NestedFeedbackAccumulatorHypothesisTransitionModule(
            _TinyEngine(),
            capacity=3,
            nested_feedback_accumulator_controller=controller,
        )

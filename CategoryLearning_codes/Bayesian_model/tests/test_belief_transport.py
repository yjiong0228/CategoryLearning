from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from CategoryLearning_codes.Bayesian_model.inference import run_inference_backend
from CategoryLearning_codes.Bayesian_model.model import ModelContext, StateModel
from CategoryLearning_codes.Bayesian_model.model.readout import (
    read_choice_probabilities_from_model,
    resolve_executed_hypothesis,
)
from CategoryLearning_codes.Bayesian_model.model.modules.hypothesis_transition.contracts import (
    HypothesisSelection,
)
from CategoryLearning_codes.Bayesian_model.model.modules.hypothesis_transition.feedback_reactive import (
    FeedbackReactiveHypothesisTransitionModule,
)
from CategoryLearning_codes.Bayesian_model.simulation.parameters import (
    apply_fixed_hyperparams_to_engine_config,
    infer_fixed_hyperparams_from_engine_config,
)
from CategoryLearning_codes.Bayesian_model.simulation.provenance import build_model_provenance


EXECUTION_ENABLED_PATH = (
    "engine.modules.hypo_transitions_mod.kwargs."
    "persistent_execution.enabled"
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


def test_mass_preserving_transport_moves_only_dropped_belief_mass() -> None:
    _, module = _module(
        "mass_preserving_similarity_transport",
        global_search=0.30,
    )
    posterior = np.asarray([0.80, 0.15, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
    selection = _one_replacement()
    module._pending_transition = {"posterior": posterior}

    prior = module.assign_prior(None, selection)

    np.testing.assert_allclose(
        prior,
        [0.80, 0.15, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0],
    )
    assert module._pending_transition["prior_assignment_method"] == (
        "mass_preserving_similarity_transport"
    )
    assert module._pending_transition["prior_transport_fraction"] == pytest.approx(
        1.0 / 3.0
    )
    assert module._pending_transition["newcomer_prior_mass"] == pytest.approx(0.05)


def test_mass_preserving_transport_semantically_splits_aggregate_removed_mass() -> None:
    _, module = _module(
        "mass_preserving_similarity_transport",
        global_search=0.30,
    )
    posterior = np.asarray([0.70, 0.20, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0])
    selection = HypothesisSelection.from_active_sets(
        [0, 1, 2],
        [0, 3, 4],
        replacement_pairs=((1, 3), (2, 4)),
    )
    module._pending_transition = {"posterior": posterior}

    prior = module.assign_prior(None, selection)

    assert prior[0] == pytest.approx(0.70)
    assert np.sum(prior[[3, 4]]) == pytest.approx(0.30)
    assert np.all(prior[[3, 4]] > 0.0)
    assert np.all(prior[[1, 2, 5, 6, 7]] == 0.0)


def test_mass_preserving_transport_is_exact_carryover_without_replacement() -> None:
    _, module = _module("mass_preserving_similarity_transport")
    posterior = np.asarray([0.80, 0.15, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
    module._pending_transition = {"posterior": posterior}
    selection = HypothesisSelection.from_active_sets([0, 1, 2], [0, 1, 2])

    prior = module.assign_prior(None, selection)

    np.testing.assert_allclose(prior, posterior)
    assert module._pending_transition["newcomer_prior_mass"] == 0.0


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









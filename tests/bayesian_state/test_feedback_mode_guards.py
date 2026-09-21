"""Keep legacy overlapping feedback evidence out of new ternary runs."""

import numpy as np
import pytest

from src.Bayesian_state.hypothesis_space.observation_model import (
    ContinuousPartition,
    DiscreteRulePartition,
    ObservationLikelihood,
)
from src.Bayesian_state.model import ModelContext, StateModel
from src.Bayesian_state.simulation.provenance import build_model_provenance


@pytest.fixture(scope="module")
def partition():
    return ContinuousPartition(4, 4, boundary_dykstra_backend="python")


@pytest.mark.parametrize("mode", ["category_feedback", "category", "legacy"])
def test_partial_feedback_requires_explicit_legacy_mode(partition, mode):
    evaluator = ObservationLikelihood(
        partition, distance_mode="prototype", feedback_likelihood_mode=mode,
    )
    with pytest.raises(ValueError, match="legacy_category_feedback"):
        evaluator.process(([0.2, 0.3, 0.4, 0.5], 1, 0.5), [0, 1], beta=5.)


def test_partition_default_cannot_bypass_partial_feedback_guard(partition):
    with pytest.raises(ValueError, match="hierarchical_pairing"):
        partition.calc_likelihood(
            [0], ([[0.2, 0.3, 0.4, 0.5]], [1], [0.5]),
            distance_mode="prototype",
        )


def test_explicit_legacy_preserves_historical_evidence(partition, monkeypatch):
    # An independent historical reference, including the overlapping zero event.
    probabilities = np.tile(np.array([.1, .2, .3, .4])[:, None], (1, 3))
    monkeypatch.setattr(
        partition, "get_category_probabilities", lambda **kwargs: probabilities,
    )
    actual = partition.calc_likelihood(
        [0], ([[.2, .3, .4, .5]] * 3, [1, 1, 1], [1., .5, 0.]),
        distance_mode="prototype", normalized=False,
        feedback_likelihood_mode="legacy_category_feedback",
    )
    np.testing.assert_allclose(actual[:, 0], [.1, .5, .9])


@pytest.mark.parametrize("n_cats", [2, 4])
def test_binary_category_evidence_is_unchanged(n_cats):
    partition = ContinuousPartition(4, n_cats)
    probabilities = np.full((n_cats, 2), 0.3 / (n_cats - 1))
    probabilities[0] = [.7, .7]
    actual = partition._feedback_likelihood_from_category_probabilities(
        0, probabilities, ([None, None], [1, 1], [1., 0.]),
    )
    np.testing.assert_allclose(actual, [.7, .3])


def test_explicit_legacy_mode_is_recorded_in_provenance(partition):
    config = {
        "partition": {"class": "src.Bayesian_state.hypothesis_space.observation_model.continuous_partition.ContinuousPartition",
                      "kwargs": {"n_dims": 4, "n_cats": 4}},
        "likelihood": {"distance_mode": "prototype",
                       "feedback_likelihood_mode": "legacy_category_feedback"},
    }
    evaluator = ObservationLikelihood(partition, **config["likelihood"])
    assert evaluator.feedback_likelihood_mode == "legacy_category_feedback"
    provenance = build_model_provenance(config, repeat_aggregation="mean_probability")
    assert provenance["resolved"]["encoding"]["feedback_likelihood_mode"] == "legacy_category_feedback"


def test_discrete_task_does_not_accept_continuous_legacy_mode():
    with pytest.raises(ValueError, match="feedback_likelihood_mode"):
        ObservationLikelihood(
            DiscreteRulePartition(), feedback_likelihood_mode="legacy_category_feedback",
        )


@pytest.mark.parametrize("mode", ["category_feedback", "legacy_category_feedback"])
def test_condition3_rejects_non_joint_configuration_before_trials(partition, mode):
    config = {
        "likelihood": {"distance_mode": "prototype", "feedback_likelihood_mode": mode},
        "modules": {"memory": {"class": "src.Bayesian_state.model.modules.memory.BayesianMemoryModule"}},
        "agenda": ["memory"],
    }
    with pytest.raises(ValueError, match="condition 3 requires hierarchical_pairing"):
        StateModel(config, context=ModelContext(condition=3), partition=partition)

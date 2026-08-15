from __future__ import annotations

import math

import numpy as np
import pytest

from src.Bayesian_state.simulation.execution import (
    compute_metrics_from_category_probabilities,
)
from src.Bayesian_state.simulation.provenance import build_model_provenance
from src.Bayesian_state.simulation.results import SingleRunResult
from src.Bayesian_state.simulation.runner import aggregate_simulation_runs


def _run(selected_probability: float, seed: int) -> SingleRunResult:
    choices = np.asarray([1, 1, 2, 2, 1, 2], dtype=int)
    probabilities = np.empty((choices.size, 2), dtype=float)
    probabilities[:, 0] = np.where(
        choices == 1,
        selected_probability,
        1.0 - selected_probability,
    )
    probabilities[:, 1] = 1.0 - probabilities[:, 0]
    metrics = compute_metrics_from_category_probabilities(
        probabilities,
        choices=choices,
        feedback=np.ones(choices.size, dtype=float),
        categories=choices,
        target_probs=np.eye(2, dtype=float)[choices - 1],
        window_size=2,
        loss_metric="choice_nll",
    )
    return SingleRunResult(
        params={},
        mean_error=float(metrics["mean_error"]),
        metrics_by_mode={"prior_t": metrics},
        selection_prediction_mode="prior_t",
        loss_metric="choice_nll",
        loss_delta=None,
        trajectory_seed=seed,
    )


def test_probability_repeat_aggregation_scores_the_mean_prediction() -> None:
    runs = [_run(0.9, 1), _run(0.5, 2)]
    result = aggregate_simulation_runs(
        runs,
        params={},
        subject_id=1,
        condition=1,
        window_size=2,
        selection_prediction_mode="prior_t",
        simulation_repeats=2,
        simulation_point_seed=123,
        keep_logs=True,
        compute_statistics=False,
        repeat_aggregation="mean_probability",
    )

    assert result.mean_error == pytest.approx(-math.log(0.7))
    assert result.aggregation_diagnostics["mean_run_error"] == pytest.approx(
        (-math.log(0.9) - math.log(0.5)) / 2.0
    )
    assert result.mean_error < result.aggregation_diagnostics["mean_run_error"]
    np.testing.assert_allclose(
        result.metrics_by_mode["prior_t"]["pred_category_probs"][:, 0],
        np.where(np.asarray([1, 1, 2, 2, 1, 2]) == 1, 0.7, 0.3),
    )


def test_mean_loss_remains_the_backward_compatible_default() -> None:
    runs = [_run(0.9, 1), _run(0.5, 2)]
    result = aggregate_simulation_runs(
        runs,
        params={},
        subject_id=1,
        condition=1,
        window_size=2,
        selection_prediction_mode="prior_t",
        simulation_repeats=2,
        simulation_point_seed=123,
        keep_logs=False,
        compute_statistics=False,
    )
    assert result.repeat_aggregation == "mean_loss"
    assert result.mean_error == pytest.approx(
        (-math.log(0.9) - math.log(0.5)) / 2.0
    )


def test_model_provenance_records_capacity_and_stochastic_initialization() -> None:
    engine = {
        "provenance": {"model_id": "test"},
        "partition": {"class": "example.Partition", "kwargs": {"n_cats": 2}},
        "inference": {"backend": "particle_filter", "particle_count": 128},
        "likelihood": {
            "distance_mode": "boundary",
            "beta_source": "fixed",
            "default_beta": 5.0,
        },
        "modules": {
            "hypo_transitions_mod": {
                "class": "example.Transition",
                "kwargs": {"capacity": 3},
            },
            "beta_mod": {"class": "example.Beta", "kwargs": {"beta_init": 5.0}},
        },
        "choice_readout": {"kwargs": {"method": "expectation", "power": 1.0}},
        "agenda": ["hypo_transitions_mod", "beta_mod"],
    }
    provenance = build_model_provenance(
        engine,
        repeat_aggregation="mean_probability",
    )
    assert provenance["resolved"]["workspace_capacity"] == 3
    initialization = provenance["resolved"]["initialization"]
    assert initialization["method"] == "prior_weighted_without_replacement"
    assert initialization["integrated_by_particle_filter"] is True
    assert len(provenance["model_config_sha256"]) == 64

"""Probability aggregation and scientific-boundary tests for the bounded study."""
from pathlib import Path

import numpy as np
import yaml

from src.Bayesian_state.workflows.analysis.model_1031_finalization import (
    aggregate, bank, engine_config, energy_score, path_features, project, task_feedback,
)


POLICY = yaml.safe_load(Path("configs/shared/model1031_finalization_20261101.yaml").read_text())


def test_sequence_likelihood_is_averaged_before_taking_log():
    records = [dict(nll=-np.log(.2), train_nll=-np.log(.5)),
               dict(nll=-np.log(.06), train_nll=-np.log(.2))]
    np.testing.assert_allclose(aggregate(records)["nll"], -np.log(.13))
    np.testing.assert_allclose(aggregate(records, conditional=True)["nll"], -np.log(.26 / .7))
    assert not np.isclose(aggregate(records, conditional=True)["nll"], -np.log((.4 + .3) / 2))


def test_nested_projection_and_canonical_configuration_are_identical():
    p = bank(POLICY)[0]
    compact = project(p, POLICY["variants"]["compact"])
    for condition in (1, 2, 3):
        assert engine_config(condition, compact, POLICY["variants"]["full"]) == engine_config(
            condition, p, POLICY["variants"]["compact"])
    assert compact["learning_rate"] == compact["execution_inertia"] == 0.
    constant = project(p, POLICY["variants"]["constant_search"])
    assert constant["error_gain"] == constant["failure_decay"] == 0.


def test_bank_is_deterministic_and_preserves_prespecified_pilot_points():
    from src.Bayesian_state.workflows.analysis.model_1031_subject_pilot import bank as pilot_bank

    previous = yaml.safe_load(Path(POLICY["search"]["previous_prespecified_bank"]).read_text())
    points = bank(POLICY)
    assert len(points) == POLICY["search"]["initial_candidates"]
    assert points == bank(POLICY)
    assert points[:16] == [p["parameters"] for p in pilot_bank(previous)]


def test_cond3_metrics_keep_species_family_and_score_distinct():
    categories = np.array([1, 2, 3, 4])
    choices = np.array([2, 1, 4, 3])
    np.testing.assert_array_equal(task_feedback(choices, categories, 3), .5)
    np.testing.assert_array_equal(task_feedback(choices, categories, 2), 0.)
    features = path_features(choices, categories, 3, block=2)
    np.testing.assert_allclose(features[0, :2], 0.)
    np.testing.assert_allclose(features[0, -2:], [1., .5])
    assert energy_score(np.repeat(features, 3, axis=0), features[0]) == 0.

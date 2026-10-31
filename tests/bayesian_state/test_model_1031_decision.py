"""Decision safeguards for finite, unequal-length subject trajectories."""
from pathlib import Path

import numpy as np
import yaml

from src.Bayesian_state.workflows.analysis.model_1031_decision import choose, summarize


POLICY = yaml.safe_load(Path("configs/shared/model1031_finalization_20261101_v2.yaml").read_text())


def rows():
    return [dict(subject=c, condition=c, variant=v, test_nll_per_trial=1.,
                 mc_se_per_trial=.0001, energy=.2, max_system_weight=.25,
                 numerical_status="within_working_tolerance")
            for c in (1, 2, 3) for v in ("full", "compact", "constant_search")]


def test_equal_predictions_choose_fewer_active_parameters():
    result = choose(rows(), POLICY)
    assert result["chosen"] == "constant_search"
    assert all(c["eligible"] for c in result["checks"].values())


def test_one_condition_failure_prevents_global_average_from_hiding_it():
    data = rows()
    for row in data:
        if row["variant"] == "constant_search":
            row["test_nll_per_trial"] += .04 if row["condition"] == 3 else -.02
    result = choose(data, POLICY)
    assert not result["checks"]["constant_search"]["eligible"]
    assert result["chosen"] == "compact"


def test_numerical_uncertainty_and_generation_are_separate_safeguards():
    data = rows()
    for row in data:
        if row["variant"] == "constant_search":
            row["mc_se_per_trial"] = .1
        if row["variant"] == "compact":
            row["energy"] = .23
    result = choose(data, POLICY)
    assert result["chosen"] == "full"
    assert result["checks"]["constant_search"]["nll_upper_difference"] > .01
    assert result["checks"]["compact"]["relative_energy_deterioration"] > .1


def test_summary_weights_subjects_equally_not_by_trial_count():
    data = rows()
    for row in data:
        row["test_trials"] = row["subject"] * 100
        row["test_nll_per_trial"] = row["subject"]
    result = summarize(data)
    np.testing.assert_allclose([s["nll"] for s in result.values()], 2.)

"""Decision safeguards for finite, unequal-length subject trajectories."""
from pathlib import Path

import numpy as np
import yaml

from src.Bayesian_state.workflows.analysis.model_1031_decision import choose, summarize, independent_seed_aggregate
from src.Bayesian_state.workflows.analysis.model_1031_finalization import aggregate


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


def test_correlated_particle_count_copies_do_not_inflate_independent_repeats():
    independent = [dict(nll=x, train_nll=.5 * x, job=dict(seed=i, particles=256))
                   for i, x in enumerate((2., 2.3, 2.8, 3.))]
    duplicated = independent + [dict(r, job=dict(r["job"], particles=512)) for r in independent]
    grouped = independent_seed_aggregate(duplicated)
    reference = aggregate(independent, conditional=True)
    np.testing.assert_allclose([grouped["nll"], grouped["mc_se"]], [reference["nll"], reference["mc_se"]])
    assert grouped["independent_seeds"] == 4 and grouped["raw_systems"] == 8
    assert aggregate(duplicated, conditional=True)["mc_se"] < grouped["mc_se"]


def test_shared_seed_error_is_not_reduced_as_if_subjects_were_independent():
    data = rows()
    for row in data:
        row["mc_se_per_trial"] = .02
    np.testing.assert_allclose([r["numerical_se"] for r in summarize(data).values()], .02)

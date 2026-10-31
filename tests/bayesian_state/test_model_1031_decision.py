"""Decision safeguards for finite, unequal-length subject trajectories."""
from pathlib import Path

import numpy as np
import yaml

from src.Bayesian_state.workflows.analysis.model_1031_decision import (
    choose, fixed_configuration_score, summarize, independent_seed_aggregate,
)
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


def test_equal_parameter_count_prefers_fewer_eligible_control_mechanisms():
    data = [dict(subject=c, condition=c, variant=v, test_nll_per_trial=1. + (.005 if v == "constant_retention" else 0.),
                 mc_se_per_trial=.0001, energy=.2, max_system_weight=.25,
                 numerical_status="within_working_tolerance")
            for c in (1, 2, 3) for v in ("compact", "constant_retention", "similarity_retrieval")]
    decision = choose(data, POLICY)
    assert decision["chosen"] == "constant_retention"
    assert decision["complexity"]["constant_retention"]["optional_controls"] == 1


def test_exact_shared_configurations_cancel_comparison_error():
    data = rows()
    for row in data:
        row["candidate"] = dict(id=f"shared_{row['subject']}")
        row["mc_se_per_trial"] = .1
    result = choose(data, POLICY)
    assert result["chosen"] == "constant_search"
    assert result["checks"]["constant_search"]["numerical_se_of_mean_difference"] == 0.
    assert result["checks"]["constant_search"]["exact_shared_configurations"] == 3


def test_fixed_configurations_pool_independent_systems_from_both_structures():
    records = [dict(nll=x, train_nll=.5 * x,
                    job=dict(id=str(i), subject=1, config="same", seed=i))
               for i, x in enumerate([2., 3., 4., 5.])]
    rows = records + [dict(nll=100., train_nll=50., job=dict(id="other", subject=2, config="same", seed=0))]
    result = fixed_configuration_score(rows, 1, "same")
    reference = aggregate(records, conditional=True)
    np.testing.assert_allclose([result["nll"], result["mc_se"]], [reference["nll"], reference["mc_se"]])
    assert result["pooled_job_ids"] == ["0", "1", "2", "3"]

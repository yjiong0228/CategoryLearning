"""Check interpretation-sensitive boundaries of the registered evidence matrix."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.Bayesian_state.evaluation import model_0923_finalization_evidence as evidence


def test_accuracy_counts_exact_categories_and_keeps_the_declared_phase():
    conditional = dict(choices=np.array([1, 2, 2, 4]), categories=np.array([1, 2, 3, 4]))
    autonomous = dict(choices=np.array([[1, 2, 1, 4], [2, 3, 3, 4]]),
                      categories=conditional["categories"], feedback=np.ones((2, 4)) * .5)
    score = evidence.phase_accuracy(conditional, autonomous, 2, 4)
    assert score["observed_accuracy"] == .5
    assert score["autonomous_accuracy"] == .75
    assert score["autonomous_accuracy_mcse"] == .25
    assert score["paths_reaching_observed_accuracy"] == 1.
    assert score["autonomous_accuracy_q05"] == pytest.approx(.525)


def contrast_table():
    return pd.DataFrame([dict(subject=s, contrast="precision", evidence="behavior",
        reading="not_applicable", window="test", domain="not_applicable",
        subset="all_trials", metric="choice_nll", difference=d, mcse=e, trials=t)
        for s, d, e, t in ((108, 0., 0., 192), (205, -.03, .003, 224), (306, .06, .004, 256))])


def test_subjects_receive_equal_weight_and_two_uncertainties_stay_separate():
    table = evidence.equal_subject_summary(contrast_table(), [108, 205, 306]).iloc[0]
    assert table.equal_subject_difference == pytest.approx(.01)
    assert table.equal_subject_mcse == pytest.approx(.005 / 3)
    assert table.between_subject_sd == pytest.approx(np.std([0, -.03, .06], ddof=1))
    assert table.between_subject_sd > 20 * table.equal_subject_mcse
    assert table.subjects == 3


@pytest.mark.parametrize("fault", ["missing", "duplicate", "nonfinite", "negative_error"])
def test_incomplete_or_invalid_contrasts_cannot_produce_a_three_person_average(fault):
    table = contrast_table()
    if fault == "missing":
        table = table.iloc[:2]
    elif fault == "duplicate":
        table.loc[2, "subject"] = 205
    elif fault == "nonfinite":
        table.loc[1, "difference"] = np.nan
    else:
        table.loc[1, "mcse"] = -.01
    with pytest.raises(ValueError):
        evidence.equal_subject_summary(table, [108, 205, 306])


def test_participant_simulation_covariance_cannot_be_silently_ignored():
    bank = {108: dict(conditional={1, 2}, autonomous={11, 12}),
            205: dict(conditional={3, 4}, autonomous={13, 14})}
    evidence.check_disjoint_subject_seeds(bank)
    bank[205]["autonomous"].add(12)
    with pytest.raises(ValueError):
        evidence.check_disjoint_subject_seeds(bank)


def test_empty_oral_stratum_keeps_the_full_denominator_and_no_average():
    table = contrast_table().assign(evidence="oral", reports=[0, 12, 24])
    table.loc[0, ["difference", "mcse"]] = np.nan
    result = evidence.equal_subject_summary(table, [108, 205, 306]).iloc[0]
    assert result.subjects == 3 and result.contributing_subjects == 2
    assert np.isnan(result.equal_subject_difference) and np.isnan(result.equal_subject_mcse)
    table.loc[0, "reports"] = 1
    with pytest.raises(ValueError):
        evidence.equal_subject_summary(table, [108, 205, 306])


def reuse_fixture():
    frozen = dict(status="frozen_before_test", smoke=False, settings=dict(subjects=[108]),
        nominees=[dict(subject=108, model=arm, candidate=dict(id=arm, hyperparams={
            evidence.equivalence.RATE: 0., evidence.equivalence.BETA: 10., "capacity": 2}))
            for arm in ("POOL", "SEARCH")])
    specification = dict(summary="old/summary", fit="new/fit", oral_midpoint="old/midpoint",
                         oral_relative="old/relative")
    proof = dict(status="complete", subject=108, new_full_evaluation_executed=False,
        new_full_state_replay_executed=False, search_reestimated=True,
        original_summary=str(evidence.ROOT / specification["summary"]),
        revised_fit=str(evidence.ROOT / specification["fit"]),
        reused_oral={r: str(evidence.ROOT / specification["oral_" + r]) for r in evidence.READINGS})
    return proof, specification, frozen, deepcopy(frozen)


def test_equivalence_is_point_specific_and_binds_both_fixed_oral_readings():
    proof, spec, old, new = reuse_fixture()
    evidence.check_reuse(proof, spec, old, new)
    proof["reused_oral"]["relative"] = str(evidence.ROOT / "another/coding")
    with pytest.raises(ValueError):
        evidence.check_reuse(proof, spec, old, new)
    proof, spec, old, new = reuse_fixture()
    new["nominees"][0]["candidate"]["hyperparams"][evidence.equivalence.RATE] = .01
    with pytest.raises(ValueError):
        evidence.check_reuse(proof, spec, old, new)


def test_missing_input_hash_is_not_substituted_by_current_contents(tmp_path):
    file = tmp_path / "artifact.csv"
    file.write_text("old")
    sha = evidence.digest(file)
    file.write_text("changed")
    with pytest.raises(ValueError):
        evidence.EvidenceReader().bind(file, sha)

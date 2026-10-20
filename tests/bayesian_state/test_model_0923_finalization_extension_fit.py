"""Protect the recommendation gate, saved starts and executed extension identity."""
from copy import deepcopy

import pytest

from src.Bayesian_state.optimization.search.adaptive_proposals import candidate
from src.Bayesian_state.workflows.analysis import model_0923_finalization_extension_fit as fit


def decision():
    return (dict(status="frozen_recommendation", candidate="SEARCH_LOG25", common_h="ONLINE",
                 precision_storage="RETAIN", execution="MIX", rationale="Recorded development trade-off.",
                 extension_subjects=[110, 209, 309], previous_extension_influence="unknown"),
            dict(status="complete", complete_primary_matrix=True, subjects=[108, 205, 306],
                 collected_cells=9, missing_subjects=[]))


@pytest.mark.parametrize("name", list(fit.CANDIDATES))
def test_any_registered_candidate_can_be_recommended_without_requiring_every_metric_to_win(name):
    recommendation, evidence = decision()
    recommendation["candidate"] = name
    assert fit.validate_recommendation(recommendation, evidence) == name


@pytest.mark.parametrize("fault", ["not_frozen", "new_candidate", "new_h", "partial_matrix",
                                  "fewer_cells", "missing_subject", "new_sample", "unknown_influence_lost"])
def test_formal_extension_cannot_run_on_a_provisional_or_changed_decision(fault):
    recommendation, evidence = decision()
    if fault == "not_frozen":
        recommendation["status"] = "provisional"
    elif fault == "new_candidate":
        recommendation["candidate"] = "SEARCH_LOG50"
    elif fault == "new_h":
        recommendation["common_h"] = "SEPARATE"
    elif fault == "partial_matrix":
        evidence["complete_primary_matrix"] = False
    elif fault == "fewer_cells":
        evidence["collected_cells"] = 3
    elif fault == "missing_subject":
        evidence["missing_subjects"] = [306]
    elif fault == "new_sample":
        recommendation["extension_subjects"] = [111, 210, 310]
    else:
        recommendation["previous_extension_influence"] = "independent"
    with pytest.raises(ValueError):
        fit.validate_recommendation(recommendation, evidence)


def neutral_fixture():
    points = [candidate(dict(coordinate=i), "saved_support_start") for i in range(40)]
    return dict(count_per_subject=40, banks=[dict(subject=s, rows=deepcopy(points),
        required_reference_id=points[0]["id"]) for s in (110, 209, 309)])


def test_single_recommended_arm_receives_the_saved_points_and_required_reference():
    setup = neutral_fixture()
    settings = dict(subjects=[110, 209, 309], arms={"SEARCH": {}})
    banks, anchors = fit.selected_banks(setup, settings, False)
    assert set(banks) == {(s, "SEARCH") for s in settings["subjects"]}
    assert all(len(rows) == 40 for rows in banks.values())
    assert all(anchors[key] == rows[0] for key, rows in banks.items())
    banks[110, "SEARCH"][0]["hyperparams"]["coordinate"] = -1
    assert setup["banks"][0]["rows"][0]["hyperparams"]["coordinate"] == 0
    settings["subjects"] = [110]
    smoke, _ = fit.selected_banks(setup, settings, True)
    assert smoke[110, "SEARCH"] == setup["banks"][0]["rows"][:6]


@pytest.mark.parametrize("fault", ["changed_point", "duplicate_point", "missing_point", "changed_reference"])
def test_saved_start_identity_and_budget_cannot_drift(fault):
    setup = neutral_fixture()
    row = setup["banks"][0]
    if fault == "changed_point":
        row["rows"][0]["hyperparams"]["coordinate"] = 9
    elif fault == "duplicate_point":
        row["rows"][1] = deepcopy(row["rows"][0])
    elif fault == "missing_point":
        row["rows"].pop()
    else:
        row["required_reference_id"] = row["rows"][1]["id"]
    with pytest.raises(ValueError):
        fit.selected_banks(setup, dict(subjects=[110], arms={"SEARCH": {}}), False)


def job(name="SEARCH_LOG25", smoke=False):
    arm, mode, scale = fit.CANDIDATES[name]
    return dict(extension_candidate=name, extension_scope="software_validation" if smoke else "extension_validation",
        extension_recommendation=None if smoke else "registered/recommendation.json", smoke=smoke,
        model=arm, arm_settings=dict(precision=mode, rate_scale=scale, upper=25.))


@pytest.mark.parametrize("name", list(fit.CANDIDATES))
def test_worker_binds_the_recommended_precision_and_proposal(name):
    fit.check_job(job(name))
    fit.check_job(job(name, smoke=True))


@pytest.mark.parametrize("fault", ["mode", "scale", "arm", "scope", "recommendation", "smoke"])
def test_mismatched_worker_is_rejected_before_cognitive_execution(fault):
    item = job()
    if fault == "mode":
        item["arm_settings"]["precision"] = "fast"
    elif fault == "scale":
        item["arm_settings"]["rate_scale"] = 10.
    elif fault == "arm":
        item["model"] = "POOL"
    elif fault == "scope":
        item["extension_scope"] = "development_comparison"
    elif fault == "recommendation":
        item["extension_recommendation"] = None
    else:
        item["smoke"] = True
    with pytest.raises(ValueError):
        fit.check_job(item)


def test_smoke_config_cannot_be_used_as_a_formal_fit():
    config = dict(setup="unused", recommendation=None, subjects=[110], parallel_budget=1,
                  scope="software_validation", software_candidate="SEARCH_LOG25")
    with pytest.raises(ValueError, match="--smoke"):
        fit.prepare_settings(config, False)

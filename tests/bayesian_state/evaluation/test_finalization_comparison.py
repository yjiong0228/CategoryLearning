"""Protect task/seed comparability while allowing the declared precision change."""
from copy import deepcopy

import pandas as pd
import pytest

from src.Bayesian_state.evaluation import model_0923_finalization_comparison as behavior
from src.Bayesian_state.workflows.analysis import model_0923_finalization_oral as oral


def policy():
    settings = dict(subjects=[108], arms={a: dict(precision="fast", rate_scale=10., upper=25.,
                                                 rates=[0., .1]) for a in ("POOL", "SEARCH")},
                    scope="common_policy_development_belief_refit", parallel_budget=64,
                    belief_mechanism="ONLINE", belief_intervention="online_retention_evidence",
                    seed=42, budgets=dict(evaluate=dict(seeds=8, particles=512)))
    old = dict(smoke=False, settings=settings, source_sha256={"shared.py": "same"})
    new = deepcopy(old)
    new["settings"].update(finalization_precision_family="LOG25",
                           finalization_scope="development_comparison", parallel_budget=28)
    for arm in new["settings"]["arms"].values():
        arm.update(precision="log", rate_scale=1.)
    new["source_sha256"][behavior.JOINT] = behavior.digest(behavior.ROOT / behavior.JOINT)
    protocols = [dict(deepcopy(f), initial_banks=[
        dict(subject=108, model=a, rows=[dict(id="reference", hyperparams=dict(gamma=.6))])
        for a in ("POOL", "SEARCH")]) for f in (old, new)]
    return [old, new], protocols, [dict(gamma=[0., .6, 1.]), dict(gamma=[0., .6, 1.])]


def test_precision_comparison_preserves_policy_despite_worker_cap_change():
    frozen, protocols, spaces = policy()
    assert behavior.match_policy(*frozen, *protocols, *spaces) == ("ADD10", "LOG25")
    assert behavior.contrast_kind(("ADD10", "LOG25"), ("SEARCH", "SEARCH")).startswith("precision")
    assert behavior.contrast_kind(("LOG25", "LOG25"), ("POOL", "SEARCH")).startswith("proposal")


@pytest.mark.parametrize("fault", ["seed", "particles", "source", "extra_source", "starts",
                                  "duplicate", "support", "upper", "rate_scale", "rates",
                                  "belief", "scope", "smoke", "missing_adapter"])
def test_rejects_changes_beyond_declared_precision_and_scheduling(fault):
    frozen, protocols, spaces = policy()
    new = frozen[1]
    if fault == "seed": new["settings"]["seed"] = 43
    elif fault == "particles": new["settings"]["budgets"]["evaluate"]["particles"] = 256
    elif fault == "source": new["source_sha256"]["shared.py"] = "changed"
    elif fault == "extra_source": new["source_sha256"]["another_model.py"] = "added"
    elif fault == "starts": protocols[1]["initial_banks"][0]["rows"][0]["hyperparams"]["gamma"] = 1.
    elif fault == "duplicate": protocols[1]["initial_banks"].append(deepcopy(protocols[1]["initial_banks"][0]))
    elif fault == "support": spaces[1]["gamma"] = [0., 1.]
    elif fault == "upper": new["settings"]["arms"]["SEARCH"]["upper"] = 50.
    elif fault == "rate_scale": new["settings"]["arms"]["SEARCH"]["rate_scale"] = 10.
    elif fault == "rates": new["settings"]["arms"]["SEARCH"]["rates"] = [0., 1.]
    elif fault == "belief": new["settings"]["belief_mechanism"] = "SEPARATE"
    elif fault == "scope": new["settings"]["finalization_scope"] = "software_validation"
    elif fault == "smoke": new["smoke"] = True
    elif fault == "missing_adapter": new["source_sha256"].pop(behavior.JOINT)
    for key in ("settings", "source_sha256", "smoke"):
        protocols[1][key] = deepcopy(new[key])
    with pytest.raises(ValueError):
        behavior.match_policy(*frozen, *protocols, *spaces)


@pytest.mark.parametrize("families,arms", [
    (("ADD10", "LOG25"), ("POOL", "SEARCH")),
    (("LOG25", "ADD10"), ("SEARCH", "SEARCH")),
    (("ADD10", "ADD10"), ("SEARCH", "SEARCH")),
])
def test_contrast_requires_one_explicit_direction(families, arms):
    with pytest.raises(ValueError):
        behavior.contrast_kind(families, arms)


def identities():
    frozen, _, _ = policy()
    return [dict(scientific=dict(particles=512, trials=640, training_trials=448,
                                  timing="after_choice_before_current_feedback"),
                 archived_pf_sha256="same", task_sha256={"training": "same"},
                 oral_manifest_sha256="same", mechanism=dict(online_belief=True,
                 retain_learned_precision=True), structural_intervention="retained_rule_precision",
                 parameter_estimation_scope="prefix", parameter_candidate_policy="same",
                 arm_settings=f["settings"]["arms"]["SEARCH"], model_sources=f["source_sha256"])
            for f in frozen]


def test_precision_oral_readout_accepts_only_matched_task_and_reports():
    before, after = identities()
    oral.match_identities(before, after, ("ADD10", "LOG25"))
    after["scientific"]["timing"] = "after_feedback"
    with pytest.raises(ValueError):
        oral.match_identities(before, after, ("ADD10", "LOG25"))


@pytest.mark.parametrize("key,value", [
    ("oral_manifest_sha256", "another_reading"),
    ("task_sha256", {"training": "different"}),
    ("mechanism", {"online_belief": False, "retain_learned_precision": True}),
    ("parameter_estimation_scope", "whole"),
])
def test_oral_readout_rejects_coding_or_selection_changes(key, value):
    before, after = identities()
    after[key] = value
    with pytest.raises(ValueError):
        oral.match_identities(before, after, ("ADD10", "LOG25"))


def test_proposal_label_normalization_preserves_original_trial_and_coverage_fields():
    trials = pd.DataFrame(dict(arm=["POOL"], state_trial=[17], text=["neck short"],
                               phase=["training"], full_report=[True]))
    rows = [dict(arm="POOL", full_reports=1)]
    scored = (trials, rows, {"same": "estimates"})
    normalized = oral.normalized_group(scored, "POOL")
    assert trials.arm.tolist() == ["POOL"] and rows[0]["arm"] == "POOL"
    pd.testing.assert_frame_equal(trials.drop(columns="arm"), normalized[0].drop(columns="arm"))
    assert normalized[1][0]["full_reports"] == 1 and normalized[2] is scored[2]
    with pytest.raises(ValueError):
        oral.normalized_group(scored, "SEARCH")

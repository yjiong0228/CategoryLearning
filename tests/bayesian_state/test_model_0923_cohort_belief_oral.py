"""Cohort oral contrasts preserve mechanism, reporting time and filter identity."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.Bayesian_state.workflows.analysis import model_0923_cohort_belief_oral as oral


def records(mechanism, arm="SEARCH"):
    intervention = oral.behavior.cohort.driver.INTERVENTIONS[mechanism]
    context = dict(parameter_estimation_scope="prefix", evaluation_use=oral.states.USE,
        parameter_candidate_policy="projected_common_search_policy_behavior_refit",
        structural_intervention="retained_rule_precision", belief_mechanism=mechanism,
        belief_intervention=intervention, cohort_fit="fit", cohort_summary="summary",
        cohort_manifest_sha256="hash", refit_nomination_sha256="nomination",
        fixed_coordinates={"transfer": .01}, selected_parameters=[dict(subject=107, arm=arm, parameter_id="point")])
    job = dict(context, model=arm, reported_arm=("SEPARATE_" if mechanism == "SEPARATE" else "") + arm,
        cohort_belief_mechanism=mechanism, probe_mechanism=mechanism, online_belief=mechanism == "ONLINE",
        candidate=dict(hyperparams={oral.behavior.cohort.separate.GAMMA: .6}))
    record = dict(subject=107, arm=arm, proposal_arm=arm, particles=4, trials=8, particle_transitions=32,
        parameter_estimation_scope="prefix", evaluation_use=oral.states.USE, exact_behavior_replay=True,
        timing="after_choice_before_current_feedback", data_use="behavior_only; no_oral_input",
        online_belief=mechanism == "ONLINE", online_evidence_matches_retained_state=True,
        online_reconstructions=28, separate_belief=mechanism == "SEPARATE", belief_updates=28,
        belief_gamma=.6, belief_matches_saved_state=True, belief_intervention=intervention)
    protocol = dict(context, scientific_scope=True, settings=dict(seeds_per_arm=2))
    manifest = dict(context, window_definition="original_chronological_boundary")
    return job, record, protocol, manifest


@pytest.mark.parametrize("mechanism", ["ONLINE", "SEPARATE"])
@pytest.mark.parametrize("arm", ["POOL", "SEARCH"])
def test_both_cohort_mechanisms_and_proposals_have_explicit_state_identity(mechanism, arm):
    oral.check_state(*records(mechanism, arm), mechanism)


@pytest.mark.parametrize("mechanism", ["ONLINE", "SEPARATE"])
@pytest.mark.parametrize("index,key,value", [(0, "cohort_belief_mechanism", "OTHER"),
    (0, "reported_arm", "OTHER"), (0, "parameter_estimation_scope", "full"),
    (1, "evaluation_use", "other"), (1, "particle_transitions", 28),
    (1, "proposal_arm", "POOL"), (1, "exact_behavior_replay", False),
    (1, "timing", "after_feedback"), (1, "data_use", "oral_fitting"),
    (2, "belief_intervention", "other"), (3, "belief_mechanism", "OTHER")])
def test_changed_execution_or_data_use_is_rejected(mechanism, index, key, value):
    args = list(records(mechanism)); args[index][key] = value
    with pytest.raises(ValueError): oral.check_state(*args, mechanism)


@pytest.mark.parametrize("mechanism,key,value", [("ONLINE", "online_reconstructions", 32),
    ("ONLINE", "online_evidence_matches_retained_state", False),
    ("SEPARATE", "belief_updates", 32), ("SEPARATE", "belief_gamma", 1),
    ("SEPARATE", "belief_matches_saved_state", False)])
def test_completed_feedback_counts_and_gamma_remain_bound(mechanism, key, value):
    args = list(records(mechanism)); args[1][key] = value
    with pytest.raises(ValueError): oral.check_state(*args, mechanism)


def state_fixture(tmp_path):
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))
    job, record, protocol, manifest = records("SEPARATE")
    source = tmp_path / "state"; manifest["source"] = str(source)
    folder = tmp_path / "data"; save(folder / "prepared.json", dict(subject=107))
    files = []
    for i in range(2):
        original = tmp_path / "evaluation" / f"worker_{i}.json"
        save(original, dict(model="SEARCH", folder=str(folder)))
        own = dict(deepcopy(job), original_job=str(original))
        save(source / "jobs" / f"worker_{i}.json", own)
        save(source / "states" / f"worker_{i}.json", dict(record, seeds=[i]))
        files.append(original)
    save(source / "protocol.json", protocol)
    return manifest, files, save


def test_selected_state_group_matches_actual_evaluation_filters(tmp_path):
    manifest, files, _ = state_fixture(tmp_path)
    bound = oral.validate_states(manifest, 107, "SEARCH", "SEPARATE", files)
    assert len(bound) == 5


@pytest.mark.parametrize("fault", ["context", "missing", "duplicate", "different", "smoke"])
def test_mismatched_states_and_validation_fixtures_cannot_enter_scientific_comparison(tmp_path, fault):
    manifest, files, save = state_fixture(tmp_path)
    source = Path(manifest["source"])
    if fault == "context": manifest["refit_nomination_sha256"] = "changed"
    elif fault == "missing": (source / "states" / "worker_1.json").unlink()
    elif fault == "duplicate":
        p = source / "jobs" / "worker_1.json"; value = json.loads(p.read_text())
        value["original_job"] = str(files[0]); save(p, value)
    elif fault == "different": files = files[:1]
    elif fault == "smoke":
        p = source / "protocol.json"; value = json.loads(p.read_text())
        value["scientific_scope"] = False; save(p, value)
    with pytest.raises(ValueError): oral.validate_states(manifest, 107, "SEARCH", "SEPARATE", files)


def binding_fixture(tmp_path, monkeypatch):
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))
    identities, manifests, metadata = [], [], {}
    for mechanism in oral.behavior.MECHANISMS:
        base = tmp_path / mechanism; fit, summary, source = [base / p for p in ("fit", "summary", "states")]
        fixed = {oral.behavior.cohort.separate.TRANSFER: .01}
        point = dict(id=mechanism, hyperparams=fixed)
        settings = dict(belief_mechanism=mechanism, belief_intervention=oral.behavior.cohort.driver.INTERVENTIONS[mechanism],
            scope="common_policy_development_belief_refit", parallel_budget=64, fixed_coordinates=fixed,
            subjects=[107], arms={"POOL": {}, "SEARCH": {}}, budgets={"evaluate": {"particles": 512, "seeds": 8}})
        hashes = {"shared.py": "hash"}
        frozen = dict(status="frozen_before_test", smoke=False, settings=settings, source_sha256=hashes,
                      nominees=[dict(subject=107, model=a, candidate=point) for a in settings["arms"]])
        protocol = dict(smoke=False, settings=settings, source_sha256=hashes, initial_banks=[
            dict(subject=107, model=a, rows=[dict(id="common", hyperparams=fixed)]) for a in settings["arms"]])
        save(fit / "nominees.json", frozen); save(fit / "protocol.json", protocol)
        nomination = oral.digest(fit / "nominees.json")
        save(fit / "completion.json", dict(status="complete", smoke=False, test_choices_used=False, nomination_sha256=nomination))
        save(fit / "parameter_spaces.json", {"gamma": [0, .6, 1]})
        save(summary / "cohort_manifest.json", dict(status="fixture"))
        save(source / "protocol.json", dict(settings=dict(summary=str(summary), source=str(base / "evaluation"))))
        manifest = dict(cohort_fit=str(fit), cohort_summary=str(summary), source=str(source),
            cohort_manifest_sha256=oral.digest(summary / "cohort_manifest.json"), refit_nomination_sha256=nomination,
            fixed_coordinates=fixed, belief_mechanism=mechanism, belief_intervention=settings["belief_intervention"],
            selected_parameters=[dict(subject=107, arm=a, parameter_id=mechanism) for a in settings["arms"]])
        identities.append(dict(candidate=deepcopy(point), model_sources=hashes))
        manifests.append(manifest)
        metadata[str(summary)] = dict(manifest, refit_replay_inputs={})
    monkeypatch.setattr(oral.states, "source_jobs", lambda settings, source, smoke:
                        ([], metadata[settings["summary"]]))
    monkeypatch.setattr(oral, "validate_states", lambda *args: {})
    return identities, manifests, metadata, save


def test_binds_nominees_and_common_cohort_search_before_oral_comparison(tmp_path, monkeypatch):
    identities, manifests, _, _ = binding_fixture(tmp_path, monkeypatch)
    inputs = oral.bind_refits(*identities, *manifests, 107, "SEARCH")
    assert len(inputs) == 12


@pytest.mark.parametrize("fault", ["later_choices", "nominee", "summary", "fixed", "state_nominee", "support"])
def test_oral_contrast_rejects_changed_fit_or_selection_source(tmp_path, monkeypatch, fault):
    identities, manifests, metadata, save = binding_fixture(tmp_path, monkeypatch)
    revised = manifests[1]; fit = Path(revised["cohort_fit"])
    if fault == "later_choices":
        p = fit / "completion.json"; value = json.loads(p.read_text()); value["test_choices_used"] = True; save(p, value)
    elif fault == "nominee": identities[1]["candidate"]["id"] = "changed"
    elif fault == "summary": revised["cohort_manifest_sha256"] = "changed"
    elif fault == "fixed": revised["fixed_coordinates"] = {}
    elif fault == "state_nominee": metadata[revised["cohort_summary"]]["refit_nomination_sha256"] = "changed"
    elif fault == "support": save(fit / "parameter_spaces.json", {"gamma": [0, 1]})
    with pytest.raises(ValueError): oral.bind_refits(*identities, *manifests, 107, "SEARCH")

"""Cohort oral replay preserves behavioral nominees, task boundaries and scope."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.Bayesian_state.workflows.analysis import model_0923_cohort_belief_states as states
from src.Bayesian_state.utils.seeding import stable_seed


def execution(mechanism, arm, condition):
    prefix = states.comparison_helpers().cohort.separate.PREFIX
    gamma = states.comparison_helpers().cohort.separate.GAMMA
    transfer = states.comparison_helpers().cohort.separate.TRANSFER
    point = dict(id="point", hyperparams={gamma: .6, prefix + "intervention.decay": .8, transfer: .01})
    settings = dict(arms={arm: dict(precision="learned")}, mechanism=dict(kind="E"), fixed_coordinates={transfer: .01})
    folder = Path("/tmp/cohort_belief_state_test")
    planned = dict(phase="evaluate", smoke=False, model=arm, candidate=deepcopy(point), seeds=[12], particles=4,
                   folder=str(folder), arm_settings=settings["arms"][arm], mechanism=settings["mechanism"])
    metadata = dict(subject=107, condition=condition, trials=8, training_trials=4, test_trials=4)
    record = dict(metadata, **{k: deepcopy(planned[k]) for k in ("candidate", "model", "seeds", "particles")})
    intervention = states.comparison_helpers().cohort.driver.INTERVENTIONS[mechanism]
    actual = dict(deepcopy(planned), structural_intervention="retained_rule_precision", retain_precision=True,
        retention_reference_inputs={}, belief_intervention=intervention, online_belief=mechanism == "ONLINE",
        probe_mechanism=mechanism, cohort_belief_mechanism=mechanism, fixed_coordinates=settings["fixed_coordinates"])
    precision = dict(structural_intervention="retained_rule_precision", retained_precision=True, transitions=32)
    belief = dict(enabled=True, belief_intervention=intervention, fixed_coordinates=settings["fixed_coordinates"],
        exact_transition_timing=True, evidence_matches_retained_state=True, reconstructions=28,
        gamma=.6, delta=[.8], updates=28, completed_feedback_timing_exact=True, belief_matches_saved_state=True)
    return [planned, actual, point, settings, mechanism, 107, arm, 12, 4, folder, metadata,
            record, precision, belief, {}]


@pytest.mark.parametrize("mechanism", ["ONLINE", "SEPARATE"])
@pytest.mark.parametrize("arm", ["POOL", "SEARCH"])
@pytest.mark.parametrize("condition", [1, 2, 3])
def test_original_evaluation_can_be_replayed(mechanism, arm, condition):
    states.check_filter(*execution(mechanism, arm, condition))


@pytest.mark.parametrize("mechanism", ["ONLINE", "SEPARATE"])
@pytest.mark.parametrize("index,key,value", [(1, "candidate", {"id": "changed"}), (1, "cohort_belief_mechanism", "OTHER"),
    (0, "smoke", True), (0, "particles", 8), (0, "seeds", [13]), (10, "subject", 207),
    (11, "condition", 3), (11, "trials", 16), (11, "training_trials", 5),
    (12, "transitions", 28), (13, "enabled", False)])
def test_changed_parameter_or_data_boundary_is_rejected(mechanism, index, key, value):
    args = execution(mechanism, "SEARCH", 2); args[index][key] = value
    with pytest.raises(ValueError): states.check_filter(*args)


def context(mechanism):
    return dict(parameter_estimation_scope="prefix", evaluation_use=states.USE,
        parameter_candidate_policy="projected_common_search_policy_behavior_refit",
        structural_intervention="retained_rule_precision", belief_mechanism=mechanism,
        belief_intervention=states.comparison_helpers().cohort.driver.INTERVENTIONS[mechanism],
        cohort_fit="fit", cohort_summary="summary", cohort_manifest_sha256="summary-hash",
        refit_nomination_sha256="nomination-hash", fixed_coordinates={"transfer": .01},
        selected_parameters=[dict(subject=107, arm="POOL", parameter_id="point")], scientific_scope=True)


@pytest.mark.parametrize("mechanism", ["ONLINE", "SEPARATE"])
def test_context_describes_the_executed_belief_mechanism(mechanism):
    metadata, note = states.estimation_context(context(mechanism))
    assert metadata["belief_mechanism"] == mechanism and mechanism in note
    assert metadata["window_definition"] == "original_chronological_boundary"


@pytest.mark.parametrize("fault", ["scientific_scope", "refit_nomination_sha256", "cohort_manifest_sha256",
                                   "selected_parameters", "cohort_summary", "parameter_estimation_scope"])
def test_smoke_or_incomplete_selection_cannot_enter_oral_inference(fault):
    protocol = context("SEPARATE")
    protocol[fault] = False if fault == "scientific_scope" else ""
    with pytest.raises(ValueError): states.estimation_context(protocol)


def source_fixture(tmp_path, monkeypatch, mechanism):
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))

    source, summary, fit, data = [tmp_path / name for name in ("evaluation", "summary", "fit", "data")]
    sample = execution(mechanism, "POOL", 1)
    original = dict(sample[3], subjects=[107, 207, 306], belief_mechanism=mechanism,
        arms={a: dict(precision="learned") for a in ("POOL", "SEARCH")},
        particle_overrides={306: 8}, budgets={"evaluate": dict(particles=4, seeds=2)},
        seed=19, protocol="fixture")
    plans, nominees = [], []
    hashes = {"frozen.py": "hash"}
    for subject, condition in ((107, 1), (207, 2), (306, 3)):
        folder = data / f"subject_{subject}"
        metadata = dict(subject=subject, condition=condition, trials=8, training_trials=4, test_trials=4)
        save(folder / "prepared.json", metadata)
        for arm in original["arms"]:
            point = execution(mechanism, arm, condition)[2]
            nominees.append(dict(subject=subject, model=arm, candidate=point))
            for i in range(2):
                planned, actual, _, _, _, _, _, _, _, _, _, record, precision, belief, _ = execution(mechanism, arm, condition)
                particles = 8 if subject == 306 else 4
                seed = stable_seed(dict(protocol="fixture", base=19, stage="evaluate", subject=subject, repeat=i))
                name = f"evaluate_S{subject}_{arm}_point_C{i:03d}"
                target = source / "conditional" / "chunks" / name
                planned.update(name=name, folder=str(folder), target=str(target), source_sha256=hashes,
                               seeds=[seed], particles=particles)
                actual.update(planned); record.update(metadata, seeds=[seed], particles=particles)
                precision["transitions"] = particles * 8
                belief.update(reconstructions=particles * 7, updates=particles * 7)
                plans.append(planned); save(source / "jobs" / (name + ".json"), actual)
                sidecar = ".online.json" if mechanism == "ONLINE" else ".belief.json"
                for suffix, value in ((".json", record), (".precision.json", precision), (sidecar, belief)):
                    save(target.with_suffix(suffix), value)
    manifest = dict(fit=str(fit), evaluation=str(source))
    frozen = dict(settings=original, data=str(data), source_sha256=hashes, nominees=nominees)
    save(fit / "cohort_belief_protocol.json", dict(input_sha256={}))
    save(fit / "nominees.json", frozen)
    save(source / "protocol.json", dict(jobs=plans))
    save(source / "completion.json", dict(status="complete"))
    save(summary / "cohort_manifest.json", dict(status="fixture"))
    monkeypatch.setattr(states.comparison_helpers().retained, "load_summary", lambda path: (manifest, frozen, {}, {}))
    monkeypatch.setattr(states.comparison_helpers(), "bind_summary", lambda *args: {})
    settings = dict(summary=str(summary), sample_role="development", workers=128,
        subjects=original["subjects"], arms=list(original["arms"]), seeds_per_arm=2)
    return settings, source, frozen


@pytest.mark.parametrize("mechanism", ["ONLINE", "SEPARATE"])
def test_full_bank_keeps_all_subjects_and_smoke_preserves_both_arms(tmp_path, monkeypatch, mechanism):
    settings, source, _ = source_fixture(tmp_path, monkeypatch, mechanism)
    files, metadata = states.source_jobs(settings, source, False)
    assert len(files) == 12 and len(metadata["selected_parameters"]) == 6
    assert metadata["scientific_scope"] is True and metadata["belief_mechanism"] == mechanism
    files, metadata = states.source_jobs(settings, source, True)
    assert len(files) == 2 and {states.load(p)["model"] for p in files} == {"POOL", "SEARCH"}
    assert metadata["scientific_scope"] is False


@pytest.mark.parametrize("fault", ["missing_filter", "selected_parameter", "subjects", "seeds"])
def test_selection_preserves_the_complete_original_filter_bank(tmp_path, monkeypatch, fault):
    settings, source, frozen = source_fixture(tmp_path, monkeypatch, "SEPARATE")
    if fault == "missing_filter": next((source / "jobs").glob("*.json")).unlink()
    elif fault == "selected_parameter": frozen["nominees"][0]["candidate"]["id"] = "changed"
    elif fault == "subjects": settings["subjects"] = [107]
    elif fault == "seeds": settings["seeds_per_arm"] = 1
    with pytest.raises(ValueError): states.source_jobs(settings, source, False)

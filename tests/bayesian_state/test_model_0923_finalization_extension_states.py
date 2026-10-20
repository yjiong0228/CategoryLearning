"""Extension states preserve the selected candidate, history and evidence scope."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from src.Bayesian_state.workflows.analysis import model_0923_finalization_extension_states as states
from src.Bayesian_state.utils.seeding import stable_seed

spec = importlib.util.spec_from_file_location(
    "extension_state_fixtures", Path(__file__).with_name("test_model_0923_cohort_belief_states.py"))
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


def execution(candidate, condition):
    arm, mode, scale = states.extension.CANDIDATES[candidate]
    args = fixtures.execution("ONLINE", arm, condition)
    definition = dict(precision=mode, rate_scale=scale, upper=25.)
    args[3]["arms"][arm] = definition
    extra = dict(extension_scope="extension_validation", extension_candidate=candidate,
                 extension_setup="registered_setup", extension_recommendation="recommendation.json")
    args[3].update(extra)
    for job in args[:2]:
        job["arm_settings"] = deepcopy(definition)
    args[1].update(extra)
    return args


@pytest.mark.parametrize("candidate", sorted(states.extension.CANDIDATES))
@pytest.mark.parametrize("condition", [1, 2, 3])
def test_original_filter_preserves_frozen_candidate(candidate, condition):
    states.check_filter(*execution(candidate, condition))


@pytest.mark.parametrize("index,key,value", [
    (1, "extension_candidate", "SEARCH_ADD10"), (1, "extension_scope", "software_validation"),
    (1, "extension_recommendation", "another.json"), (1, "extension_setup", "another_setup"),
    (1, "model", "POOL"), (1, "candidate", {"id": "another"}), (1, "seeds", [13]),
    (1, "particles", 8), (11, "training_trials", 5), (11, "trials", 9),
    (12, "transitions", 28), (13, "exact_transition_timing", False),
])
def test_parameter_history_or_feedback_changes_are_rejected(index, key, value):
    args = execution("SEARCH_LOG25", 2)
    args[index][key] = value
    with pytest.raises(ValueError):
        states.check_filter(*args)


def context(tmp_path):
    fit, summary = tmp_path / "fit", tmp_path / "summary"
    fit.mkdir(); summary.mkdir()
    (fit / "nominees.json").write_text("{}")
    (summary / "extension_manifest.json").write_text("{}")
    return dict(parameter_estimation_scope="prefix", evaluation_use=states.USE,
        parameter_candidate_policy=states.POLICY, belief_mechanism="ONLINE",
        belief_intervention="online_retention_evidence", structural_intervention="retained_rule_precision",
        extension_candidate="SEARCH_LOG25", extension_scope="extension_validation", scientific_scope=True,
        previous_development_influence="unknown", settings=dict(subjects=[110, 209, 309], arms=["SEARCH"]),
        extension_fit=str(fit), extension_summary=str(summary), extension_recommendation="recommendation.json",
        extension_manifest_sha256=states.digest(summary / "extension_manifest.json"),
        refit_nomination_sha256=states.digest(fit / "nominees.json"), fixed_coordinates={"transfer": .01},
        selected_parameters=[dict(subject=s, arm="SEARCH", parameter_id="point") for s in (110, 209, 309)],
        refit_replay_inputs={})


def test_oral_context_retains_uncertainty_and_original_scope(tmp_path):
    metadata, note = states.estimation_context(context(tmp_path))
    assert metadata["previous_development_influence"] == "unknown"
    assert metadata["window_definition"] == "original_chronological_boundary"
    assert "带噪声" in note and "既往开发影响未确定" in note


@pytest.mark.parametrize("key,value", [
    ("scientific_scope", False), ("extension_scope", "software_validation"),
    ("previous_development_influence", "strict_holdout"), ("extension_candidate", "SEARCH_BAD"),
    ("belief_mechanism", "SEPARATE"), ("selected_parameters", []),
    ("settings", {"subjects": [110, 209], "arms": ["SEARCH"]}),
    ("settings", {"subjects": [110, 209, 309], "arms": ["POOL", "SEARCH"]}),
])
def test_short_or_mislabelled_result_cannot_enter_oral_inference(tmp_path, key, value):
    protocol = context(tmp_path); protocol[key] = value
    with pytest.raises(ValueError):
        states.estimation_context(protocol)


def test_changed_nominee_cannot_enter_oral_inference(tmp_path):
    protocol = context(tmp_path)
    (Path(protocol["extension_fit"]) / "nominees.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="nomination changed"):
        states.estimation_context(protocol)


def source_fixture(tmp_path, monkeypatch, candidate):
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    summary, fit, data, source = [tmp_path / n for n in ("summary", "fit", "data", "evaluation")]
    arm = states.extension.CANDIDATES[candidate][0]
    original = deepcopy(execution(candidate, 1)[3])
    original.update(subjects=[110, 209, 309], seed=19, protocol="fixture",
        particle_overrides={309: 8}, budgets={"evaluate": dict(particles=4, seeds=2)})
    plans, nominees = [], []
    hashes = {"frozen.py": "hash"}
    for subject, condition in ((110, 1), (209, 2), (309, 3)):
        args = execution(candidate, condition)
        point = args[2]
        nominees.append(dict(subject=subject, model=arm, candidate=point))
        folder = data / f"subject_{subject}"
        meta = dict(args[10], subject=subject)
        save(folder / "prepared.json", meta)
        particles = original["particle_overrides"].get(subject, 4)
        for i in range(2):
            seed = stable_seed(dict(protocol="fixture", base=19, stage="evaluate", subject=subject, repeat=i))
            name = f'evaluate_S{subject}_{arm}_{point["id"]}_C{i:03d}'
            target = source / "conditional" / name
            updates = dict(name=name, folder=str(folder), seeds=[seed], particles=particles,
                           target=str(target), source_sha256=hashes)
            planned, actual = [dict(deepcopy(job), **updates) for job in args[:2]]
            plans.append(planned)
            save(source / "jobs" / (name + ".json"), actual)
            record = dict(args[11], **meta, seeds=[seed], particles=particles)
            save(target.with_suffix(".json"), record)
            save(target.with_suffix(".precision.json"), dict(args[12], transitions=particles * 8))
            save(target.with_suffix(".online.json"), dict(args[13], reconstructions=particles * 7))
    save(source / "protocol.json", dict(jobs=plans))
    save(fit / "extension_protocol.json", dict(input_sha256={}))
    save(fit / "nominees.json", {})
    save(summary / "extension_manifest.json", {})
    frozen = dict(settings=original, data=str(data), source_sha256=hashes, nominees=nominees)
    monkeypatch.setattr(states, "load_summary", lambda _: (dict(fit=str(fit), evaluation=str(source)), frozen, {}))
    settings = dict(summary=str(summary), subjects=[110, 209, 309], arms=[arm],
                    seeds_per_arm=2, sample_role="development", workers=128)
    return settings, source, frozen


@pytest.mark.parametrize("candidate", sorted(states.extension.CANDIDATES))
def test_all_original_filters_and_registered_participants_are_selected(tmp_path, monkeypatch, candidate):
    settings, source, _ = source_fixture(tmp_path, monkeypatch, candidate)
    files, metadata = states.source_jobs(settings, source, False)
    assert len(files) == 6 and len(metadata["selected_parameters"]) == 3
    assert metadata["extension_candidate"] == candidate and metadata["scientific_scope"] is True
    assert metadata["previous_development_influence"] == "unknown"
    short, context = states.source_jobs(settings, source, True)
    assert len(short) == 1 and context["selected_parameters"][0]["subject"] == 110
    assert context["scientific_scope"] is False


@pytest.mark.parametrize("fault", ["seed", "missing", "duplicate_nominee", "subject_subset", "two_arms"])
def test_replay_selection_cannot_change_filter_bank(tmp_path, monkeypatch, fault):
    settings, source, frozen = source_fixture(tmp_path, monkeypatch, "SEARCH_LOG25")
    file = next((source / "jobs").glob("*.json"))
    if fault == "seed":
        value = json.loads(file.read_text()); value["seeds"] = [999]; file.write_text(json.dumps(value))
    elif fault == "missing":
        file.unlink()
    elif fault == "duplicate_nominee":
        frozen["nominees"].append(deepcopy(frozen["nominees"][0]))
    elif fault == "subject_subset":
        settings["subjects"] = [110]
    else:
        settings["arms"] = ["POOL", "SEARCH"]
    with pytest.raises(ValueError):
        states.source_jobs(settings, source, False)

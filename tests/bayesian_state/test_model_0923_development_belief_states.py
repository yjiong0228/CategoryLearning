"""The registered observer preserves complete original filter and trial scope."""
import importlib.util
import json
from pathlib import Path

import pytest

from src.Bayesian_state.workflows.analysis import model_0923_development_belief_states as states

# Reuse the existing independent evaluation fixtures rather than defining a
# second version of valid feedback counts, seeds or parameter-selection rules.
spec = importlib.util.spec_from_file_location(
    "development_belief_state_fixtures", Path(__file__).with_name("test_model_0923_cohort_belief_states.py"))
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


def execution(mechanism, arm, condition):
    args = fixtures.execution(mechanism, arm, condition)
    args[3]["development_cohort_fit"] = "original_completed_fit"
    args[1].update(development_cohort_scope=states.SCOPE, development_cohort_fit="original_completed_fit")
    return args


@pytest.mark.parametrize("mechanism", ["ONLINE", "SEPARATE"])
@pytest.mark.parametrize("arm", ["POOL", "SEARCH"])
@pytest.mark.parametrize("condition", [1, 2, 3])
def test_registered_filter_preserves_the_original_audit(mechanism, arm, condition):
    states.check_filter(*execution(mechanism, arm, condition))


@pytest.mark.parametrize("index,key,value", [
    (1, "development_cohort_scope", "other"),
    (1, "development_cohort_fit", "different_participants"),
    (1, "candidate", {"id": "different"}),
    (0, "particles", 8),
    (0, "seeds", [99]),
    (10, "subject", 108),
    (11, "training_trials", 5),
    (12, "transitions", 16),
])
def test_registration_keeps_existing_identity_and_boundary_rejections(index, key, value):
    args = execution("SEPARATE", "SEARCH", 2)
    args[index][key] = value
    with pytest.raises(ValueError):
        states.check_filter(*args)


@pytest.mark.parametrize("mechanism", ["ONLINE", "SEPARATE"])
def test_registered_source_keeps_every_original_filter(tmp_path, monkeypatch, mechanism):
    settings, source, frozen = fixtures.source_fixture(tmp_path, monkeypatch, mechanism)
    fit = Path(frozen["data"]).parent / "fit"
    frozen["settings"]["development_cohort_fit"] = "original_completed_fit"
    (fit / "nominees.json").write_text(json.dumps(frozen))
    summary = Path(settings["summary"])
    (summary / "development_manifest.json").write_text(json.dumps({"fixture": True}))
    for path in (source / "jobs").glob("*.json"):
        job = json.loads(path.read_text())
        job.update(development_cohort_scope=states.SCOPE, development_cohort_fit="original_completed_fit")
        path.write_text(json.dumps(job))
    monkeypatch.setattr(states.summary_helpers(), "bind_registration", lambda directory: {})
    with states.registered_observer():
        files, metadata = states.cohort.source_jobs(settings, source, False)
        assert len(files) == 12 and len(metadata["selected_parameters"]) == 6
        assert metadata["development_cohort_fit"] == "original_completed_fit"
        assert str(states.ROOT / states.COHORT) in metadata["refit_replay_inputs"]
        files, metadata = states.cohort.source_jobs(settings, source, True)
        assert len(files) == 2 and metadata["scientific_scope"] is False
        assert {states.cohort.load(p)["model"] for p in files} == {"POOL", "SEARCH"}


@pytest.mark.parametrize("fault", ["scope", "manifest", "scientific_scope", "selection"])
def test_oral_context_requires_registration_and_full_scientific_scope(fault):
    protocol = fixtures.context("SEPARATE")
    protocol.update(development_cohort_scope=states.SCOPE, development_cohort_fit="original_completed_fit",
                    development_manifest_sha256="bound_manifest")
    context, note = states.estimation_context(protocol)
    assert context["development_cohort_scope"] == states.SCOPE and "带噪声" in note
    key = dict(scope="development_cohort_scope", manifest="development_manifest_sha256",
               scientific_scope="scientific_scope", selection="selected_parameters")[fault]
    protocol[key] = False if fault == "scientific_scope" else ""
    with pytest.raises(ValueError):
        states.estimation_context(protocol)


def test_observer_context_restores_the_original_functions_after_failure():
    previous = {name: getattr(states.cohort, name) for name in
                ("ENTRY", "check_filter", "source_jobs", "estimation_context")}
    with pytest.raises(RuntimeError):
        with states.registered_observer():
            raise RuntimeError("deliberate")
    assert all(getattr(states.cohort, name) is value for name, value in previous.items())


def test_paired_oral_adapter_restores_original_state_audit_after_failure(tmp_path, monkeypatch):
    from src.Bayesian_state.workflows.analysis import model_0923_cohort_belief_oral as oral
    previous = oral.states

    def fail(*args):
        assert oral.states is states.cohort
        assert oral.states.source_jobs is states.source_jobs
        assert oral.states.estimation_context is states.estimation_context
        raise RuntimeError("deliberate")

    monkeypatch.setattr(oral, "run", fail)
    with pytest.raises(RuntimeError):
        states.compare(tmp_path / "old", tmp_path / "new", tmp_path / "comparison", 108, "SEARCH")
    assert oral.states is previous

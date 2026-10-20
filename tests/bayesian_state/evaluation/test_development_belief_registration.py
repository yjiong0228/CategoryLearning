"""Participant scope must survive the fit-to-summary transition."""
import json

import pytest

from src.Bayesian_state.evaluation import model_0923_development_belief_refit as summary


def registered_fixture(tmp_path):
    fit, evaluation = (tmp_path / name for name in ("fit", "evaluation"))
    source = summary.registration.ENTRY
    hashes = {source: summary.digest(summary.ROOT / source)}
    settings = dict(subjects=[108], development_cohort_fit="completed_behavior_fit")
    declaration = dict(scope=summary.registration.SCOPE, input_sha256={}, **settings)
    job = dict(development_cohort_scope=summary.registration.SCOPE,
               development_cohort_fit=settings["development_cohort_fit"],
               source_sha256=hashes)
    for directory in (fit, evaluation):
        (directory / "jobs").mkdir(parents=True)
        (directory / "jobs" / "one.json").write_text(json.dumps(job))
    (fit / "nominees.json").write_text(json.dumps(dict(settings=settings, source_sha256=hashes)))
    (fit / "development_cohort_protocol.json").write_text(json.dumps(declaration))
    return fit, evaluation


def test_registration_binds_both_roles_and_detects_later_changes(tmp_path):
    fit, evaluation = registered_fixture(tmp_path)
    inputs = summary.registered_inputs(fit, evaluation)
    assert all(str((p / "jobs" / "one.json").resolve()) in inputs for p in (fit, evaluation))
    (evaluation / "jobs" / "one.json").write_text("{}")
    with pytest.raises(ValueError, match="input changed"):
        summary.verify_inputs(inputs)


@pytest.mark.parametrize("fault", ["scope", "cohort", "implementation", "missing_evaluation"])
def test_registration_rejects_mismatched_worker_provenance(tmp_path, fault):
    fit, evaluation = registered_fixture(tmp_path)
    file = evaluation / "jobs" / "one.json"
    job = json.loads(file.read_text())
    if fault == "missing_evaluation":
        file.unlink()
    else:
        if fault == "scope":
            job["development_cohort_scope"] = "unregistered"
        elif fault == "cohort":
            job["development_cohort_fit"] = "different_participants"
        else:
            job["source_sha256"] = {}
        file.write_text(json.dumps(job))
    with pytest.raises(ValueError):
        summary.registered_inputs(fit, evaluation)


def test_registered_policy_restores_original_driver_after_error():
    driver = summary.cohort.driver
    previous = {name: getattr(driver, name) for name in ("CONFIG_KEYS", "prepare_settings", "configure")}
    with pytest.raises(RuntimeError, match="deliberate"):
        with summary.registered_policy():
            assert driver.prepare_settings is summary.registration.prepare_settings
            assert driver.CONFIG_KEYS == summary.registration.CONFIG_KEYS
            raise RuntimeError("deliberate")
    assert all(getattr(driver, name) is value for name, value in previous.items())


def test_summary_registration_binds_the_existing_score_manifest(tmp_path):
    output = tmp_path / "summary"
    output.mkdir()
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(dict(status="complete")))
    summary.save_registration(output, {}, role="summary")
    inputs = summary.bind_registration(output)
    assert str((output / "development_manifest.json").resolve()) in inputs
    manifest.write_text(json.dumps(dict(status="changed")))
    with pytest.raises(ValueError, match="intact registered"):
        summary.bind_registration(output)

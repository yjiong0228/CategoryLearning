from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from src.Bayesian_state.workflows.analysis import model_0923_finalization_states as states

spec = importlib.util.spec_from_file_location(
    "finalization_state_fixtures", Path(__file__).with_name("test_model_0923_cohort_belief_states.py"))
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


def execution(mechanism, arm, condition, family):
    args = fixtures.execution(mechanism, arm, condition)
    precision, scale, upper = states.FAMILIES[family]
    definition = dict(precision=precision, rate_scale=scale, upper=upper)
    args[3]["arms"][arm] = definition
    args[3].update(finalization_precision_family=family, finalization_scope="development_comparison",
                   development_cohort_fit="registered_fit")
    for job in (args[0], args[1]):
        job["arm_settings"] = deepcopy(definition)
    args[1].update(finalization_precision_family=family, finalization_scope="development_comparison",
                   development_cohort_fit="registered_fit",
                   development_cohort_scope="completed_behavior_fit_development_cohort")
    return args


@pytest.mark.parametrize("mechanism", ["ONLINE", "SEPARATE"])
@pytest.mark.parametrize("arm", ["POOL", "SEARCH"])
@pytest.mark.parametrize("condition", [1, 2, 3])
@pytest.mark.parametrize("family", ["ADD10", "LOG25"])
def test_fitted_precision_and_original_filter_identity_are_preserved(mechanism, arm, condition, family):
    states.check_filter(*execution(mechanism, arm, condition, family))


@pytest.mark.parametrize("index,key,value", [
    (1, "finalization_precision_family", "ADD10"),
    (1, "finalization_scope", "software_validation"),
    (1, "development_cohort_fit", "another_sample"),
    (1, "seeds", [99]),
    (1, "particles", 8),
    (1, "candidate", {"id": "changed"}),
    (11, "training_trials", 5),
    (13, "updates", 20),
])
def test_metadata_changes_do_not_bypass_original_filter_checks(index, key, value):
    args = execution("SEPARATE", "SEARCH", 2, "LOG25")
    args[index][key] = value
    with pytest.raises(ValueError):
        states.check_filter(*args)


def test_completed_original_reference_retains_its_untagged_family():
    args = execution("ONLINE", "POOL", 1, "ADD10")
    for key in ("finalization_precision_family", "finalization_scope"):
        args[3].pop(key)
        args[1].pop(key)
    states.check_filter(*args)
    args[1]["finalization_precision_family"] = "LOG25"
    with pytest.raises(ValueError):
        states.check_filter(*args)


def test_smoke_replay_cannot_be_used_for_oral_inference(tmp_path):
    file = tmp_path/"manifest.json"
    file.write_text("{}")
    protocol = fixtures.context("ONLINE")
    protocol.update(finalization_precision_family="LOG25", precision_source_kind="joint_finalization",
                    precision_summary_manifest=str(file), precision_summary_sha256=states.cohort.digest(file))
    context, note = states.estimation_context(protocol)
    assert context["finalization_precision_family"] == "LOG25" and "带噪声" in note
    protocol["scientific_scope"] = False
    with pytest.raises(ValueError):
        states.estimation_context(protocol)
    protocol["scientific_scope"] = True
    file.write_text('{"changed": true}')
    with pytest.raises(ValueError):
        states.estimation_context(protocol)


def test_reference_cannot_be_relabelled_as_a_log_update(tmp_path):
    file = tmp_path/"manifest.json"
    file.write_text("{}")
    protocol = dict(finalization_precision_family="LOG25", precision_source_kind="completed_additive_reference",
                    precision_summary_manifest=str(file), precision_summary_sha256=states.cohort.digest(file))
    with pytest.raises(ValueError):
        states.check_precision_context(protocol)


def test_observer_restores_shared_functions_after_failure():
    previous = {name: getattr(states.cohort, name) for name in
                ("ENTRY", "check_filter", "source_jobs", "estimation_context")}
    with pytest.raises(RuntimeError):
        with states.precision_observer():
            raise RuntimeError("interrupted")
    assert all(getattr(states.cohort, name) is value for name, value in previous.items())

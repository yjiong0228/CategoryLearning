from copy import deepcopy

import pytest

from src.Bayesian_state.workflows.analysis import model_0923_finalization_joint as joint


def policy():
    return dict(
        subjects=[108], seed=991, budgets={"screen": {"particles": 128, "seeds": 2}},
        initial_candidates=40,
        arms={name: dict(precision="fast", rate_scale=10., upper=25., rates=[0., .01, .1])
              for name in ("POOL", "SEARCH")})


def test_log_mode_preserves_all_parameter_supports_and_numerical_policy():
    original = policy()
    saved = deepcopy(original)
    changed = joint.precision_settings(original, "LOG25", "development_comparison")
    for name in ("POOL", "SEARCH"):
        assert changed["arms"][name] == dict(precision="log", rate_scale=1., upper=25., rates=[0., .01, .1])
        changed["arms"][name] = original["arms"][name]
    for key in ("finalization_precision_family", "finalization_scope"):
        changed.pop(key)
    assert changed == saved
    assert original == saved


def test_additive_reproduction_preserves_the_complete_original_policy():
    original = policy()
    result = joint.precision_settings(original, "ADD10", "development_comparison")
    assert {k: result[k] for k in original} == original


@pytest.mark.parametrize("family,scope", [("LOG100", "development_comparison"), ("LOG25", "unknown")])
def test_undeclared_candidates_and_scope_are_rejected(family, scope):
    with pytest.raises(ValueError):
        joint.precision_settings(policy(), family, scope)


def test_changed_reference_definition_is_rejected():
    original = policy()
    original["arms"]["SEARCH"]["upper"] = 100.
    with pytest.raises(ValueError, match="reference"):
        joint.precision_settings(original, "LOG25", "development_comparison")


def job():
    return dict(finalization_precision_family="LOG25", finalization_scope="development_comparison",
                smoke=False, development_cohort_scope=joint.registration.SCOPE,
                arm_settings=dict(precision="log", rate_scale=1., upper=25.))


def test_worker_rejects_wrong_precision_and_software_as_science():
    current = job()
    joint.check_job(current)
    current["arm_settings"]["precision"] = "fast"
    with pytest.raises(ValueError, match="precision"):
        joint.check_job(current)
    current = job()
    current["finalization_scope"] = "software_validation"
    with pytest.raises(ValueError, match="smoke"):
        joint.check_job(current)
    current["smoke"] = True
    joint.check_job(current)


def test_summary_adapter_restores_original_shared_driver_after_failure():
    from src.Bayesian_state.evaluation import model_0923_cohort_belief_refit as scores
    previous = (scores.driver.CONFIG_KEYS, scores.driver.prepare_settings, scores.driver.configure)
    with pytest.raises(RuntimeError):
        with joint.summary_policy():
            assert scores.driver.prepare_settings is joint.prepare_settings
            raise RuntimeError("interrupt summary")
    assert (scores.driver.CONFIG_KEYS, scores.driver.prepare_settings, scores.driver.configure) == previous

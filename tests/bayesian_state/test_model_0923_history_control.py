"""Guard the information boundary of the observed-history control."""
from types import SimpleNamespace

import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis import model_0923_history_control as control


def snapshot():
    rules = np.array([[1., 0.], [0., 1.]])
    return dict(weights=np.array([.5, .5]), belief=rules.copy(), active=rules.copy(),
                choice_weights=rules.copy(), dominant=rules.copy(), beta=rules*3,
                prediction=np.array([[.8, .2], [.1, .9]]), search=np.array([.2, .3]),
                global_search=np.array([.5, .5]), failure=np.array([1., 2.]),
                admission_event=np.zeros(2))


def test_collector_preserves_paths_and_distinguishes_path_participation():
    collector = control.UniformHistoryCollector(np.array([2]))
    state = snapshot()
    collector(0, "predictive", state)
    collector(0, "filtered", state)
    arrays = collector.arrays()
    np.testing.assert_allclose(arrays["correct_paths"], [[.2, .9]])
    np.testing.assert_allclose(arrays["choice_weights"], [[.5, .5]])
    np.testing.assert_allclose(arrays["effective_participation"], [1.])


def test_collector_rejects_choice_conditioning():
    collector = control.UniformHistoryCollector(np.array([2]))
    state = snapshot()
    state["weights"] = np.array([.2, .8])
    with pytest.raises(AssertionError):
        collector(0, "predictive", state)


def test_collector_rejects_state_change_between_pre_feedback_phases():
    collector = control.UniformHistoryCollector(np.array([2]))
    state = snapshot()
    collector(0, "predictive", state)
    changed = snapshot()
    changed["prediction"] = changed["prediction"][::-1]
    with pytest.raises(AssertionError):
        collector(0, "filtered", changed)


def test_collector_requires_complete_phases():
    collector = control.UniformHistoryCollector(np.array([2]))
    collector(0, "predictive", snapshot())
    with pytest.raises(ValueError, match="incomplete"):
        collector.arrays()


@pytest.mark.parametrize("is_control", [True, False])
def test_filter_keeps_observed_history_and_only_changes_analysis_controls(monkeypatch, is_control):
    arrays = SimpleNamespace(stimulus=np.zeros((4, 4)), choices=np.array([1, 2, 1, 2]),
                             feedback=np.array([0., .5, 1., 0.]))
    inputs = dict(engine={"marker": 7}, subject=335, condition=3, arrays=arrays,
                  paths={"processed_dir": "unused"})
    captured = {}
    def fake_filter(**kwargs):
        captured.update(kwargs)
        kwargs["engine_config"]["marker"] = 8
        return "result"
    monkeypatch.setattr(control, "run_state_model_particle_filter", fake_filter)
    assert control.run_filter(inputs, 10, 4, control=is_control, max_trials=3) == "result"
    assert captured["condition_on_observed_choice"] is (not is_control)
    assert captured["resample_threshold_fraction"] == (0 if is_control else .5)
    np.testing.assert_array_equal(captured["choices"], [1, 2, 1])
    np.testing.assert_array_equal(captured["feedback"], [0, .5, 1])
    assert inputs["engine"] == {"marker": 7}
    assert "categories" not in captured

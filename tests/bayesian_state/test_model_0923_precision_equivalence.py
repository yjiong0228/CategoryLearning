from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from src.Bayesian_state.model.modules import precision_interventions as precision
from src.Bayesian_state.workflows.analysis import model_0923_precision_equivalence as reuse


def points():
    original = dict(status="frozen_before_test", smoke=False, settings=dict(subjects=[108]),
        nominees=[dict(subject=108, model=arm, candidate=dict(id=arm,
            hyperparams={reuse.RATE: 0., reuse.BETA: 10., "capacity": 2}))
            for arm in ("POOL", "SEARCH")])
    return original, deepcopy(original)


def test_only_exact_zero_rate_points_allow_reuse():
    before, after = points()
    assert set(reuse.check_points(before, after)) == {(108, "POOL"), (108, "SEARCH")}


@pytest.mark.parametrize("fault", ["nonzero", "capacity", "identity", "missing", "duplicate", "smoke"])
def test_changed_selected_points_cannot_reuse_predictions(fault):
    before, after = points()
    point = after["nominees"][0]["candidate"]
    if fault == "nonzero": point["hyperparams"][reuse.RATE] = 1e-12
    elif fault == "capacity": point["hyperparams"]["capacity"] = 3
    elif fault == "identity": point["id"] = "another"
    elif fault == "missing": after["nominees"].pop()
    elif fault == "duplicate": after["nominees"].append(deepcopy(after["nominees"][0]))
    elif fault == "smoke": after["smoke"] = True
    with pytest.raises(ValueError):
        reuse.check_points(before, after)


def test_zero_rate_intervention_bypasses_coordinate_transform(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Zero-rate feedback must not enter additive/log coordinate arithmetic")
    monkeypatch.setattr(precision, "precision_step", forbidden)
    gradient = np.array([-.7, .2])

    class Likelihood:
        def feedback_precision_gradient(self, stimulus, indices, beta):
            return gradient.copy()

    class Beta:
        def __init__(self):
            self.learning_rate = 0.
            self.beta_min, self.beta_max = .01, 25.
            self.beta = np.array([10., 10.])
            self.engine = SimpleNamespace(observation_likelihood=Likelihood(),
                get_module=lambda role: SimpleNamespace(trial_index=2))

        def _update_beta_feedback_gradient(self, stimulus, choice, feedback, indices):
            value = self.engine.observation_likelihood.feedback_precision_gradient(stimulus, indices, self.beta)
            self.last_gradient = value
            self.beta += self.learning_rate * value

    results = []
    for mode in ("fast", "log"):
        with precision.precision_intervention(Beta, mode=mode, upper=25.):
            model = Beta()
            for feedback in (0., .5, 1.):
                model._update_beta_feedback_gradient(None, 1, feedback, [0, 1])
                np.testing.assert_array_equal(model.last_gradient, gradient)
            results.append(model.beta)
    np.testing.assert_array_equal(results[0], results[1])
    np.testing.assert_array_equal(results[0], [10., 10.])

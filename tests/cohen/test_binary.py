"""Independent formula checks and frozen pre-refactor numeric references."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.Cohen.binary import FeatureMixtureStrict
from src.Cohen.cohen_schneidman import FeatureMixtureStrict as LegacyImport
from src.Cohen.data import prepare_dataframe, sti_to_bits
from src.Cohen.features import normalize
from src.Cohen.fitting import FitConfig, fit_paper_style
from src.Cohen.trace import run_session_with_formula_trace


@pytest.fixture
def reference():
    return json.loads(Path(__file__).with_name("legacy_reference.json").read_text())


def test_frozen_legacy_session_and_fit(reference):
    q = reference
    X, y, r = (np.asarray(q[k]) for k in ("X", "y", "choices"))
    m = FeatureMixtureStrict(3, beta=2.3, eta=.17, seed=19, memory_window=2)
    result = m.run_session(X, y, r)
    for key, expected in q["session"].items():
        np.testing.assert_allclose(result[key], expected, rtol=1e-12, atol=1e-13)
    for key, expected in q["final"].items():
        np.testing.assert_allclose(getattr(m, key), expected, atol=1e-13)
    fitted, info = fit_paper_style(X, y, r, config=FitConfig(
        population_size=4, generations=2, anneal_steps=2, seed=23), verbose=False)
    for key in ("alpha_pos", "alpha_neg", "beta", "eta"):
        np.testing.assert_allclose(getattr(fitted, key), q["fit"][key], atol=1e-12)
    assert info["negative_log_likelihood"] == pytest.approx(q["fit"]["nll"], abs=1e-12)
    assert LegacyImport is FeatureMixtureStrict


def test_eq2_matches_direct_bayes():
    model = FeatureMixtureStrict(3, seed=3, gamma=.7)
    dpos, dneg = model.category_distribution(1), model.category_distribution(-1)
    prior_pos = 1 / (1 + np.exp(model.gamma))
    bayes = dpos * prior_pos / (dpos * prior_pos + dneg * (1 - prior_pos))
    np.testing.assert_allclose(model.prob_pos(model.patterns), bayes, atol=1e-14)
    assert dpos.sum() == pytest.approx(1)
    assert dneg.sum() == pytest.approx(1)


@pytest.mark.parametrize("rule,factor", [("si_ratio", 1 / 2.7), ("paper_eq3", 1), ("exact_gradient", 2.7)])
def test_learning_conventions_against_finite_difference(rule, factor):
    m = FeatureMixtureStrict(2, beta=2.7, eta=.02, seed=9, update_rule=rule)
    x, y = np.array([1., -1.]), -1
    pos, neg, gamma = m.alpha_pos.copy(), m.alpha_neg.copy(), m.gamma
    numerical = []
    eps = 1e-6
    for name in ("alpha_pos", "alpha_neg"):
        g = []
        for j in range(m.n_features):
            getattr(m, name)[j] += eps
            upper = np.log(m.prob_label(x, [y])[0])
            getattr(m, name)[j] -= 2 * eps
            lower = np.log(m.prob_label(x, [y])[0])
            getattr(m, name)[j] += eps
            g.append((upper - lower) / (2 * eps))
        numerical.append(np.asarray(g))
    m.gamma = gamma + eps
    upper = np.log(m.prob_label(x, [y])[0])
    m.gamma = gamma - eps
    lower = np.log(m.prob_label(x, [y])[0])
    m.gamma = gamma
    m.learn_from_feedback(x, y)
    np.testing.assert_allclose(m.alpha_pos, normalize(pos + m.eta * factor * numerical[0] / m.beta), atol=1e-10)
    np.testing.assert_allclose(m.alpha_neg, normalize(neg + m.eta * factor * numerical[1] / m.beta), atol=1e-10)
    assert m.gamma == pytest.approx(gamma + m.eta * (upper - lower) / (2 * eps), abs=1e-10)


def test_trace_and_chunked_history(reference):
    X, y, r = (np.asarray(reference[k]) for k in ("X", "y", "choices"))
    full = FeatureMixtureStrict(3, seed=8, memory_window=3)
    split = full.copy()
    result, formulas = run_session_with_formula_trace(full, X, y, r, include_after_update=True)
    first = split.run_session(X[:3], y[:3], r[:3])
    continued = split.copy()
    tail = continued.run_session(X[3:], y[3:], r[3:])
    np.testing.assert_allclose(result["p_pos"], np.r_[first["p_pos"], tail["p_pos"]])
    np.testing.assert_allclose(full.alpha_pos, continued.alpha_pos)
    assert len(formulas) == len(X)
    assert "after feedback" in formulas[0]


def test_log_likelihood_has_no_artificial_floor():
    m = FeatureMixtureStrict(1, alpha_pos=[1], alpha_neg=[1], gamma=100)
    out = m.run_session([[1]], [1], [1], update=False)
    assert out["log_likelihood"][0] == pytest.approx(-100)


@pytest.mark.parametrize("kwargs", [{"memory_window": 0}, {"beta": np.nan}, {"eta": -1}, {"alpha_pos": [1, 2]}, {"update_rule": "unknown"}])
def test_invalid_model_rejected(kwargs):
    with pytest.raises(ValueError):
        FeatureMixtureStrict(2, **kwargs)


def test_data_and_length_errors_are_not_silently_mapped():
    with pytest.raises(ValueError, match="categories 1 and 2"):
        prepare_dataframe(pd.DataFrame({"stiID": [1], "category": [3], "choice": [4]}), 2, True)
    with pytest.raises(ValueError):
        sti_to_bits(0)
    with pytest.raises(ValueError):
        sti_to_bits(33)
    with pytest.raises(ValueError):
        FeatureMixtureStrict(2).run_session([[1, 1], [-1, 1]], [1], [1])
    with pytest.raises(ValueError):
        fit_paper_style(np.ones((2, 2)), [1, 1], [1, 1], fit_first_n=0)
    with pytest.raises(ValueError):
        FitConfig(generations=0).validate()

"""Check integration, likelihood derivatives, feedback semantics and timing."""
import numpy as np
import pytest
from scipy.special import logsumexp

from src.Cohen.continuous import ContinuousFeatureMixture
from src.Cohen.feedback import feedback_mask, task_feedback
from src.Cohen.fitting import FitConfig, fit_continuous


def test_continuous_integration_matches_analytic_one_dimensional_density():
    beta = 3.
    m = ContinuousFeatureMixture(n_dimensions=1, beta=beta, alpha=[[1], [-1]], quadrature_order=16)
    z, expectation = m._statistics()
    np.testing.assert_allclose(z, np.log(np.sinh(beta) / beta), atol=1e-12)
    np.testing.assert_allclose(expectation[:, 0], [1 / np.tanh(beta) - 1 / beta, -1 / np.tanh(beta) + 1 / beta], atol=1e-12)
    integrated = np.exp(m.log_weights) @ m.category_density(m.nodes)
    np.testing.assert_allclose(integrated, 1., atol=1e-12)
    p = m.predict_proba([[.49], [.51]])
    assert p[0, 0] != p[1, 0]
    np.testing.assert_allclose(m.features([[.2]]), [[-.6]])


@pytest.mark.parametrize("condition,choice,feedback,mask", [
    (1, 1, 0, [False, True]), (1, 2, 1, [False, True]),
    (2, 1, 0, [False, True, True, True]), (2, 4, 1, [False, False, False, True]),
    (3, 1, .5, [False, True, False, False]), (3, 4, .5, [False, False, True, False]),
    (3, 2, 0, [False, False, True, True]), (3, 3, 0, [True, True, False, False]),
])
def test_feedback_information(condition, choice, feedback, mask):
    np.testing.assert_array_equal(feedback_mask(choice, feedback, condition), mask)


@pytest.mark.parametrize("condition,choice,feedback", [(1, 1, 0), (2, 2, 0), (3, 2, .5), (3, 4, 0)])
def test_feedback_gradient_by_finite_differences(condition, choice, feedback):
    k = 2 if condition == 1 else 4
    m = ContinuousFeatureMixture(n_dimensions=2, n_categories=k, beta=2.3, seed=17)
    x = np.array([.27, .78])
    mask = feedback_mask(choice, feedback, condition)
    ga, gb = m.evidence_gradients(x, mask)
    eps = 1e-6

    def score():
        return logsumexp(m.log_probabilities(x)[0, mask])

    for c in range(k):
        for j in range(m.n_features):
            m.alpha[c, j] += eps
            plus = score()
            m.alpha[c, j] -= 2 * eps
            minus = score()
            m.alpha[c, j] += eps
            assert ga[c, j] == pytest.approx((plus - minus) / (2 * eps), abs=1e-8)
        if c < k - 1:
            m.bias[c] += eps
            plus = score()
            m.bias[c] -= 2 * eps
            minus = score()
            m.bias[c] += eps
            assert gb[c] == pytest.approx((plus - minus) / (2 * eps), abs=1e-8)
    assert gb[-1] == 0


@pytest.mark.parametrize("condition", [1, 2, 3])
def test_simulation_replays_its_own_history_and_continues_across_chunks(condition):
    k = 2 if condition == 1 else 4
    m = ContinuousFeatureMixture(n_dimensions=2, n_categories=k, beta=2, eta=.1, seed=5)
    X = np.random.default_rng(9).uniform(size=(12, 2))
    y = np.arange(12) % k + 1
    generated = m.copy().run_session(X, None, None, condition, y, simulation_seed=42)
    again = m.copy().run_session(X, None, None, condition, y, simulation_seed=42)
    replay = m.copy().run_session(X, generated["choices"], generated["feedback"], condition, y)
    for key in ("probabilities", "alpha_before_feedback", "bias_before_feedback", "choices", "feedback"):
        np.testing.assert_array_equal(generated[key], again[key])
        np.testing.assert_array_equal(generated[key], replay[key])
    np.testing.assert_allclose(replay["probabilities"].sum(axis=1), 1., atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(replay["alpha_before_feedback"], axis=2), 1., atol=1e-14)
    part = m.copy()
    part.run_session(X[:6], generated["choices"][:6], generated["feedback"][:6], condition)
    tail = part.copy().run_session(X[6:], generated["choices"][6:], generated["feedback"][6:], condition)
    np.testing.assert_allclose(replay["probabilities"][6:], tail["probabilities"])


def test_true_labels_are_only_used_for_scoring_and_feedback_is_after_choice():
    m = ContinuousFeatureMixture(n_dimensions=2, n_categories=4, seed=19)
    X = np.array([[.2, .8], [.3, .9], [.8, .2]])
    a = m.copy().run_session(X, [1, 2, 3], [0, 1, 0], 2, [2, 2, 4])
    b = m.copy().run_session(X, [1, 2, 3], [0, 1, 0], 2, [4, 3, 1])
    np.testing.assert_array_equal(a["probabilities"], b["probabilities"])
    changed = m.copy().run_session(X, [1, 2, 3], [1, 1, 0], 2)
    np.testing.assert_array_equal(a["probabilities"][0], changed["probabilities"][0])
    assert not np.allclose(a["probabilities"][1], changed["probabilities"][1])


def test_fit_uses_only_prefix_choices_and_feedback():
    X = np.random.default_rng(2).uniform(size=(5, 2))
    cfg = FitConfig(population_size=3, generations=1, anneal_steps=1, seed=12)
    kwargs = dict(model_config={"n_dimensions": 2, "quadrature_order": 6}, fit_first_n=3, config=cfg)
    a, info = fit_continuous(X, [1, 2, 1, 2, 1], [1, 0, 1, 0, 1], 1, **kwargs)
    b, other = fit_continuous(X, [1, 2, 1, 1, 2], [1, 0, 1, 1, 0], 1, **kwargs)
    np.testing.assert_array_equal(a.alpha, b.alpha)
    assert (a.beta, a.eta) == (b.beta, b.eta)
    assert info == other
    out = a.copy().run_session(X[:3], [1, 2, 1], [1, 0, 1], 1)
    assert -out["log_likelihood"].sum() == pytest.approx(info["negative_log_likelihood"])


def test_continuous_rejects_bad_domains_and_masks():
    m = ContinuousFeatureMixture(n_dimensions=2)
    with pytest.raises(ValueError):
        m.features([-.01, .2])
    with pytest.raises(ValueError):
        m.run_session([[.1, .2]], [3], [1], 1)
    with pytest.raises(ValueError):
        m.learn_from_feedback([.1, .2], 1, .5, 1)
    with pytest.raises(ValueError):
        m.evidence_gradients([.1, .2], np.array([False, False]))
    with pytest.raises(ValueError):
        ContinuousFeatureMixture(quadrature_order=1)


@pytest.mark.parametrize("rule", ["si_ratio", "paper_eq3", "exact_gradient"])
def test_discrete_measure_recovers_binary_updates(rule):
    """Replace only the integration measure to check the two-category reduction."""
    from src.Cohen.binary import FeatureMixtureStrict

    b = FeatureMixtureStrict(2, beta=2.8, eta=.07, gamma=.2, seed=7, update_rule=rule)
    c = ContinuousFeatureMixture(n_dimensions=2, n_categories=2, beta=b.beta, eta=b.eta,
                                 alpha=[b.alpha_pos, b.alpha_neg], bias=[-b.gamma, 0], update_rule=rule)
    c.nodes = (b.patterns + 1) / 2
    c.F_grid = b.F_all.copy()
    c.log_weights = np.full(len(b.patterns), -np.log(len(b.patterns)))
    np.testing.assert_allclose(c.predict_proba(c.nodes)[:, 0], b.prob_pos(b.patterns), atol=1e-14)
    x = np.array([1., -1.])
    # Choosing category 2 and receiving 0 fully identifies category 1.
    c.learn_from_feedback((x + 1) / 2, 2, 0, 1)
    b.learn_from_feedback(x, 1)
    np.testing.assert_allclose(c.alpha, [b.alpha_pos, b.alpha_neg], atol=1e-14)
    assert -c.bias[0] == pytest.approx(b.gamma, abs=1e-14)


def test_parallel_search_matches_serial_and_reports_evaluation_counts():
    X = np.random.default_rng(4).uniform(size=(6, 2))
    kwargs = dict(model_config={"n_dimensions": 2, "quadrature_order": 3})
    base = dict(population_size=4, generations=2, anneal_steps=2, seed=15)
    history = []
    serial, info = fit_continuous(X, [1, 2, 1, 1, 2, 1], [1, 0, 0, 1, 1, 0], 1,
        config=FitConfig(**base), **kwargs)
    parallel, other = fit_continuous(X, [1, 2, 1, 1, 2, 1], [1, 0, 0, 1, 1, 0], 1,
        config=FitConfig(**base, n_jobs=2), progress_callback=history.append, **kwargs)
    np.testing.assert_array_equal(serial.alpha, parallel.alpha)
    assert (serial.beta, serial.eta, info) == (parallel.beta, parallel.eta, other)
    assert [row["n_evaluations"] for row in history] == [4, 8, 10]
    assert np.all(np.diff([row["best_nll"] for row in history]) <= 0)

"""Definitions and rendered lines must agree; no model runtime is imported."""
from types import SimpleNamespace

import numpy as np
import pytest

from evaluation import figures
from evaluation.plot_results import predictive_bands, rolling_accuracy, validate_inputs

def subject(condition=1):
    truth = np.array([0, 1, 0, 1, 0, 1])
    choice = np.array([0, 0, 0, 1, 1, 1])
    return SimpleNamespace(subject_id=122, condition=condition, choices=choice,
        feedback=(choice == truth).astype(float), true_categories=truth,
        category_to_response=np.tile(np.arange(2 if condition == 1 else 4), (6, 1)),
        score_mask=np.array([0, 1, 1, 1, 1, 1], bool), valid=np.ones(6, bool),
        session_ids=np.ones(6, int))


def capture_plot(tmp_path, monkeypatch, data, feedback=None):
    captured = {}
    def save(fig, path):
        captured["figure"] = fig
        return path
    monkeypatch.setattr(figures, "_save", save)
    correct = (data.choices == data.true_categories).astype(float)
    choice_bands = predictive_bands(np.stack([correct, correct]), 3, valid=data.score_mask)
    fb = correct if feedback is None else feedback
    credit_bands = predictive_bands(np.stack([fb, fb]), 3, valid=data.score_mask)
    figures.plot_behavior(data, choice_bands, tmp_path / "behavior.png", window=3,
                          conditional_bands=choice_bands, feedback_bands=credit_bands)
    return captured["figure"]


@pytest.mark.parametrize("condition", [1, 2])
def test_binary_feedback_panel_is_omitted_only_when_truly_redundant(tmp_path, monkeypatch, condition):
    data = subject(condition)
    fig = capture_plot(tmp_path, monkeypatch, data)
    assert len(fig.axes) == 2
    assert "feedback" in fig._suptitle.get_text().lower()
    fig.clear()


def test_observed_credit_difference_keeps_third_panel(tmp_path, monkeypatch):
    data = subject()
    data.feedback[1] = 1
    fig = capture_plot(tmp_path, monkeypatch, data)
    assert len(fig.axes) == 3
    fig.clear()


def test_ternary_partial_credit_is_not_correctness(tmp_path, monkeypatch):
    data = subject(3)
    data.feedback[1] = .5
    fig = capture_plot(tmp_path, monkeypatch, data, data.feedback)
    assert len(fig.axes) == 3
    assert "feedback" in fig.axes[2].get_ylabel().lower()
    assert not np.array_equal(fig.axes[1].lines[1].get_ydata(), fig.axes[2].lines[1].get_ydata())
    fig.clear()


def test_observed_is_piecewise_linear_trailing_mean_with_expanding_warmup(tmp_path, monkeypatch):
    values = np.array([1, 0, 0, 0, 1, 1], float)
    np.testing.assert_allclose(rolling_accuracy(values, 4), [1, .5, 1/3, .25, .25, .5])
    data = subject()
    fig = capture_plot(tmp_path, monkeypatch, data)
    for ax in fig.axes:
        line = ax.lines[0]
        assert line.get_drawstyle() == "default"
        np.testing.assert_allclose(line.get_ydata(),
            rolling_accuracy(data.choices == data.true_categories, 3, valid=data.score_mask,
                             session_ids=data.session_ids), equal_nan=True)
    fig.clear()


def test_rolling_mean_gaps_and_session_boundaries_have_no_hidden_smoothing():
    np.testing.assert_allclose(rolling_accuracy([1, 0, 1, 0], 3, valid=[1, 0, 1, 1],
        session_ids=[1, 1, 1, 2]), [1, 1, 1, 0])


def synthetic_bundle(condition, permuted=False):
    c, n = (2 if condition == 1 else 4), 4
    truth = np.array([0, 0, 1, 1] if c == 2 else [0, 0, 2, 3])
    chosen = np.array([0, 1, 0, 1] if c == 2 else [0, 1, 3, 0])
    mapping = np.tile(np.roll(np.arange(c), 1) if permuted else np.arange(c), (n, 1))
    response = mapping[np.arange(n), chosen]
    credit = (chosen == truth).astype(float)
    if condition == 3:
        credit[(chosen != truth) & (chosen // 2 == truth // 2)] = .5
    probabilities = np.full((1, n, c), 1/c)
    arrays = dict(observed_choices=response, observed_feedback=credit.copy(), true_categories=truth,
        valid=np.ones(n, np.uint8), score_mask=np.array([0, 1, 1, 1], np.uint8),
        conditional_score_mask=np.array([0, 1, 1, 1], np.uint8), session_ids=np.ones(n, int),
        category_to_response=mapping, stimuli=np.zeros((n, 4)), trial_ids=np.arange(n),
        conditional_category_probability_repeats=probabilities,
        conditional_choice_probability_repeats=np.full((1, n), 1/c), autonomous_choices=response[None],
        autonomous_feedback=credit[None].copy(), autonomous_valid=np.ones((1, n), np.uint8),
        autonomous_category_probabilities=probabilities.copy(), pf_seeds=np.array([1]),
        simulation_seeds=np.array([2]), band_seed=np.array(3))
    kinds = ["uncommitted", "example", "generalization", "catalogue_rule"]
    manifest = dict(subject_id=122, condition=condition, parameters={}, config={}, provenance={"test": True},
        schema_version="gm1031-evaluation-inputs-v1", scientific_model_contract="1031-unified-v2",
        semantic_axis={"kinds": kinds}, state_metadata=[{"kind": k} for k in kinds])
    return arrays, manifest


@pytest.mark.parametrize("condition", [1, 2, 3])
@pytest.mark.parametrize("permuted", [False, True])
def test_saved_autonomous_feedback_cannot_silently_disagree_with_task(condition, permuted):
    arrays, manifest = synthetic_bundle(condition, permuted)
    validate_inputs(arrays, manifest)
    arrays["autonomous_feedback"][0, 2] = 0 if arrays["autonomous_feedback"][0, 2] == 1 else 1
    with pytest.raises(ValueError, match="autonomous_feedback.*task"):
        validate_inputs(arrays, manifest)

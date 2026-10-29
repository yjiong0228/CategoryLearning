"""Mechanism, timing and independent-reference checks for SUSTAIN."""

import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from src.SUSTAIN import Parameters, SUSTAIN, run_sequence


def test_first_trial_and_second_trial_have_hand_calculated_probabilities():
    model = SUSTAIN([2], parameters=Parameters(r=1, beta=1, d=2, eta=.25))
    first = model.step([1, 0], 0)
    np.testing.assert_allclose(first.prediction.probabilities, [.5, .5])
    assert first.prediction.winner is None
    np.testing.assert_allclose(model.weights, [[.25, 0]])
    np.testing.assert_allclose(model.lambdas, [1.25])
    second = model.step([1, 0], 0)
    np.testing.assert_allclose(second.prediction.probabilities, [1 / (1 + np.exp(-.5)), 1 / (1 + np.exp(.5))])
    assert not second.recruited
    np.testing.assert_allclose(model.weights, [[.4375, 0]])


def test_current_feedback_changes_learning_but_not_current_prediction():
    left, right = SUSTAIN([2]), SUSTAIN([2])
    for model in (left, right):
        model.step([1, 0], 0)
    a, b = left.step([0, 1], 0), right.step([0, 1], 1)
    np.testing.assert_array_equal(a.prediction.probabilities, b.prediction.probabilities)
    assert not a.recruited and b.recruited
    assert len(left.clusters) == 1 and len(right.clusters) == 2


def test_new_cluster_wins_after_feedback_and_uses_recomputed_output():
    model = SUSTAIN([2], parameters=Parameters(r=1, beta=1, d=2, eta=.25))
    model.step([1, 0], 0)
    old_weights = model.weights.copy()
    result = model.step([0, 1], 1)
    assert result.prediction.winner == 0 and result.learning_winner == 1
    new_output = 1 / (1 + np.exp(-1.25))
    np.testing.assert_allclose(model.weights[1], [0, .25 * new_output])
    np.testing.assert_array_equal(model.weights[0], old_weights[0])
    np.testing.assert_allclose(model.lambdas, [1.5])


def test_continuous_distance_is_on_the_declared_unit_scale():
    model = SUSTAIN([1], parameters=Parameters(r=0, beta=0, eta=0))
    model.step([0], 0)
    np.testing.assert_allclose(model.predict([1]).activations, [np.exp(-1)])


def test_attention_and_center_update_use_the_pre_update_distance():
    model = SUSTAIN([1, 2], parameters=Parameters(eta=.2))
    model.step([0, 1, 0], 0)
    model.step([1, 0, 1], 0)
    np.testing.assert_allclose(model.clusters, [[.2, .8, .2]])
    np.testing.assert_allclose(model.lambdas, np.full(2, 1.2 + .2 * np.exp(-1.2) * (1 - 1.2)))


def test_humble_teacher_does_not_penalize_already_sufficient_output():
    model = SUSTAIN([1], parameters=Parameters(eta=.2))
    model.step([0], 0)
    model.weights[:] = [2, -2]
    model.step([0], 0)
    np.testing.assert_array_equal(model.weights, [[2, -2]])


def test_multiclass_recruitment_checks_maximal_not_minimal_ties():
    model = SUSTAIN([2], n_categories=4)
    model.step([1, 0], 2)
    result = model.step([1, 0], 2)
    assert not result.recruited  # Three losing outputs tied at zero.
    assert result.prediction.probabilities.shape == (4,)
    assert result.prediction.probabilities[2] > result.prediction.probabilities[0]
    np.testing.assert_allclose(result.prediction.probabilities.sum(), 1)


def test_recruitment_variants_are_explicit_and_distinct():
    paper = SUSTAIN([1], recruitment="prediction_error")
    author = SUSTAIN([1], recruitment="cluster_label")
    for model in (paper, author):
        model.step([0], 0)
        model.weights[:] = [0, 1]  # Output predicts a category different from the cluster label.
    assert paper.step([0], 0).recruited
    assert not author.step([0], 0).recruited


def test_unsupervised_threshold_and_frozen_state():
    model = SUSTAIN([1], n_categories=None, parameters=Parameters(tau=.9))
    assert model.step([0]).recruited
    assert model.step([1]).recruited
    before = model.state_dict()
    result = model.step([.5], learn=False)
    assert not result.recruited
    assert result.prediction.probabilities.size == 0
    assert before == model.state_dict()


def test_extreme_attention_and_competition_remain_finite():
    model = SUSTAIN([1, 1], parameters=Parameters(r=1000, beta=1000, d=1000),
                    initial_lambdas=[1000, 2000])
    model.step([0, 0], 0)
    prediction = model.predict([1, 1])
    assert np.isfinite(prediction.probabilities).all()
    assert np.isfinite(prediction.log_probabilities).all()
    np.testing.assert_allclose(prediction.probabilities.sum(), 1)


def test_sequence_generation_replay_and_state_snapshots():
    x = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    y = np.array([0, 1, 0, 1])
    a, b = SUSTAIN([2]), SUSTAIN([2])
    generated = run_sequence(a, x, y, sample_choices=True, seed=4, trace=True)
    replay = run_sequence(b, x, y, choices=generated.choices, trace=True)
    np.testing.assert_array_equal(generated.probabilities, replay.probabilities)
    assert generated.traces == replay.traces
    assert generated.traces[0]["post_feedback"]["lambdas"] == [1 + a.parameters.eta]
    assert a.state_dict() == b.state_dict()
    np.testing.assert_allclose(replay.negative_log_likelihood,
                               -np.log(replay.probabilities[np.arange(4), generated.choices]).sum())


def test_reset_reproduces_random_ties_with_fixed_seed():
    model = SUSTAIN([2], ties="random", seed=31)
    model.step([1, 0], 0)
    model.step([1, 0], 1)
    state = model.state_dict()
    winners = [model.predict([1, 0]).winner for _ in range(16)]
    assert set(winners) == {0, 1}
    model.reset()
    model.step([1, 0], 0)
    model.step([1, 0], 1)
    assert model.state_dict() == state
    assert [model.predict([1, 0]).winner for _ in range(16)] == winners


def test_test_trials_freeze_learning_and_random_likelihood_is_rejected():
    model = SUSTAIN([2])
    result = run_sequence(model, [[1,0],[0,1],[0,1]], [0,1,1],
                          learn_mask=np.array([True,False,False]), trace=True)
    assert result.traces[0]["post_feedback"] == result.traces[2]["post_feedback"]
    assert result.traces[1]["learning_winner"] is None
    with pytest.raises(ValueError,match="not marginalized"):
        run_sequence(SUSTAIN([2],ties="random"), [[1,0]], [0], choices=[0])


def test_underflowed_probability_retains_a_finite_log_likelihood():
    model = SUSTAIN([1], parameters=Parameters(d=2000,eta=1))
    model.step([0],0)
    result = run_sequence(model, [[0]], [0], choices=[1])
    assert result.probabilities[0,1] == 0
    assert result.negative_log_likelihood == 2000


@pytest.mark.parametrize("kwargs", [dict(eta=1.1), dict(r=-1), dict(beta=float('nan')), dict(tau=1.1)])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        Parameters(**kwargs)


@pytest.mark.parametrize("x", [[.5, .5], [1, 1], [float('nan'), 0], [-1, 1], [1]])
def test_invalid_nominal_stimuli(x):
    with pytest.raises(ValueError):
        SUSTAIN([2]).step(x, 0)


def test_invalid_sequence_is_rejected_before_state_changes():
    model = SUSTAIN([2])
    with pytest.raises(ValueError):
        run_sequence(model, [[1, 0], [.5, .5]], [0, 1])
    assert len(model.clusters) == 0


@pytest.mark.parametrize("recruitment", ["prediction_error", "cluster_label"])
def test_author_reference_probabilities_and_states(recruitment):
    fixture = json.loads((Path(__file__).parent / "fixtures/author_reference.json").read_text())
    for case in fixture["cases"]:
        model = SUSTAIN(fixture["dimensions"], fixture["n_categories"], Parameters(**fixture["parameters"]),
                        recruitment=recruitment)
        for index, trial in enumerate(case["trials"]):
            result = model.step(trial["stimulus"], trial["target"])
            context = f"problem {case['problem']}, trial {index + 1}"
            np.testing.assert_allclose(result.prediction.probabilities, trial["probabilities"], atol=1e-12, rtol=1e-12, err_msg=context)
            for name in ("lambdas", "clusters", "weights", "category_positions"):
                np.testing.assert_allclose(getattr(model, name), trial[name], atol=1e-12, rtol=1e-12, err_msg=context + name)


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript is unavailable; cross-language numeric check not run")
def test_legacy_r_agrees_on_nominal_no_recruitment_sequence(tmp_path):
    # Reviewed scope: strip choice/feedback, fix ties, use nominal one-hot
    # features, and avoid later recruitment (where the two update rules differ).
    root = Path(__file__).resolve().parents[2]
    reference = root / "src/SUSTAIN/slpSUSTAIN_trace.R"
    if not reference.exists():
        pytest.skip("Historical untracked R reference is not present in this checkout")
    code = '''
args <- commandArgs(trailingOnly=TRUE)
source(args[1])
tr <- matrix(c(1,1,0,1,0, 0,0,1,1,0, 0,1,0,1,0), nrow=3, byrow=TRUE)
colnames(tr) <- c("ctrl","x1","x2","c1","c2")
st <- list(dims=c(2), lambda=c(1), w=NA, cluster=NA, colskip=1,
           r=2, beta=1, d=3, eta=.2, tau=0)
out <- slpSUSTAIN_trace(st,tr,xtdo=TRUE,ties="first")
write.table(out$xtdo[,1:2],args[2],row.names=FALSE,col.names=FALSE,sep=",")
write.table(out$lambda_hist,args[3],row.names=FALSE,col.names=FALSE,sep=",")
'''
    script = tmp_path / "reference.R"
    script.write_text(code)
    probabilities, lambdas = tmp_path / "p.csv", tmp_path / "lambda.csv"
    subprocess.run(["Rscript", str(script), str(reference), str(probabilities), str(lambdas)], check=True, capture_output=True, text=True)
    result = run_sequence(SUSTAIN([2], parameters=Parameters(r=2,beta=1,d=3,eta=.2,tau=0)),
                          [[1,0],[0,1],[1,0]], [0,0,0], trace=True)
    np.testing.assert_allclose(result.probabilities, np.loadtxt(probabilities, delimiter=","), atol=1e-12)
    np.testing.assert_allclose([t["post_feedback"]["lambdas"][0] for t in result.traces], np.loadtxt(lambdas, delimiter=","), atol=1e-12)

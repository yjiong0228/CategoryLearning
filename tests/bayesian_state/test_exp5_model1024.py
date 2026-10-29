"""Discrete emission/gradient, structural search and shared closed-loop contracts."""
from itertools import product
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.Bayesian_state.hypothesis_space.observation_model.exp5_partition import Exp5ParityPartition
from src.Bayesian_state.hypothesis_space.observation_model.discrete_rule_partition import DiscreteRulePartition
from src.Bayesian_state.hypothesis_space.observation_model.likelihood import ObservationLikelihood
from src.Bayesian_state.inference.dispatcher import run_inference_backend
from src.Bayesian_state.model.config import ModelConfig
from src.Bayesian_state.model.model_1024 import model_1024_runtime
from src.Bayesian_state.simulation.exp5_data import load_exp5_subject
from src.Bayesian_state.simulation.probabilistic import run_autonomous_probabilistic_learning
from src.Bayesian_state.workflows.runs.run_model_1024_exp5 import run, load_config, verify_generated_history

CONFIG = Path("configs/exp5/model1024_smoke.yaml")
PATTERNS = np.array(list(product((-1., 1.), repeat=5)))


@pytest.fixture
def engine():
    return load_config(CONFIG)[1]


def test_complete_catalogue_and_correct_rule_on_every_stimulus():
    partition = Exp5ParityPartition()
    assert partition.length == 62
    correct = next(h.index for h in partition.hypothesis_space if h.dims == (1, 3) and h.polarity == 1)
    actual = [partition.get_category_assignment(correct, x) + 1 for x in PATTERNS]
    np.testing.assert_array_equal(actual, np.where(PATTERNS[:, 1] == PATTERNS[:, 3], 1, 2))
    for h in partition.hypothesis_space:
        probabilities = partition.get_category_probabilities(h.index, (PATTERNS, None, None), 2.)
        labels = partition.rule_geometry.category_assignments(h.index, PATTERNS)
        np.testing.assert_allclose(probabilities[labels, np.arange(32)], 1/(1+np.exp(-2.)))
        np.testing.assert_allclose(probabilities.sum(axis=0), 1.)


@pytest.mark.parametrize("choice,feedback", [(1, 0.), (1, 1.), (2, 0.), (2, 1.)])
def test_all_rules_precision_gradient_matches_log_likelihood_difference(choice, feedback):
    likelihood = ObservationLikelihood(Exp5ParityPartition(), distance_mode="rule")
    hypotheses = np.arange(62)
    beta = np.linspace(.1, 8., 62)
    step = 1e-5
    for x in PATTERNS:
        observation = (x, choice, feedback)
        actual = likelihood.feedback_precision_gradient(observation, hypotheses, beta)
        high = likelihood.binary_feedback_likelihood(observation, hypotheses, beta+step)
        low = likelihood.binary_feedback_likelihood(observation, hypotheses, beta-step)
        np.testing.assert_allclose(actual, (np.log(high)-np.log(low))/(2*step), atol=2e-9, rtol=1e-7)
        assert np.isfinite(likelihood.feedback_precision_gradient(observation, hypotheses, 25.)).all()


def test_local_graph_and_exact_functional_relations_are_distinct():
    partition = Exp5ParityPartition()
    version, programs, edit, confusion = partition.rule_search_resources(
        dict(position=1., polarity=2.), distance_mode="rule")
    assert version.endswith("_v1") and edit.shape == (62, 62) and confusion.shape == (62, 62, 2, 2)
    np.testing.assert_allclose(confusion.sum(axis=(2, 3)), 1.)
    np.testing.assert_allclose(np.trace(confusion, axis1=2, axis2=3), partition.get_similarity_matrix())
    lookup = {(h.dims, h.polarity): h.index for h in programs}
    h = lookup[((1, 3), 1)]
    assert edit[h, lookup[((1,), 1)]] == 1.
    assert edit[h, lookup[((1, 4), 1)]] == 1.
    assert edit[h, lookup[((1, 2, 3), 1)]] == 1.
    assert edit[h, lookup[((1, 3), -1)]] == 2.
    assert np.isinf(edit[h, lookup[((0, 2, 4), 1)]])
    reached = {0}
    while True:
        updated = reached | set(np.flatnonzero(np.isfinite(edit[list(reached)]).any(axis=0)))
        if updated == reached:
            break
        reached = updated
    assert len(reached) == 62
    with pytest.raises(ValueError):
        edit.flags.writeable = True
    with pytest.raises(ValueError):
        confusion.flags.writeable = True


@pytest.mark.parametrize("stimulus", [[0, 1, 1, 1, 1], [np.nan, 1, 1, 1, 1], [1, 1, 1, 1]])
def test_strict_inputs_are_opt_in_and_legacy_coercion_is_preserved(stimulus):
    with pytest.raises(ValueError, match="Exp5 stimuli"):
        Exp5ParityPartition().get_category_assignment(0, stimulus)
    if len(stimulus) == 5:
        assert DiscreteRulePartition().get_category_assignment(0, stimulus) == 1


@pytest.mark.parametrize("mutation", ["perception", "precision", "feedback", "agenda", "oral", "partition"])
def test_task_contract_rejects_silent_model_changes(engine, mutation):
    if mutation == "perception":
        engine["modules"]["perception_mod"]["class"] = "src.Bayesian_state.model.modules.perception.PerceptionModule"
    elif mutation == "precision":
        engine["modules"]["beta_mod"]["kwargs"]["beta_update_mode"] = "legacy"
    elif mutation == "feedback":
        engine["likelihood"]["feedback_likelihood_mode"] = "hierarchical_pairing"
    elif mutation == "agenda":
        engine["agenda"][-2:] = ["beta_mod", "memory_mod"]
    elif mutation == "oral":
        engine["oral_readout"] = {"method": "rule_content_v1", "kwargs": dict(
            mention_probability=.8, other_probability=.02, body_naming_probability=.5,
            body_reference_center=.75, body_reference_width=.1)}
    else:
        engine["partition"]["kwargs"]["n_dims"] = 4
    with pytest.raises(ValueError):
        ModelConfig.from_mapping(engine)


def _generate(engine, n=12, seed=52):
    x = PATTERNS[:n]
    q = np.eye(2)[(x[:, 1] != x[:, 3]).astype(int)]
    return run_autonomous_probabilistic_learning(engine_config=engine, subject_id=503, stimulus=x,
                                               target_probabilities=q, trajectory_seed=seed)


@pytest.mark.parametrize("kappa", [0., 2.])
def test_seed_prefix_feedback_and_exact_state_replay(engine, kappa):
    engine["modules"]["hypo_transitions_mod"]["kwargs"]["execution_inertia"] = kappa
    case = load_exp5_subject(Path("configs/exp5/model1024_inputs.yaml"), 503)
    with model_1024_runtime():
        first, repeated, prefix = _generate(engine), _generate(engine), _generate(engine, n=5)
        active, executed = verify_generated_history(engine, case, first)
    trajectory = first.trajectory
    for field in ("choices", "feedback", "prior", "posterior", "beta", "observed_probabilities"):
        np.testing.assert_array_equal(getattr(trajectory, field), getattr(repeated.trajectory, field))
        np.testing.assert_array_equal(getattr(trajectory, field)[:5], getattr(prefix.trajectory, field))
    np.testing.assert_array_equal(trajectory.perceived_stimulus, trajectory.stimulus)
    labels = np.where(trajectory.stimulus[:, 1] == trajectory.stimulus[:, 3], 1, 2)
    np.testing.assert_array_equal(trajectory.feedback, trajectory.choices == labels)
    assert active[np.arange(len(active)), executed].all()
    assert np.all((active.sum(axis=1) >= 1) & (active.sum(axis=1) <= 3))


def test_current_feedback_and_future_choices_do_not_leak_into_pre_choice_predictions(engine):
    engine["inference"]["particle_count"] = 4
    with model_1024_runtime():
        generated = _generate(engine)
        y, r = generated.trajectory.choices.copy(), generated.trajectory.feedback.copy()
        kwargs = dict(engine_config=engine, subject_id=503, condition=1, stimulus=PATTERNS[:12], inference_seed=72)
        first = run_inference_backend(**kwargs, choices=y, feedback=r)
        r[5:] = 1 - r[5:]
        changed_feedback = run_inference_backend(**kwargs, choices=y, feedback=r)
        y[5:] = 3-y[5:]
        changed_choices = run_inference_backend(**kwargs, choices=y, feedback=r)
    np.testing.assert_array_equal(first.marginal_probabilities[:6], changed_feedback.marginal_probabilities[:6])
    np.testing.assert_array_equal(first.marginal_probabilities[:6], changed_choices.marginal_probabilities[:6])
    assert not np.array_equal(first.marginal_probabilities[6:], changed_feedback.marginal_probabilities[6:])


def test_real_entry_provenance_and_overwrite_protection(tmp_path):
    settings = yaml.safe_load(CONFIG.read_text())
    for key in ("input_config", "engine_config"):
        settings[key] = str((CONFIG.parent / settings[key]).resolve())
    settings.update(max_trials=6, autonomous_paths=1)
    config = tmp_path / "smoke.yaml"
    config.write_text(yaml.safe_dump(settings))
    output = tmp_path / "smoke"
    report = run(config, output)
    assert report["status"] == "closed_loop_smoke_passed" and all(report["checks"].values())
    assert report["executed_trials"] == 6 and report["recovery_smoke"]["paths"] == 1
    with np.load(output / "observed.npz", allow_pickle=False) as result:
        assert result["pre_choice_probabilities"].shape == (6, 2)
        assert "categories" not in result
    with np.load(output / "autonomous_000.npz", allow_pickle=False) as result:
        assert result["active"].shape == (6, 62)
    with pytest.raises(FileExistsError):
        run(config, output)
    settings["max_trials"] = 128
    config.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match="max_trials"):
        load_config(config)

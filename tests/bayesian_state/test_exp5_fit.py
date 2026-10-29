"""Full-history likelihood, fixed coordinates, task seeds and source isolation."""
from pathlib import Path
import tarfile

import numpy as np
import pytest
import yaml

from src.Bayesian_state.metrics.trajectory_shape import sequence_score
from src.Bayesian_state.workflows.analysis import model_1024_exp5_fit as fit

CONFIG = Path("configs/exp5/model1024_fit_501.yaml")
ENGINE = Path("configs/exp5/model_struct/model_1024_rule1.yaml")


def test_pooled_predictions_preserve_sequence_likelihood_and_causal_weights():
    p = np.array([[[.9, .1], [.2, .8], [.7, .3]], [[.3, .7], [.6, .4], [.2, .8]]])
    choices = np.array([1, 2, 2])
    pooled = fit.pooled_predictions(p, choices)
    observed = p[:, np.arange(3), choices-1]
    np.testing.assert_allclose(-np.log(pooled[np.arange(3), choices-1]).mean(), sequence_score(observed)["nll"])
    np.testing.assert_allclose(pooled[0], [.6, .4])
    np.testing.assert_allclose(pooled[1], .75*p[0, 1] + .25*p[1, 1])
    changed = p.copy()
    changed[:, 2] = [.5, .5]
    np.testing.assert_array_equal(fit.pooled_predictions(changed, choices)[:2], pooled[:2])
    changed_choice = choices.copy()
    changed_choice[1:] = 3-choices[1:]
    np.testing.assert_array_equal(fit.pooled_predictions(p, changed_choice)[:2], pooled[:2])


def test_parameter_coordinates_keep_fixed_cognitive_assumptions():
    base = yaml.safe_load(ENGINE.read_text())
    point = fit.canonical(dict(fit.anchor(base), error_gain=0, failure_decay=.99, capacity=6, inertia=0), "test")
    assert point["hyperparams"]["failure_decay"] == 0
    changed = fit.engine_at(base, point, 512)
    assert changed["inference"]["particle_count"] == 512
    assert changed["modules"]["hypo_transitions_mod"]["kwargs"]["max_capacity"] == 6
    assert changed["likelihood"] == base["likelihood"]
    assert changed["partition"] == base["partition"]
    assert changed["modules"]["memory_mod"] == base["modules"]["memory_mod"]
    assert base["inference"]["particle_count"] == 8


@pytest.mark.parametrize("key,value", [("p0", 0), ("failure_decay", 1), ("capacity", 2.5), ("capacity", 62),
                                       ("beta_init", 26), ("learning_rate", -1), ("inertia", float("nan"))])
def test_parameter_domain_errors_precede_worker_launch(key, value):
    base = yaml.safe_load(ENGINE.read_text())
    with pytest.raises(ValueError):
        fit.canonical(dict(fit.anchor(base), **{key: value}), "bad")


def test_jobs_pair_seeds_separate_roles_and_drop_historical_job_trees(tmp_path):
    policy = yaml.safe_load(CONFIG.read_text())
    base = yaml.safe_load(ENGINE.read_text())
    points = [fit.canonical(dict(fit.anchor(base), inertia=k), "test") for k in (0, 2)]
    points[0].update(score={"nll": .5}, jobs=[{"old": "job"}], mean_nll=.5)
    protocol = dict(policy=policy, runtime=str(tmp_path))
    jobs = fit.jobs_for(protocol, tmp_path, points, "confirm", "confirmation", 512, 2)
    assert jobs[0]["seeds"] == jobs[2]["seeds"]
    assert jobs[1]["seeds"] == jobs[3]["seeds"]
    assert set(jobs[0]["candidate"]) == {"id", "hyperparams", "sources", "origin"}
    other = fit.jobs_for(protocol, tmp_path, points, "audit", "audit", 2048, 2)
    assert {j["seeds"][0] for j in jobs}.isdisjoint({j["seeds"][0] for j in other})
    assert jobs[0]["worker_entry"] == str(tmp_path / fit.ENTRY)


def test_source_archive_preserves_exact_bytes_when_worktree_changes(tmp_path, monkeypatch):
    root = tmp_path / "root"
    source = root / "src/Bayesian_state/example.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n")
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(fit, "ROOT", root)
    runtime, hashes = fit.freeze_sources(output)
    source.write_text("VALUE = 2\n")
    frozen = Path(runtime) / "src/Bayesian_state/example.py"
    assert frozen.read_text() == "VALUE = 1\n"
    assert fit.digest(frozen) == hashes["src/Bayesian_state/example.py"]
    with tarfile.open(output / "source_snapshot.tar") as archive:
        assert archive.extractfile("src/Bayesian_state/example.py").read() == b"VALUE = 1\n"


def test_incomplete_repeat_bank_cannot_be_scored(tmp_path):
    jobs = [dict(name="missing", output=str(tmp_path), target=str(tmp_path / "missing"))]
    assert fit.collect(jobs, "observed_probability") is None


def test_expired_fit_budget_is_recorded_as_partial(tmp_path):
    policy = yaml.safe_load(CONFIG.read_text())
    policy.update(input_config=str(Path("configs/exp5/model1024_inputs.yaml").resolve()),
                  engine_config=str(ENGINE.resolve()))
    policy["budget"]["wall_hours"] = 0
    config = tmp_path / "expired.yaml"
    config.write_text(yaml.safe_dump(policy))
    output = tmp_path / "run"
    fit.prepare(config, output, smoke=True)
    result = fit.run(output)
    assert result["status"] == "bounded_partial" and result["stop_reason"] == "wall_budget"
    assert result["successful_jobs"] == 0
    assert (output / "report_v1/report.md").exists()
    with pytest.raises(FileExistsError):
        fit.run(output)

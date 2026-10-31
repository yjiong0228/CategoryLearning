"""Saved-array verification and prefix-conditioned state pooling."""
import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis.model_1031_finalization import digest, save_arrays, write_new
from src.Bayesian_state.workflows.reports.audit_model_1031_finalization import (
    generation_summaries, merge_state_records, nested_search_diagnostics, verify_jobs,
)


def test_state_systems_use_current_prefix_likelihood_after_choice(tmp_path):
    save_arrays(tmp_path / "inputs/s1.npz", choices=[1, 1])
    records = []
    for i, chosen in enumerate(([.8, .1], [.2, .9])):
        p = np.column_stack([chosen, 1 - np.asarray(chosen)])
        state = np.repeat(np.eye(4)[[1 if i == 0 else 3]], 2, axis=0)
        path = tmp_path / "jobs" / f"j{i}.npz"
        save_arrays(path, choice_probabilities=p, filtered_representation_execution=state)
        record = dict(job=dict(output=str(tmp_path), id=f"j{i}", kind="fit", subject=1, train_trials=1),
            nll=-float(np.log(chosen).sum()), train_nll=-float(np.log(chosen[0])), cpu_seconds=.1,
            array_sha256=digest(path))
        write_new(path.with_suffix(".json"), record)
        records.append(record)
    result = merge_state_records(tmp_path, records, 1)["summaries"]["filtered_representation_execution"]
    example = (.8 + .08 / (.08 + .18)) / 2
    np.testing.assert_allclose(result["whole"], [0., example, 0., 1 - example])
    assert not np.isclose(example, .5)
    checked, audit = verify_jobs(tmp_path)
    assert len(checked) == 2 and audit["max_recomputed_nll_error"] == 0.


def test_generation_summary_includes_failed_paths_and_rejects_missing_seeds(tmp_path):
    save_arrays(tmp_path / "inputs/s1.npz", choices=[1, 2], categories=[1, 2])
    save_arrays(tmp_path / "jobs/g1.npz", choices=[[1, 2], [1, 1], [2, 1]],
                representation_group=[[1, 1], [2, 2], [3, 3]])
    records = [dict(job=dict(kind="generate", id="g1", subject=1, config="same", seeds=[2500, 2501, 2502]))]
    rows = [dict(subject=1, condition=1, variant="compact", candidate=dict(id="same"))]
    policy = dict(seed=0, generation=dict(paths_per_candidate=3))
    result = generation_summaries(tmp_path, records, rows, policy, "development", ["compact"])[0]
    assert result["paths"] == 3 and len(result["feature_names"]) == 9
    np.testing.assert_allclose(result["accuracy_quantiles_05_50_95"], [.05, .5, .95])
    assert result["observed_accuracy"] == 1.
    records[0]["job"]["seeds"] = [2500, 2501]
    with pytest.raises(ValueError, match="every registered path"):
        generation_summaries(tmp_path, records, rows, policy, "development", ["compact"])


def test_nested_search_warning_requires_comparable_precision_and_clear_loss():
    groups = [dict(variant=v, subject=dict(subject=1, condition=1, train_trials=100), particles=256,
                   candidates=[dict(nll=nll, mc_se=.01, repeats=4)])
              for v, nll in (("full", 100.), ("compact", 95.))]
    check = nested_search_diagnostics(groups, .01)[0]
    assert check["status"] == "finite_search_warning"
    groups[1]["particles"] = 512
    assert nested_search_diagnostics(groups, .01)[0]["status"] == "different_numerical_precision"
    groups[1]["particles"] = 256
    groups[0]["candidates"][0]["mc_se"] = 3.
    assert nested_search_diagnostics(groups, .01)[0]["status"] == "no_detected_dominance_failure"

"""Saved-array verification and prefix-conditioned state pooling."""
import numpy as np

from src.Bayesian_state.workflows.analysis.model_1031_finalization import digest, save_arrays, write_new
from src.Bayesian_state.workflows.reports.audit_model_1031_finalization import merge_state_records, verify_jobs


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

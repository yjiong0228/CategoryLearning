"""Training precomputation preserves pair-specific search and a future-data gate."""
import json
import time

import pytest

from src.Bayesian_state.workflows.analysis import model_1031_prefit as prefit


def test_training_pairs_redirect_only_completion_tables(tmp_path, monkeypatch):
    pairs = [["no_inertia", "full"], ["bayes_retention", "full"]]
    (tmp_path / "evaluation_prefit_plan.json").write_text(json.dumps(dict(workers=2, variant_pairs=pairs)))
    calls = []

    def fit(output, role, variants):
        calls.append((role, variants))
        prefit.core.write_new(output / "evaluation_parameters.json", dict(variants=variants))
        prefit.core.write_new(output / f"ordinary_{variants[0]}.json", dict(unchanged=True))

    monkeypatch.setattr(prefit.core, "fit_role", fit)
    original_write = prefit.core.write_new
    prefit.run_training_pairs(tmp_path)
    assert calls == [("evaluation", pair) for pair in pairs]
    assert prefit.core.write_new is original_write
    assert not (tmp_path / "evaluation_parameters.json").exists()
    for pair in pairs:
        assert json.loads(prefit.prefit_path(tmp_path, pair[0]).read_text())["variants"] == pair
        assert (tmp_path / f"ordinary_{pair[0]}.json").exists()


def test_selected_pair_requires_structure_freeze_and_exact_variant_order(tmp_path):
    with pytest.raises(FileNotFoundError):
        prefit.reuse_selected_prefit(tmp_path)
    selection = dict(selected_variant="bayes_retention", evaluation_variants=["bayes_retention", "full"],
                     frozen_at_utc="2026-11-01T00:00:00+00:00")
    (tmp_path / "structure_selection.json").write_text(json.dumps(selection))
    assert not prefit.reuse_selected_prefit(tmp_path)
    path = prefit.prefit_path(tmp_path, "bayes_retention")
    path.parent.mkdir()
    path.write_text(json.dumps(dict(groups=[dict(variant="bayes_retention"), dict(variant="full")])))
    assert prefit.reuse_selected_prefit(tmp_path)
    assert json.loads((tmp_path / "evaluation_parameters.json").read_text()) == json.loads(path.read_text())
    selection["evaluation_variants"].reverse()
    (tmp_path / "structure_selection.json").write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="different model pair"):
        prefit.reuse_selected_prefit(tmp_path)


def test_gate_requires_closed_worker_group_and_cpu_reader_ignores_partial_line(tmp_path):
    assert prefit.await_prefits(tmp_path, time.time() + 1) is None
    (tmp_path / "evaluation_prefit_plan.json").write_text("{}")
    with pytest.raises(TimeoutError):
        prefit.await_prefits(tmp_path, time.time() - 1)
    finished = dict(workers_stopped=True, exit_code=0)
    (tmp_path / "evaluation_prefit_finished.json").write_text(json.dumps(finished))
    assert prefit.await_prefits(tmp_path, time.time() + 1) == finished
    (tmp_path / "supervisor_heartbeat_1.jsonl").write_text('{"cpu_seconds": 30}\n{"cpu_seconds":')
    assert prefit.main_counter(tmp_path) == 30.

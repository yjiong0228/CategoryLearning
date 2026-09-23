"""Report preparation must preserve categories, timing and ambiguous phenotypes."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.Bayesian_state.workflows.analysis.prepare_model_0923 import prepare_inventory


def _write(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _inputs(tmp_path):
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    cohort = tmp_path / "cohort.json"
    cohort.write_text(json.dumps({"subjects": [101]}))
    summary = [{"subject": 101, "condition": 1, "task": 1, "n": 3}]
    _write(analysis / "subjects.csv", summary)
    behavior = tmp_path / "behavior.csv"
    _write(behavior, [{"subject": 101, "criterion": 3, "delta_bic": 2.0}])
    trials = [{"iSub": 101, "condition": 1, "iSession": 1, "iBlock": 1,
               "iTrial": i, "trial": i, "choice": choice, "text": "same words",
               "oral_valid": True, "scored": i != 1}
              for i, choice in enumerate([1, 2, 1], 1)]
    _write(analysis / "trials.csv", trials)
    return dict(cohort_config=cohort, analysis_dir=analysis, behavior_summary=behavior,
                output_dir=tmp_path / "output"), trials


def test_inventory_keeps_choice_specific_reports_and_both_validity_counts(tmp_path):
    kwargs, _ = _inputs(tmp_path)
    manifest = prepare_inventory(**kwargs)
    assert manifest["unique_condition_choice_reports"] == 2
    assert manifest["encoded_reports"] == 3
    assert manifest["scored_encoded_reports"] == 2
    with (kwargs["output_dir"] / "subjects.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert row["shape_evidence"] == "unresolved"
    assert row["learning_type"] == "not_assigned"
    with (kwargs["output_dir"] / "report_review.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert {r["catalogue_coverage"] for r in rows} == {"unreviewed"}
    assert {r["text"] for r in rows} == {"same words"}
    with pytest.raises(FileExistsError):
        prepare_inventory(**kwargs)


def test_inventory_rejects_duplicate_trials_before_creating_output(tmp_path):
    kwargs, trials = _inputs(tmp_path)
    trials[-1] = trials[0].copy()
    _write(kwargs["analysis_dir"] / "trials.csv", trials)
    with pytest.raises(ValueError, match="duplicate trial key"):
        prepare_inventory(**kwargs)
    assert not kwargs["output_dir"].exists()

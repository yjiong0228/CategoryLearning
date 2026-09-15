from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from src.Bayesian_state import run_model_evaluation as evaluation_cli
from src.Bayesian_state.evaluation.evaluator import ModelEvaluator


def _result(distance_mode="boundary", backend="particle_filter", **resolved):
    return {
        "subject_id": 129,
        "condition": 1,
        "model_provenance": {
            "resolved": {
                "likelihood": {"distance_mode": distance_mode},
                "inference": {"backend": backend},
                **resolved,
            }
        },
        "representative_run": {},
    }


@pytest.mark.parametrize(
    "info, expected",
    [
        (_result(), {"region": [129]}),
        (_result("prototype"), {"center": [129]}),
        (_result(None, encoding={"distance_mode": "boundary"}), {"region": [129]}),
    ],
)
def test_oral_auto_uses_saved_encoding(info, expected):
    assert evaluation_cli.resolve_oral_modes({129: info}, "auto") == expected


def test_oral_auto_separates_subjects_with_different_encodings():
    results = {129: _result(), 101: _result("prototype"), 105: _result()}
    assert evaluation_cli.resolve_oral_modes(results, "auto") == {
        "center": [101],
        "region": [105, 129],
    }


@pytest.mark.parametrize("info", [{}, _result("unsupported")])
def test_oral_auto_requires_known_encoding(info):
    with pytest.raises(ValueError, match="129.*--oral-mode"):
        evaluation_cli.resolve_oral_modes({129: info}, "auto")


def test_oral_auto_rejects_conflicting_saved_encoding():
    with pytest.raises(ValueError, match="129.*conflict"):
        evaluation_cli.resolve_oral_modes(
            {129: _result("prototype", encoding={"distance_mode": "boundary"})},
            "auto",
        )


@pytest.mark.parametrize("mode", ["center", "region"])
def test_explicit_oral_mode_supports_legacy_and_comparison_runs(mode):
    assert evaluation_cli.resolve_oral_modes({129: {}, 101: _result()}, mode) == {
        mode: [101, 129]
    }


def test_pf_detection_uses_provenance_without_state_logs():
    assert ModelEvaluator.is_particle_filter_result(_result())
    assert not ModelEvaluator.is_particle_filter_result(_result(backend="trajectory"))


@pytest.mark.parametrize(
    "backend, flags, expect_trajectory",
    [
        ("particle_filter", [], False),
        ("particle_filter", ["--include-trajectory"], True),
        ("trajectory", [], True),
        ("trajectory", ["--skip-trajectory"], False),
    ],
)
def test_cli_trajectory_outputs_follow_backend_and_explicit_flags(
    tmp_path: Path, monkeypatch, backend, flags, expect_trajectory
):
    input_dir = tmp_path / "simulation"
    subject_dir = input_dir / "subjects"
    subject_dir.mkdir(parents=True)
    (subject_dir / "subject_129.json").write_text(
        json.dumps(_result(backend=backend)), encoding="utf-8"
    )
    output_dir = tmp_path / "evaluation"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_model_evaluation",
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--skip-basic", "--skip-behavior-ppc", "--skip-oral",
            *flags,
        ],
    )
    evaluation_cli.main()

    for directory in ("trajectory_accuracy", "trajectory_posterior"):
        assert (output_dir / directory).exists() == expect_trajectory
    if backend == "particle_filter" and not flags:
        manifest = json.loads((output_dir / "evaluation_manifest.json").read_text())
        skipped = {entry["name"] for entry in manifest if entry["status"] == "skipped"}
        assert {"trajectory_accuracy", "trajectory_posterior"} <= skipped

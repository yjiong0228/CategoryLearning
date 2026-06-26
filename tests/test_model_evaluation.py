from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import numpy as np

from src.Bayesian_state.run_model_evaluation import load_simulation_results
from src.Bayesian_state.utils.model_evaluation import ModelEval


def test_load_simulation_results_cli_window_overrides_saved_window() -> None:
    root = Path("tmp") / f"model_eval_{uuid.uuid4().hex}"
    subject_dir = root / "subjects"
    subject_dir.mkdir(parents=True)
    payload = {
        "subject_id": 101,
        "condition": 1,
        "simulation": {"window_size": 8},
        "selection": {
            "selection_prediction_mode": "prior_t",
            "selection_meta": {"window_size": 8},
        },
        "representative_run": {
            "metrics_by_mode": {
                "prior_t": {
                    "true_acc": [1.0, 0.0, 1.0],
                    "pred_acc": [0.8, 0.4, 0.7],
                    "sliding_true_acc": [0.5],
                    "sliding_pred_acc": [0.6],
                    "sliding_pred_acc_std": [0.1],
                }
            }
        },
    }
    try:
        (subject_dir / "subject_101.json").write_text(json.dumps(payload), encoding="utf-8")

        loaded = load_simulation_results(root, window_size=32)

        assert loaded[101]["window_size"] == 32
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_exp_accuracy_alpha_override_recomputes_saved_curve() -> None:
    evaluator = ModelEval()
    info = {
        "condition": 1,
        "window_size": 8,
        "true_acc": [1.0, 0.0],
        "pred_acc": [0.8, 0.2],
        "exp_accuracy_alpha": 0.5,
        "exp_true_acc": [0.75, 0.375],
        "exp_pred_acc": [0.65, 0.425],
    }

    got = evaluator.compute_exponential_accuracy_metrics(info, exp_accuracy_alpha=0.25)

    assert got["exp_accuracy_alpha"] == 0.25
    assert np.allclose(got["exp_true_acc"], [0.625, 0.46875])
    assert np.allclose(got["exp_pred_acc"], [0.575, 0.48125])


def test_window_override_recomputes_sliding_accuracy() -> None:
    evaluator = ModelEval()
    info = {
        "window_size": 2,
        "true_acc": [np.nan, 1.0, 0.0, 1.0, 1.0],
        "pred_acc": [np.nan, 0.8, 0.2, 0.6, 0.4],
        "sliding_true_acc": [0.5, 0.5, 1.0],
        "sliding_pred_acc": [0.5, 0.4, 0.5],
        "sliding_pred_acc_std": [0.1, 0.1, 0.1],
    }

    got = evaluator.compute_accuracy_metrics(info, window_size=3)

    assert got["window_size"] == 3
    assert np.allclose(got["sliding_true_acc"], [2.0 / 3.0, 2.0 / 3.0])
    assert np.allclose(got["sliding_pred_acc"], [0.5333333333333333, 0.4])

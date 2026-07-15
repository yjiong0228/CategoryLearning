from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import numpy as np

from src.Bayesian_state.run_model_evaluation import load_simulation_results
from src.Bayesian_state.utils.model_evaluation import ModelEval
from src.Bayesian_state.utils.stream import StreamList


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


def test_dynamic_profile_and_active_set_log_parsing() -> None:
    info = {
        "strategy_counts_log": [
            {
                "policy_probabilities": {
                    "stable": 0.4,
                    "conservative": 0.3,
                    "aggressive": 0.2,
                    "stubborn": 0.1,
                },
                "selected_policy_method": "stable",
                "profile_policy": {
                    "retained_count": 3,
                    "newcomer_count": 1,
                },
                "active_total": 4,
            },
            {
                "policy_probabilities": {
                    "stable": 0.1,
                    "conservative": 0.2,
                    "aggressive": 0.6,
                    "stubborn": 0.1,
                },
                "selected_policy_method": "aggressive",
                "profile_policy": {
                    "retained_count": 1,
                    "newcomer_count": 3,
                },
                "active_total": 4,
            },
        ]
    }

    activation = ModelEval._profile_activation_data(info)
    counts = ModelEval._active_set_count_rows(info)

    assert activation is not None
    assert activation["policies"] == ["conservative", "stable", "aggressive", "stubborn"]
    assert np.allclose(activation["probabilities"].sum(axis=1), 1.0)
    assert activation["selected"] == ["stable", "aggressive"]
    assert counts[["retained", "newcomer", "total"]].to_dict("records") == [
        {"retained": 3.0, "newcomer": 1.0, "total": 4.0},
        {"retained": 1.0, "newcomer": 3.0, "total": 4.0},
    ]


def test_predictive_band_includes_full_range_and_uses_lowest_error_run() -> None:
    root = Path("tmp") / f"model_eval_band_{uuid.uuid4().hex}"
    subject_dir = root / "subjects"
    subject_dir.mkdir(parents=True)
    stream_path = root / "subject_118_raw_runs.pkl.gz"
    metric_template = {
        "sliding_true_acc": [0.6, 0.4],
        "sliding_pred_acc_std": [0.1, 0.1],
    }
    runs = [
        {
            "run_index": 0,
            "mean_error": 0.4,
            "selection_prediction_mode": "prior_t",
            "metrics_by_mode": {
                "prior_t": {**metric_template, "sliding_pred_acc": [0.2, 0.3]}
            },
        },
        {
            "run_index": 1,
            "mean_error": 0.1,
            "selection_prediction_mode": "prior_t",
            "metrics_by_mode": {
                "prior_t": {**metric_template, "sliding_pred_acc": [0.8, 0.7]}
            },
        },
    ]
    try:
        StreamList(str(stream_path), 0).extend(runs)
        payload = {
            "subject_id": 118,
            "condition": 1,
            "simulation": {"window_size": 2},
            "selection": {"selection_prediction_mode": "prior_t"},
            "representative_run": {"metrics_by_mode": {"prior_t": runs[0]["metrics_by_mode"]["prior_t"]}},
            "raw_runs_ref": {"path": "../subject_118_raw_runs.pkl.gz", "count": 2},
        }
        subject_path = subject_dir / "subject_118.json"
        subject_path.write_text(json.dumps(payload), encoding="utf-8")

        band = ModelEval()._predictive_accuracy_band_data(subject_path, eval_prediction_mode="prior_t")

        assert np.allclose(band["q00"], [0.2, 0.3])
        assert np.allclose(band["q100"], [0.8, 0.7])
        assert np.allclose(band["best_curve"], [0.8, 0.7])
        assert band["best_run_index"] == 1
    finally:
        shutil.rmtree(root, ignore_errors=True)

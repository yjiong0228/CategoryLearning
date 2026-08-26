from __future__ import annotations

import numpy as np
import pytest

from src.Bayesian_state.evaluation.evaluator import ModelEvaluator
from src.Bayesian_state.hypothesis_space import ContinuousPartition
from src.Bayesian_state.simulation.provenance import build_model_provenance


def _engine_config(distance_mode: str = "boundary") -> dict:
    return {
        "partition": {
            "class": (
                "src.Bayesian_state.hypothesis_space.observation_model."
                "continuous_partition.ContinuousPartition"
            ),
            "kwargs": {
                "n_dims": 4,
                "n_cats": 4,
                "boundary_distance_method": "kkt_active_set_projection",
                "boundary_distance_tolerance": 1e-9,
                "boundary_projection_iterations": 100,
                "label_permutation_policy": "identity_only",
                "similarity_n_samples": 8,
            },
        },
        "likelihood": {"distance_mode": distance_mode},
        "modules": {},
        "agenda": [],
    }


def _info() -> dict:
    partition = ContinuousPartition(4, 4, similarity_n_samples=8)
    distribution = np.zeros(partition.length)
    distribution[0] = 1.0
    trials = {
        "choice": [1, 2, 3, 4],
        "category": [1, 2, 3, 4],
        "feature1": [0.1, 0.2, 0.3, 0.4],
        "feature2": [0.2, 0.3, 0.4, 0.5],
        "feature3": [0.3, 0.4, 0.5, 0.6],
        "feature4": [0.4, 0.5, 0.6, 0.7],
    }
    return {
        "condition": 2,
        "subject_trials": trials,
        "window_size": 2,
        "eval_prediction_mode": "prior_t",
        "prior_log": [distribution.tolist() for _ in range(4)],
        "best_step_results": [
            {"perceived_stimulus": [0.1, 0.2, 0.3, 0.4]}
            for _ in range(4)
        ],
        "beta_log": [
            np.full(partition.length, float(index + 1)).tolist()
            for index in range(4)
        ],
        "model_provenance": build_model_provenance(
            _engine_config(), repeat_aggregation="mean_probability"
        ),
    }


def test_provenance_records_resolved_encoding() -> None:
    provenance = build_model_provenance(
        _engine_config(), repeat_aggregation="mean_probability"
    )
    assert provenance["schema_version"] == 2
    encoding = provenance["resolved"]["encoding"]
    assert encoding["distance_mode"] == "boundary"
    assert encoding["boundary_distance_method"] == "kkt_active_set_projection"
    assert encoding["label_permutation_policy"] == "identity_only"


def test_family_recompute_uses_trial_beta_and_saved_mode(monkeypatch) -> None:
    calls: list[tuple[float, str]] = []
    original = ContinuousPartition.get_category_probabilities

    def recording(self, hypo, data, beta, distance_mode=None, **kwargs):
        calls.append((float(beta), str(distance_mode)))
        return original(
            self,
            hypo,
            data,
            beta,
            distance_mode=distance_mode,
            **kwargs,
        )

    monkeypatch.setattr(ContinuousPartition, "get_category_probabilities", recording)
    result = ModelEvaluator().compute_family_accuracy_metrics(_info())
    assert result["pred_family_acc"].shape == (4,)
    assert calls == [(2.0, "boundary"), (3.0, "boundary"), (4.0, "boundary")]


def test_family_recompute_cli_mode_override_has_priority(monkeypatch) -> None:
    modes: list[str] = []
    original = ContinuousPartition.get_category_probabilities

    def recording(self, hypo, data, beta, distance_mode=None, **kwargs):
        modes.append(str(distance_mode))
        return original(
            self,
            hypo,
            data,
            beta,
            distance_mode=distance_mode,
            **kwargs,
        )

    monkeypatch.setattr(ContinuousPartition, "get_category_probabilities", recording)
    ModelEvaluator().compute_family_accuracy_metrics(
        _info(), distance_mode="prototype"
    )
    assert modes == ["prototype", "prototype", "prototype"]


def test_family_recompute_rejects_missing_provenance() -> None:
    info = _info()
    info.pop("model_provenance")
    with pytest.raises(ValueError, match="partition provenance"):
        ModelEvaluator().compute_family_accuracy_metrics(info)

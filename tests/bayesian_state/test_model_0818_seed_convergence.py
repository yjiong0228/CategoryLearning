from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.run_model_0818_seed_convergence import (
    build_seed_jobs,
    seed_cache_path,
)
from src.Bayesian_state.optimization.seed_convergence import (
    bootstrap_pairwise_delta_nll,
    evaluate_seed_convergence,
    summarize_independent_seed_halves,
    summarize_nested_seed_budgets,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/specific_models/model_0818_seed_convergence.yaml"
HIGH_CONFIG = (
    ROOT / "configs/specific_models/model_0818_high_budget_convergence.yaml"
)


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _constant_banks(seed_count: int = 16) -> tuple[dict[str, np.ndarray], np.ndarray]:
    choices = np.asarray([1, 2], dtype=int)
    candidate_a = np.tile(
        np.asarray([[[0.9, 0.1], [0.2, 0.8]]], dtype=float),
        (seed_count, 1, 1),
    )
    candidate_b = np.tile(
        np.asarray([[[0.7, 0.3], [0.4, 0.6]]], dtype=float),
        (seed_count, 1, 1),
    )
    return {"candidate_a": candidate_a, "candidate_b": candidate_b}, choices


def test_config_declares_128_unique_candidate_paired_seed_tasks() -> None:
    config = _yaml(CONFIG)
    design = config["design"]
    datasets = [
        {"dataset_id": dataset_id, "subject_id": 103, "true_profile_id": "x"}
        for dataset_id in config["dataset_ids"]
    ]
    profiles = [{"profile_id": f"p{index}"} for index in range(4)]
    jobs = build_seed_jobs(
        datasets=datasets,
        profiles=profiles,
        particle_count=design["particle_count"],
        filter_seed_count=design["filter_seed_count"],
        base_seed=20260818,
        seed_family=design["seed_family"],
    )

    assert len(jobs) == config["execution"]["expected_task_count"] == 128
    assert config["design"]["n_jobs"] == 128
    assert config["design"]["nested_checkpoints"] == [2, 4, 8, 16]
    assert len(
        {
            (
                job["dataset"]["dataset_id"],
                job["profile"]["profile_id"],
                job["filter_seed"],
            )
            for job in jobs
        }
    ) == 128
    for dataset_id in config["dataset_ids"]:
        candidate_seed_lists = []
        for profile in profiles:
            candidate_seed_lists.append(
                [
                    job["filter_seed"]
                    for job in jobs
                    if job["dataset"]["dataset_id"] == dataset_id
                    and job["profile"]["profile_id"] == profile["profile_id"]
                ]
            )
        assert all(
            seeds == candidate_seed_lists[0] for seeds in candidate_seed_lists[1:]
        )
        assert len(set(candidate_seed_lists[0])) == 16


def test_high_budget_config_declares_1024_paired_seed_tasks() -> None:
    config = _yaml(HIGH_CONFIG)
    design = config["design"]
    datasets = [
        {"dataset_id": dataset_id, "subject_id": 103, "true_profile_id": "x"}
        for dataset_id in config["dataset_ids"]
    ]
    profiles = [{"profile_id": f"p{index}"} for index in range(4)]
    jobs = build_seed_jobs(
        datasets=datasets,
        profiles=profiles,
        particle_count=design["particle_count"],
        filter_seed_count=design["filter_seed_count"],
        base_seed=20260818,
        seed_family=design["seed_family"],
    )

    assert design["particle_count"] == 128
    assert design["filter_seed_count"] == 128
    assert design["nested_checkpoints"] == [16, 32, 64, 128]
    assert design["independent_halves"] == [64, 64]
    assert design["n_jobs"] == 128
    assert len(jobs) == config["execution"]["expected_task_count"] == 1024
    assert len(
        {
            (
                job["dataset"]["dataset_id"],
                job["profile"]["profile_id"],
                job["filter_seed"],
            )
            for job in jobs
        }
    ) == 1024


def test_seed_cache_path_is_unique_by_dataset_profile_and_seed(tmp_path: Path) -> None:
    left = seed_cache_path(tmp_path, "namespace", "dataset", "p0", 123)
    right = seed_cache_path(tmp_path, "namespace", "dataset", "p1", 123)

    assert left != right
    assert left.name == "seed_123.npz"
    assert "per_seed" in left.parts


def test_nested_seed_scores_use_exact_prefixes_and_mean_probabilities() -> None:
    choices = np.asarray([1, 2], dtype=int)
    banks = {
        "candidate_a": np.asarray(
            [
                [[0.9, 0.1], [0.2, 0.8]],
                [[0.9, 0.1], [0.2, 0.8]],
                [[0.7, 0.3], [0.4, 0.6]],
                [[0.7, 0.3], [0.4, 0.6]],
            ],
            dtype=float,
        ),
        "candidate_b": np.tile(
            np.asarray([[[0.6, 0.4], [0.3, 0.7]]], dtype=float),
            (4, 1, 1),
        ),
    }

    scores = summarize_nested_seed_budgets(banks, choices, [2, 4])
    a_b2 = scores[
        scores["fit_profile_id"].eq("candidate_a")
        & scores["filter_seed_count"].eq(2)
    ].iloc[0]
    a_b4 = scores[
        scores["fit_profile_id"].eq("candidate_a")
        & scores["filter_seed_count"].eq(4)
    ].iloc[0]

    assert a_b2["total_nll"] == pytest.approx(-np.log(0.9) - np.log(0.8))
    assert a_b4["total_nll"] == pytest.approx(-np.log(0.8) - np.log(0.7))
    assert a_b2["seed_subset"] == "prefix_0_1"
    assert a_b4["seed_subset"] == "prefix_0_3"


def test_independent_halves_and_bootstrap_preserve_candidate_pairing() -> None:
    banks, choices = _constant_banks()

    half_scores, half_summary = summarize_independent_seed_halves(banks, choices)
    first = bootstrap_pairwise_delta_nll(
        banks,
        choices,
        replicates=200,
        confidence_level=0.95,
        bootstrap_seed=1234,
    )
    second = bootstrap_pairwise_delta_nll(
        banks,
        choices,
        replicates=200,
        confidence_level=0.95,
        bootstrap_seed=1234,
    )

    assert len(half_scores) == 4
    assert half_summary["same_winner"] is True
    assert half_summary["maximum_absolute_candidate_nll_difference"] == 0.0
    pd.testing.assert_frame_equal(first, second)
    assert first.loc[0, "ci_half_width"] == pytest.approx(0.0)
    assert bool(first.loc[0, "ci_excludes_zero"]) is True


def test_predeclared_seed_convergence_gates_do_not_authorize_formal_recovery() -> None:
    checkpoint_scores = pd.DataFrame(
        [
            {
                "dataset_id": dataset_id,
                "fit_profile_id": profile,
                "filter_seed_count": seed_count,
                "total_nll": base + (0.05 if seed_count == 16 else 0.0),
            }
            for dataset_id in ("d0", "d1")
            for profile, base in (("p0", 10.0), ("p1", 11.0))
            for seed_count in (8, 16)
        ]
    )
    intervals = pd.DataFrame(
        [
            {
                "dataset_id": dataset_id,
                "ci_half_width": 0.1,
                "ci_excludes_zero": True,
            }
            for dataset_id in ("d0", "d1")
        ]
    )
    gates = _yaml(CONFIG)["convergence_gates"]

    changes, summary = evaluate_seed_convergence(
        checkpoint_scores, intervals, gates
    )

    assert np.allclose(changes["absolute_nll_change"], 0.05)
    assert summary["budget_status"] == "seed_convergence_gate_passed"
    assert summary["paired_particle_count_comparison_authorized"] is True
    assert summary["formal_recovery_authorized"] is False
    assert summary["observed_data_fit_authorized"] is False


def test_probability_bank_rejects_non_normalized_probabilities() -> None:
    banks, choices = _constant_banks(seed_count=4)
    banks["candidate_a"][0, 0] = [0.5, 0.6]

    with pytest.raises(ValueError, match="sum to one"):
        summarize_nested_seed_budgets(banks, choices, [2, 4])

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_model_0813_pf_choice_layer_audit import (
    RuntimeDesign,
    _layer_metrics,
    _validate_probabilities,
    summarize_effects,
)


def test_choice_layer_probability_validation_and_metrics():
    probability = np.asarray(
        [[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]], dtype=float
    )
    checked = _validate_probabilities(
        probability, layer_id="test", trial_count=3
    )
    metrics = _layer_metrics(checked, np.asarray([1, 2, 1]))

    expected = -np.mean(np.log([0.8, 0.7, 0.6]))
    assert metrics["mean_nll"] == expected
    assert metrics["total_nll"] == expected * 3
    assert metrics["minimum_selected_probability"] == 0.6
    assert metrics["maximum_normalization_error"] == 0.0


def test_choice_layer_summary_uses_disjoint_panels_and_frozen_gates():
    comparison = {
        "contrast_id": "mechanism_a",
        "mechanism_id": "OBS-X",
        "comparator_layer": "before",
        "mechanism_layer": "after",
        "interpretation_role": "test",
    }
    rows = []
    subject_effects = np.linspace(0.008, 0.015, 8)
    seed_offsets = np.asarray([-2, -1, 1, 2, -1.5, -0.5, 0.5, 1.5]) * 1e-5
    for subject_id, base_effect in enumerate(subject_effects, start=1):
        for repeat, offset in enumerate(seed_offsets):
            rows.append(
                {
                    "subject_id": subject_id,
                    "filter_repeat": repeat,
                    "trial_count": 128,
                    "contrast_id": "mechanism_a",
                    "delta_mean_nll": base_effect + offset,
                    "comparator_total_nll": 0.5 * 128,
                    "mechanism_total_nll": (0.5 - base_effect - offset) * 128,
                }
            )
    scores = pd.DataFrame(rows)
    runtime = RuntimeDesign(
        subjects=tuple(range(1, 9)),
        trials_per_subject=128,
        particle_count=128,
        seed_indices=tuple(range(8)),
        training_seed_indices=(0, 1, 2, 3),
        validation_seed_indices=(4, 5, 6, 7),
        n_jobs=1,
        bootstrap_repeats=500,
    )
    design = {
        "bootstrap_confidence": 0.95,
        "base_seed": 20260814,
        "stability_gates": {
            "minimum_train_validation_subject_spearman": 0.70,
            "minimum_subject_sign_agreement": 0.75,
            "require_aggregate_sign_agreement": True,
            "maximum_median_paired_mean_nll_mcse": 0.001,
        },
        "practical_effect_rule": {
            "baseline_mean_nll_fraction": 0.01,
            "paired_seed_sd_multiplier": 2.0,
        },
    }

    subjects, contrasts, summary = summarize_effects(
        scores, [comparison], runtime, design
    )
    assert len(subjects) == 8
    row = contrasts.iloc[0]
    assert row["train_validation_subject_spearman"] == 1.0
    assert row["subject_split_sign_agreement"] == 1.0
    assert bool(row["all_numerical_stability_gates_pass"])
    assert row["conditional_triage"] == "advance_conditional_benefit"
    assert summary["numerically_stable_contrast_n"] == 1


def test_choice_layer_summary_marks_unstable_sign_reversal():
    comparison = {
        "contrast_id": "mechanism_b",
        "mechanism_id": "OBS-Y",
        "comparator_layer": "before",
        "mechanism_layer": "after",
        "interpretation_role": "test",
    }
    rows = []
    for subject_id in range(1, 9):
        for repeat in range(8):
            effect = 0.01 if repeat < 4 else -0.01
            rows.append(
                {
                    "subject_id": subject_id,
                    "filter_repeat": repeat,
                    "trial_count": 128,
                    "contrast_id": "mechanism_b",
                    "delta_mean_nll": effect,
                    "comparator_total_nll": 64.0,
                    "mechanism_total_nll": (0.5 - effect) * 128,
                }
            )
    runtime = RuntimeDesign(
        subjects=tuple(range(1, 9)),
        trials_per_subject=128,
        particle_count=128,
        seed_indices=tuple(range(8)),
        training_seed_indices=(0, 1, 2, 3),
        validation_seed_indices=(4, 5, 6, 7),
        n_jobs=1,
        bootstrap_repeats=100,
    )
    design = {
        "bootstrap_confidence": 0.95,
        "base_seed": 20260814,
        "stability_gates": {
            "minimum_train_validation_subject_spearman": 0.70,
            "minimum_subject_sign_agreement": 0.75,
            "require_aggregate_sign_agreement": True,
            "maximum_median_paired_mean_nll_mcse": 0.001,
        },
        "practical_effect_rule": {
            "baseline_mean_nll_fraction": 0.01,
            "paired_seed_sd_multiplier": 2.0,
        },
    }
    _, contrasts, _ = summarize_effects(
        pd.DataFrame(rows), [comparison], runtime, design
    )
    row = contrasts.iloc[0]
    assert not bool(row["aggregate_sign_agreement"])
    assert not bool(row["all_numerical_stability_gates_pass"])
    assert row["conditional_triage"] == "unresolved_numerical"

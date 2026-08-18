from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.run_model_0815_p0_persistent_execution_counterfactual import (
    BETA_SCOPE_PATH,
    EXECUTION_PATH,
    build_variant_engine,
    summarize_contrast,
    summarize_variant,
    validate_variant_bank,
)


ROOT = Path(__file__).resolve().parents[2]


def _get_path(root: dict, path: str):
    value = root
    for part in path.split("."):
        value = value[part]
    return value


def _base_engine() -> dict:
    return yaml.safe_load(
        (ROOT / "configs/model_struct/pmh_model_cond1_0815_p0.yaml").read_text(
            encoding="utf-8"
        )
    )


def _variant(variant_id: str, enabled: bool, scope: str) -> dict:
    return {
        "variant_id": variant_id,
        "persistent_execution_enabled": enabled,
        "beta_update_scope": scope,
    }


def test_three_model_bank_separates_execution_from_beta_scope() -> None:
    base = _base_engine()
    variants = [
        _variant("off", False, "active_hypotheses"),
        _variant("on_active", True, "active_hypotheses"),
        _variant("on_executed", True, "executed_hypothesis"),
    ]
    engines = validate_variant_bank(base, variants)
    assert _get_path(engines["off"], EXECUTION_PATH) is False
    assert _get_path(engines["off"], BETA_SCOPE_PATH) == "active_hypotheses"
    assert _get_path(engines["on_active"], EXECUTION_PATH) is True
    assert _get_path(engines["on_active"], BETA_SCOPE_PATH) == "active_hypotheses"
    assert _get_path(engines["on_executed"], EXECUTION_PATH) is True
    assert _get_path(engines["on_executed"], BETA_SCOPE_PATH) == "executed_hypothesis"
    with pytest.raises(ValueError, match="undefined"):
        build_variant_engine(base, _variant("invalid", False, "executed_hypothesis"))


def _panel(choice_probability: float) -> dict[str, np.ndarray]:
    choices = np.asarray([0, 0, 1, 1], dtype=int)
    base = np.column_stack(
        [
            np.where(choices == 0, choice_probability, 1.0 - choice_probability),
            np.where(choices == 1, choice_probability, 1.0 - choice_probability),
        ]
    )
    probability = np.stack([base, base, base, base])
    prior = np.tile(np.asarray([[[0.7, 0.3]] * 4]), (4, 1, 1))
    return {
        "choice_probability": probability,
        "marginal_prior": prior,
        "pre_choice_ess": np.full((4, 4), 48.0),
        "post_choice_ess": np.full((4, 4), 40.0),
        "resampled": np.zeros((4, 4), dtype=bool),
        "predictive_strategy_exploit": np.ones((4, 4)),
        "predictive_strategy_local_explore": np.zeros((4, 4)),
        "predictive_strategy_global_explore": np.zeros((4, 4)),
        "predictive_execution_switch_event_probability": np.zeros((4, 4)),
        "predictive_execution_dwell_trials": np.ones((4, 4)),
        "filter_seed": np.asarray([11, 12, 13, 14], dtype=np.uint64),
        "repeat_index": np.arange(4),
        "observed_choice_index": choices,
        "valid_trial_mask": np.ones(4, dtype=bool),
    }


def test_paired_counterfactual_delta_is_positive_for_better_mechanism() -> None:
    comparator_row, comparator = summarize_variant(
        _panel(0.60), subject_id=103, variant_id="off", particle_count=64
    )
    mechanism_row, mechanism = summarize_variant(
        _panel(0.70), subject_id=103, variant_id="on", particle_count=64
    )
    row, seed_rows = summarize_contrast(
        comparator_row,
        comparator,
        mechanism_row,
        mechanism,
        contrast={"contrast_id": "execution", "interpretation": "test"},
        numerical_gates={
            "maximum_paired_mean_nll_mcse": 0.001,
            "maximum_disjoint_half_delta_difference": 0.003,
        },
        practical_rule={
            "baseline_mean_nll_fraction": 0.01,
            "paired_seed_sd_multiplier": 2.0,
        },
    )
    assert row["paired_delta_mean_nll"] > 0.0
    assert row["numerically_stable"] is True
    assert row["predictive_geometry_prior_js"] == pytest.approx(0.0)
    assert len(seed_rows) == 4

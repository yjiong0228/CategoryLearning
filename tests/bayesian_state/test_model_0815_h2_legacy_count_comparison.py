from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.run_model_0815_h2_legacy_count_comparison import (
    LEGACY_CLASS,
    summarize_contrast,
    summarize_variant,
    validate_engine_pair,
    validate_legacy_engine,
)
from src.Bayesian_state.model import ModelContext, StateModel


ROOT = Path(__file__).resolve().parents[2]
ADAPTIVE_CONFIG = (
    ROOT / "configs/model_struct/pmh_model_cond1_0815_h1_adaptive_controller.yaml"
)
LEGACY_CONFIG = (
    ROOT / "configs/model_struct/pmh_model_cond1_0815_h2_legacy_count.yaml"
)


def _engine(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_legacy_h_preserves_exact_variable_count_policy_and_common_architecture() -> None:
    adaptive = _engine(ADAPTIVE_CONFIG)
    legacy = _engine(LEGACY_CONFIG)
    validate_legacy_engine(legacy)
    validate_engine_pair(adaptive, legacy)

    transition = legacy["modules"]["hypo_transitions_mod"]
    assert transition["class"] == LEGACY_CLASS
    assert transition["kwargs"]["init_num"] == 2
    assert transition["kwargs"]["max_active_hypotheses"] == 3
    assert [item["amount"] for item in transition["kwargs"]["strategies"]] == [
        "random_4",
        "opp_random_4",
    ]
    for key in ("likelihood", "choice_readout", "output_noise", "inference", "agenda"):
        assert legacy[key] == adaptive[key]
    for key in ("perception_mod", "memory_mod", "beta_mod"):
        assert legacy["modules"][key] == adaptive["modules"][key]


def test_legacy_transition_emits_pf_diagnostics_without_changing_policy() -> None:
    model = StateModel(
        _engine(LEGACY_CONFIG),
        context=ModelContext(condition=1, subject_id=103),
    )
    transition = model.engine.modules["hypo_transitions_mod"]
    before = transition.active.copy()
    model.begin_trial(np.asarray([0.2, 0.4, 0.6, 0.8], dtype=float))
    event = transition.transition_log[-1]

    after = transition.active.copy()
    expected_newcomers = int(np.sum(~np.isin(after, before)))
    assert event["replacement_count"] == expected_newcomers
    assert event["swap_event"] == (not np.array_equal(np.sort(before), np.sort(after)))
    assert event["swap_probability"] in (0.0, 1.0)
    assert event["diagnostic_probability_semantics"] == "realized_particle_indicator"
    assert event["retained_count"] + event["explored_count"] == event["active_total"]


def _panel(correct_probability: float, active_total: float) -> dict[str, np.ndarray]:
    choices = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=int)
    rows = np.column_stack(
        [
            np.where(choices == 0, correct_probability, 1.0 - correct_probability),
            np.where(choices == 1, correct_probability, 1.0 - correct_probability),
        ]
    )
    probability = np.stack([rows] * 4)
    prior = np.tile(np.asarray([[[0.6, 0.3, 0.1]] * 8]), (4, 1, 1))
    active = np.tile(
        np.asarray([[[active_total / 3.0] * 3] * 8]), (4, 1, 1)
    )
    traces = np.full((4, 8), 0.25)
    return {
        "choice_probability": probability,
        "marginal_prior": prior,
        "marginal_active_probability": active,
        "pre_choice_ess": np.full((4, 8), 28.0),
        "post_choice_ess": np.full((4, 8), 24.0),
        "resampled": np.zeros((4, 8), dtype=bool),
        "predictive_transition_rate": traces,
        "predictive_search_range": traces,
        "predictive_swap_probability": traces,
        "swap_event_probability": traces,
        "replacement_count": traces,
        "replacement_fraction": traces,
        "filter_seed": np.asarray([11, 12, 13, 14], dtype=np.uint64),
        "repeat_index": np.arange(4),
        "observed_choice_index": choices,
        "valid_trial_mask": np.ones(8, dtype=bool),
    }


def test_positive_legacy_minus_adaptive_delta_favors_adaptive() -> None:
    legacy_row, legacy = summarize_variant(
        _panel(0.60, 2.0),
        subject_id=103,
        variant_id="legacy",
        particle_count=32,
        train_trials=4,
        target_hypothesis=0,
    )
    adaptive_row, adaptive = summarize_variant(
        _panel(0.70, 3.0),
        subject_id=103,
        variant_id="adaptive",
        particle_count=32,
        train_trials=4,
        target_hypothesis=0,
    )
    row, seed_rows = summarize_contrast(
        legacy_row,
        legacy,
        adaptive_row,
        adaptive,
        train_trials=4,
        practical_fraction=0.01,
        seed_noise_multiplier=2.0,
    )

    assert row["paired_delta_nll_heldout"] > 0.0
    assert row["positive_seed_fraction_heldout"] == pytest.approx(1.0)
    assert row["heldout_active_probability_mae"] > 0.0
    assert len(seed_rows) == 16

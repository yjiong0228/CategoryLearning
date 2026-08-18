from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.run_model_0815_b0_adaptive_controller_pilot import (
    ADAPTIVE_CLASS,
    DUAL_MEMORY_CLASS,
    FIXED_CLASS,
    H1_PROFILE,
    build_training_matched_fixed_engine,
    derive_training_match,
    event_probability_to_slot_rate,
    summarize_contrast,
    summarize_variant,
    validate_minimal_adaptive_engine,
)
from src.Bayesian_state.model import ModelContext, StateModel


ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG = (
    ROOT / "configs/model_struct/pmh_model_cond1_0815_b0_adaptive_controller.yaml"
)
H1_MODEL_CONFIG = (
    ROOT / "configs/model_struct/pmh_model_cond1_0815_h1_adaptive_controller.yaml"
)


def _base_engine() -> dict:
    return yaml.safe_load(MODEL_CONFIG.read_text(encoding="utf-8"))


def _h1_engine() -> dict:
    return yaml.safe_load(H1_MODEL_CONFIG.read_text(encoding="utf-8"))


def test_minimal_adaptive_baseline_assembles_without_optional_leakage() -> None:
    engine = _base_engine()
    validate_minimal_adaptive_engine(engine)
    model = StateModel(engine, context=ModelContext(condition=1, subject_id=103))
    transition = model.engine.modules["hypo_transitions_mod"]
    beta = model.engine.modules["beta_mod"]

    assert engine["modules"]["hypo_transitions_mod"]["class"] == ADAPTIVE_CLASS
    assert transition.capacity == 3
    assert transition.failure_accumulator_enabled is True
    assert transition.persistent_execution_enabled is False
    assert beta.decrease_rate == 0.0
    assert beta.correct_additive == 0.0
    assert model.observation_likelihood.distance_mode == "boundary"
    assert model.observation_likelihood.default_beta == 10.0


def test_fixed_comparator_changes_only_transition_controller_and_matches_event_rate() -> None:
    adaptive = _base_engine()
    event_probability = 0.42
    global_search = 0.31
    fixed = build_training_matched_fixed_engine(
        adaptive,
        event_probability=event_probability,
        global_search=global_search,
    )
    kwargs = fixed["modules"]["hypo_transitions_mod"]["kwargs"]

    assert fixed["modules"]["hypo_transitions_mod"]["class"] == FIXED_CLASS
    assert kwargs["m"] == pytest.approx(
        event_probability_to_slot_rate(event_probability, capacity=3)
    )
    assert 1.0 - (1.0 - kwargs["m"]) ** 3 == pytest.approx(event_probability)
    assert kwargs["g"] == pytest.approx(global_search)
    for key in (
        "likelihood",
        "choice_readout",
        "output_noise",
        "agenda",
    ):
        assert fixed[key] == adaptive[key]
    for key in ("perception_mod", "memory_mod", "beta_mod"):
        assert fixed["modules"][key] == adaptive["modules"][key]

    model = StateModel(fixed, context=ModelContext(condition=1, subject_id=103))
    transition = model.engine.modules["hypo_transitions_mod"]
    assert transition.dynamic_rate is False
    assert transition.dynamic_range is False
    assert transition.current_event_probability == pytest.approx(event_probability)


def test_h1_baseline_uses_leaky_memory_and_one_dynamic_beta() -> None:
    engine = _h1_engine()
    validate_minimal_adaptive_engine(engine)
    model = StateModel(engine, context=ModelContext(condition=1, subject_id=103))
    memory = engine["modules"]["memory_mod"]
    beta = model.engine.modules["beta_mod"]

    assert engine["provenance"]["architecture_profile"] == H1_PROFILE
    assert memory["class"] == DUAL_MEMORY_CLASS
    assert memory["kwargs"]["gamma"] == pytest.approx(0.80)
    assert memory["kwargs"]["w0"] == pytest.approx(0.0)
    assert model.observation_likelihood.beta_source == "action"
    assert beta.beta_init == pytest.approx(5.0)
    assert beta.beta_max == pytest.approx(25.0)
    assert beta.decrease_rate == pytest.approx(0.15)
    assert beta.correct_additive == pytest.approx(1.0)
    assert beta.update_scope == "active_hypotheses"


def test_h1_fixed_comparator_preserves_memory_and_unified_dynamic_beta() -> None:
    adaptive = _h1_engine()
    fixed = build_training_matched_fixed_engine(
        adaptive,
        event_probability=0.36,
        global_search=0.24,
    )

    assert fixed["likelihood"] == adaptive["likelihood"]
    assert fixed["modules"]["memory_mod"] == adaptive["modules"]["memory_mod"]
    assert fixed["modules"]["beta_mod"] == adaptive["modules"]["beta_mod"]
    assert fixed["modules"]["hypo_transitions_mod"]["class"] == FIXED_CLASS


def _panel(correct_probability: float, *, adaptive_controls: bool) -> dict[str, np.ndarray]:
    choices = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=int)
    row = np.column_stack(
        [
            np.where(choices == 0, correct_probability, 1.0 - correct_probability),
            np.where(choices == 1, correct_probability, 1.0 - correct_probability),
        ]
    )
    probability = np.stack([row] * 4)
    prior = np.tile(np.asarray([[[0.6, 0.3, 0.1]] * choices.size]), (4, 1, 1))
    if adaptive_controls:
        event = np.asarray([0.0, 0.2, 0.4, 0.6, 0.3, 0.5, 0.4, 0.2])
        global_search = np.asarray([0.05, 0.1, 0.2, 0.3, 0.25, 0.2, 0.15, 0.1])
    else:
        event = np.asarray([0.0] + [0.4] * 7)
        global_search = np.asarray([0.2] * 8)
    event_panel = np.tile(event, (4, 1))
    global_panel = np.tile(global_search, (4, 1))
    return {
        "choice_probability": probability,
        "marginal_prior": prior,
        "pre_choice_ess": np.full((4, 8), 48.0),
        "post_choice_ess": np.full((4, 8), 40.0),
        "resampled": np.zeros((4, 8), dtype=bool),
        "predictive_transition_rate": np.full((4, 8), 0.1),
        "predictive_search_range": global_panel,
        "predictive_swap_probability": event_panel,
        "predictive_swap_event_probability": event_panel,
        "predictive_strategy_exploit": 1.0 - event_panel,
        "predictive_strategy_local_explore": event_panel * (1.0 - global_panel),
        "predictive_strategy_global_explore": event_panel * global_panel,
        "filter_seed": np.asarray([11, 12, 13, 14], dtype=np.uint64),
        "repeat_index": np.arange(4),
        "observed_choice_index": choices,
        "valid_trial_mask": np.ones(8, dtype=bool),
    }


def test_training_match_excludes_initialization_and_uses_only_train_trials() -> None:
    match = derive_training_match(
        _panel(0.65, adaptive_controls=True),
        train_trials=4,
        exclude_initialization_trial=True,
        capacity=3,
    )
    assert match["event_probability"] == pytest.approx(np.mean([0.2, 0.4, 0.6]))
    assert match["global_search"] == pytest.approx(np.mean([0.1, 0.2, 0.3]))
    assert match["matched_trial_count"] == 3


def test_heldout_paired_delta_is_positive_for_better_adaptive_predictions() -> None:
    fixed_row, fixed = summarize_variant(
        _panel(0.60, adaptive_controls=False),
        subject_id=103,
        variant_id="fixed",
        particle_count=64,
        train_trials=4,
    )
    adaptive_row, adaptive = summarize_variant(
        _panel(0.70, adaptive_controls=True),
        subject_id=103,
        variant_id="adaptive",
        particle_count=64,
        train_trials=4,
    )
    row, seed_rows = summarize_contrast(
        fixed_row,
        fixed,
        adaptive_row,
        adaptive,
        train_trials=4,
        practical_fraction=0.01,
        seed_noise_multiplier=2.0,
    )

    assert row["paired_delta_nll_heldout"] > 0.0
    assert row["ensemble_delta_nll_heldout"] > 0.0
    assert row["positive_seed_fraction_heldout"] == 1.0
    assert row["heldout_predictive_geometry_prior_js"] == pytest.approx(0.0)
    assert len(seed_rows) == 16

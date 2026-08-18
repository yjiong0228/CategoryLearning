from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.run_model_0815_h3_feedback_reactive_comparison import (
    build_projected_reactive_engine,
    project_feedback_reactive_controls,
    validate_reactive_pair,
)
from src.Bayesian_state.model import ModelContext, StateModel


ROOT = Path(__file__).resolve().parents[2]
ADAPTIVE_CONFIG = (
    ROOT / "configs/model_struct/pmh_model_cond1_0815_h1_adaptive_controller.yaml"
)
REACTIVE_CONFIG = (
    ROOT / "configs/model_struct/pmh_model_cond1_0815_h3_feedback_reactive.yaml"
)


def _engine(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _panel(event: np.ndarray, global_search: np.ndarray) -> dict[str, np.ndarray]:
    event = np.asarray(event, dtype=float)
    global_search = np.asarray(global_search, dtype=float)
    trial_n = event.size
    seed_n = 2
    probability = np.tile(np.asarray([[[0.6, 0.4]] * trial_n]), (seed_n, 1, 1))
    prior = np.tile(
        np.asarray([[[0.5, 0.3, 0.2]] * trial_n]), (seed_n, 1, 1)
    )
    event_panel = np.vstack([event, event + 0.02])
    global_panel = np.vstack([global_search, global_search + 0.02])
    return {
        "choice_probability": probability,
        "marginal_prior": prior,
        "pre_choice_ess": np.full((seed_n, trial_n), 28.0),
        "post_choice_ess": np.full((seed_n, trial_n), 24.0),
        "resampled": np.zeros((seed_n, trial_n), dtype=bool),
        "predictive_transition_rate": np.full((seed_n, trial_n), 0.1),
        "predictive_search_range": global_panel,
        "predictive_swap_probability": event_panel,
        "predictive_swap_event_probability": event_panel,
        "predictive_strategy_exploit": 1.0 - event_panel,
        "predictive_strategy_local_explore": event_panel * (1.0 - global_panel),
        "predictive_strategy_global_explore": event_panel * global_panel,
        "filter_seed": np.asarray([11, 12], dtype=np.uint64),
        "repeat_index": np.arange(seed_n),
        "observed_choice_index": np.zeros(trial_n, dtype=int),
        "valid_trial_mask": np.ones(trial_n, dtype=bool),
    }


def test_projection_uses_previous_feedback_and_train_trials_only() -> None:
    event = np.asarray([0.0, 0.2, 0.6, 0.4, 0.99, 0.99, 0.99, 0.99])
    global_search = np.asarray([0.0, 0.1, 0.3, 0.5, 0.99, 0.99, 0.99, 0.99])
    feedback = np.asarray([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    projection = project_feedback_reactive_controls(
        _panel(event, global_search),
        feedback,
        train_trials=4,
        enforce_error_not_less=False,
    )

    # The second seed is exactly 0.02 higher than the first.
    assert projection["event_after_correct"] == pytest.approx(
        np.mean([0.2, 0.4, 0.22, 0.42])
    )
    assert projection["event_after_error"] == pytest.approx(np.mean([0.6, 0.62]))
    assert projection["global_search"] == pytest.approx(
        np.mean([0.1, 0.3, 0.5, 0.12, 0.32, 0.52])
    )
    assert projection["calibration_trial_count"] == 3
    assert projection["uses_choice_nll"] is False
    assert projection["uses_heldout_trials"] is False


def test_projected_reactive_engine_changes_only_h_and_assembles() -> None:
    adaptive = _engine(ADAPTIVE_CONFIG)
    reactive = build_projected_reactive_engine(
        _engine(REACTIVE_CONFIG),
        {
            "event_after_correct": 0.20,
            "event_after_error": 0.50,
            "initial_event_probability": 0.20,
            "global_search": 0.30,
        },
    )
    validate_reactive_pair(adaptive, reactive)
    model = StateModel(
        reactive,
        context=ModelContext(condition=1, subject_id=103),
    )
    transition = model.engine.modules["hypo_transitions_mod"]

    assert transition.event_after_correct == pytest.approx(0.20)
    assert transition.event_after_error == pytest.approx(0.50)
    assert transition.current_g == pytest.approx(0.30)
    assert transition.capacity == 3
    assert transition.persistent_execution_enabled is False

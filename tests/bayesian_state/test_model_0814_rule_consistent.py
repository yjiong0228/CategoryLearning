from __future__ import annotations

from pathlib import Path

import yaml

from src.Bayesian_state.model import ModelContext, StateModel


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "configs"
    / "model_struct"
    / "pmh_model_cond1_0814_rule_consistent.yaml"
)


def test_rule_consistent_candidate_assembles_with_separated_precision_layers() -> None:
    engine_config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    model = StateModel(
        engine_config,
        context=ModelContext(condition=1, subject_id=103),
    )

    likelihood = model.observation_likelihood
    assert likelihood.distance_mode == "boundary"
    assert likelihood.beta_source == "fixed"
    assert likelihood.default_beta == 5.0

    transition = model.engine.modules["hypo_transitions_mod"]
    assert transition.capacity == 3
    assert transition.persistent_execution_enabled is True

    readout = engine_config["choice_readout"]["kwargs"]
    assert readout == {
        "method": "expectation",
        "power": 1.0,
        "strategy_confidence_gain": 0.0,
    }

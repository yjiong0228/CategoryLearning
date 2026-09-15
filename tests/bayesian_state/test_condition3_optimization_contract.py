"""Condition 3 fitting must retain the approved joint-memory architecture."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.Bayesian_state.model import ModelContext, ModuleRole, StateModel
from src.Bayesian_state.model.modules.pairing_memory import HierarchicalPairingMemoryModule
from src.Bayesian_state.optimization.model_0826 import build_model_0826_cell_engine


ROOT = Path(__file__).resolve().parents[2]


def _condition3_config():
    config = yaml.safe_load(
        (ROOT / "configs/exp123/model_struct/pmh_model_cond3_0826.yaml").read_text()
    )
    config["modules"]["perception_mod"]["kwargs"] = {
        "features": 4, "mean": [0.0] * 4, "std": [0.0] * 4, "module_seed": 19,
    }
    config["modules"]["hypo_transitions_mod"]["kwargs"]["module_seed"] = 19
    return config


def test_condition3_pmh_builder_retains_joint_learning_on_partial_feedback():
    base = _condition3_config()
    original = deepcopy(base)

    config = build_model_0826_cell_engine(base, "PMH")
    model = StateModel(config, context=ModelContext(condition=3))
    model.begin_trial(np.asarray([0.13, 0.27, 0.63, 0.89]))
    model.complete_trial(1, 0.5)

    memory = model.engine.get_module(ModuleRole.MEMORY)
    assert isinstance(memory, HierarchicalPairingMemoryModule)
    np.testing.assert_allclose(memory.joint.sum(axis=1), model.engine.posterior)
    assert not np.allclose(memory.pairing_marginal(), 1.0 / 3.0)
    assert base == original


@pytest.mark.parametrize("cell", ["P", "PM", "PH"])
def test_condition3_builder_rejects_unapproved_architecture_cells(cell):
    with pytest.raises(ValueError, match="condition 3.*PMH"):
        build_model_0826_cell_engine(_condition3_config(), cell)


@pytest.mark.parametrize("memory_class", [
    "src.Bayesian_state.model.modules.memory.DualMemoryModule",
    "src.Bayesian_state.model.modules.memory.BayesianMemoryModule",
])
def test_hierarchical_cell_builder_rejects_incompatible_memory(memory_class):
    config = _condition3_config()
    config["modules"]["memory_mod"]["class"] = memory_class

    with pytest.raises(ValueError, match="HierarchicalPairingMemoryModule"):
        build_model_0826_cell_engine(config, "PMH")


def test_joint_memory_requires_explicit_hierarchical_feedback_mode():
    config = _condition3_config()
    config["likelihood"]["feedback_likelihood_mode"] = "category_feedback"

    with pytest.raises(ValueError, match="hierarchical_pairing"):
        build_model_0826_cell_engine(config, "PMH")

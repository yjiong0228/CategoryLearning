"""Joint transport preserves the existing rule-marginal implementation."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.Bayesian_state.model import ModelContext, ModuleRole, StateModel
from src.Bayesian_state.model.modules.hypothesis_transition.contracts import HypothesisSelection
from src.Bayesian_state.model.modules.pairing_memory import transport_joint_belief


@pytest.fixture(scope="module")
def transition():
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(
        (root / "configs/exp123/model_struct/pmh_model_cond3_0826.yaml").read_text()
    )
    config["modules"]["perception_mod"]["kwargs"] = {
        "features": 4, "mean": [0.0] * 4, "std": [0.0] * 4, "module_seed": 19,
    }
    config["modules"]["hypo_transitions_mod"]["kwargs"]["init_hypotheses"] = [0, 37, 91]
    model = StateModel(config, context=ModelContext(condition=3))
    module = model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
    module._ensure_geometry()
    return module


@pytest.mark.parametrize("method, scalar_method", [
    ("similarity_transport", "_similarity_transport_prior"),
    ("mass_preserving_similarity_transport", "_mass_preserving_similarity_transport_prior"),
    ("pairwise_mass_transfer", "_pairwise_mass_transfer_prior"),
])
@pytest.mark.parametrize("global_fraction", [0.0, 0.37, 1.0])
@pytest.mark.parametrize("after, replacements", [
    ([0, 37, 91], []),
    ([0, 10, 37], [(91, 10)]),
    ([4, 10, 90], [(0, 4), (37, 10), (91, 90)]),
])
def test_joint_transport_matches_actual_scalar_rule_prior(
    transition, method, scalar_method, global_fraction, after, replacements,
):
    joint = np.zeros((transition.total_hypo, 3))
    joint[[0, 37, 91]] = [[0.14, 0.04, 0.02], [0.03, 0.21, 0.06], [0.10, 0.15, 0.25]]
    selection = HypothesisSelection.from_active_sets(
        [0, 37, 91], after, replacement_pairs=replacements,
    )
    transition.current_g = global_fraction
    scalar, _ = getattr(transition, scalar_method)(joint.sum(axis=1), selection)

    lifted, fallback = transport_joint_belief(
        joint, selection, method=method, local_kernel=transition._local_kernel,
        base_prior=transition.base_prior, global_fraction=global_fraction,
    )

    np.testing.assert_allclose(lifted.sum(axis=1), scalar, rtol=1e-13, atol=1e-15)
    assert not fallback
    assert np.all(np.isfinite(lifted))
    assert np.all(lifted >= 0.0)
    assert lifted.sum() == pytest.approx(1.0, abs=1e-14)

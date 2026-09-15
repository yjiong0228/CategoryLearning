"""Four-label binary-feedback PMH extension, including prediction timing."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.Bayesian_state.inference.dispatcher import run_inference_backend


@pytest.fixture
def case():
    config = yaml.safe_load(Path("configs/exp123/model_struct/pmh_model_cond1_0826.yaml").read_text())
    config["partition"]["kwargs"]["n_cats"] = 4
    config["inference"]["particle_count"] = 2
    frame = pd.read_csv("data/exp123/processed/Task2_processed.csv")
    frame = frame.loc[frame.iSub.eq(229)].iloc[:8]
    return dict(engine_config=config, subject_id=229, condition=2,
                stimulus=frame[[f"feature{i}" for i in range(1, 5)]].to_numpy(),
                choices=frame.choice.to_numpy(), feedback=frame.feedback.to_numpy(),
                inference_seed=20260909)


@pytest.mark.parametrize("execution", [False, True])
def test_condition2_normalization_determinism_and_prechoice_timing(case, execution):
    case["engine_config"]["modules"]["hypo_transitions_mod"]["kwargs"]["persistent_execution"]["enabled"] = execution
    result = run_inference_backend(**case)
    repeated = run_inference_backend(**case)
    p = result.marginal_probabilities
    assert p.shape == (8, 4)
    assert np.isfinite(p).all() and (p >= 0).all()
    np.testing.assert_allclose(p.sum(axis=1), 1., atol=1e-12)
    np.testing.assert_array_equal(p, repeated.marginal_probabilities)
    np.testing.assert_allclose(result.marginal_hypothesis_prior.sum(axis=1), 1.)
    # No future feedback/choice can affect any prediction through that trial.
    changed = dict(case, choices=case["choices"].copy(), feedback=case["feedback"].copy())
    changed["choices"][-1] = (changed["choices"][-1] % 4) + 1
    changed["feedback"][-1] = 1 - changed["feedback"][-1]
    np.testing.assert_array_equal(p, run_inference_backend(**changed).marginal_probabilities)


def test_condition2_uniform_output_lapse(case):
    np.testing.assert_allclose(run_inference_backend(**case, output_lapse=1.).marginal_probabilities, .25)


def test_condition2_rejects_partial_feedback_and_binary_audit(case):
    case["feedback"][0] = .5
    with pytest.raises(ValueError, match="binary feedback"):
        run_inference_backend(**case)
    case["feedback"][0] = 0.
    case["engine_config"]["inference"]["choice_transmission_audit"] = True
    with pytest.raises(ValueError, match="audit.*condition 1"):
        run_inference_backend(**case)


def test_condition3_requires_hierarchical_pairing_configuration(case):
    case["condition"] = 3
    with pytest.raises(ValueError, match="hierarchical_pairing"):
        run_inference_backend(**case)

"""Choice-conditioned report pooling preserves the fitted behavioral process."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.Bayesian_state.inference.backends.particle_filter import (
    _pool_oral_report_probabilities,
    run_state_model_particle_filter,
)


def _case(condition=1, *, oral=False, execution=False):
    config = yaml.safe_load(Path(
        f"configs/exp123/model_struct/pmh_model_cond{condition}_0826.yaml"
    ).read_text())
    config["modules"]["hypo_transitions_mod"]["kwargs"]["persistent_execution"]["enabled"] = execution
    if oral:
        config["oral_readout"] = {"method": "rule_content_v1", "kwargs": {
            "mention_probability": .8, "other_probability": .05,
            "body_naming_probability": .5, "body_reference_center": .75,
            "body_reference_width": .1, "extra_probability": .2,
            "extra_selection": "salience",
        }}
    return dict(
        engine_config=config, subject_id=101, condition=condition,
        stimulus=np.array([[.8, .2, .7, .3], [.2, .7, .3, .8],
                           [.6, .3, .8, .1], [.3, .8, .2, .7]]),
        choices=np.array([1, 2, 1, 2]), feedback=np.array([1., 0., 1., 0.]),
        particle_count=2, choice_readout_power=1., filter_seed=2026101108,
        resample_threshold_fraction=1.,
    )


def test_joint_pooling_conditions_particle_weights_on_choice():
    # Equally likely trajectories predict this response with .9 and .1.
    # They respectively produce disjoint reports A and B, so the report
    # probability after the response is .9/.1 rather than .5/.5.
    pooled = _pool_oral_report_probabilities(
        np.array([.5, .5]), np.array([.9, .1]),
        np.array([[.9, 0.], [0., .1]]),
    )
    np.testing.assert_allclose(pooled, [.9, .1], atol=1e-12)
    assert np.isnan(_pool_oral_report_probabilities(
        np.array([.5, .5]), np.zeros(2), np.zeros((2, 2))
    )).all()


@pytest.mark.parametrize("condition", [1, 2, 3])
@pytest.mark.parametrize("execution", [False, True])
def test_reports_preserve_behavior_rng_and_resampling(condition, execution):
    original = run_state_model_particle_filter(**_case(condition, execution=execution))
    reported = run_state_model_particle_filter(**_case(condition, oral=True, execution=execution))
    assert "oral_report_probabilities" not in original.observation_probabilities
    for namespace in ("observation_probabilities", "state_probabilities", "latent_summaries", "diagnostics"):
        before, after = getattr(original, namespace), getattr(reported, namespace)
        for key, value in before.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, after[key], err_msg=f"{namespace}.{key}")
    np.testing.assert_array_equal(original.final_weights, reported.final_weights)
    probabilities = reported.observation_probabilities["oral_report_probabilities"]
    codes = reported.observation_probabilities["oral_report_codes"]
    assert probabilities.shape == (4, len(codes))
    assert np.isfinite(probabilities).all() and (probabilities >= 0).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1., atol=1e-12)
    assert reported.observation_probabilities["oral_report_valid_mask"].all()
    if condition == 3:
        assert reported.metadata["oral_readout"]["probability_coordinate"] == "choice"


def test_current_feedback_does_not_change_current_report_prediction():
    case = _case(3, oral=True)
    original = run_state_model_particle_filter(**case)
    changed = deepcopy(case)
    changed["feedback"][-1] = .5
    alternate = run_state_model_particle_filter(**changed)
    np.testing.assert_array_equal(
        original.observation_probabilities["oral_report_probabilities"],
        alternate.observation_probabilities["oral_report_probabilities"],
    )


def test_invalid_trials_have_missing_report_rows():
    case = _case(oral=True)
    case["valid_trial_mask"] = [True, False, True, True]
    result = run_state_model_particle_filter(**case)
    mask = result.observation_probabilities["oral_report_valid_mask"]
    np.testing.assert_array_equal(mask, case["valid_trial_mask"])
    probabilities = result.observation_probabilities["oral_report_probabilities"]
    assert np.isnan(probabilities[1]).all()
    assert np.isfinite(probabilities[mask]).all()


@pytest.mark.parametrize("unsupported", [
    {"output_lapse": .1}, {"strategy_confidence_gain": 1.},
    {"rule_commitment_confidence_gain": 1.},
])
def test_report_prediction_requires_explicit_joint_model_for_choice_transform(unsupported):
    case = _case(oral=True)
    case.update(unsupported)
    with pytest.raises(ValueError, match="untransformed rule-mixture"):
        run_state_model_particle_filter(**case)


def test_simulation_exposes_report_predictions_only_when_enabled():
    from src.Bayesian_state.simulation.execution import evaluate_state_model_run

    case = _case(oral=True)
    case["engine_config"]["inference"]["particle_count"] = 2
    common = dict(
        subject_id=101, condition=1,
        arrays=(case["stimulus"], case["choices"], case["feedback"],
                np.where(case["feedback"] == 1., case["choices"], 3 - case["choices"])),
        params={}, processed_data_dir=Path("data/exp123/processed"), window_size=2,
        keep_logs=True, prediction_mode="prior_t", selection_prediction_mode="prior_t",
        loss_metric="choice_nll", trajectory_seed=case["filter_seed"],
    )
    reported = evaluate_state_model_run(engine_config_template=case["engine_config"], **common)
    config = deepcopy(case["engine_config"])
    config.pop("oral_readout")
    original = evaluate_state_model_run(engine_config_template=config, **common)
    assert "oral_report_probabilities" not in original.state_log
    assert reported.state_log["oral_report_probabilities"].shape == (
        4, len(reported.state_log["oral_report_codes"])
    )
    assert reported.state_log["oral_report_valid_mask"].all()
    assert reported.state_log["oral_readout_metadata"]["timing"] == "after_observed_choice_before_current_feedback"
    np.testing.assert_array_equal(original.state_log["marginal_prior"], reported.state_log["marginal_prior"])

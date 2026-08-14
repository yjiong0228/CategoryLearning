from __future__ import annotations

import numpy as np

from scripts.run_model_0813_pf_mechanism_reachability import (
    EXPECTED_EXACT_VARIANTS,
    MECHANISM_PROBE_MAP,
    _array_max_diff,
    _compare_outputs,
    _variant_engine,
)


def _minimal_engine() -> dict:
    return {
        "likelihood": {"distance_mode": "prototype"},
        "modules": {
            "perception_mod": {
                "class": "src.Bayesian_state.model.modules.perception.PerceptionModule"
            },
            "beta_mod": {
                "class": "src.Bayesian_state.model.modules.beta.BetaModule",
                "kwargs": {
                    "decrease_rate": 0.15,
                    "correct_additive": 1.0,
                    "update_scope": "executed_hypothesis",
                    "use_prior_scaling": False,
                    "prior_beta_scale": 0.0,
                },
            },
            "hypo_transitions_mod": {
                "class": (
                    "src.Bayesian_state.model.modules.hypothesis_transition."
                    "dynamic_adaptive_control."
                    "DynamicAdaptiveControlHypothesisTransitionModule"
                ),
                "kwargs": {
                    "capacity": 3,
                    "continuous_controller": {
                        "state": {"failure_decay": 0.60},
                        "exploration": {"mastery_weight": 1.0},
                        "range": {"global_max": 0.80},
                        "execution": {"enabled": True, "switch_scale": 0.20},
                    },
                },
            },
            "memory_mod": {
                "class": "src.Bayesian_state.model.modules.memory.DualMemoryModule",
                "kwargs": {"gamma": 0.80, "w0": 0.15},
            },
        },
        "choice_readout": {
            "kwargs": {
                "method": "sharpened_expectation",
                "power": 4.0,
                "strategy_confidence_gain": 2.0,
            }
        },
        "output_noise": {"kwargs": {"enabled": True, "base_lapse": 0.02}},
    }


def test_variant_mutations_are_scoped_and_leave_input_unchanged() -> None:
    baseline = _minimal_engine()
    variant = _variant_engine(baseline, "execution_off_linked")
    assert baseline["modules"]["hypo_transitions_mod"]["kwargs"][
        "continuous_controller"
    ]["execution"]["enabled"] is True
    assert variant["modules"]["hypo_transitions_mod"]["kwargs"][
        "continuous_controller"
    ]["execution"]["enabled"] is False
    assert variant["modules"]["beta_mod"]["kwargs"]["update_scope"] == (
        "active_hypotheses"
    )


def test_compare_outputs_distinguishes_exact_and_changed_predictions() -> None:
    choices = np.asarray([1, 2], dtype=int)
    baseline = {
        "observation.prior_t": np.asarray([[0.6, 0.4], [0.3, 0.7]]),
        "state.active_probability": np.ones((2, 2)),
    }
    exact = _compare_outputs(
        baseline, baseline, choices, tolerance=1e-12
    )
    assert exact["all_public_arrays_exact"] is True
    assert exact["max_abs_choice_probability_diff"] == 0.0

    changed_values = {key: value.copy() for key, value in baseline.items()}
    changed_values["observation.prior_t"][0] = [0.5, 0.5]
    changed = _compare_outputs(
        baseline, changed_values, choices, tolerance=1e-12
    )
    assert changed["all_public_arrays_exact"] is False
    assert np.isclose(changed["max_abs_choice_probability_diff"], 0.1)
    assert changed["first_changed_trial_zero_based"] == 0


def test_array_max_diff_supports_boolean_diagnostics() -> None:
    assert _array_max_diff(
        np.asarray([True, False]), np.asarray([True, True])
    ) == 1.0
    assert _array_max_diff(
        np.asarray([np.inf]), np.asarray([-np.inf])
    ) == np.inf


def test_all_registered_numerical_mechanisms_have_declared_probes() -> None:
    expected = {
        "REP-02",
        "COG-01",
        "COG-02",
        "COG-03",
        "COG-04",
        "COG-05",
        "COG-09",
        "COG-10",
        "COG-11",
        "COG-12",
        "COG-13",
        "OBS-01",
        "OBS-02",
        "OBS-03",
    }
    assert set(MECHANISM_PROBE_MAP) == expected
    assert {
        "readout_power_one",
        "readout_expectation",
        "dormant_signal_scaling",
        "dormant_beta_prior_scale",
        "dormant_output_controls",
        "disabled_optional_blocks",
    } == set(EXPECTED_EXACT_VARIANTS)

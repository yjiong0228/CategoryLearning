from __future__ import annotations

import numpy as np

from src.Bayesian_state.utils.newplan_posterior_predictive import (
    DynamicRhoConfig,
    run_conditioned_condition1_rollouts,
)


def _engine_config() -> dict:
    return {
        "partition": {
            "class": "src.Bayesian_state.problems.partitions.Partition",
            "kwargs": {
                "n_dims": 4,
                "n_cats": 2,
                "include_label_reversals": True,
            },
        },
        "modules": {
            "perception_mod": {
                "class": (
                    "src.Bayesian_state.problems.modules.perception."
                    "PerceptionModule"
                ),
                "kwargs": {
                    "features": 4,
                    "mean": [0.0, 0.0, 0.0, 0.0],
                    "std": [0.0, 0.0, 0.0, 0.0],
                },
            },
            "beta_mod": {
                "class": (
                    "src.Bayesian_state.problems.modules.beta.BetaModule"
                ),
                "kwargs": {
                    "beta_init": 5.0,
                    "beta_min": 0.1,
                    "beta_max": 25.0,
                    "decrease_rate": 0.15,
                    "correct_additive": 0.5,
                    "beta_update_mode": "probabilistic_feedback",
                    "use_prior_scaling": False,
                    "prior_beta_scale": 0.0,
                },
            },
            "hypo_transitions_mod": {
                "class": (
                    "src.Bayesian_state.problems.modules."
                    "minimal_hypo_transition.FeedbackSwapHypothesisModule"
                ),
                "kwargs": {"capacity": 5, "theta": 0.0},
            },
            "likelihood_mod": {
                "class": (
                    "src.Bayesian_state.problems.modules.likelihood."
                    "LikelihoodModule"
                ),
                "kwargs": {"distance_mode": "prototype"},
            },
            "memory_mod": {
                "class": (
                    "src.Bayesian_state.problems.modules.memory."
                    "DualMemoryModule"
                ),
                "kwargs": {"gamma": 0.55, "w0": 0.10},
            },
        },
        "agenda": [
            "perception_mod",
            "hypo_transitions_mod",
            "likelihood_mod",
            "memory_mod",
            "beta_mod",
        ],
    }


def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stimulus = np.asarray(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.8, 0.2, 0.7, 0.1],
            [0.4, 0.9, 0.1, 0.6],
            [0.7, 0.6, 0.2, 0.8],
            [0.3, 0.4, 0.8, 0.9],
            [0.9, 0.1, 0.6, 0.2],
            [0.2, 0.8, 0.4, 0.7],
            [0.6, 0.3, 0.9, 0.2],
        ]
    )
    categories = np.asarray([1, 2, 1, 2, 1, 2, 1, 2], dtype=int)
    choices = categories.copy()
    feedback = np.ones_like(categories, dtype=float)
    return stimulus, categories, choices, feedback


def test_conditioned_rollouts_are_reproducible_and_autonomous() -> None:
    stimulus, categories, choices, feedback = _data()
    kwargs = {
        "engine_config": _engine_config(),
        "subject_id": 999,
        "stimulus": stimulus,
        "categories": categories,
        "observed_prefix_choices": choices[:4],
        "observed_prefix_feedback": feedback[:4],
        "particle_count": 8,
        "rollout_count": 12,
        "rho": 2.0,
        "filter_seed": 41,
        "rollout_seed": 42,
    }
    first = run_conditioned_condition1_rollouts(**kwargs)
    second = run_conditioned_condition1_rollouts(**kwargs)
    assert np.array_equal(first.choices, second.choices)
    assert np.array_equal(first.feedback, second.feedback)
    assert np.allclose(first.probabilities, second.probabilities)
    assert np.array_equal(
        first.prefix_choice_probabilities,
        second.prefix_choice_probabilities,
    )
    assert np.array_equal(
        first.prefix_observed_choice_probability,
        second.prefix_observed_choice_probability,
    )
    assert first.prefix_log_predictive_density == (
        second.prefix_log_predictive_density
    )
    assert first.choices.shape == (12, 4)
    assert np.array_equal(
        first.feedback,
        (
            first.choices
            == categories[4:][None, :]
        ).astype(np.int8),
    )
    assert np.allclose(first.probabilities.sum(axis=2), 1.0)
    assert np.isclose(first.boundary_weights.sum(), 1.0)
    assert np.allclose(
        first.prefix_choice_probabilities.sum(axis=1),
        1.0,
    )
    assert np.isclose(
        first.prefix_log_predictive_density,
        np.log(first.prefix_observed_choice_probability).sum(),
    )


def test_future_observations_are_not_part_of_the_api() -> None:
    stimulus, categories, choices, feedback = _data()
    first = run_conditioned_condition1_rollouts(
        engine_config=_engine_config(),
        subject_id=999,
        stimulus=stimulus,
        categories=categories,
        observed_prefix_choices=choices[:4],
        observed_prefix_feedback=feedback[:4],
        particle_count=6,
        rollout_count=8,
        rho=2.0,
        filter_seed=7,
        rollout_seed=9,
    )
    altered_categories = categories.copy()
    altered_categories[4:] = 3 - altered_categories[4:]
    second = run_conditioned_condition1_rollouts(
        engine_config=_engine_config(),
        subject_id=999,
        stimulus=stimulus,
        categories=altered_categories,
        observed_prefix_choices=choices[:4],
        observed_prefix_feedback=feedback[:4],
        particle_count=6,
        rollout_count=8,
        rho=2.0,
        filter_seed=7,
        rollout_seed=9,
    )
    # The first future choice is made before future feedback exists, so it is
    # identical under the same random stream.  The changed task category then
    # changes feedback and may legitimately send later learning down another
    # autonomous path.
    assert np.array_equal(first.choices[:, 0], second.choices[:, 0])
    assert not np.array_equal(first.feedback[:, 0], second.feedback[:, 0])


def test_trialwise_epsilon_schedule_is_applied_to_future_choices() -> None:
    stimulus, categories, choices, feedback = _data()
    schedule = np.zeros(len(categories), dtype=float)
    schedule[4] = 1.0
    result = run_conditioned_condition1_rollouts(
        engine_config=_engine_config(),
        subject_id=999,
        stimulus=stimulus,
        categories=categories,
        observed_prefix_choices=choices[:4],
        observed_prefix_feedback=feedback[:4],
        particle_count=6,
        rollout_count=64,
        rho=2.0,
        epsilon_schedule=schedule,
        filter_seed=7,
        rollout_seed=11,
    )
    assert np.allclose(result.probabilities[:, 0], 0.5)
    assert np.any(np.abs(result.probabilities[:, 1:] - 0.5) > 1e-6)


def test_learning_update_gate_changes_autonomous_suffix() -> None:
    stimulus, categories, choices, feedback = _data()
    common = {
        "engine_config": _engine_config(),
        "subject_id": 999,
        "stimulus": stimulus,
        "categories": categories,
        "observed_prefix_choices": choices[:4],
        "observed_prefix_feedback": feedback[:4],
        "particle_count": 8,
        "rollout_count": 16,
        "rho": 2.0,
        "filter_seed": 13,
        "rollout_seed": 17,
    }
    frozen = run_conditioned_condition1_rollouts(
        learning_update_probability=0.0,
        **common,
    )
    learning = run_conditioned_condition1_rollouts(
        learning_update_probability=1.0,
        **common,
    )
    assert not np.allclose(frozen.probabilities, learning.probabilities)


def test_zero_acquisition_hazard_stays_novice_and_uniform() -> None:
    stimulus, categories, choices, feedback = _data()
    result = run_conditioned_condition1_rollouts(
        engine_config=_engine_config(),
        subject_id=999,
        stimulus=stimulus,
        categories=categories,
        observed_prefix_choices=choices[:4],
        observed_prefix_feedback=feedback[:4],
        particle_count=8,
        rollout_count=32,
        rho=2.0,
        acquisition_hazard=0.0,
        filter_seed=19,
        rollout_seed=23,
    )
    assert np.allclose(result.prefix_acquired_probability, 0.0)
    assert not np.any(result.boundary_acquired)
    assert not np.any(result.generated_acquired)
    assert np.allclose(result.probabilities, 0.5)


def test_certain_acquisition_matches_ordinary_readout() -> None:
    stimulus, categories, choices, feedback = _data()
    common = {
        "engine_config": _engine_config(),
        "subject_id": 999,
        "stimulus": stimulus,
        "categories": categories,
        "observed_prefix_choices": choices[:4],
        "observed_prefix_feedback": feedback[:4],
        "particle_count": 8,
        "rollout_count": 16,
        "rho": 2.0,
        "filter_seed": 29,
        "rollout_seed": 31,
    }
    ordinary = run_conditioned_condition1_rollouts(**common)
    acquired = run_conditioned_condition1_rollouts(
        acquisition_hazard=1.0,
        **common,
    )
    assert np.array_equal(ordinary.choices, acquired.choices)
    assert np.array_equal(ordinary.feedback, acquired.feedback)
    assert np.allclose(ordinary.probabilities, acquired.probabilities)
    assert np.allclose(acquired.prefix_acquired_probability, 1.0)
    assert np.all(acquired.boundary_acquired)
    assert np.all(acquired.generated_acquired)


def test_acquisition_state_is_irreversible_within_each_rollout() -> None:
    stimulus, categories, choices, feedback = _data()
    result = run_conditioned_condition1_rollouts(
        engine_config=_engine_config(),
        subject_id=999,
        stimulus=stimulus,
        categories=categories,
        observed_prefix_choices=choices[:4],
        observed_prefix_feedback=feedback[:4],
        particle_count=16,
        rollout_count=64,
        rho=2.0,
        acquisition_hazard=0.25,
        filter_seed=37,
        rollout_seed=41,
    )
    transitions = np.diff(
        result.generated_acquired.astype(np.int8),
        axis=1,
    )
    assert np.all(transitions >= 0)
    novice = ~result.generated_acquired
    assert np.allclose(result.probabilities[novice], 0.5)


def test_partial_pre_acquisition_lapse_preserves_informed_choices() -> None:
    stimulus, categories, choices, feedback = _data()
    result = run_conditioned_condition1_rollouts(
        engine_config=_engine_config(),
        subject_id=999,
        stimulus=stimulus,
        categories=categories,
        observed_prefix_choices=choices[:4],
        observed_prefix_feedback=feedback[:4],
        particle_count=8,
        rollout_count=32,
        rho=2.0,
        acquisition_hazard=0.0,
        pre_acquisition_lapse=0.5,
        filter_seed=43,
        rollout_seed=47,
    )
    assert not np.any(result.generated_acquired)
    assert np.any(np.abs(result.probabilities - 0.5) > 1e-6)
    assert np.all(result.probabilities >= 0.25 - 1e-6)
    assert np.all(result.probabilities <= 0.75 + 1e-6)


def test_deterministic_dynamic_rho_follows_the_requested_trend() -> None:
    stimulus, categories, choices, feedback = _data()
    result = run_conditioned_condition1_rollouts(
        engine_config=_engine_config(),
        subject_id=999,
        stimulus=stimulus,
        categories=categories,
        observed_prefix_choices=choices[:4],
        observed_prefix_feedback=feedback[:4],
        particle_count=8,
        rollout_count=16,
        rho=2.0,
        dynamic_rho=DynamicRhoConfig(
            start=0.5,
            end=4.0,
            volatility=0.0,
            persistence=0.9,
            start_log_sd=0.0,
            gain_log_sd=0.0,
            volatility_log_sd=0.0,
            trend_reference_trials=len(categories),
        ),
        filter_seed=53,
        rollout_seed=59,
    )
    expected = np.exp(
        np.linspace(np.log(0.5), np.log(4.0), len(categories))
    )
    assert np.allclose(
        result.prefix_rho_posterior_mean,
        expected[:4],
    )
    assert np.allclose(
        result.generated_rho,
        expected[4:][None, :],
    )
    assert np.all(np.diff(result.generated_rho, axis=1) > 0.0)


def test_stochastic_dynamic_rho_is_reproducible_and_can_reverse() -> None:
    stimulus, categories, choices, feedback = _data()
    kwargs = {
        "engine_config": _engine_config(),
        "subject_id": 999,
        "stimulus": stimulus,
        "categories": categories,
        "observed_prefix_choices": choices[:4],
        "observed_prefix_feedback": feedback[:4],
        "particle_count": 16,
        "rollout_count": 64,
        "rho": 2.0,
        "dynamic_rho": DynamicRhoConfig(
            start=0.25,
            end=2.0,
            volatility=0.6,
            persistence=0.8,
            start_log_sd=0.2,
            gain_log_sd=0.2,
            volatility_log_sd=0.2,
        ),
        "filter_seed": 61,
        "rollout_seed": 67,
    }
    first = run_conditioned_condition1_rollouts(**kwargs)
    second = run_conditioned_condition1_rollouts(**kwargs)
    assert np.array_equal(first.generated_rho, second.generated_rho)
    assert np.array_equal(first.choices, second.choices)
    changes = np.diff(first.generated_rho, axis=1)
    assert np.any(changes > 0.0)
    assert np.any(changes < 0.0)
    assert np.all(first.generated_rho >= 0.05)
    assert np.all(first.generated_rho <= 20.0)


def test_dynamic_rho_and_acquisition_boundary_are_mutually_exclusive() -> None:
    stimulus, categories, choices, feedback = _data()
    with np.testing.assert_raises(ValueError):
        run_conditioned_condition1_rollouts(
            engine_config=_engine_config(),
            subject_id=999,
            stimulus=stimulus,
            categories=categories,
            observed_prefix_choices=choices[:4],
            observed_prefix_feedback=feedback[:4],
            particle_count=8,
            rollout_count=16,
            rho=2.0,
            acquisition_hazard=0.1,
            dynamic_rho=DynamicRhoConfig(
                start=0.5,
                end=2.0,
                volatility=0.1,
                persistence=0.9,
            ),
        )

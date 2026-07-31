from __future__ import annotations

import numpy as np

from src.Bayesian_state.problems.modules.perception import PerceptionModule
from src.Bayesian_state.utils.newplan_particle_filter import (
    effective_sample_size,
    run_newplan_particle_filter,
    systematic_resample,
)


def _engine_config(theta: float, perception_std: float = 0.05) -> dict:
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
                    "std": [perception_std] * 4,
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
                    "src.Bayesian_state.problems.modules.minimal_hypo_transition."
                    "FeedbackSwapHypothesisModule"
                ),
                "kwargs": {"capacity": 5, "theta": float(theta)},
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


def _trajectory(n_trials: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = np.asarray(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.8, 0.2, 0.7, 0.1],
            [0.4, 0.9, 0.1, 0.6],
            [0.7, 0.6, 0.2, 0.8],
        ],
        dtype=float,
    )
    stimulus = np.vstack([base[index % len(base)] for index in range(n_trials)])
    choices = np.asarray(
        [1 if index % 3 else 2 for index in range(n_trials)], dtype=int
    )
    feedback = np.asarray(
        [0.0 if index % 2 else 1.0 for index in range(n_trials)], dtype=float
    )
    return stimulus, choices, feedback


def test_effective_sample_size_boundaries() -> None:
    assert np.isclose(effective_sample_size([0.25] * 4), 4.0)
    assert np.isclose(effective_sample_size([1.0, 0.0, 0.0, 0.0]), 1.0)


def test_systematic_resample_is_deterministic_and_monotone() -> None:
    ancestors = systematic_resample([0.1, 0.2, 0.3, 0.4], uniform=0.25)
    assert ancestors.tolist() == [0, 2, 2, 3]
    concentrated = systematic_resample([0.0, 0.0, 0.1, 0.9], uniform=0.5)
    assert np.all(np.diff(concentrated) >= 0)
    assert np.sum(concentrated == 3) >= 3


def test_perception_module_reseed_future_is_reproducible() -> None:
    engine = type("Engine", (), {"observation": None})()
    first = PerceptionModule(
        engine,
        features=4,
        mean=[0.0] * 4,
        std=[0.1] * 4,
        module_seed=3,
    )
    stimulus = np.full(4, 0.5, dtype=float)
    _ = first.sample(stimulus)
    first.reseed_future(19)
    after_reseed = first.sample(stimulus)

    second = PerceptionModule(
        engine,
        features=4,
        mean=[0.0] * 4,
        std=[0.1] * 4,
        module_seed=19,
    )
    assert np.allclose(after_reseed, second.sample(stimulus))


def test_epsilon_one_keeps_uniform_weights_and_exact_predictions() -> None:
    stimulus, choices, feedback = _trajectory()
    result = run_newplan_particle_filter(
        engine_config=_engine_config(theta=0.0),
        subject_id=999,
        stimulus=stimulus,
        choices=choices,
        feedback=feedback,
        particle_count=8,
        rho=2.0,
        epsilon=1.0,
        filter_seed=41,
        resample_threshold_fraction=1.0,
    )

    assert np.allclose(result.marginal_probabilities, 0.5)
    assert np.allclose(result.pre_choice_ess, 8.0)
    assert np.allclose(result.post_choice_ess, 8.0)
    assert not np.any(result.resampled)
    assert np.allclose(result.filtered_swap_event_probability, 0.0)


def test_trialwise_epsilon_schedule_controls_readout() -> None:
    stimulus, choices, feedback = _trajectory()
    schedule = np.zeros(len(choices), dtype=float)
    schedule[:3] = 1.0
    result = run_newplan_particle_filter(
        engine_config=_engine_config(theta=0.0),
        subject_id=999,
        stimulus=stimulus,
        choices=choices,
        feedback=feedback,
        particle_count=8,
        rho=2.0,
        epsilon_schedule=schedule,
        filter_seed=43,
    )
    assert np.allclose(result.marginal_probabilities[:3], 0.5)
    assert np.any(
        np.abs(result.marginal_probabilities[3:] - 0.5) > 1e-6
    )


def test_stochastic_learning_update_gate_changes_future_predictions() -> None:
    stimulus, choices, feedback = _trajectory(n_trials=16)
    common = {
        "engine_config": _engine_config(theta=0.0),
        "subject_id": 999,
        "stimulus": stimulus,
        "choices": choices,
        "feedback": feedback,
        "particle_count": 8,
        "rho": 2.0,
        "filter_seed": 47,
    }
    frozen = run_newplan_particle_filter(
        learning_update_probability=0.0,
        **common,
    )
    learning = run_newplan_particle_filter(
        learning_update_probability=1.0,
        **common,
    )
    assert np.allclose(
        frozen.marginal_probabilities[0],
        learning.marginal_probabilities[0],
    )
    assert not np.allclose(
        frozen.marginal_probabilities[1:],
        learning.marginal_probabilities[1:],
    )


def test_filter_is_reproducible_and_resampling_preserves_valid_output() -> None:
    stimulus, choices, feedback = _trajectory(n_trials=16)
    kwargs = {
        "engine_config": _engine_config(theta=1.0),
        "subject_id": 999,
        "stimulus": stimulus,
        "choices": choices,
        "feedback": feedback,
        "particle_count": 12,
        "rho": 2.0,
        "epsilon": 0.0,
        "filter_seed": 73,
        "resample_threshold_fraction": 1.0,
    }
    first = run_newplan_particle_filter(**kwargs)
    second = run_newplan_particle_filter(**kwargs)

    assert np.array_equal(first.marginal_probabilities, second.marginal_probabilities)
    assert np.array_equal(first.post_choice_ess, second.post_choice_ess)
    assert np.array_equal(first.resampled, second.resampled)
    assert np.allclose(first.marginal_probabilities.sum(axis=1), 1.0)
    assert np.all((first.marginal_probabilities >= 0.0) & (first.marginal_probabilities <= 1.0))
    assert first.marginal_hypothesis_prior.shape[0] == first.marginal_probabilities.shape[0]
    assert np.allclose(first.marginal_hypothesis_prior.sum(axis=1), 1.0)
    assert np.all(
        (first.marginal_active_probability >= 0.0)
        & (first.marginal_active_probability <= 1.0)
    )
    assert np.any(first.resampled)
    assert all(
        1 <= item["unique_ancestors"] <= first.particle_count
        for item in first.resampling_log
    )


def test_current_choice_does_not_change_its_own_prediction() -> None:
    stimulus, choices, feedback = _trajectory(n_trials=8)
    alternative_choices = choices.copy()
    alternative_choices[0] = 3 - alternative_choices[0]
    common = {
        "engine_config": _engine_config(theta=0.75),
        "subject_id": 999,
        "stimulus": stimulus,
        "feedback": feedback,
        "particle_count": 10,
        "rho": 2.0,
        "epsilon": 0.0,
        "filter_seed": 101,
        "resample_threshold_fraction": 0.5,
    }
    first = run_newplan_particle_filter(choices=choices, **common)
    second = run_newplan_particle_filter(choices=alternative_choices, **common)

    assert np.array_equal(
        first.marginal_probabilities[0],
        second.marginal_probabilities[0],
    )
    assert not np.array_equal(
        first.marginal_probabilities[1:],
        second.marginal_probabilities[1:],
    )

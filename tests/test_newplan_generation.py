from __future__ import annotations

import numpy as np

from src.Bayesian_state.utils.newplan_generation import (
    generate_condition1_trajectory,
)


def _engine_config(theta: float) -> dict:
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
                "class": "src.Bayesian_state.problems.modules.perception.PerceptionModule",
                "kwargs": {
                    "features": 4,
                    "mean": [0.0, 0.0, 0.0, 0.0],
                    "std": [0.0, 0.0, 0.0, 0.0],
                },
            },
            "beta_mod": {
                "class": "src.Bayesian_state.problems.modules.beta.BetaModule",
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
                "kwargs": {
                    "capacity": 5,
                    "theta": float(theta),
                },
            },
            "likelihood_mod": {
                "class": "src.Bayesian_state.problems.modules.likelihood.LikelihoodModule",
                "kwargs": {"distance_mode": "prototype"},
            },
            "memory_mod": {
                "class": "src.Bayesian_state.problems.modules.memory.DualMemoryModule",
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


def test_autonomous_generation_is_causal_normalized_and_reproducible() -> None:
    stimulus = np.asarray(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.8, 0.2, 0.7, 0.1],
            [0.4, 0.9, 0.1, 0.6],
            [0.7, 0.6, 0.2, 0.8],
            [0.3, 0.4, 0.8, 0.9],
            [0.9, 0.1, 0.6, 0.2],
        ]
    )
    categories = np.asarray([1, 2, 1, 2, 1, 2], dtype=int)

    first = generate_condition1_trajectory(
        engine_config=_engine_config(theta=1.0),
        subject_id=999,
        stimulus=stimulus,
        categories=categories,
        epsilon=0.1,
        rho=2.0,
        trajectory_seed=41,
    )
    second = generate_condition1_trajectory(
        engine_config=_engine_config(theta=1.0),
        subject_id=999,
        stimulus=stimulus,
        categories=categories,
        epsilon=0.1,
        rho=2.0,
        trajectory_seed=41,
    )

    assert np.array_equal(first.choices, second.choices)
    assert np.array_equal(first.feedback, second.feedback)
    assert np.array_equal(first.prior, second.prior)
    assert np.array_equal(first.posterior, second.posterior)
    assert np.allclose(first.cognitive_probabilities.sum(axis=1), 1.0)
    assert np.allclose(first.observed_probabilities.sum(axis=1), 1.0)
    assert np.allclose(first.prior.sum(axis=1), 1.0)
    assert np.allclose(first.posterior.sum(axis=1), 1.0)
    assert np.array_equal(
        first.feedback,
        (first.choices == categories).astype(float),
    )
    assert first.transition_log[0]["feedback_used"] is None
    assert first.transition_log[0]["swap_event"] is False
    for trial_idx in range(1, len(first.transition_log)):
        assert (
            first.transition_log[trial_idx]["feedback_used"]
            == first.feedback[trial_idx - 1]
        )
    assert not any(item["fallback"] for item in first.transition_log)


def test_epsilon_one_generates_uniform_observed_probabilities() -> None:
    result = generate_condition1_trajectory(
        engine_config=_engine_config(theta=0.0),
        subject_id=999,
        stimulus=np.full((4, 4), 0.25, dtype=float),
        categories=np.asarray([1, 2, 1, 2], dtype=int),
        epsilon=1.0,
        rho=1.0,
        trajectory_seed=7,
    )

    assert np.allclose(result.observed_probabilities, 0.5)
    assert not any(item["swap_event"] for item in result.transition_log)

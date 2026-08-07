from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from src.Bayesian_state.optimization.optimizer_common import (
    TrialArrays,
    evaluate_state_model_run,
)
from src.Bayesian_state.problems.modules.finite_workspace_transition import (
    AdaptiveFiniteWorkspaceTransitionModule,
)


class _TinyPartition:
    """Small partition sufficient for engine/particle integration tests."""

    VALID_DISTANCE_MODES = ("prototype",)

    def __init__(self, **kwargs):
        del kwargs
        self.length = 6
        self.n_cats = 2
        positions = np.arange(self.length, dtype=float)
        self._similarity = np.exp(-np.abs(positions[:, None] - positions[None, :]))

    @property
    def similarity_matrix(self) -> np.ndarray:
        return self._similarity

    def _probability(self, hypo: int, stimulus: np.ndarray) -> np.ndarray:
        x = float(np.asarray(stimulus, dtype=float).reshape(-1)[0])
        boundary = (int(hypo) + 1.0) / (self.length + 1.0)
        category_one = x <= boundary if int(hypo) % 2 == 0 else x > boundary
        return np.asarray([0.85, 0.15] if category_one else [0.15, 0.85])

    def get_category_probabilities(self, hypo, data, beta, distance_mode, **kwargs):
        del beta, distance_mode, kwargs
        return self._probability(int(hypo), np.asarray(data[0][0]))[:, None]

    def calc_likelihood(
        self,
        hypos,
        data,
        beta,
        distance_mode,
        normalized,
        **kwargs,
    ):
        del beta, distance_mode, normalized, kwargs
        choice = int(data[1][0]) - 1
        feedback = float(data[2][0])
        compatible = choice if feedback >= 0.5 else 1 - choice
        return np.asarray(
            [[self._probability(int(hypo), data[0][0])[compatible] for hypo in hypos]],
            dtype=float,
        )


class _TinyEngine:
    def __init__(self):
        self.set_size = 6
        self.prior = np.full(6, 1.0 / 6.0)
        self.posterior = None
        self.likelihood = np.ones(6)
        self.partition = _TinyPartition()
        self.modules = {}
        self.hypotheses_mask = None


def _engine_config() -> dict:
    return {
        "partition": {"class": _TinyPartition, "kwargs": {}},
        "inference": {
            "backend": "particle_filter",
            "particle_count": 8,
            "resample_threshold_fraction": 0.95,
        },
        "modules": {
            "perception_mod": {
                "class": "src.Bayesian_state.problems.modules.perception.PerceptionModule",
                "kwargs": {"features": 1, "mean": [0.0], "std": [0.0]},
            },
            "beta_mod": {
                "class": "src.Bayesian_state.problems.modules.beta.BetaModule",
                "kwargs": {
                    "beta_init": 5.0,
                    "decrease_rate": 0.0,
                    "correct_additive": 0.0,
                    "beta_update_mode": "probabilistic_feedback",
                    "use_prior_scaling": False,
                },
            },
            "hypo_transitions_mod": {
                "class": (
                    "src.Bayesian_state.problems.modules.finite_workspace_transition."
                    "AdaptiveFiniteWorkspaceTransitionModule"
                ),
                "kwargs": {
                    "capacity": 2,
                    "m": 0.35,
                    "m_phi": 0.4,
                    "m_beta_surprise": 0.6,
                    "surprise_center": 0.8,
                    "surprise_scale": 0.5,
                    "g": 0.35,
                },
            },
            "likelihood_mod": {
                "class": "src.Bayesian_state.problems.modules.likelihood.LikelihoodModule",
                "kwargs": {"distance_mode": "prototype"},
            },
            "memory_mod": {
                "class": "src.Bayesian_state.problems.modules.memory.DualMemoryModule",
                "kwargs": {"gamma": 0.7, "w0": 0.4},
            },
        },
        "choice_readout": {
            "kwargs": {"method": "sharpened_expectation", "power": 2.0}
        },
        "output_noise": {
            "kwargs": {
                "enabled": True,
                "base_lapse": 0.05,
                "post_error_lapse": 0.0,
                "low_accuracy_lapse": 0.0,
                "latent_volatility_lapse": 0.0,
                "lapse_target": "uniform",
            }
        },
        "agenda": [
            "perception_mod",
            "hypo_transitions_mod",
            "likelihood_mod",
            "memory_mod",
            "beta_mod",
        ],
    }


def test_adaptive_rate_uses_previous_feedback_and_restores_state():
    engine = _TinyEngine()
    module = AdaptiveFiniteWorkspaceTransitionModule(
        engine,
        capacity=2,
        init_hypotheses=[0, 1],
        m=0.2,
        m_phi=0.5,
        m_beta_surprise=0.8,
        surprise_center=0.4,
        surprise_scale=0.5,
        g=1.0,
        module_seed=7,
    )

    module.process()
    predictive_prior = module.predictive_prior.copy()
    engine.likelihood = np.asarray([0.25, 0.75, 0.0, 0.0, 0.0, 0.0])
    raw_posterior = predictive_prior * engine.likelihood
    engine.posterior = raw_posterior / raw_posterior.sum()
    expected_surprise = -math.log(float(np.sum(predictive_prior * engine.likelihood)))
    expected_logit = module.baseline_logit + 0.8 * (
        (expected_surprise - 0.4) / 0.5
    )

    module.process()

    assert np.isclose(module.feedback_surprise, expected_surprise)
    assert np.isclose(module.control_logit, expected_logit)
    assert np.isclose(module.current_m, 1.0 / (1.0 + math.exp(-expected_logit)))
    assert module.active.size == 2
    assert np.isclose(engine.prior.sum(), 1.0)
    assert np.count_nonzero(engine.hypotheses_mask) == 2

    saved = module.state_dict()
    saved_active = module.active.copy()
    module.process()
    module.load_state_dict(saved)
    assert np.array_equal(module.active, saved_active)
    assert module.trial_index == saved["trial_index"]


def test_standard_runner_dispatches_model_0806_to_particle_backend():
    n_trials = 18
    stimulus = np.linspace(0.05, 0.95, n_trials)[:, None]
    categories = np.where(stimulus[:, 0] < 0.5, 1, 2)
    choices = np.where(np.arange(n_trials) % 4 == 0, 3 - categories, categories)
    feedback = (choices == categories).astype(float)
    target_probs = np.eye(2, dtype=float)[categories - 1]
    arrays = TrialArrays(
        stimulus=stimulus,
        choices=choices,
        feedback=feedback,
        categories=categories,
        target_probs=target_probs,
    )

    result = evaluate_state_model_run(
        subject_id=1,
        condition=1,
        arrays=arrays,
        params={},
        engine_config_template=_engine_config(),
        processed_data_dir=Path("."),
        window_size=4,
        keep_logs=True,
        prediction_mode="prior_t",
        selection_prediction_mode="prior_t",
        loss_metric="choice_brier",
        trajectory_seed=20260806,
    )

    metrics = result.metrics_by_mode["prior_t"]
    probabilities = np.asarray(metrics["pred_category_probs"])
    assert probabilities.shape == (n_trials, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert not bool(np.asarray(metrics["valid_trial_mask"])[0])
    assert np.isfinite(result.mean_error)
    assert result.state_log is not None
    assert np.asarray(result.state_log["transition_rate"]).shape == (n_trials,)
    assert np.asarray(result.state_log["replacement_fraction"]).shape == (n_trials,)
    assert np.asarray(result.state_log["newcomer_distance"]).shape == (n_trials,)
    assert result.transition_counts is not None
    assert len(result.transition_counts) == n_trials

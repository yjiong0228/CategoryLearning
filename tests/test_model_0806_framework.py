from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.Bayesian_state.optimization.optimizer_common import (
    TrialArrays,
    evaluate_state_model_run,
)
from src.Bayesian_state.inference_engine.dispatcher import run_inference_backend
from src.Bayesian_state.inference_engine.results import (
    InferenceResult,
    TrajectoryInferenceResult,
)
from src.Bayesian_state.problems.modules.hypo_transition.dynamic_continuous import (
    DynamicContinuousHypothesisTransitionModule,
)
from src.Bayesian_state.problems.modules.hypo_transition.dynamic_discrete import (
    DynamicDiscreteHypothesisTransitionModule,
)
from src.Bayesian_state.problems.modules.hypo_transition.static import (
    StaticFeedbackSwapHypothesisTransitionModule,
    StaticWorkspaceHypothesisTransitionModule,
    StaticHypothesisTransitionModule,
)
from src.Bayesian_state.problems.modules.readout import (
    read_oral_report,
    read_reaction_time,
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
                    "src.Bayesian_state.problems.modules."
                    "hypo_transition.dynamic_continuous."
                    "DynamicContinuousHypothesisTransitionModule"
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
    module = DynamicContinuousHypothesisTransitionModule(
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


def test_dynamic_search_range_uses_previous_feedback_and_restores_state():
    engine = _TinyEngine()
    module = DynamicContinuousHypothesisTransitionModule(
        engine,
        capacity=2,
        init_hypotheses=[0, 1],
        m=0.0,
        g=0.35,
        range_controller={
            "g_phi": 0.5,
            "g_beta_surprise": 0.6,
            "g_beta_uncertainty": 0.0,
            "g_surprise_center": 0.4,
            "g_surprise_scale": 0.5,
        },
        module_seed=9,
    )

    module.process()
    predictive_prior = module.predictive_prior.copy()
    engine.likelihood = np.asarray([0.25, 0.75, 0.0, 0.0, 0.0, 0.0])
    posterior = predictive_prior * engine.likelihood
    engine.posterior = posterior / posterior.sum()
    surprise = -math.log(float(np.sum(predictive_prior * engine.likelihood)))
    expected_logit = module.g_baseline_logit + 0.6 * ((surprise - 0.4) / 0.5)

    module.process()

    assert np.isclose(module.g_control_logit, expected_logit)
    assert np.isclose(module.current_g, 1.0 / (1.0 + math.exp(-expected_logit)))
    assert np.isclose(module.transition_log[-1]["predictive_g"], module.current_g)
    saved = module.state_dict()
    module.g_control_logit = -20.0
    module.current_g = 0.0
    module.load_state_dict(saved)
    assert np.isclose(module.g_control_logit, expected_logit)
    assert np.isclose(module.current_g, saved["current_g"])


def test_static_bounded_workspace_uses_common_transition_contract():
    engine = _TinyEngine()
    module = StaticWorkspaceHypothesisTransitionModule(
        engine,
        capacity=2,
        init_hypotheses=[0, 1],
        m=0.0,
        g=1.0,
        module_seed=4,
    )

    module.process()

    assert module.active.size == 2
    assert np.isclose(np.sum(engine.prior), 1.0)
    assert module.last_transition_result is not None
    assert module.last_transition_result.diagnostics["strategy_mode"] == "static"


def test_static_strategy_uses_common_selection_then_prior_contract():
    engine = _TinyEngine()
    engine.posterior = np.asarray([0.7, 0.3, 0.0, 0.0, 0.0, 0.0])
    engine.observation = (np.asarray([0.2]), 1, 1.0)
    module = StaticHypothesisTransitionModule(
        engine,
        module_seed=11,
        selection_strategy={
            "method": "strategy_chain",
            "init_num": 2,
            "init_hypotheses": [0, 1],
            "max_active_hypotheses": 2,
            "strategies": [
                {
                    "label": "retain",
                    "amount": "fixed",
                    "value": 1,
                    "method": "top_posterior",
                    "pool": "active",
                },
                {
                    "label": "explore",
                    "amount": "fixed",
                    "value": 1,
                    "method": "random",
                    "pool": "inactive",
                },
            ],
        },
        prior_assignment={
            "method": "conservative_carryover",
            "newcomer_mass": 0.1,
        },
    )

    module.process()

    result = module.last_transition_result
    assert result is not None
    assert result.diagnostics["strategy_mode"] == "static"
    assert result.selection.active_after.size == 2
    assert result.selection.newcomers.size == 1
    assert np.isclose(np.sum(result.prior_after), 1.0)
    assert np.all(result.prior_after[engine.hypotheses_mask == 0.0] == 0.0)


def test_dynamic_discrete_has_explicit_trial_level_strategy_state():
    engine = _TinyEngine()
    engine.posterior = np.asarray([0.65, 0.35, 0.0, 0.0, 0.0, 0.0])
    engine.observation = (np.asarray([0.2]), 1, 1.0)
    module = DynamicDiscreteHypothesisTransitionModule(
        engine,
        module_seed=12,
        init_num=2,
        init_hypotheses=[0, 1],
        max_active_hypotheses=2,
        state_controller={
            "method": "feedback_gated_softmax",
            "features": {
                "recent_accuracy_window": 2,
                "accuracy_delta_window": 2,
                "padding": "chance",
                "feedback_mode": "exact",
                "trial_progress_scale": 10,
            },
            "activation": {"temperature": 1.0},
            "states": [
                {
                    "id": "only_state",
                    "activation": {"bias": 0.0},
                    "strategies": [
                        {
                            "amount": "fixed",
                            "value": 1,
                            "method": "top_posterior",
                            "pool": "active",
                        },
                        {
                            "amount": "fixed",
                            "value": 1,
                            "method": "random",
                            "pool": "inactive",
                        },
                    ],
                    "post_to_prior": {
                        "method": "conservative_carryover",
                        "newcomer_mass": 0.1,
                    },
                }
            ],
        },
    )

    module.process()

    assert module.strategy_counts_log[-1]["selected_state"] == "only_state"
    assert module.strategy_counts_log[-1]["strategy_mode"] == "dynamic_discrete"
    assert module.last_transition_result is not None
    assert (
        module.last_transition_result.selection.diagnostics["selected_state"]
        == "only_state"
    )


def test_transition_mode_boundaries_fail_fast():
    with pytest.raises(ValueError, match="does not accept a state controller"):
        StaticHypothesisTransitionModule(
            _TinyEngine(),
            strategies=[
                {
                    "amount": "fixed",
                    "value": 1,
                    "method": "random",
                    "pool": "active",
                }
            ],
            state_controller={"states": []},
        )

    with pytest.raises(ValueError, match="requires at least one trial-varying control"):
        DynamicContinuousHypothesisTransitionModule(
            _TinyEngine(),
            capacity=2,
            m=0.2,
            g=0.35,
        )


def test_feedback_swap_is_a_static_reactive_strategy():
    engine = _TinyEngine()
    engine.observation = (np.asarray([0.2]), 1, 0.0)
    module = StaticFeedbackSwapHypothesisTransitionModule(
        engine,
        capacity=2,
        theta=1.0,
        init_hypotheses=[0, 1],
        module_seed=15,
    )

    module.process()
    assert not module.transition_log[-1]["swap_event"]

    engine.posterior = np.asarray([0.9, 0.1, 0.0, 0.0, 0.0, 0.0])
    engine.observation = (np.asarray([0.3]), 1, 1.0)
    module.process()

    event = module.transition_log[-1]
    assert event["strategy_mode"] == "static"
    assert event["swap_event"]
    assert event["dropped_hypothesis"] == 1
    assert event["new_hypothesis"] in {2, 3, 4, 5}
    assert np.isclose(engine.prior[event["new_hypothesis"]], 0.5)
    assert module.last_transition_result is not None


def test_model_structure_module_class_paths_are_importable():
    for config_path in sorted(Path("configs/model_struct").glob("*.yaml")):
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        for module_config in (payload.get("modules") or {}).values():
            class_path = module_config.get("class")
            if not isinstance(class_path, str):
                continue
            module_name, class_name = class_path.rsplit(".", 1)
            loaded_module = importlib.import_module(module_name)
            assert hasattr(loaded_module, class_name), (
                f"{config_path} references missing class {class_path}"
            )


def test_all_hypothesis_transition_config_paths_are_current_and_resolvable():
    old_fragments = (
        "hypo_transition.core",
        "hypo_transition.static_strategy.",
        "hypo_transition.profile_dynamic",
        "hypo_transition.continuous_dynamic",
        "hypo_transition.finite_workspace",
        "hypo_transition.strategy_chain",
        "modules.hypo_transitions.",
        "modules.static_hypo_transition.",
        "modules.profile_dynamic_hypo_transition.",
        "modules.continuous_dynamic_hypo_transition.",
        "modules/finite_workspace_transition",
        "modules/hypo_transition_strategies",
    )

    def visit(value, config_path: Path):
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key == "class" and isinstance(child, str) and (
                    "modules.hypo_transition." in child
                ):
                    module_name, class_name = child.rsplit(".", 1)
                    loaded_module = importlib.import_module(module_name)
                    assert hasattr(loaded_module, class_name), (
                        f"{config_path} references missing H class {child}"
                    )
                if key == "path" and isinstance(child, str) and (
                    "hypo_transition/candidates" in child
                ):
                    resolved_path = (config_path.parent / child).resolve()
                    assert resolved_path.is_file(), (
                        f"{config_path} references missing H candidate {child}"
                    )
                visit(child, config_path)
        elif isinstance(value, list):
            for child in value:
                visit(child, config_path)

    for root in (Path("configs"), Path("configs_exp4"), Path("configs_exp5")):
        for config_path in sorted(root.rglob("*.yaml")):
            raw = config_path.read_text(encoding="utf-8")
            assert not any(fragment in raw for fragment in old_fragments), config_path
            if "state_controller" in raw:
                assert "DynamicDiscreteHypothesisTransitionModule" in raw, config_path
            visit(yaml.safe_load(raw) or {}, config_path)


@pytest.mark.parametrize(
    "config_name, expected_mode",
    [
        ("pmh_model_cond1.yaml", "static"),
        ("pmh_model_cond1_active_set.yaml", "static"),
        ("pmh_model_cond1_v13.yaml", "dynamic_discrete"),
        ("pmh_model_cond1_v14.yaml", "dynamic_discrete"),
        ("pmh_model_cond1_0806.yaml", "dynamic_continuous"),
    ],
)
def test_canonical_transition_configs_run_the_declared_mode(
    config_name: str,
    expected_mode: str,
):
    config_path = Path("configs/model_struct") / config_name
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    transition_config = payload["modules"]["hypo_transitions_mod"]
    module_name, class_name = transition_config["class"].rsplit(".", 1)
    transition_class = getattr(importlib.import_module(module_name), class_name)
    engine = _TinyEngine()
    engine.observation = (np.asarray([0.2]), 1, 1.0)
    transition = transition_class(engine, **transition_config.get("kwargs", {}))

    transition.process()

    assert transition.strategy_mode == expected_mode
    assert transition.last_transition_result is not None
    assert np.isclose(np.sum(engine.prior), 1.0)


def test_0806_hyper_candidates_bind_static_and_dynamic_classes_to_controls():
    config_path = Path("configs/hyper_cd_cfg/pmh_cond1_hyper_cd_0806.yaml")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_payload = yaml.safe_load(
        Path("configs/model_struct/pmh_model_cond1_0806.yaml").read_text(
            encoding="utf-8"
        )
    )
    base_kwargs = model_payload["modules"]["hypo_transitions_mod"]["kwargs"]
    candidates = payload["hyperparam_space"]["__profile_candidate__"]["values"]
    class_key = "engine.modules.hypo_transitions_mod.class"
    rate_key = "engine.modules.hypo_transitions_mod.kwargs.rate_controller"
    range_key = "engine.modules.hypo_transitions_mod.kwargs.range_controller"

    assert len(candidates) == 28
    static_candidates = [
        item for item in candidates if "StaticWorkspace" in item[class_key]
    ]
    assert len(static_candidates) == 1
    static_candidate = static_candidates[0]
    assert static_candidate[rate_key]["m_beta_surprise"] == 0.0
    assert static_candidate[rate_key]["m_beta_uncertainty"] == 0.0
    assert static_candidate[range_key]["g_beta_surprise"] == 0.0
    assert static_candidate[range_key]["g_beta_uncertainty"] == 0.0

    for candidate in candidates:
        has_dynamic_control = any(
            float(candidate[rate_key][key]) > 0.0
            for key in ("m_beta_surprise", "m_beta_uncertainty")
        ) or any(
            float(candidate[range_key][key]) > 0.0
            for key in ("g_beta_surprise", "g_beta_uncertainty")
        )
        assert has_dynamic_control == (
            "DynamicContinuousHypothesisTransitionModule" in candidate[class_key]
        )

        module_name, class_name = candidate[class_key].rsplit(".", 1)
        transition_class = getattr(importlib.import_module(module_name), class_name)
        transition_kwargs = dict(base_kwargs)
        transition_kwargs["rate_controller"] = candidate[rate_key]
        transition_kwargs["range_controller"] = candidate[range_key]
        engine = _TinyEngine()
        engine.observation = (np.asarray([0.2]), 1, 1.0)
        transition = transition_class(engine, **transition_kwargs)
        transition.process()
        assert np.isclose(np.sum(engine.prior), 1.0)


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
    assert np.asarray(result.state_log["search_range"]).shape == (n_trials,)


def test_dispatcher_runs_single_trajectory_backend():
    n_trials = 8
    stimulus = np.linspace(0.1, 0.9, n_trials)[:, None]
    categories = np.where(stimulus[:, 0] < 0.5, 1, 2)
    feedback = np.ones(n_trials, dtype=float)
    config = _engine_config()
    config["inference"] = {"backend": "trajectory"}

    result = run_inference_backend(
        engine_config=config,
        subject_id=1,
        condition=1,
        stimulus=stimulus,
        choices=categories,
        feedback=feedback,
        inference_seed=20260806,
        processed_data_dir=Path("."),
    )

    assert isinstance(result, InferenceResult)
    assert isinstance(result, TrajectoryInferenceResult)
    assert result.backend == "trajectory"
    assert result.state_probabilities["hypothesis_prior"] is result.prior_log
    assert np.asarray(result.prior_log).shape == (n_trials, 6)
    assert np.asarray(result.posterior_log).shape == (n_trials, 6)
    assert len(result.step_log) == n_trials


def test_standard_runner_trajectory_uses_shared_choice_readout():
    n_trials = 10
    stimulus = np.linspace(0.1, 0.9, n_trials)[:, None]
    categories = np.where(stimulus[:, 0] < 0.5, 1, 2)
    feedback = np.ones(n_trials, dtype=float)
    config = _engine_config()
    config["inference"] = {"backend": "trajectory"}
    arrays = TrialArrays(
        stimulus=stimulus,
        choices=categories,
        feedback=feedback,
        categories=categories,
        target_probs=np.eye(2, dtype=float)[categories - 1],
    )

    result = evaluate_state_model_run(
        subject_id=1,
        condition=1,
        arrays=arrays,
        params={},
        engine_config_template=config,
        processed_data_dir=Path("."),
        window_size=4,
        keep_logs=False,
        prediction_mode="prior_t",
        selection_prediction_mode="prior_t",
        loss_metric="choice_brier",
        trajectory_seed=20260806,
    )

    metrics = result.metrics_by_mode["prior_t"]
    probabilities = np.asarray(metrics["pred_category_probs"], dtype=float)
    valid = np.asarray(metrics["valid_trial_mask"], dtype=bool)
    assert np.allclose(probabilities[valid].sum(axis=1), 1.0)
    assert metrics["choice_readout_method"] == "sharpened_expectation"
    assert np.isfinite(result.mean_error)


def test_particle_backend_uses_common_inference_result_contract():
    n_trials = 6
    stimulus = np.linspace(0.1, 0.9, n_trials)[:, None]
    categories = np.where(stimulus[:, 0] < 0.5, 1, 2)
    config = _engine_config()
    config["modules"]["hypo_transitions_mod"]["kwargs"]["range_controller"] = {
        "g_phi": 0.0,
        "g_beta_surprise": 0.5,
        "g_beta_uncertainty": 0.0,
        "g_surprise_center": 0.0,
        "g_surprise_scale": 1.0,
    }
    result = run_inference_backend(
        engine_config=config,
        subject_id=1,
        condition=1,
        stimulus=stimulus,
        choices=categories,
        feedback=np.ones(n_trials, dtype=float),
        inference_seed=20260806,
        choice_readout_power=2.0,
        output_lapse=0.05,
        processed_data_dir=Path("."),
    )

    assert isinstance(result, InferenceResult)
    assert result.backend == "particle_filter"
    assert result.observation_probabilities["prior_t"] is result.marginal_probabilities
    search_range = np.asarray(result.latent_summaries["search_range"], dtype=float)
    assert search_range.shape == (n_trials,)
    assert np.isclose(search_range[0], 0.35)
    assert np.any(search_range[1:] > search_range[0])
    assert np.allclose(result.marginal_probabilities.sum(axis=1), 1.0)


def test_rt_and_oral_readouts_return_normalized_measurements():
    rt = read_reaction_time(
        [0.8, 0.2],
        trial_index=4,
        replacement_fraction=0.5,
        newcomer_distance=0.25,
        config={
            "intercept": 1.0,
            "choice_uncertainty": 0.2,
            "replacement_fraction": 0.4,
            "newcomer_distance": 0.3,
            "scale": 0.5,
            "degrees_of_freedom": 6.0,
        },
    )
    assert np.isfinite(rt.log_location)
    assert rt.scale == 0.5
    assert 0.0 <= rt.choice_uncertainty <= 1.0

    oral = read_oral_report(
        [0.75, 0.25],
        [[0.9, 0.1], [0.2, 0.8]],
        reliability=0.8,
    )
    assert np.all(oral.probabilities >= 0.0)
    assert np.isclose(np.sum(oral.probabilities), 1.0)

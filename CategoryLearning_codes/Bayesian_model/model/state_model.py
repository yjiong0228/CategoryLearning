"""StateModel 的 trial 生命周期与高层模型接口。"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import numpy as np

from .assembly import build_engine, build_observation_likelihood, build_partition
from .config import ModelConfig, ModelContext
from .engine import IndexedSet
from .modules.base_module import ModulePhase, ModuleRole

if TYPE_CHECKING:
    from .readout import ChoicePrediction


@dataclass(frozen=True)
class PreparedTrial:
    """Model state after pre-choice processing and before an outcome exists."""

    trial_index: int
    stimulus: np.ndarray
    perceived_stimulus: np.ndarray
    prior: np.ndarray
    beta: np.ndarray | None
    log: dict[str, Any]


@dataclass
class GeneratedBehaviorTrajectory:
    """One model-generated behavioral trajectory under a fixed task schedule."""

    stimulus: np.ndarray
    perceived_stimulus: np.ndarray
    choices: np.ndarray
    feedback: np.ndarray
    cognitive_probabilities: np.ndarray
    observed_probabilities: np.ndarray
    prior: np.ndarray
    posterior: np.ndarray
    beta: np.ndarray
    step_log: list[dict[str, Any]]
    transition_log: list[dict[str, Any]]
    choice_seed: int | None


class StateModel:
    """装配 engine，并管理观察拟合与自主生成共享的 trial 生命周期。"""

    def __init__(
        self,
        engine_config: Mapping[str, Any] | ModelConfig,
        *,
        context: ModelContext,
        partition: Any | None = None,
        hypotheses_set: IndexedSet | None = None,
        module_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        """构造一个模型轨迹；结构配置与运行上下文必须显式传入。"""

        if not isinstance(context, ModelContext):
            raise TypeError("context must be a ModelContext.")
        self.model_config = ModelConfig.from_mapping(
            engine_config.to_dict()
            if isinstance(engine_config, ModelConfig)
            else engine_config
        )
        self.engine_config = self.model_config.to_dict()
        self.context = context
        self.condition = context.condition
        self.subject_id = context.subject_id
        self.processed_data_dir = context.processed_data_dir
        self.dataset_paths = context.dataset_paths

        for name, overrides in dict(module_overrides or {}).items():
            if name not in self.engine_config["modules"]:
                raise ValueError(f"module_overrides refers to unknown module {name!r}.")
            if not isinstance(overrides, Mapping):
                raise TypeError(f"module override {name!r} must be a mapping.")
            self.engine_config["modules"][name].update(dict(overrides))

        # Initialize partition
        self.partition_model = partition
        if self.partition_model is None:
            self.partition_model = build_partition(self.engine_config, self.condition)
        self.n_cats = int(
            getattr(
                self.partition_model,
                "n_cats",
                2 if self.condition == 1 else 4,
            )
        )
        # Initialize hypotheses set (length = partition_model.length)
        self.hypotheses_set = (
            IndexedSet(list(range(self.partition_model.length)))
            if hypotheses_set is None
            else hypotheses_set
        )
        self.observation_likelihood = build_observation_likelihood(
            self.engine_config,
            self.partition_model,
        )

        self.engine = build_engine(
            self.engine_config,
            hypotheses_set=self.hypotheses_set,
            partition=self.partition_model,
            observation_likelihood=self.observation_likelihood,
            context=self.context,
        )
        self._pending_trial: PreparedTrial | None = None
        self._completed_trial_count = 0
        self.posterior_log: list[np.ndarray] = []
        self.prior_log: list[np.ndarray] = []
        self.step_log: list[dict[str, Any]] = []

    def _store_trajectory_logs(
        self,
        posterior_log: list[np.ndarray],
        prior_log: list[np.ndarray],
        step_log: list[dict[str, Any]],
    ) -> None:
        """保存最近一次轨迹运行的日志。"""

        self.posterior_log = posterior_log
        self.prior_log = prior_log
        self.step_log = step_log

    def begin_trial(
        self,
        stimulus: Sequence[float] | np.ndarray,
        *,
        module_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> PreparedTrial:
        """Run the causal pre-choice phase for one trial.

        The model sees only the physical stimulus here.  Choice and feedback
        remain unavailable until :meth:`complete_trial`, which makes the same
        lifecycle usable for observed behavior and autonomous generation.
        """

        if self._pending_trial is not None:
            raise RuntimeError("complete the current trial before beginning another one.")
        physical = np.asarray(stimulus, dtype=float).reshape(-1)
        if physical.size == 0 or not np.all(np.isfinite(physical)):
            raise ValueError("stimulus must be a non-empty finite vector.")

        engine = self.engine
        if engine.prior is None:
            raise ValueError("model prior is not initialized.")
        engine.begin_trial(physical)
        engine.run_phase(
            ModulePhase.PRE_CHOICE,
            module_kwargs=module_kwargs,
        )

        perceived = np.asarray(engine.observation[0], dtype=float).copy()
        prior = np.asarray(engine.prior, dtype=float).copy()
        beta_value = getattr(engine, "beta", None)
        beta = None if beta_value is None else np.asarray(beta_value, dtype=float).copy()
        log: dict[str, Any] = {"perceived_stimulus": perceived.copy()}
        mask = getattr(engine, "hypotheses_mask", None)
        if mask is not None:
            log["active_indices"] = np.flatnonzero(
                np.asarray(mask, dtype=float) > 0.0
            ).astype(int)
        transition = engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        if transition is not None and bool(
            getattr(transition, "persistent_execution_enabled", False)
        ):
            log.update(
                {
                    "executed_hypothesis": int(transition.executed_hypothesis),
                    "execution_switch_event": bool(
                        transition.current_execution_switch_event
                    ),
                    "execution_switch_probability": float(
                        transition.current_execution_switch_probability
                    ),
                    "execution_dwell_trials": int(transition.execution_dwell_trials),
                }
            )
        mapping = engine.get_module(ModuleRole.MAPPING)
        if mapping is not None:
            orientation = np.asarray(
                getattr(mapping, "orientation_probability"), dtype=float
            ).copy()
            log["orientation_probability"] = orientation
            executed = log.get("executed_hypothesis")
            if executed is not None:
                log["executed_orientation_probability"] = float(
                    orientation[int(executed)]
                )
        prepared = PreparedTrial(
            trial_index=int(self._completed_trial_count),
            stimulus=physical.copy(),
            perceived_stimulus=perceived,
            prior=prior,
            beta=beta,
            log=log,
        )
        self._pending_trial = prepared
        return prepared

    def predict_choice(
        self,
        *,
        choices: Sequence[int] | np.ndarray,
        feedback: Sequence[float] | np.ndarray,
        choice_readout_config: Mapping[str, Any],
        output_noise_config: Mapping[str, Any],
        rng: np.random.Generator,
        sticky_state: dict[str, Any],
        post_error_lapse_state: float,
        latent_volatility_value: float = 0.0,
    ) -> "ChoicePrediction":
        """Return the observable choice distribution for the prepared trial."""

        if self._pending_trial is None:
            raise RuntimeError("begin_trial() must be called before predict_choice().")
        from .readout import predict_choice_from_model

        return predict_choice_from_model(
            self,
            self._pending_trial.perceived_stimulus,
            trial_idx=self._pending_trial.trial_index,
            choices=choices,
            feedback=feedback,
            choice_readout_config=choice_readout_config,
            output_noise_config=output_noise_config,
            rng=rng,
            sticky_state=sticky_state,
            post_error_lapse_state=post_error_lapse_state,
            latent_volatility_value=latent_volatility_value,
        )

    def complete_trial(
        self,
        choice: int,
        feedback: float,
        *,
        update_state: bool = True,
        module_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Consume a choice/outcome and run the post-choice learning phase."""

        prepared = self._pending_trial
        if prepared is None:
            raise RuntimeError("begin_trial() must be called before complete_trial().")
        choice_value = int(choice)
        if choice_value < 1 or choice_value > int(self.n_cats):
            raise ValueError(
                f"choice must be 1-indexed in [1, {self.n_cats}], got {choice!r}."
            )
        feedback_value = float(feedback)
        if not np.isfinite(feedback_value) or not 0.0 <= feedback_value <= 1.0:
            raise ValueError("feedback must be a finite value in [0, 1].")

        observation = (
            prepared.perceived_stimulus.copy(),
            choice_value,
            feedback_value,
        )
        self.engine.observation = observation
        self.engine.record_outcome(observation)

        if update_state:
            self.engine.compute_likelihood()
            self.engine.run_phase(
                ModulePhase.POST_CHOICE,
                module_kwargs=module_kwargs,
            )
        else:
            self.engine.posterior = prepared.prior.copy()
        posterior = getattr(self.engine, "posterior", None)
        if posterior is None:
            raise ValueError("post-choice agenda did not produce engine.posterior.")
        posterior_snapshot = np.asarray(posterior, dtype=float).copy()
        log = dict(prepared.log)
        log["choice"] = choice_value
        log["feedback"] = feedback_value
        mapping = self.engine.get_module(ModuleRole.MAPPING)
        if mapping is not None:
            orientation = np.asarray(
                getattr(mapping, "orientation_probability"), dtype=float
            ).copy()
            log["orientation_probability_post"] = orientation
            executed = log.get("executed_hypothesis")
            if executed is not None:
                log["executed_orientation_probability_post"] = float(
                    orientation[int(executed)]
                )
        self._pending_trial = None
        self._completed_trial_count += 1
        return posterior_snapshot, prepared.prior.copy(), log

    def fit_step_by_step(self, data: list | np.ndarray, **kwargs):
        """Condition the model state on observed choice/feedback trials."""

        module_kwargs = kwargs.get("module_kwargs", {})
        if data is None:
            raise ValueError("No trial data were supplied to fit_step_by_step().")
        resolved_data = data
        step_log: list[dict[str, Any]] = []
        posterior_log: list[np.ndarray] = []
        prior_log: list[np.ndarray] = []
        for datum in resolved_data:
            if len(datum) < 3:
                raise ValueError(
                    "each observed trial must contain (stimulus, choice, feedback)."
                )
            self.begin_trial(datum[0], module_kwargs=module_kwargs)
            posterior, prior, log = self.complete_trial(
                int(datum[1]),
                float(datum[2]),
                module_kwargs=module_kwargs,
            )
            step_log.append(log)
            posterior_log.append(posterior)
            prior_log.append(prior)

        self._store_trajectory_logs(posterior_log, prior_log, step_log)
        return posterior_log, prior_log

    def generate_step_by_step(
        self,
        stimulus: Sequence[Sequence[float]] | np.ndarray,
        feedback_provider: Callable[[int, int], float],
        *,
        choice_seed: int | None = None,
        choice_readout_config: Mapping[str, Any] | None = None,
        output_noise_config: Mapping[str, Any] | None = None,
        module_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> GeneratedBehaviorTrajectory:
        """Generate choices and learn from task-produced feedback trial by trial."""

        if not callable(feedback_provider):
            raise TypeError("feedback_provider must be callable.")
        physical = np.asarray(stimulus, dtype=float)
        if physical.ndim != 2 or physical.shape[0] == 0:
            raise ValueError("stimulus must be a non-empty 2-D array.")
        if not np.all(np.isfinite(physical)):
            raise ValueError("stimulus contains non-finite values.")

        from .readout import (
            resolve_choice_readout_config,
            resolve_output_noise_config,
        )

        readout_config = (
            resolve_choice_readout_config(None, self.engine_config)
            if choice_readout_config is None
            else resolve_choice_readout_config(
                {"choice_readout": {"kwargs": dict(choice_readout_config)}},
                self.engine_config,
            )
        )
        noise_config = (
            resolve_output_noise_config(None, self.engine_config)
            if output_noise_config is None
            else resolve_output_noise_config(
                {"output_noise": {"kwargs": dict(output_noise_config)}},
                self.engine_config,
            )
        )
        rng = np.random.default_rng(choice_seed)
        n_trials = int(physical.shape[0])
        choices = np.zeros(n_trials, dtype=int)
        feedback = np.full(n_trials, np.nan, dtype=float)
        perceived = np.zeros_like(physical, dtype=float)
        cognitive = np.zeros((n_trials, self.n_cats), dtype=float)
        observed = np.zeros((n_trials, self.n_cats), dtype=float)
        prior_log: list[np.ndarray] = []
        posterior_log: list[np.ndarray] = []
        beta_log: list[np.ndarray] = []
        step_log: list[dict[str, Any]] = []
        sticky_state: dict[str, Any] = {}
        post_error_lapse_state = 0.0

        for trial_index, trial_stimulus in enumerate(physical):
            prepared = self.begin_trial(
                trial_stimulus,
                module_kwargs=module_kwargs,
            )
            transition = self.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
            latent_volatility = float(
                getattr(transition, "latent_volatility_state", 0.0)
                if transition is not None
                else 0.0
            )
            prediction = self.predict_choice(
                choices=choices,
                feedback=feedback,
                choice_readout_config=readout_config,
                output_noise_config=noise_config,
                rng=rng,
                sticky_state=sticky_state,
                post_error_lapse_state=post_error_lapse_state,
                latent_volatility_value=latent_volatility,
            )
            post_error_lapse_state = prediction.post_error_lapse_state
            choice = int(rng.choice(self.n_cats, p=prediction.observed_probabilities)) + 1
            outcome = float(feedback_provider(trial_index, choice))
            posterior, prior, log = self.complete_trial(
                choice,
                outcome,
                module_kwargs=module_kwargs,
            )

            perceived[trial_index] = prepared.perceived_stimulus
            choices[trial_index] = choice
            feedback[trial_index] = outcome
            cognitive[trial_index] = prediction.cognitive_probabilities
            observed[trial_index] = prediction.observed_probabilities
            prior_log.append(prior)
            posterior_log.append(posterior)
            beta_log.append(
                np.full(self.engine.set_size, np.nan, dtype=float)
                if prepared.beta is None
                else prepared.beta.copy()
            )
            log.update(
                {
                    "cognitive_choice_probabilities": prediction.cognitive_probabilities.copy(),
                    "choice_probabilities": prediction.observed_probabilities.copy(),
                    "choice_readout": dict(prediction.readout_details),
                    "output_lapse": float(prediction.output_lapse),
                }
            )
            step_log.append(log)

        self._store_trajectory_logs(posterior_log, prior_log, step_log)
        transition = self.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        transition_log = (
            [dict(item) for item in getattr(transition, "transition_log", [])]
            if transition is not None
            else []
        )
        return GeneratedBehaviorTrajectory(
            stimulus=physical.copy(),
            perceived_stimulus=perceived,
            choices=choices,
            feedback=feedback,
            cognitive_probabilities=cognitive,
            observed_probabilities=observed,
            prior=np.asarray(prior_log, dtype=float),
            posterior=np.asarray(posterior_log, dtype=float),
            beta=np.asarray(beta_log, dtype=float),
            step_log=step_log,
            transition_log=transition_log,
            choice_seed=None if choice_seed is None else int(choice_seed),
        )


__all__ = [
    "GeneratedBehaviorTrajectory",
    "PreparedTrial",
    "StateModel",
]

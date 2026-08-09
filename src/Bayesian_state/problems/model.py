"""
Base Model
"""
import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
import numpy as np
from .base_problem import BaseSet, BaseEngine
from .partitions import Partition
from ..utils.paths import PROCESSED_DATA_DIR

if TYPE_CHECKING:
    from .modules.readout import ChoicePrediction


def _get_class_from_string(class_path: str):
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _build_partition_from_config(engine_config: dict, condition: int):
    partition_cfg = engine_config.get("partition")
    if partition_cfg is None:
        n_dims = int(engine_config.get("n_dims", 4))
        n_cats = int(engine_config.get("n_cats", 2 if condition == 1 else 4))
        return Partition(n_dims, n_cats)

    if not isinstance(partition_cfg, dict):
        raise ValueError("engine_config.partition must be a mapping when provided.")

    class_path = partition_cfg.get("class")
    if class_path is None:
        raise ValueError("engine_config.partition must include a class path.")

    partition_class = _get_class_from_string(class_path) if isinstance(class_path, str) else class_path
    kwargs = dict(partition_cfg.get("kwargs", {}) or {})
    return partition_class(**kwargs)


@dataclass(frozen=True)
class PreparedTrial:
    """Model state after pre-choice processing and before an outcome exists."""

    trial_index: int
    stimulus: np.ndarray
    perceived_stimulus: np.ndarray
    prior: np.ndarray
    beta: np.ndarray | None
    log: Dict[str, Any]


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
    step_log: list[Dict[str, Any]]
    transition_log: list[Dict[str, Any]]
    choice_seed: int | None



class StateModel:
    """
    State Model, initialize an engine
    Refreshes the engine step by step:
        StateModel ----[data]----> engine
        engine ----[posterior]----> StateModel
    """

    def __init__(self, engine_config, **kwargs):
        """
        """

        if engine_config is None:
            raise ValueError("engine_config must be a dict; got None. Check MODEL_STRUCT and model_choice settings.")
        # Initialize attributes
        self.engine_config = deepcopy(engine_config)
        self.all_centers = None
        self.data = None
        self.hypotheses_set = BaseSet([])
        self.observation_set = BaseSet([])

        self.condition = kwargs.get("condition", 1)
        self.subject_id = kwargs.pop("subject_id", None)
        processed_data_dir = kwargs.pop("processed_data_dir", None)
        self.dataset_paths = kwargs.pop("dataset_paths", None)
        if processed_data_dir is None:
            self.processed_data_dir = PROCESSED_DATA_DIR.resolve()
        else:
            self.processed_data_dir = Path(processed_data_dir).resolve()

        # Initialize partition
        self.partition_model = kwargs.get(
            "partition", _build_partition_from_config(self.engine_config, self.condition))
        self.n_cats = int(getattr(self.partition_model, "n_cats", 2 if self.condition == 1 else 4))
        # Initialize hypotheses set (length = partition_model.length)
        self.hypotheses_set = kwargs.get(
            "space", BaseSet(list(range(self.partition_model.length))))

        # Merge module overrides provided via kwargs
        for key, value in kwargs.items():
            if key in self.engine_config.get("modules", {}):
                self.engine_config["modules"][key].update(value)

        # initialize engine
        self.engine = BaseEngine(
            self.engine_config["agenda"],
            hypotheses_set=self.hypotheses_set,
            partition=self.partition_model,
        )
        # expose shared context for modules (e.g., PerceptionModule auto loading)
        self.engine.subject_id = self.subject_id
        self.engine.processed_data_dir = self.processed_data_dir
        self.engine.dataset_paths = self.dataset_paths
        # build modules for the engine
        self.engine.build_modules(self.engine_config["modules"])
        self._pending_trial: PreparedTrial | None = None
        self._completed_trial_count = 0


    def save(self, posterior_log, prior_log, step_log):
        """
        保存结果
        """
        self.posterior_log = posterior_log
        self.prior_log = prior_log
        self.step_log = step_log
        

    @staticmethod
    def _pre_choice_module(name: str) -> bool:
        return name in {"perception_mod", "hypo_transitions_mod"}

    def _run_agenda_phase(
        self,
        *,
        pre_choice: bool,
        module_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        agenda = self.engine.agenda
        if agenda is None:
            raise ValueError("inference agenda is not defined.")
        unknown = [name for name in agenda if name not in self.engine.modules]
        if unknown:
            raise ValueError(f"inference agenda contains unknown modules: {unknown}.")

        kwargs_by_module = module_kwargs or {}
        for module_name in agenda:
            if self._pre_choice_module(module_name) != bool(pre_choice):
                continue
            module = self.engine.modules[module_name]
            if not hasattr(module, "process"):
                raise TypeError(f"Module {module_name} has no process() method.")
            current_kwargs = dict(kwargs_by_module.get(module_name, {}))
            if module_name == "hypo_transitions_mod":
                current_kwargs["defer_outcome_recording"] = True
            module.process(**current_kwargs)

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
        engine.prior = (
            np.asarray(engine.posterior, dtype=float).copy()
            if engine.posterior is not None
            else np.asarray(engine.prior, dtype=float).copy()
        )
        # None marks the as-yet unobserved outcome; it is not a synthetic
        # choice or feedback value and must never enter post-choice modules.
        engine.observation = (physical.copy(), None, None)
        self._run_agenda_phase(
            pre_choice=True,
            module_kwargs=module_kwargs,
        )

        perceived = np.asarray(engine.observation[0], dtype=float).copy()
        prior = np.asarray(engine.prior, dtype=float).copy()
        beta_value = getattr(engine, "beta", None)
        beta = None if beta_value is None else np.asarray(beta_value, dtype=float).copy()
        log: Dict[str, Any] = {"perceived_stimulus": perceived.copy()}
        mask = getattr(engine, "hypotheses_mask", None)
        if mask is not None:
            log["active_indices"] = np.flatnonzero(
                np.asarray(mask, dtype=float) > 0.0
            ).astype(int)
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
        sticky_state: Dict[str, Any],
        post_error_lapse_state: float,
        latent_volatility_value: float = 0.0,
    ) -> "ChoicePrediction":
        """Return the observable choice distribution for the prepared trial."""

        if self._pending_trial is None:
            raise RuntimeError("begin_trial() must be called before predict_choice().")
        from .modules.readout import predict_choice_from_model

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
    ) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
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
        transition = self.engine.modules.get("hypo_transitions_mod")
        if transition is not None and hasattr(transition, "record_outcome"):
            transition.record_outcome(observation)

        if update_state:
            self._run_agenda_phase(
                pre_choice=False,
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
        self._pending_trial = None
        self._completed_trial_count += 1
        return posterior_snapshot, prepared.prior.copy(), log

    def fit_step_by_step(self, data: List | np.ndarray, **kwargs):
        """Condition the model state on observed choice/feedback trials."""

        module_kwargs = kwargs.get("module_kwargs", {})
        resolved_data = self.data if data is None else data
        if resolved_data is None:
            raise ValueError("No trial data were supplied to fit_step_by_step().")
        step_log: list[Dict[str, Any]] = []
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

        self.save(posterior_log, prior_log, step_log)
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

        from .modules.readout import (
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
        step_log: list[Dict[str, Any]] = []
        sticky_state: Dict[str, Any] = {}
        post_error_lapse_state = 0.0

        for trial_index, trial_stimulus in enumerate(physical):
            prepared = self.begin_trial(
                trial_stimulus,
                module_kwargs=module_kwargs,
            )
            transition = self.engine.modules.get("hypo_transitions_mod")
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

        self.save(posterior_log, prior_log, step_log)
        transition = self.engine.modules.get("hypo_transitions_mod")
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

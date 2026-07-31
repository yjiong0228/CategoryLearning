"""Online particle filtering for the stochastic minimal new-plan model.

The particle filter is a numerical integration layer.  It does not add a
cognitive state or change the swap-one transition.  Each particle is one
possible latent trajectory of perception noise, active-set initialization,
swap events, tie breaking, and newcomer sampling.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .optimizer_common import stable_seed


@dataclass
class NewPlanParticleFilterResult:
    marginal_probabilities: np.ndarray
    marginal_hypothesis_prior: np.ndarray
    marginal_active_probability: np.ndarray
    pre_choice_ess: np.ndarray
    post_choice_ess: np.ndarray
    resampled: np.ndarray
    resampling_unique_ancestors: np.ndarray
    filtered_swap_probability: np.ndarray
    filtered_swap_event_probability: np.ndarray
    final_weights: np.ndarray
    particle_swap_counts: np.ndarray
    resampling_log: list[dict[str, Any]]
    particle_count: int
    resample_threshold_fraction: float
    filter_seed: int


@dataclass
class _CognitiveSnapshot:
    prior: np.ndarray
    posterior: np.ndarray | None
    likelihood: np.ndarray | None
    hypotheses_mask: np.ndarray
    beta: np.ndarray
    transition_active: np.ndarray
    transition_old_active: np.ndarray
    transition_previous_feedback: float | None
    transition_trial_index: int
    memory_state: dict[str, np.ndarray]
    memory_baseline_state: dict[str, float]
    memory_mask: np.ndarray
    memory_prior: np.ndarray


def effective_sample_size(weights: Sequence[float] | np.ndarray) -> float:
    values = np.asarray(weights, dtype=float).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("weights must be a non-empty finite non-negative vector.")
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("weights must have positive mass.")
    normalized = values / total
    return 1.0 / float(np.sum(np.square(normalized)))


def systematic_resample(
    weights: Sequence[float] | np.ndarray,
    uniform: float,
) -> np.ndarray:
    """Return systematic-resampling ancestor indices.

    ``uniform`` is expressed on ``[0, 1)`` and converted to the first offset
    on ``[0, 1 / n)``.  This makes the function deterministic and easy to
    pair across candidate models.
    """

    values = np.asarray(weights, dtype=float).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("weights must be a non-empty finite non-negative vector.")
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("weights must have positive mass.")
    unit_uniform = float(uniform)
    if not np.isfinite(unit_uniform) or not 0.0 <= unit_uniform < 1.0:
        raise ValueError("uniform must be finite and lie in [0, 1).")
    normalized = values / total
    cumulative = np.cumsum(normalized)
    cumulative[-1] = 1.0
    n_particles = int(values.size)
    positions = (
        unit_uniform / float(n_particles)
        + np.arange(n_particles, dtype=float) / float(n_particles)
    )
    return np.searchsorted(cumulative, positions, side="right").astype(int)


def _normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError("Probability vector must be finite and non-negative.")
    total = float(np.sum(array))
    if total <= 0.0:
        raise ValueError("Probability vector has zero mass.")
    return array / total


def _particle_seed(filter_seed: int, particle_index: int, role: str) -> int:
    return stable_seed(
        {
            "seed_role": role,
            "filter_seed": int(filter_seed),
            "particle_index": int(particle_index),
        }
    )


def _future_seed(
    filter_seed: int,
    trial_index: int,
    particle_index: int,
    role: str,
) -> int:
    return stable_seed(
        {
            "seed_role": role,
            "filter_seed": int(filter_seed),
            "trial_index": int(trial_index),
            "particle_index": int(particle_index),
        }
    )


def _particle_config(
    engine_config: Mapping[str, Any],
    filter_seed: int,
    particle_index: int,
) -> dict[str, Any]:
    config = deepcopy(dict(engine_config))
    modules = config.setdefault("modules", {})
    transition = modules.get("hypo_transitions_mod")
    perception = modules.get("perception_mod")
    if not isinstance(transition, Mapping) or not isinstance(perception, Mapping):
        raise ValueError(
            "Particle filtering requires perception_mod and hypo_transitions_mod."
        )
    transition_kwargs = transition.setdefault("kwargs", {})
    perception_kwargs = perception.setdefault("kwargs", {})
    transition_kwargs["module_seed"] = _particle_seed(
        filter_seed, particle_index, "newplan_pf_transition_initial"
    )
    perception_kwargs["module_seed"] = _particle_seed(
        filter_seed, particle_index, "newplan_pf_perception_initial"
    )
    return config


def _snapshot(model: Any) -> _CognitiveSnapshot:
    engine = model.engine
    transition = engine.modules["hypo_transitions_mod"]
    memory = engine.modules["memory_mod"]
    beta_mod = engine.modules["beta_mod"]
    posterior = getattr(engine, "posterior", None)
    likelihood = getattr(engine, "likelihood", None)
    return _CognitiveSnapshot(
        prior=np.asarray(engine.prior, dtype=float).copy(),
        posterior=(
            None
            if posterior is None
            else np.asarray(posterior, dtype=float).copy()
        ),
        likelihood=(
            None
            if likelihood is None
            else np.asarray(likelihood, dtype=float).copy()
        ),
        hypotheses_mask=np.asarray(engine.hypotheses_mask, dtype=float).copy(),
        beta=np.asarray(beta_mod.beta, dtype=float).copy(),
        transition_active=np.asarray(transition.active, dtype=int).copy(),
        transition_old_active=np.asarray(transition.old_active, dtype=int).copy(),
        transition_previous_feedback=(
            None
            if transition.previous_feedback is None
            else float(transition.previous_feedback)
        ),
        transition_trial_index=int(transition.trial_index),
        memory_state={
            str(key): np.asarray(value, dtype=float).copy()
            for key, value in memory.state.items()
        },
        memory_baseline_state={
            str(key): float(value)
            for key, value in memory.baseline_state.items()
        },
        memory_mask=np.asarray(memory.mask, dtype=float).copy(),
        memory_prior=np.asarray(memory.prior, dtype=float).copy(),
    )


def _restore(
    model: Any,
    snapshot: _CognitiveSnapshot,
    *,
    filter_seed: int,
    trial_index: int,
    particle_index: int,
) -> None:
    engine = model.engine
    transition = engine.modules["hypo_transitions_mod"]
    perception = engine.modules["perception_mod"]
    memory = engine.modules["memory_mod"]
    beta_mod = engine.modules["beta_mod"]

    engine.prior = snapshot.prior.copy()
    engine.posterior = (
        None if snapshot.posterior is None else snapshot.posterior.copy()
    )
    engine.likelihood = (
        None if snapshot.likelihood is None else snapshot.likelihood.copy()
    )
    engine.hypotheses_mask = snapshot.hypotheses_mask.copy()

    transition.active = snapshot.transition_active.copy()
    transition.old_active = snapshot.transition_old_active.copy()
    transition.previous_feedback = snapshot.transition_previous_feedback
    transition.trial_index = int(snapshot.transition_trial_index)
    transition.transition_log.clear()
    transition.reseed_future(
        _future_seed(
            filter_seed,
            trial_index,
            particle_index,
            "newplan_pf_transition_after_resample",
        )
    )
    perception.reseed_future(
        _future_seed(
            filter_seed,
            trial_index,
            particle_index,
            "newplan_pf_perception_after_resample",
        )
    )

    beta_mod.beta = snapshot.beta.copy()
    beta_mod.beta_log.clear()
    engine.beta = beta_mod.beta

    memory.state = {
        key: value.copy() for key, value in snapshot.memory_state.items()
    }
    engine.state = memory.state
    memory.baseline_state = dict(snapshot.memory_baseline_state)
    memory.mask = snapshot.memory_mask.copy()
    memory.prior = snapshot.memory_prior.copy()


def _choice_probability(
    model: Any,
    perceived_stimulus: np.ndarray,
    rho: float,
    epsilon: float,
) -> np.ndarray:
    engine = model.engine
    n_categories = int(model.n_cats)
    prior = np.asarray(engine.prior, dtype=float)
    beta = np.asarray(engine.beta, dtype=float)
    active = np.flatnonzero(
        np.asarray(engine.hypotheses_mask, dtype=float) > 0.0
    )
    if active.size == 0:
        raise RuntimeError("Particle has an empty active hypothesis set.")
    rho_value = float(rho)
    if not np.isfinite(rho_value) or rho_value <= 0.0:
        raise ValueError("rho must be finite and positive.")
    log_readout = rho_value * np.log(
        np.clip(prior[active], 1e-300, None)
    )
    log_readout -= float(np.max(log_readout))
    readout = _normalize(np.exp(log_readout))
    cognitive = np.zeros(n_categories, dtype=float)
    for weight, hypothesis in zip(readout, active):
        raw = model.partition_model.get_category_probabilities(
            hypo=int(hypothesis),
            data=([perceived_stimulus], [1], [1.0]),
            beta=float(beta[hypothesis]),
            distance_mode=getattr(engine, "distance_mode", "prototype"),
        )
        probability = _normalize(np.asarray(raw[:, 0], dtype=float))
        cognitive += float(weight) * probability
    cognitive = _normalize(cognitive)
    return _normalize(
        (1.0 - float(epsilon)) * cognitive
        + float(epsilon) / float(n_categories)
    )


def run_newplan_particle_filter(
    *,
    engine_config: Mapping[str, Any],
    subject_id: int,
    stimulus: Sequence[Sequence[float]] | np.ndarray,
    choices: Sequence[int] | np.ndarray,
    feedback: Sequence[float] | np.ndarray,
    particle_count: int,
    rho: float,
    epsilon: float = 0.0,
    epsilon_schedule: Sequence[float] | np.ndarray | None = None,
    learning_update_probability: float = 1.0,
    filter_seed: int = 20260730,
    resample_threshold_fraction: float = 0.5,
    valid_trial_mask: Sequence[bool] | np.ndarray | None = None,
    processed_data_dir: Path | str | None = None,
    dataset_paths: Mapping[str, Path | str] | None = None,
) -> NewPlanParticleFilterResult:
    """Filter one observed condition-1 trajectory using bootstrap particles."""

    from ..problems import StateModel

    x = np.asarray(stimulus, dtype=float)
    observed_choices = np.asarray(choices, dtype=int).reshape(-1)
    observed_feedback = np.asarray(feedback, dtype=float).reshape(-1)
    if x.ndim != 2:
        raise ValueError("stimulus must be a 2-D array.")
    n_trials = int(x.shape[0])
    if observed_choices.shape[0] != n_trials or observed_feedback.shape[0] != n_trials:
        raise ValueError("stimulus, choices, and feedback must have equal trial counts.")
    if not np.all(np.isin(observed_choices, [1, 2])):
        raise ValueError("Condition-1 choices must be encoded as 1 or 2.")
    if not np.all(np.isfinite(observed_feedback)) or np.any(
        (observed_feedback < 0.0) | (observed_feedback > 1.0)
    ):
        raise ValueError("feedback must contain finite values in [0, 1].")
    n_particles = int(particle_count)
    if n_particles < 2:
        raise ValueError("particle_count must be at least 2.")
    rho_value = float(rho)
    epsilon_value = float(epsilon)
    threshold_fraction = float(resample_threshold_fraction)
    update_probability = float(learning_update_probability)
    if not np.isfinite(rho_value) or rho_value <= 0.0:
        raise ValueError("rho must be finite and positive.")
    if not np.isfinite(epsilon_value) or not 0.0 <= epsilon_value <= 1.0:
        raise ValueError("epsilon must lie in [0, 1].")
    if epsilon_schedule is None:
        epsilon_by_trial = np.full(n_trials, epsilon_value, dtype=float)
    else:
        epsilon_by_trial = np.asarray(
            epsilon_schedule, dtype=float
        ).reshape(-1)
        if epsilon_by_trial.size != n_trials:
            raise ValueError(
                "epsilon_schedule length must match the number of trials."
            )
        if (
            not np.all(np.isfinite(epsilon_by_trial))
            or np.any(epsilon_by_trial < 0.0)
            or np.any(epsilon_by_trial > 1.0)
        ):
            raise ValueError("epsilon_schedule values must lie in [0, 1].")
    if (
        not np.isfinite(threshold_fraction)
        or not 0.0 < threshold_fraction <= 1.0
    ):
        raise ValueError("resample_threshold_fraction must lie in (0, 1].")
    if (
        not np.isfinite(update_probability)
        or not 0.0 <= update_probability <= 1.0
    ):
        raise ValueError("learning_update_probability must lie in [0, 1].")
    if valid_trial_mask is None:
        valid = np.ones(n_trials, dtype=bool)
    else:
        valid = np.asarray(valid_trial_mask, dtype=bool).reshape(-1)
        if valid.shape[0] != n_trials:
            raise ValueError("valid_trial_mask length does not match trials.")

    models = []
    shared_partition = None
    shared_space = None
    for particle_index in range(n_particles):
        config = _particle_config(engine_config, int(filter_seed), particle_index)
        kwargs: dict[str, Any] = {
            "condition": 1,
            "subject_id": int(subject_id),
            "processed_data_dir": processed_data_dir,
            "dataset_paths": dataset_paths,
        }
        if shared_partition is not None:
            kwargs["partition"] = shared_partition
            kwargs["space"] = shared_space
        model = StateModel(config, **kwargs)
        if shared_partition is None:
            shared_partition = model.partition_model
            shared_space = model.hypotheses_set
        models.append(model)

    n_hypotheses = int(models[0].engine.set_size)
    marginal_hypothesis_prior = np.zeros((n_trials, n_hypotheses), dtype=float)
    marginal_active_probability = np.zeros((n_trials, n_hypotheses), dtype=float)

    weights = np.full(n_particles, 1.0 / float(n_particles), dtype=float)
    particle_swap_counts = np.zeros(n_particles, dtype=int)
    marginal = np.zeros((n_trials, 2), dtype=float)
    pre_ess = np.zeros(n_trials, dtype=float)
    post_ess = np.zeros(n_trials, dtype=float)
    resampled = np.zeros(n_trials, dtype=bool)
    unique_ancestors = np.full(n_trials, n_particles, dtype=int)
    filtered_swap_probability = np.zeros(n_trials, dtype=float)
    filtered_swap_event_probability = np.zeros(n_trials, dtype=float)
    resampling_log: list[dict[str, Any]] = []

    for trial_index in range(n_trials):
        particle_predictions = np.zeros((n_particles, 2), dtype=float)
        particle_priors = np.zeros((n_particles, n_hypotheses), dtype=float)
        particle_active = np.zeros((n_particles, n_hypotheses), dtype=float)
        swap_probabilities = np.zeros(n_particles, dtype=float)
        swap_events = np.zeros(n_particles, dtype=float)

        for particle_index, model in enumerate(models):
            engine = model.engine
            if engine.posterior is not None:
                engine.prior = np.asarray(engine.posterior, dtype=float).copy()
            engine.observation = (
                x[trial_index].copy(),
                int(observed_choices[trial_index]),
                float(observed_feedback[trial_index]),
            )
            engine.modules["perception_mod"].process()
            perceived = np.asarray(engine.observation[0], dtype=float).copy()
            transition = engine.modules["hypo_transitions_mod"]
            transition.process()
            event = transition.transition_log[-1]
            particle_priors[particle_index] = _normalize(
                np.asarray(engine.prior, dtype=float)
            )
            particle_active[particle_index] = (
                np.asarray(engine.hypotheses_mask, dtype=float) > 0.0
            ).astype(float)
            swap_probabilities[particle_index] = float(
                event["swap_probability"]
            )
            swap_events[particle_index] = float(bool(event["swap_event"]))
            particle_predictions[particle_index] = _choice_probability(
                model,
                perceived,
                rho=rho_value,
                epsilon=float(epsilon_by_trial[trial_index]),
            )

        pre_ess[trial_index] = effective_sample_size(weights)
        marginal_hypothesis_prior[trial_index] = np.sum(
            weights[:, None] * particle_priors,
            axis=0,
        )
        marginal_hypothesis_prior[trial_index] = _normalize(
            marginal_hypothesis_prior[trial_index]
        )
        marginal_active_probability[trial_index] = np.sum(
            weights[:, None] * particle_active,
            axis=0,
        )
        marginal[trial_index] = np.sum(
            weights[:, None] * particle_predictions,
            axis=0,
        )
        marginal[trial_index] = _normalize(marginal[trial_index])

        choice_index = int(observed_choices[trial_index]) - 1
        if valid[trial_index]:
            weights *= np.clip(
                particle_predictions[:, choice_index],
                1e-12,
                1.0,
            )
            weight_total = float(np.sum(weights))
            if not np.isfinite(weight_total) or weight_total <= 0.0:
                weights.fill(1.0 / float(n_particles))
            else:
                weights /= weight_total
        post_ess[trial_index] = effective_sample_size(weights)
        filtered_swap_probability[trial_index] = float(
            np.sum(weights * swap_probabilities)
        )
        filtered_swap_event_probability[trial_index] = float(
            np.sum(weights * swap_events)
        )
        particle_swap_counts += swap_events.astype(int)

        for particle_index, model in enumerate(models):
            engine = model.engine
            update_seed = _future_seed(
                int(filter_seed),
                trial_index,
                particle_index,
                "newplan_learning_update_gate",
            )
            update_occurs = bool(
                np.random.default_rng(update_seed).random()
                < update_probability
            )
            if update_occurs:
                engine.modules["likelihood_mod"].process()
                engine.modules["memory_mod"].process()
                engine.modules["beta_mod"].process()
            else:
                engine.posterior = np.asarray(
                    engine.prior, dtype=float
                ).copy()
            engine.modules["beta_mod"].beta_log.clear()
            engine.modules["hypo_transitions_mod"].transition_log.clear()

        if post_ess[trial_index] < threshold_fraction * float(n_particles):
            resampling_seed = _future_seed(
                int(filter_seed),
                trial_index,
                0,
                "newplan_pf_systematic_resampling",
            )
            uniform = float(np.random.default_rng(resampling_seed).random())
            ancestors = systematic_resample(weights, uniform)
            snapshots = [_snapshot(models[int(index)]) for index in ancestors]
            copied_swap_counts = particle_swap_counts[ancestors].copy()
            for particle_index, snapshot in enumerate(snapshots):
                _restore(
                    models[particle_index],
                    snapshot,
                    filter_seed=int(filter_seed),
                    trial_index=trial_index,
                    particle_index=particle_index,
                )
            particle_swap_counts = copied_swap_counts
            weights.fill(1.0 / float(n_particles))
            resampled[trial_index] = True
            unique_ancestors[trial_index] = int(np.unique(ancestors).size)
            resampling_log.append(
                {
                    "trial_index": int(trial_index),
                    "post_choice_ess": float(post_ess[trial_index]),
                    "unique_ancestors": int(np.unique(ancestors).size),
                    "uniform": uniform,
                    "ancestors": ancestors.tolist(),
                }
            )

    return NewPlanParticleFilterResult(
        marginal_probabilities=marginal,
        marginal_hypothesis_prior=marginal_hypothesis_prior,
        marginal_active_probability=marginal_active_probability,
        pre_choice_ess=pre_ess,
        post_choice_ess=post_ess,
        resampled=resampled,
        resampling_unique_ancestors=unique_ancestors,
        filtered_swap_probability=filtered_swap_probability,
        filtered_swap_event_probability=filtered_swap_event_probability,
        final_weights=weights.copy(),
        particle_swap_counts=particle_swap_counts.copy(),
        resampling_log=resampling_log,
        particle_count=n_particles,
        resample_threshold_fraction=threshold_fraction,
        filter_seed=int(filter_seed),
    )


__all__ = [
    "NewPlanParticleFilterResult",
    "effective_sample_size",
    "run_newplan_particle_filter",
    "systematic_resample",
]

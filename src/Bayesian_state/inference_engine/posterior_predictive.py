"""Conditioned autonomous rollouts for the minimal condition-1 model.

The observed prefix is used only to infer a distribution over the latent
state at a prediction boundary.  Future choices and feedback are then
generated autonomously on the subject's actual physical stimulus/category
schedule.  No observed future choice or feedback enters a rollout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .backends.particle_filter import (
    _choice_probability,
    _future_seed,
    _normalize,
    _particle_config,
    _restore,
    _snapshot,
    effective_sample_size,
    systematic_resample,
)
from ..utils.seeding import stable_seed


@dataclass(frozen=True)
class DynamicRhoConfig:
    """Population specification for a continuous stochastic readout.

    ``rho`` controls how strongly the current hypothesis posterior is
    concentrated before it is mapped to choices.  The population mean is a
    log-linear trend from ``start`` on trial 1 to ``end`` at the shared
    ``trend_reference_trials`` point.  Each particle carries shrinkage-style
    random effects for its start, gain, and volatility, plus an AR(1)
    deviation around that trend.
    """

    start: float
    end: float
    volatility: float
    persistence: float
    start_log_sd: float = 0.35
    gain_log_sd: float = 0.35
    volatility_log_sd: float = 0.50
    trend_reference_trials: int = 128
    min_rho: float = 0.05
    max_rho: float = 20.0


@dataclass
class ConditionedRolloutResult:
    choices: np.ndarray
    feedback: np.ndarray
    probabilities: np.ndarray
    ancestor_indices: np.ndarray
    boundary_weights: np.ndarray
    prefix_pre_choice_ess: np.ndarray
    prefix_post_choice_ess: np.ndarray
    prefix_resampled: np.ndarray
    prefix_choice_probabilities: np.ndarray
    prefix_observed_choice_probability: np.ndarray
    prefix_log_predictive_density: float
    prefix_acquired_probability: np.ndarray
    boundary_acquired: np.ndarray
    generated_acquired: np.ndarray
    prefix_rho_posterior_mean: np.ndarray
    boundary_rho: np.ndarray
    boundary_rho_start: np.ndarray
    boundary_rho_gain: np.ndarray
    boundary_rho_volatility: np.ndarray
    generated_rho: np.ndarray
    split_index: int
    particle_count: int
    rollout_count: int
    filter_seed: int
    rollout_seed: int


def run_conditioned_condition1_rollouts(
    *,
    engine_config: Mapping[str, Any],
    subject_id: int,
    stimulus: Sequence[Sequence[float]] | np.ndarray,
    categories: Sequence[int] | np.ndarray,
    observed_prefix_choices: Sequence[int] | np.ndarray,
    observed_prefix_feedback: Sequence[float] | np.ndarray,
    particle_count: int,
    rollout_count: int,
    rho: float,
    epsilon: float = 0.0,
    epsilon_schedule: Sequence[float] | np.ndarray | None = None,
    learning_update_probability: float = 1.0,
    acquisition_hazard: float | None = None,
    pre_acquisition_lapse: float = 1.0,
    dynamic_rho: DynamicRhoConfig | None = None,
    filter_seed: int = 20260801,
    rollout_seed: int = 20260802,
    resample_threshold_fraction: float = 0.5,
    processed_data_dir: Path | str | None = None,
    dataset_paths: Mapping[str, Path | str] | None = None,
) -> ConditionedRolloutResult:
    """Filter an observed prefix, then autonomously generate the suffix.

    When ``acquisition_hazard`` is supplied, each counterfactual trajectory
    starts in a novice readout regime and can cross one irreversible
    acquisition boundary.  Before that boundary choices use the ordinary
    cognitive readout mixed with ``pre_acquisition_lapse`` uninformed
    responding; after it choices use the ordinary cognitive readout.
    Evidence accumulation continues in both regimes, so this is a one-time
    access/readout change-point rather than a recurrent strategy-state
    controller.

    When ``dynamic_rho`` is supplied, the static ``rho`` is replaced by a
    positive trialwise concentration process.  Its population trend is
    monotone, while its persistent stochastic deviation can rise and fall.
    The change-point and dynamic-rho mechanisms are intentionally mutually
    exclusive in this implementation.
    """

    from ..problems import StateModel

    x = np.asarray(stimulus, dtype=float)
    y = np.asarray(categories, dtype=int).reshape(-1)
    prefix_choices = np.asarray(observed_prefix_choices, dtype=int).reshape(-1)
    prefix_feedback = np.asarray(
        observed_prefix_feedback, dtype=float
    ).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("stimulus must be 2-D and aligned with categories.")
    if not np.all(np.isin(y, [1, 2])):
        raise ValueError("Condition-1 categories must be encoded as 1 or 2.")
    if prefix_choices.shape[0] != prefix_feedback.shape[0]:
        raise ValueError("Observed prefix choices and feedback must align.")
    split_index = int(prefix_choices.shape[0])
    if split_index <= 0 or split_index >= x.shape[0]:
        raise ValueError("Observed prefix must be non-empty and shorter than stimulus.")
    if not np.all(np.isin(prefix_choices, [1, 2])):
        raise ValueError("Observed prefix choices must be encoded as 1 or 2.")
    if (
        not np.all(np.isfinite(prefix_feedback))
        or np.any((prefix_feedback < 0.0) | (prefix_feedback > 1.0))
    ):
        raise ValueError("Observed prefix feedback must lie in [0, 1].")
    expected_feedback = (
        prefix_choices == y[:split_index]
    ).astype(float)
    if not np.array_equal(prefix_feedback, expected_feedback):
        raise ValueError(
            "Observed prefix feedback does not match choices and categories."
        )

    n_particles = int(particle_count)
    n_rollouts = int(rollout_count)
    if n_particles < 2 or n_rollouts < 2:
        raise ValueError("particle_count and rollout_count must be at least 2.")
    threshold_fraction = float(resample_threshold_fraction)
    update_probability = float(learning_update_probability)
    hazard_value = (
        None if acquisition_hazard is None else float(acquisition_hazard)
    )
    if hazard_value is not None and dynamic_rho is not None:
        raise ValueError(
            "acquisition_hazard and dynamic_rho are mutually exclusive."
        )
    novice_lapse = float(pre_acquisition_lapse)
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
    if (
        hazard_value is not None
        and (
            not np.isfinite(hazard_value)
            or not 0.0 <= hazard_value <= 1.0
        )
    ):
        raise ValueError("acquisition_hazard must lie in [0, 1].")
    if (
        not np.isfinite(novice_lapse)
        or not 0.0 <= novice_lapse <= 1.0
    ):
        raise ValueError("pre_acquisition_lapse must lie in [0, 1].")
    rho_value = float(rho)
    epsilon_value = float(epsilon)
    if not np.isfinite(rho_value) or rho_value <= 0.0:
        raise ValueError("rho must be finite and positive.")
    if not np.isfinite(epsilon_value) or not 0.0 <= epsilon_value <= 1.0:
        raise ValueError("epsilon must lie in [0, 1].")
    if epsilon_schedule is None:
        epsilon_by_trial = np.full(x.shape[0], epsilon_value, dtype=float)
    else:
        epsilon_by_trial = np.asarray(
            epsilon_schedule, dtype=float
        ).reshape(-1)
        if epsilon_by_trial.size != x.shape[0]:
            raise ValueError(
                "epsilon_schedule length must match the stimulus schedule."
            )
        if (
            not np.all(np.isfinite(epsilon_by_trial))
            or np.any(epsilon_by_trial < 0.0)
            or np.any(epsilon_by_trial > 1.0)
        ):
            raise ValueError("epsilon_schedule values must lie in [0, 1].")
    if dynamic_rho is not None:
        rho_start = float(dynamic_rho.start)
        rho_end = float(dynamic_rho.end)
        rho_volatility = float(dynamic_rho.volatility)
        rho_persistence = float(dynamic_rho.persistence)
        rho_start_log_sd = float(dynamic_rho.start_log_sd)
        rho_gain_log_sd = float(dynamic_rho.gain_log_sd)
        rho_volatility_log_sd = float(
            dynamic_rho.volatility_log_sd
        )
        rho_reference_trials = int(
            dynamic_rho.trend_reference_trials
        )
        rho_min = float(dynamic_rho.min_rho)
        rho_max = float(dynamic_rho.max_rho)
        dynamic_values = np.asarray(
            [
                rho_start,
                rho_end,
                rho_volatility,
                rho_persistence,
                rho_start_log_sd,
                rho_gain_log_sd,
                rho_volatility_log_sd,
                rho_min,
                rho_max,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(dynamic_values)):
            raise ValueError("dynamic_rho values must be finite.")
        if rho_start <= 0.0 or rho_end < rho_start:
            raise ValueError(
                "dynamic_rho requires 0 < start <= end."
            )
        if rho_volatility < 0.0:
            raise ValueError(
                "dynamic_rho volatility must be non-negative."
            )
        if not 0.0 <= rho_persistence < 1.0:
            raise ValueError(
                "dynamic_rho persistence must lie in [0, 1)."
            )
        if min(
            rho_start_log_sd,
            rho_gain_log_sd,
            rho_volatility_log_sd,
        ) < 0.0:
            raise ValueError(
                "dynamic_rho random-effect scales must be non-negative."
            )
        if rho_min <= 0.0 or rho_max <= rho_min:
            raise ValueError(
                "dynamic_rho bounds must satisfy 0 < min_rho < max_rho."
            )
        if rho_reference_trials < 2:
            raise ValueError(
                "dynamic_rho trend_reference_trials must be at least 2."
            )
    else:
        rho_start = rho_value
        rho_end = rho_value
        rho_volatility = 0.0
        rho_persistence = 0.0
        rho_start_log_sd = 0.0
        rho_gain_log_sd = 0.0
        rho_volatility_log_sd = 0.0
        rho_reference_trials = 2
        rho_min = rho_value
        rho_max = rho_value

    models: list[Any] = []
    shared_partition = None
    shared_space = None
    for particle_index in range(n_particles):
        config = _particle_config(
            engine_config,
            int(filter_seed),
            particle_index,
        )
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

    weights = np.full(n_particles, 1.0 / n_particles, dtype=float)
    acquired = np.full(
        n_particles,
        hazard_value is None,
        dtype=bool,
    )
    pre_ess = np.zeros(split_index, dtype=float)
    post_ess = np.zeros(split_index, dtype=float)
    resampled = np.zeros(split_index, dtype=bool)
    prefix_acquired_probability = np.zeros(split_index, dtype=float)
    prefix_rho_posterior_mean = np.zeros(split_index, dtype=float)
    prefix_choice_probabilities = np.zeros((split_index, 2), dtype=float)
    prefix_observed_choice_probability = np.zeros(split_index, dtype=float)

    population_log_gain = float(
        np.log(rho_end) - np.log(rho_start)
    )
    particle_rho_start = np.full(
        n_particles,
        np.log(rho_start),
        dtype=float,
    )
    particle_rho_gain = np.full(
        n_particles,
        population_log_gain,
        dtype=float,
    )
    particle_rho_volatility = np.full(
        n_particles,
        rho_volatility,
        dtype=float,
    )
    particle_rho_residual = np.zeros(n_particles, dtype=float)
    if dynamic_rho is not None:
        for particle_index in range(n_particles):
            trait_rng = np.random.default_rng(
                _future_seed(
                    int(filter_seed),
                    -1,
                    particle_index,
                    "active_set_ppc_dynamic_rho_traits",
                )
            )
            particle_rho_start[particle_index] += (
                rho_start_log_sd * trait_rng.normal()
            )
            if population_log_gain > 0.0:
                particle_rho_gain[particle_index] *= np.exp(
                    rho_gain_log_sd * trait_rng.normal()
                )
            if rho_volatility > 0.0:
                particle_rho_volatility[particle_index] *= np.exp(
                    rho_volatility_log_sd * trait_rng.normal()
                )
    current_rho = np.full(n_particles, rho_value, dtype=float)
    trend_denominator = int(rho_reference_trials - 1)

    for trial_index in range(split_index):
        if hazard_value is not None:
            for particle_index in np.flatnonzero(~acquired):
                acquisition_seed = _future_seed(
                    int(filter_seed),
                    trial_index,
                    int(particle_index),
                    "active_set_ppc_acquisition_change_point",
                )
                acquired[int(particle_index)] = bool(
                    np.random.default_rng(acquisition_seed).random()
                    < hazard_value
                )
        if dynamic_rho is not None:
            tau = float(trial_index) / float(trend_denominator)
            for particle_index in range(n_particles):
                innovation_rng = np.random.default_rng(
                    _future_seed(
                        int(filter_seed),
                        trial_index,
                        particle_index,
                        "active_set_ppc_dynamic_rho_prefix_innovation",
                    )
                )
                particle_rho_residual[particle_index] = (
                    rho_persistence
                    * particle_rho_residual[particle_index]
                    + particle_rho_volatility[particle_index]
                    * innovation_rng.normal()
                )
            current_rho = np.exp(
                np.clip(
                    particle_rho_start
                    + particle_rho_gain * tau
                    + particle_rho_residual,
                    np.log(rho_min),
                    np.log(rho_max),
                )
            )
        predictions = np.zeros((n_particles, 2), dtype=float)
        for particle_index, model in enumerate(models):
            engine = model.engine
            if engine.posterior is not None:
                engine.prior = np.asarray(engine.posterior, dtype=float).copy()
            engine.observation = (
                x[trial_index].copy(),
                int(prefix_choices[trial_index]),
                float(prefix_feedback[trial_index]),
            )
            engine.modules["perception_mod"].process()
            perceived = np.asarray(engine.observation[0], dtype=float).copy()
            engine.modules["hypo_transitions_mod"].process()
            if acquired[particle_index]:
                predictions[particle_index] = _choice_probability(
                    model,
                    perceived,
                    rho=float(current_rho[particle_index]),
                    epsilon=float(epsilon_by_trial[trial_index]),
                )
            else:
                predictions[particle_index] = _choice_probability(
                    model,
                    perceived,
                    rho=float(current_rho[particle_index]),
                    epsilon=novice_lapse,
                )

        pre_ess[trial_index] = effective_sample_size(weights)
        observed_index = int(prefix_choices[trial_index]) - 1
        marginal_prediction = np.sum(
            weights[:, np.newaxis] * predictions,
            axis=0,
        )
        marginal_prediction = _normalize(marginal_prediction)
        prefix_choice_probabilities[trial_index] = marginal_prediction
        prefix_observed_choice_probability[trial_index] = float(
            np.clip(marginal_prediction[observed_index], 1e-12, 1.0)
        )
        weights *= np.clip(
            predictions[:, observed_index],
            1e-12,
            1.0,
        )
        weight_total = float(np.sum(weights))
        if not np.isfinite(weight_total) or weight_total <= 0.0:
            weights.fill(1.0 / n_particles)
        else:
            weights /= weight_total
        post_ess[trial_index] = effective_sample_size(weights)
        prefix_acquired_probability[trial_index] = float(
            np.sum(weights * acquired)
        )
        prefix_rho_posterior_mean[trial_index] = float(
            np.sum(weights * current_rho)
        )

        for particle_index, model in enumerate(models):
            engine = model.engine
            update_seed = _future_seed(
                int(filter_seed),
                trial_index,
                particle_index,
                "active_set_ppc_learning_update_gate",
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

        if post_ess[trial_index] < threshold_fraction * n_particles:
            seed = _future_seed(
                int(filter_seed),
                trial_index,
                0,
                "active_set_ppc_systematic_resampling",
            )
            uniform = float(np.random.default_rng(seed).random())
            ancestors = systematic_resample(weights, uniform)
            snapshots = [_snapshot(models[int(index)]) for index in ancestors]
            acquired = acquired[ancestors].copy()
            current_rho = current_rho[ancestors].copy()
            particle_rho_start = particle_rho_start[ancestors].copy()
            particle_rho_gain = particle_rho_gain[ancestors].copy()
            particle_rho_volatility = (
                particle_rho_volatility[ancestors].copy()
            )
            particle_rho_residual = (
                particle_rho_residual[ancestors].copy()
            )
            for particle_index, snapshot in enumerate(snapshots):
                _restore(
                    models[particle_index],
                    snapshot,
                    filter_seed=int(filter_seed),
                    trial_index=trial_index,
                    particle_index=particle_index,
                )
            weights.fill(1.0 / n_particles)
            resampled[trial_index] = True

    boundary_snapshots = [_snapshot(model) for model in models]
    boundary_acquired = acquired.copy()
    boundary_rho = current_rho.copy()
    boundary_rho_start = particle_rho_start.copy()
    boundary_rho_gain = particle_rho_gain.copy()
    boundary_rho_volatility = particle_rho_volatility.copy()
    boundary_rho_residual = particle_rho_residual.copy()
    ancestor_rng = np.random.default_rng(
        stable_seed(
            {
                "seed_role": "active_set_ppc_boundary_ancestor",
                "rollout_seed": int(rollout_seed),
                "subject_id": int(subject_id),
                "split_index": split_index,
            }
        )
    )
    ancestor_indices = ancestor_rng.choice(
        n_particles,
        size=n_rollouts,
        replace=True,
        p=weights,
    ).astype(int)

    scratch_config = _particle_config(
        engine_config,
        int(rollout_seed),
        0,
    )
    scratch = StateModel(
        scratch_config,
        condition=1,
        subject_id=int(subject_id),
        processed_data_dir=processed_data_dir,
        dataset_paths=dataset_paths,
        partition=shared_partition,
        space=shared_space,
    )
    suffix_length = int(x.shape[0] - split_index)
    generated_choices = np.zeros(
        (n_rollouts, suffix_length),
        dtype=np.int8,
    )
    generated_feedback = np.zeros(
        (n_rollouts, suffix_length),
        dtype=np.int8,
    )
    generated_probabilities = np.zeros(
        (n_rollouts, suffix_length, 2),
        dtype=np.float32,
    )
    generated_acquired = np.zeros(
        (n_rollouts, suffix_length),
        dtype=bool,
    )
    generated_rho = np.zeros(
        (n_rollouts, suffix_length),
        dtype=np.float32,
    )

    for rollout_index, ancestor in enumerate(ancestor_indices):
        trajectory_seed = stable_seed(
            {
                "seed_role": "active_set_ppc_rollout",
                "rollout_seed": int(rollout_seed),
                "subject_id": int(subject_id),
                "rollout_index": int(rollout_index),
                "ancestor": int(ancestor),
            }
        )
        _restore(
            scratch,
            boundary_snapshots[int(ancestor)],
            filter_seed=int(trajectory_seed),
            trial_index=split_index - 1,
            particle_index=int(rollout_index),
        )
        choice_rng = np.random.default_rng(
            stable_seed(
                {
                    "seed_role": "active_set_ppc_choice",
                    "trajectory_seed": int(trajectory_seed),
                }
            )
        )
        acquisition_rng = np.random.default_rng(
            stable_seed(
                {
                    "seed_role": "active_set_ppc_rollout_acquisition",
                    "trajectory_seed": int(trajectory_seed),
                }
            )
        )
        rollout_acquired = bool(boundary_acquired[int(ancestor)])
        rollout_rho_start = float(boundary_rho_start[int(ancestor)])
        rollout_rho_gain = float(boundary_rho_gain[int(ancestor)])
        rollout_rho_volatility = float(
            boundary_rho_volatility[int(ancestor)]
        )
        rollout_rho_residual = float(
            boundary_rho_residual[int(ancestor)]
        )
        rho_rng = np.random.default_rng(
            stable_seed(
                {
                    "seed_role": "active_set_ppc_rollout_dynamic_rho",
                    "trajectory_seed": int(trajectory_seed),
                }
            )
        )
        for local_index, trial_index in enumerate(
            range(split_index, x.shape[0])
        ):
            if (
                not rollout_acquired
                and hazard_value is not None
                and acquisition_rng.random() < hazard_value
            ):
                rollout_acquired = True
            if dynamic_rho is None:
                trial_rho = rho_value
            else:
                rollout_rho_residual = (
                    rho_persistence * rollout_rho_residual
                    + rollout_rho_volatility * rho_rng.normal()
                )
                tau = float(trial_index) / float(trend_denominator)
                trial_rho = float(
                    np.exp(
                        np.clip(
                            rollout_rho_start
                            + rollout_rho_gain * tau
                            + rollout_rho_residual,
                            np.log(rho_min),
                            np.log(rho_max),
                        )
                    )
                )
            engine = scratch.engine
            if engine.posterior is not None:
                engine.prior = np.asarray(engine.posterior, dtype=float).copy()
            engine.observation = (x[trial_index].copy(), 1, 1.0)
            engine.modules["perception_mod"].process()
            perceived = np.asarray(engine.observation[0], dtype=float).copy()
            engine.modules["hypo_transitions_mod"].process()
            if rollout_acquired:
                probability = _choice_probability(
                    scratch,
                    perceived,
                    rho=trial_rho,
                    epsilon=float(epsilon_by_trial[trial_index]),
                )
            else:
                probability = _choice_probability(
                    scratch,
                    perceived,
                    rho=trial_rho,
                    epsilon=novice_lapse,
                )
            choice = int(choice_rng.choice(2, p=probability)) + 1
            outcome = int(choice == int(y[trial_index]))
            engine.observation = (perceived, choice, float(outcome))
            engine.modules[
                "hypo_transitions_mod"
            ].record_outcome_feedback(float(outcome))
            update_seed = _future_seed(
                int(trajectory_seed),
                trial_index,
                rollout_index,
                "active_set_ppc_rollout_learning_update_gate",
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

            generated_choices[rollout_index, local_index] = choice
            generated_feedback[rollout_index, local_index] = outcome
            generated_probabilities[
                rollout_index, local_index
            ] = probability.astype(np.float32)
            generated_acquired[rollout_index, local_index] = rollout_acquired
            generated_rho[rollout_index, local_index] = trial_rho

    return ConditionedRolloutResult(
        choices=generated_choices,
        feedback=generated_feedback,
        probabilities=generated_probabilities,
        ancestor_indices=ancestor_indices,
        boundary_weights=weights.copy(),
        prefix_pre_choice_ess=pre_ess,
        prefix_post_choice_ess=post_ess,
        prefix_resampled=resampled,
        prefix_choice_probabilities=prefix_choice_probabilities,
        prefix_observed_choice_probability=(
            prefix_observed_choice_probability
        ),
        prefix_log_predictive_density=float(
            np.sum(np.log(prefix_observed_choice_probability))
        ),
        prefix_acquired_probability=prefix_acquired_probability,
        boundary_acquired=boundary_acquired,
        generated_acquired=generated_acquired,
        prefix_rho_posterior_mean=prefix_rho_posterior_mean,
        boundary_rho=boundary_rho,
        boundary_rho_start=boundary_rho_start,
        boundary_rho_gain=boundary_rho_gain,
        boundary_rho_volatility=boundary_rho_volatility,
        generated_rho=generated_rho,
        split_index=split_index,
        particle_count=n_particles,
        rollout_count=n_rollouts,
        filter_seed=int(filter_seed),
        rollout_seed=int(rollout_seed),
    )


__all__ = [
    "ConditionedRolloutResult",
    "DynamicRhoConfig",
    "run_conditioned_condition1_rollouts",
]

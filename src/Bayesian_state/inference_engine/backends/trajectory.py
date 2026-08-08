"""Single-trajectory inference backend for ``StateModel``."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..results import TrajectoryInferenceResult
from ...utils.seeding import inject_module_seed_from_trajectory


def run_state_model_trajectory(
    *,
    engine_config: Mapping[str, Any],
    subject_id: int,
    condition: int,
    stimulus: Sequence[Sequence[float]] | np.ndarray,
    choices: Sequence[int] | np.ndarray,
    feedback: Sequence[float] | np.ndarray,
    trajectory_seed: int | None = None,
    processed_data_dir: Path | str | None = None,
    dataset_paths: Mapping[str, Path | str] | None = None,
) -> TrajectoryInferenceResult:
    """Run one realized latent path through the configured model.

    This backend conditions on one realization of every stochastic module. It
    does not marginalize those latent paths; use the particle backend for that.
    """
    from ...problems import StateModel

    x = np.asarray(stimulus, dtype=float)
    y = np.asarray(choices, dtype=int).reshape(-1)
    outcome = np.asarray(feedback, dtype=float).reshape(-1)
    if x.ndim != 2:
        raise ValueError("stimulus must be a 2-D array.")
    if x.shape[0] != y.size or y.size != outcome.size:
        raise ValueError("stimulus, choices, and feedback must have equal trial counts.")

    resolved_config = deepcopy(dict(engine_config))
    module_seed = inject_module_seed_from_trajectory(
        resolved_config,
        trajectory_seed,
    )
    if trajectory_seed is not None:
        # Preserve reproducibility for legacy modules that still use np.random.
        np.random.seed(int(trajectory_seed))

    model = StateModel(
        resolved_config,
        condition=int(condition),
        subject_id=int(subject_id),
        processed_data_dir=processed_data_dir,
        dataset_paths=dataset_paths,
    )
    trial_sequence = [
        [trial_stimulus, int(choice), float(trial_feedback)]
        for trial_stimulus, choice, trial_feedback in zip(x, y, outcome)
    ]
    posterior_log, prior_log = model.fit_step_by_step(trial_sequence)
    step_log = getattr(model, "step_log", None)
    if step_log is None:
        raise ValueError("StateModel.step_log is missing after fit_step_by_step")

    modules = getattr(model.engine, "modules", {})
    transition = modules.get("hypo_transitions_mod")
    beta_module = modules.get("beta_mod")
    transition_counts = (
        getattr(transition, "strategy_counts_log")
        if transition is not None and hasattr(transition, "strategy_counts_log")
        else None
    )
    latent_volatility_log = (
        getattr(transition, "latent_volatility_log")
        if transition is not None and hasattr(transition, "latent_volatility_log")
        else None
    )
    beta_log = (
        getattr(beta_module, "beta_log")
        if beta_module is not None and hasattr(beta_module, "beta_log")
        else None
    )
    return TrajectoryInferenceResult(
        model=model,
        posterior_log=posterior_log,
        prior_log=prior_log,
        step_log=step_log,
        beta_log=beta_log,
        transition_counts=transition_counts,
        latent_volatility_log=latent_volatility_log,
        module_seed=module_seed,
    )


__all__ = ["run_state_model_trajectory"]

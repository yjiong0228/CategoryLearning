"""Bounded parameter estimation using a specified training prefix."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
from scipy.optimize import differential_evolution

from .model import Parameters, SUSTAIN
from .sequence import run_sequence


def fit_parameters(
    stimuli: np.ndarray,
    targets: np.ndarray,
    choices: np.ndarray,
    dimensions: tuple[int, ...],
    n_categories: int,
    *,
    base: Parameters,
    bounds: dict[str, list[float]],
    train_trials: int,
    maxiter: int,
    popsize: int,
    seed: int,
    model_options: dict | None = None,
) -> tuple[Parameters, dict]:
    """Fit r/beta/d/eta; retain initial parameters if search finds a worse point.

    Evaluation is deterministic with first-cluster tie breaking. One process is
    used. DE completion indicates optimization status, not parameter recovery.
    """
    options = dict(model_options or {})
    if options.get("ties", "first") != "first":
        raise ValueError("Likelihood fitting requires ties: first; random tie paths are not marginalized")
    if (isinstance(train_trials, bool) or int(train_trials) != train_trials
            or not 1 <= train_trials <= len(stimuli)):
        raise ValueError("train_trials must select a nonempty available prefix")
    if any(isinstance(v, bool) or int(v) != v for v in (maxiter, popsize)) or maxiter < 0 or popsize < 1:
        raise ValueError("maxiter must be nonnegative and popsize positive integers")
    train_trials = int(train_trials)
    names = ("r", "beta", "d", "eta")
    if set(bounds) != set(names):
        raise ValueError("bounds must specify exactly r, beta, d and eta")
    intervals = np.asarray([bounds[name] for name in names], dtype=float)
    if (intervals.shape != (4, 2) or not np.isfinite(intervals).all()
            or np.any(intervals[:, 0] < 0) or np.any(intervals[:, 1] <= intervals[:, 0])
            or intervals[3, 1] > 1):
        raise ValueError("Invalid parameter bounds; eta must stay in [0, 1]")
    initial = np.array([getattr(base, name) for name in names])
    if np.any(initial < intervals[:, 0]) or np.any(initial > intervals[:, 1]):
        raise ValueError("Initial parameters must lie inside the optimization bounds")

    def parameters(values: np.ndarray) -> Parameters:
        return Parameters(**{**asdict(base), **dict(zip(names, map(float, values)))})

    def objective(values: np.ndarray) -> float:
        model = SUSTAIN(dimensions, n_categories, parameters(values), **options)
        result = run_sequence(model, stimuli[:train_trials], targets[:train_trials], choices=choices[:train_trials])
        assert result.negative_log_likelihood is not None
        return result.negative_log_likelihood

    # Validate the complete input shape/values once before invoking the optimizer.
    run_sequence(SUSTAIN(dimensions, n_categories, base, **options), stimuli, targets, choices=choices)
    initial_nll = objective(initial)
    result = differential_evolution(
        objective, intervals, seed=seed, maxiter=int(maxiter), popsize=int(popsize),
        polish=False, workers=1, updating="immediate", x0=initial,
    )
    retain_initial = initial_nll < result.fun
    best = initial if retain_initial else result.x
    return parameters(best), dict(
        objective="choice_negative_log_likelihood_before_feedback",
        train_trials=int(train_trials), initial_nll=initial_nll,
        best_nll=float(min(initial_nll, result.fun)), retained_initial=bool(retain_initial),
        optimizer="scipy.optimize.differential_evolution", success=bool(result.success),
        message=str(result.message), evaluations=int(result.nfev), iterations=int(result.nit),
        bounds=bounds, maxiter=int(maxiter), popsize=int(popsize), seed=int(seed),
        workers=1, polish=False,
    )

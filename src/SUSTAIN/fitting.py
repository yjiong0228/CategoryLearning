"""Bounded parameter estimation using a specified training prefix."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import multiprocessing
import os
import time
from typing import Callable

import numpy as np
from scipy.optimize import differential_evolution
from threadpoolctl import threadpool_limits

from .model import Parameters, SUSTAIN
from .sequence import run_sequence


@dataclass
class _ChoiceObjective:
    """Picklable objective shared by serial and process-parallel searches."""

    stimuli: np.ndarray
    targets: np.ndarray
    choices: np.ndarray
    dimensions: tuple[int, ...]
    n_categories: int
    base: Parameters
    options: dict

    def parameters(self, values: np.ndarray) -> Parameters:
        return Parameters(**{**asdict(self.base), **dict(zip(("r", "beta", "d", "eta"), map(float, values)))})

    def __call__(self, values: np.ndarray) -> float:
        model = SUSTAIN(self.dimensions, self.n_categories, self.parameters(values), **self.options)
        result = run_sequence(model, self.stimuli, self.targets, choices=self.choices)
        assert result.negative_log_likelihood is not None
        return result.negative_log_likelihood


def _limit_numeric_threads() -> None:
    """Each process evaluates one history using one numeric thread."""
    threadpool_limits(limits=1)


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
    workers: int = 1,
    restart_seeds: list[int] | None = None,
    tol: float = 0.01,
    atol: float = 0.0,
    updating: str | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> tuple[Parameters, dict]:
    """Fit r/beta/d/eta; retain initial parameters if search finds a worse point.

    Evaluation is deterministic with first-cluster tie breaking. Independent
    restarts use the same initial parameter vector and are run sequentially;
    only population evaluation is parallel. The default retains the original
    one-process, one-restart behavior. Convergence is an optimization diagnostic.
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
    if isinstance(workers, bool) or int(workers) != workers or workers < 1:
        raise ValueError("workers must be a positive integer")
    if not np.isfinite([tol, atol]).all() or min(tol, atol) < 0:
        raise ValueError("tol and atol must be finite and nonnegative")
    seeds = [seed] if restart_seeds is None else list(restart_seeds)
    if not seeds or any(isinstance(s, bool) or int(s) != s or not 0 <= s < 2**32 for s in seeds):
        raise ValueError("restart_seeds must contain valid nonnegative integer seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError("restart_seeds must be distinct")
    seeds = list(map(int, seeds))
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    actual_workers = min(int(workers), available, max(5, int(popsize) * 4))
    update_mode = updating or ("immediate" if actual_workers == 1 else "deferred")
    if update_mode not in ("immediate", "deferred") or (actual_workers > 1 and update_mode != "deferred"):
        raise ValueError("Parallel search requires updating: deferred")
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

    # Validate the complete input shape/values once before invoking the optimizer.
    run_sequence(SUSTAIN(dimensions, n_categories, base, **options), stimuli, targets, choices=choices)
    objective = _ChoiceObjective(stimuli[:train_trials], targets[:train_trials], choices[:train_trials],
                                 dimensions, n_categories, base, options)
    initial_nll = objective(initial)
    results = []
    started = time.perf_counter()

    def emit(event: dict) -> None:
        if progress_callback is not None:
            progress_callback({**event, "elapsed_seconds": time.perf_counter() - started})

    pool_context = (ProcessPoolExecutor(max_workers=actual_workers,
                    mp_context=multiprocessing.get_context("spawn"), initializer=_limit_numeric_threads)
                    if actual_workers > 1 else nullcontext(None))
    with threadpool_limits(limits=1), pool_context as pool:
        for restart, restart_seed in enumerate(seeds, 1):
            emit(dict(event="restart_started", restart=restart, seed=restart_seed, initial_nll=initial_nll,
                      workers=actual_workers, population_size=max(5, int(popsize) * 4)))

            def on_generation(intermediate_result):
                current = intermediate_result
                emit(dict(event="generation", restart=restart, seed=restart_seed,
                          generation=int(current.nit), evaluations=int(current.nfev),
                          best_nll=float(current.fun), parameters=asdict(objective.parameters(current.x)),
                          population_nll_std=float(np.std(current.population_energies))))

            result = differential_evolution(
                objective, intervals, seed=restart_seed, maxiter=int(maxiter), popsize=int(popsize),
                polish=False, workers=1 if pool is None else pool.map, updating=update_mode,
                x0=initial, tol=tol, atol=atol, callback=on_generation,
            )
            retain_initial = initial_nll < result.fun
            values = initial if retain_initial else result.x
            entry = dict(restart=restart, seed=restart_seed, parameters=asdict(objective.parameters(values)),
                         best_nll=float(min(initial_nll, result.fun)), retained_initial=bool(retain_initial),
                         success=bool(result.success), message=str(result.message),
                         evaluations=int(result.nfev), iterations=int(result.nit),
                         population_nll_std=float(np.std(result.population_energies)),
                         final_population=result.population.tolist(),
                         final_population_nll=result.population_energies.tolist())
            results.append(entry)
            emit({"event": "restart_completed", **{k: v for k, v in entry.items() if not k.startswith("final_")}})
    best = min(results, key=lambda item: item["best_nll"])
    scores = sorted(entry["best_nll"] for entry in results)
    return Parameters(**best["parameters"]), dict(
        objective="choice_negative_log_likelihood_before_feedback",
        train_trials=int(train_trials), initial_nll=initial_nll,
        best_nll=best["best_nll"], retained_initial=best["retained_initial"],
        optimizer="scipy.optimize.differential_evolution", success=best["success"],
        message=best["message"], evaluations=sum(entry["evaluations"] for entry in results),
        iterations=sum(entry["iterations"] for entry in results),
        bounds=bounds, maxiter=int(maxiter), popsize=int(popsize), seed=int(seed),
        workers=actual_workers, requested_workers=int(workers), numeric_threads_per_worker=1,
        updating=update_mode, tol=float(tol), atol=float(atol), polish=False,
        selected_restart=best["restart"], restart_seeds=seeds, restarts=results,
        best_two_nll_gap=None if len(scores) < 2 else scores[1] - scores[0],
    )

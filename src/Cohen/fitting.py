"""Fitting for the Cohen–Schneidman reimplementation."""
from __future__ import annotations

from typing import Optional, Tuple, Dict, Any, Callable, TYPE_CHECKING
import numpy as np
from dataclasses import dataclass
import math
from .binary import FeatureMixtureStrict
from .features import check_pm1, feature_subsets, normalize, positive_integer, validate_trial_lengths

Array = np.ndarray

if TYPE_CHECKING:
    from .continuous import ContinuousFeatureMixture


@dataclass
class FitConfig:
    population_size: int = 40
    generations: int = 40
    elite_fraction: float = 0.2
    mutation_scale: float = 0.15
    mutation_decay: float = 0.98
    anneal_steps: int = 120
    anneal_initial_temp: float = 1.0
    anneal_final_temp: float = 1e-3
    anneal_step_scale: float = 0.08
    beta_range: Tuple[float, float] = (0.5, 12.0)
    eta_range: Tuple[float, float] = (1e-3, 0.5)
    seed: Optional[int] = None

    def validate(self) -> None:
        positive_integer(self.population_size, "population_size")
        positive_integer(self.generations, "generations")
        if not isinstance(self.anneal_steps, int) or self.anneal_steps < 0:
            raise ValueError("anneal_steps must be a nonnegative integer")
        if not np.isfinite(self.elite_fraction) or not 0 < self.elite_fraction <= 1:
            raise ValueError("elite_fraction must be in (0, 1]")
        for name in ("mutation_scale", "mutation_decay", "anneal_initial_temp",
                     "anneal_final_temp", "anneal_step_scale"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for bounds in (self.beta_range, self.eta_range):
            if len(bounds) != 2 or not np.all(np.isfinite(bounds)) or not 0 < bounds[0] <= bounds[1]:
                raise ValueError("beta_range and eta_range must be positive ordered pairs")


def _pack(ap: Array, an: Array, beta: float, eta: float) -> Array:
    return np.r_[normalize(ap), normalize(an), math.log(beta), math.log(eta)]


def _unpack(theta: Array, n_bits: int, max_order: Optional[int], memory_window: int, update_rule: str = "si_ratio") -> FeatureMixtureStrict:
    d = len(feature_subsets(n_bits, max_order))
    return FeatureMixtureStrict(
        n_bits=n_bits,
        max_order=max_order,
        alpha_pos=normalize(theta[:d]),
        alpha_neg=normalize(theta[d:2*d]),
        beta=float(np.exp(theta[2*d])),
        eta=float(np.exp(theta[2*d+1])),
        gamma=0.0,  # paper Methods: gamma_t=0 = 0
        memory_window=memory_window,
        update_rule=update_rule,
    )


def nll(theta: Array, X: Array, y: Array, r: Array, n_bits: int,
        max_order: int, memory_window: int, fit_slice: slice,
        update_rule: str = "si_ratio") -> float:
    if fit_slice.start not in (None, 0) or fit_slice.step not in (None, 1):
        raise ValueError("fit_slice must be a contiguous prefix to preserve learning history")
    m = _unpack(theta, n_bits, max_order, memory_window, update_rule)
    out = m.run_session(X[fit_slice], y[fit_slice], choices=r[fit_slice], update=True)
    return float(-np.sum(out["log_likelihood"]))


def fit_paper_style(
    X: Array,
    y: Array,
    r: Array,
    n_bits: Optional[int] = None,
    max_order: Optional[int] = None,
    memory_window: int = 1,
    fit_first_n: Optional[int] = None,
    config: Optional[FitConfig] = None,
    verbose: bool = True,
    update_rule: str = "si_ratio",
) -> Tuple[FeatureMixtureStrict, Dict[str, Any]]:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a trial-by-feature matrix")
    y = check_pm1(y, "y").reshape(-1)
    r = check_pm1(r, "r").reshape(-1)
    if n_bits is None:
        n_bits = X.shape[1]
    if max_order is None:
        max_order = n_bits
    if config is None:
        config = FitConfig()

    config.validate()
    FeatureMixtureStrict(n_bits, max_order, memory_window=memory_window,
                         seed=config.seed, update_rule=update_rule).features(X)
    validate_trial_lengths(X, y, r)
    validate_prefix(fit_first_n, len(X))
    rng = np.random.default_rng(config.seed)
    d = len(feature_subsets(n_bits, max_order))
    dim = 2 * d + 2
    fit_slice = slice(None) if fit_first_n is None else slice(0, int(fit_first_n))

    pop = np.zeros((config.population_size, dim))
    for i in range(config.population_size):
        beta = float(np.exp(rng.uniform(np.log(config.beta_range[0]), np.log(config.beta_range[1]))))
        eta = float(np.exp(rng.uniform(np.log(config.eta_range[0]), np.log(config.eta_range[1]))))
        pop[i] = _pack(rng.normal(size=d), rng.normal(size=d), beta, eta)

    best_theta, best = minimize_ga_sa(
        pop, lambda th: nll(th, X, y, r, n_bits, max_order, memory_window, fit_slice, update_rule),
        config, rng, verbose,
    )

    model = _unpack(best_theta, n_bits, max_order, memory_window, update_rule)
    info = {
        "negative_log_likelihood": best,
        "optimizer": "genetic_algorithm_plus_simulated_annealing",
        "gamma0_fixed_to_zero": True,
        "alpha_update": update_rule,
        "beta": model.beta,
        "eta": model.eta,
        "n_features": model.n_features,
        "memory_window": memory_window,
        "fit_first_n": fit_first_n,
        "note": "Independent reimplementation; learning convention is explicit because the paper and SI differ. GA/SA settings are implementation choices.",
    }
    return model, info


def validate_prefix(fit_first_n: Optional[int], n_trials: int) -> None:
    if fit_first_n is not None:
        positive_integer(fit_first_n, "fit_first_n")
        if fit_first_n > n_trials:
            raise ValueError("fit_first_n exceeds available trials")


def minimize_ga_sa(pop: Array, objective: Callable[[Array], float], config: FitConfig,
                   rng: np.random.Generator, verbose: bool = False) -> Tuple[Array, float]:
    """Bounded GA followed by annealing; final two coordinates are log beta/eta."""
    config.validate()
    dim = pop.shape[1]
    elite_n = max(1, int(round(config.population_size * config.elite_fraction)))
    best_theta = pop[0].copy()
    best = float("inf")
    mut = config.mutation_scale

    for gen in range(config.generations):
        scores = np.array([objective(th) for th in pop])
        if not np.all(np.isfinite(scores)):
            raise FloatingPointError("nonfinite optimizer objective")
        order = np.argsort(scores)
        pop, scores = pop[order], scores[order]
        if scores[0] < best:
            best = float(scores[0])
            best_theta = pop[0].copy()
        if verbose and (gen == 0 or (gen + 1) % 10 == 0 or gen + 1 == config.generations):
            print(f"GA {gen+1}/{config.generations}: best NLL={best:.6f}")
        elites = pop[:elite_n].copy()
        new = [e.copy() for e in elites]
        while len(new) < config.population_size:
            p1, p2 = elites[rng.integers(elite_n)], elites[rng.integers(elite_n)]
            mix = rng.uniform(0, 1, size=dim)
            child = mix * p1 + (1 - mix) * p2 + rng.normal(scale=mut, size=dim)
            child[-2] = np.clip(child[-2], np.log(config.beta_range[0]), np.log(config.beta_range[1]))
            child[-1] = np.clip(child[-1], np.log(config.eta_range[0]), np.log(config.eta_range[1]))
            new.append(child)
        pop = np.asarray(new)
        mut *= config.mutation_decay

    theta = best_theta.copy()
    theta_score = best
    for step in range(config.anneal_steps):
        frac = step / max(1, config.anneal_steps - 1)
        temp = config.anneal_initial_temp * (config.anneal_final_temp / config.anneal_initial_temp) ** frac
        prop = theta + rng.normal(scale=config.anneal_step_scale * (1 - 0.8 * frac), size=dim)
        prop[-2] = np.clip(prop[-2], np.log(config.beta_range[0]), np.log(config.beta_range[1]))
        prop[-1] = np.clip(prop[-1], np.log(config.eta_range[0]), np.log(config.eta_range[1]))
        ps = objective(prop)
        if not np.isfinite(ps):
            raise FloatingPointError("nonfinite optimizer objective")
        delta = ps - theta_score
        if delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-12)):
            theta, theta_score = prop, float(ps)
            if theta_score < best:
                best, best_theta = theta_score, theta.copy()
        if verbose and (step + 1) % 100 == 0:
            print(f"SA {step+1}/{config.anneal_steps}: best NLL={best:.6f}")

    return best_theta, best


def fit_continuous(X: Array, choices: Array, feedback: Array, condition: int, *,
                   model_config: dict | None = None, fit_first_n: int | None = None,
                   config: FitConfig | None = None, verbose: bool = False
                   ) -> Tuple[ContinuousFeatureMixture, Dict[str, Any]]:
    """Fit initial category weights, beta and eta using a chronological prefix.

    Initial reference-category biases are fixed to zero. The target categories
    are absent from this interface: learning conditions on choices and feedback.
    """
    from .continuous import ContinuousFeatureMixture
    from .feedback import category_count

    config = FitConfig() if config is None else config
    config.validate()
    X = np.asarray(X, dtype=float)
    choices, feedback = np.asarray(choices), np.asarray(feedback)
    validate_trial_lengths(X, choices, feedback)
    validate_prefix(fit_first_n, len(X))
    count = len(X) if fit_first_n is None else fit_first_n
    settings = dict(model_config or {})
    if any(k in settings for k in ("alpha", "bias", "n_categories")):
        raise ValueError("fitting sets alpha, initial bias and condition-specific category count")
    settings["n_categories"] = category_count(condition)
    prototype = ContinuousFeatureMixture(**settings)
    prototype.run_session(X[:count], choices[:count], feedback[:count], condition, update=False)
    d, k = prototype.n_features, prototype.n_categories
    rng = np.random.default_rng(config.seed)
    pop = []
    for _ in range(config.population_size):
        beta = np.exp(rng.uniform(*np.log(config.beta_range)))
        eta = np.exp(rng.uniform(*np.log(config.eta_range)))
        weights = np.stack([normalize(rng.normal(size=d)) for _ in range(k)])
        pop.append(np.r_[weights.ravel(), np.log(beta), np.log(eta)])

    def unpack(theta):
        model = prototype.copy()
        model.alpha = np.stack([normalize(row) for row in theta[:-2].reshape(k, d)])
        model.beta, model.eta = np.exp(theta[-2:])
        model.bias[:] = 0.0
        return model

    def objective(theta):
        out = unpack(theta).run_session(X[:count], choices[:count], feedback[:count], condition)
        return float(-out["log_likelihood"].sum())

    theta, score = minimize_ga_sa(np.asarray(pop), objective, config, rng, verbose)
    return unpack(theta), {"negative_log_likelihood": score, "n_fit_trials": count,
                           "optimizer": "genetic_algorithm_plus_simulated_annealing",
                           "update_rule": prototype.update_rule,
                           "initial_bias_fixed_to_zero": True}

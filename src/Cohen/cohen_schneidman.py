#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cohen_schneidman.py

A formula-faithful implementation of Cohen & Schneidman (2013) feature-mixture model.

Public-paper aligned parts:
- x_i in {-1,+1}
- f_J(x)=prod_{j in J} x_j
- Eq. 1: P(x|y=c)=exp(beta * alpha_c·f(x))/Z_c
- Eq. 2: P(y=1|x)=1/[1+(Z_1/Z_-1)*exp(beta*(alpha_-1-alpha_1)·f(x)+gamma)]
- Eq. 3 and SI S1-S3: gradient ascent update, followed by ||alpha_c||=1
- Main model uses memory window |W|=1
- gamma at t=0 is fixed to 0 during fitting
- Fitting objective is Eq. 5 likelihood of observed subject choices
- Optimizer is genetic algorithm + simulated annealing, matching the optimizer family described in Methods

Limit:
The paper does not publish exact GA/SA population sizes, mutation rates, or temperature schedule.
So this is formula-faithful and method-family-faithful, but not an unpublished official code clone.

CSV usage example:
    python src/Cohen/cohen_schneidman.py --csv data_exp5/processed/Rule1_processed.csv --outdir results/Cohen_model/Exp5 --fit

Faster test:
    python src/Cohen/cohen_schneidman.py --csv data_exp5/processed/Rule1_processed.csv --outdir results/Cohen_model/Exp5 --fit --population-size 8 --generations 2 --anneal-steps 2

Paper-style future prediction:
    python src/Cohen/cohen_schneidman.py --csv data_exp5/processed/Rule1_processed.csv --outdir results/Cohen_model/Exp5 --fit --fit-first-n 64
"""

from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Optional, Sequence, Tuple, List, Dict, Any
import argparse, json, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


Array = np.ndarray
DEFAULT_EXP5_CSV = Path("data_exp5/processed/Rule1_processed.csv")


# -----------------------------
# Feature basis
# -----------------------------

def all_binary_patterns(n_bits: int) -> Array:
    return np.asarray(list(product([-1.0, 1.0], repeat=n_bits)), dtype=float)


def feature_subsets(n_bits: int, max_order: Optional[int] = None) -> List[Tuple[int, ...]]:
    if max_order is None:
        max_order = n_bits
    if not (1 <= max_order <= n_bits):
        raise ValueError("max_order must be in [1, n_bits]")
    out: List[Tuple[int, ...]] = []
    for k in range(1, max_order + 1):
        out.extend(combinations(range(n_bits), k))
    return out


def transform_features(X: Array, subsets: Sequence[Tuple[int, ...]]) -> Array:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    return np.column_stack([np.prod(X[:, s], axis=1) for s in subsets])


def check_pm1(x: Array, name: str) -> Array:
    x = np.asarray(x, dtype=float)
    if not np.all(np.isin(x, [-1.0, 1.0])):
        raise ValueError(f"{name} must contain only -1 and +1")
    return x


def normalize(v: Array, eps: float = 1e-12) -> Array:
    v = np.asarray(v, dtype=float).copy()
    n = float(np.linalg.norm(v))
    if n < eps:
        v[:] = 0.0
        v[0] = 1.0
        return v
    return v / n


def logsumexp(a: Array) -> float:
    a = np.asarray(a, dtype=float)
    m = float(np.max(a))
    return m + math.log(float(np.sum(np.exp(a - m))))


def sigmoid_neg_logodds(z: Array) -> Array:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = np.exp(-z[pos]) / (1.0 + np.exp(-z[pos]))
    out[~pos] = 1.0 / (1.0 + np.exp(z[~pos]))
    return out


def feature_name(subset: Tuple[int, ...]) -> str:
    return "x" + "".join(str(i + 1) for i in subset)


# -----------------------------
# Model
# -----------------------------

@dataclass
class FeatureMixtureStrict:
    n_bits: int
    max_order: Optional[int] = None
    beta: float = 5.0
    eta: float = 0.05
    alpha_pos: Optional[Array] = None
    alpha_neg: Optional[Array] = None
    gamma: float = 0.0
    memory_window: int = 1
    seed: Optional[int] = None

    def __post_init__(self):
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        if self.eta < 0:
            raise ValueError("eta must be non-negative")
        self.subsets = feature_subsets(self.n_bits, self.max_order)
        self.n_features = len(self.subsets)
        self.patterns = all_binary_patterns(self.n_bits)
        self.F_all = transform_features(self.patterns, self.subsets)
        rng = np.random.default_rng(self.seed)
        if self.alpha_pos is None:
            self.alpha_pos = rng.normal(size=self.n_features)
        if self.alpha_neg is None:
            self.alpha_neg = rng.normal(size=self.n_features)
        self.alpha_pos = normalize(self.alpha_pos)
        self.alpha_neg = normalize(self.alpha_neg)
        self.gamma = float(self.gamma)
        self._hist_x: List[Array] = []
        self._hist_f: List[Array] = []
        self._hist_y: List[float] = []

    def copy(self):
        return FeatureMixtureStrict(
            n_bits=self.n_bits,
            max_order=self.max_order,
            beta=self.beta,
            eta=self.eta,
            alpha_pos=self.alpha_pos.copy(),
            alpha_neg=self.alpha_neg.copy(),
            gamma=self.gamma,
            memory_window=self.memory_window,
        )

    def reset_history(self):
        self._hist_x.clear()
        self._hist_f.clear()
        self._hist_y.clear()

    def features(self, X: Array) -> Array:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != self.n_bits:
            raise ValueError(f"Expected {self.n_bits} stimulus bits")
        check_pm1(X, "X")
        return transform_features(X, self.subsets)

    def _logZ(self, c: int) -> float:
        a = self.alpha_pos if c == 1 else self.alpha_neg
        return logsumexp(self.beta * (self.F_all @ a))

    def category_distribution(self, c: int) -> Array:
        a = self.alpha_pos if c == 1 else self.alpha_neg
        logits = self.beta * (self.F_all @ a)
        return np.exp(logits - logsumexp(logits))

    def expected_features(self, c: int) -> Array:
        return self.category_distribution(c) @ self.F_all

    def prob_pos(self, X: Array) -> Array:
        return self.prob_pos_features(self.features(X))

    def prob_pos_features(self, F: Array) -> Array:
        F = np.asarray(F, dtype=float)
        if F.ndim == 1:
            F = F[None, :]
        # Eq. 2 with alpha = alpha_-1 - alpha_1
        z = (self._logZ(1) - self._logZ(-1)) + self.beta * (F @ (self.alpha_neg - self.alpha_pos)) + self.gamma
        return sigmoid_neg_logodds(z)

    def prob_label(self, X: Array, y: Array) -> Array:
        y = check_pm1(y, "y").reshape(-1)
        p1 = self.prob_pos(X).reshape(-1)
        if len(y) == 1 and len(p1) > 1:
            y = np.repeat(y, len(p1))
        return np.where(y == 1.0, p1, 1.0 - p1)

    def predict(self, X: Array) -> Array:
        return np.where(self.prob_pos(X) >= 0.5, 1.0, -1.0)

    def update_batch(self, Xw: Array, yw: Array):
        Xw = np.asarray(Xw, dtype=float)
        if Xw.ndim == 1:
            Xw = Xw[None, :]
        self.update_batch_features(self.features(Xw), yw)

    def update_batch_features(self, Fw: Array, yw: Array):
        Fw = np.asarray(Fw, dtype=float)
        if Fw.ndim == 1:
            Fw = Fw[None, :]
        yw = check_pm1(yw, "yw").reshape(-1)
        Epos = self.expected_features(1)
        Eneg = self.expected_features(-1)
        p1 = self.prob_pos_features(Fw).reshape(-1)
        py = np.where(yw == 1.0, p1, 1.0 - p1)
        err = 1.0 - py

        gpos = np.zeros(self.n_features)
        gneg = np.zeros(self.n_features)
        ggam = 0.0

        for f, y, e in zip(Fw, yw, err):
            # Eq. 3: d alpha_c = - (1-P(y|x)) * (E_c[f]-f(x)) * y * c
            gpos += -e * (Epos - f) * y * 1.0
            gneg += -e * (Eneg - f) * y * (-1.0)
            # SI S3
            ggam += -e * y

        scale = 1.0 / len(yw)
        # SI: alpha_c -> alpha_c + eta * dlogL/dalpha_c, followed by
        # normalization; the eta:beta derivation writes the normalized vector
        # equivalently as beta * alpha_t + eta * gradient.
        self.alpha_pos = normalize(self.beta * self.alpha_pos + self.eta * scale * gpos)
        self.alpha_neg = normalize(self.beta * self.alpha_neg + self.eta * scale * gneg)
        self.gamma = float(self.gamma + self.eta * scale * ggam)

    def learn_from_feedback(self, x: Array, y: float):
        x = np.asarray(x, dtype=float).reshape(self.n_bits)
        f = self.features(x)[0]
        self.learn_from_feedback_features(x, f, y)

    def learn_from_feedback_features(self, x: Array, f: Array, y: float):
        x = np.asarray(x, dtype=float).reshape(self.n_bits)
        f = np.asarray(f, dtype=float).reshape(self.n_features)
        y = float(check_pm1(np.asarray([y]), "true_y")[0])
        self._hist_x.append(x.copy())
        self._hist_f.append(f.copy())
        self._hist_y.append(y)
        start = max(0, len(self._hist_y) - self.memory_window)
        self.update_batch_features(np.asarray(self._hist_f[start:]), np.asarray(self._hist_y[start:]))

    def run_session(self, stimuli: Array, true_labels: Array, choices: Optional[Array] = None, update: bool = True) -> Dict[str, Array]:
        X = np.asarray(stimuli, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        y = check_pm1(true_labels, "true_labels").reshape(-1)
        r = None if choices is None else check_pm1(choices, "choices").reshape(-1)
        F = self.features(X)

        p1s, preds, lls, pcors, pchs, gams = [], [], [], [], [], []
        for t, (x, f, yt) in enumerate(zip(X, F, y)):
            p1 = float(self.prob_pos_features(f)[0])
            pred = 1.0 if p1 >= 0.5 else -1.0
            ch = pred if r is None else float(r[t])
            pcor = p1 if yt == 1.0 else 1.0 - p1
            pch = p1 if ch == 1.0 else 1.0 - p1
            p1s.append(p1); preds.append(pred); pcors.append(pcor); pchs.append(pch)
            lls.append(math.log(max(pch, 1e-12)))
            gams.append(self.gamma)
            if update:
                self.learn_from_feedback_features(x, f, yt)

        return {
            "p_pos": np.asarray(p1s),
            "predicted_label": np.asarray(preds),
            "p_correct": np.asarray(pcors),
            "p_observed_choice": np.asarray(pchs),
            "log_likelihood": np.asarray(lls),
            "gamma_before_update": np.asarray(gams),
        }


# -----------------------------
# Eq. 5 likelihood and GA + SA fitting
# -----------------------------

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


def _pack(ap: Array, an: Array, beta: float, eta: float) -> Array:
    return np.r_[normalize(ap), normalize(an), math.log(beta), math.log(eta)]


def _unpack(theta: Array, n_bits: int, max_order: Optional[int], memory_window: int) -> FeatureMixtureStrict:
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
    )


def nll(theta: Array, X: Array, y: Array, r: Array, n_bits: int, max_order: int, memory_window: int, fit_slice: slice) -> float:
    try:
        m = _unpack(theta, n_bits, max_order, memory_window)
        out = m.run_session(X[fit_slice], y[fit_slice], choices=r[fit_slice], update=True)
        return float(-np.sum(out["log_likelihood"]))
    except Exception:
        return 1e12


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
) -> Tuple[FeatureMixtureStrict, Dict[str, Any]]:
    X = np.asarray(X, dtype=float)
    y = check_pm1(y, "y").reshape(-1)
    r = check_pm1(r, "r").reshape(-1)
    if n_bits is None:
        n_bits = X.shape[1]
    if max_order is None:
        max_order = n_bits
    if config is None:
        config = FitConfig()

    rng = np.random.default_rng(config.seed)
    d = len(feature_subsets(n_bits, max_order))
    dim = 2 * d + 2
    fit_slice = slice(None) if fit_first_n is None else slice(0, int(fit_first_n))

    pop = np.zeros((config.population_size, dim))
    for i in range(config.population_size):
        beta = float(np.exp(rng.uniform(np.log(config.beta_range[0]), np.log(config.beta_range[1]))))
        eta = float(np.exp(rng.uniform(np.log(config.eta_range[0]), np.log(config.eta_range[1]))))
        pop[i] = _pack(rng.normal(size=d), rng.normal(size=d), beta, eta)

    elite_n = max(1, int(round(config.population_size * config.elite_fraction)))
    best_theta = pop[0].copy()
    best = float("inf")
    mut = config.mutation_scale

    for gen in range(config.generations):
        scores = np.array([nll(th, X, y, r, n_bits, max_order, memory_window, fit_slice) for th in pop])
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
            child[2*d] = np.clip(child[2*d], np.log(config.beta_range[0]), np.log(config.beta_range[1]))
            child[2*d+1] = np.clip(child[2*d+1], np.log(config.eta_range[0]), np.log(config.eta_range[1]))
            new.append(child)
        pop = np.asarray(new)
        mut *= config.mutation_decay

    theta = best_theta.copy()
    theta_score = best
    for step in range(config.anneal_steps):
        frac = step / max(1, config.anneal_steps - 1)
        temp = config.anneal_initial_temp * (config.anneal_final_temp / config.anneal_initial_temp) ** frac
        prop = theta + rng.normal(scale=config.anneal_step_scale * (1 - 0.8 * frac), size=dim)
        prop[2*d] = np.clip(prop[2*d], np.log(config.beta_range[0]), np.log(config.beta_range[1]))
        prop[2*d+1] = np.clip(prop[2*d+1], np.log(config.eta_range[0]), np.log(config.eta_range[1]))
        ps = nll(prop, X, y, r, n_bits, max_order, memory_window, fit_slice)
        delta = ps - theta_score
        if delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-12)):
            theta, theta_score = prop, float(ps)
            if theta_score < best:
                best, best_theta = theta_score, theta.copy()
        if verbose and (step + 1) % 100 == 0:
            print(f"SA {step+1}/{config.anneal_steps}: best NLL={best:.6f}")

    model = _unpack(best_theta, n_bits, max_order, memory_window)
    info = {
        "negative_log_likelihood": best,
        "optimizer": "genetic_algorithm_plus_simulated_annealing",
        "gamma0_fixed_to_zero": True,
        "alpha_update": "normalize(beta * alpha + eta * gradient)",
        "beta": model.beta,
        "eta": model.eta,
        "n_features": model.n_features,
        "memory_window": memory_window,
        "fit_first_n": fit_first_n,
        "note": "Model equations follow the public paper and SI; exact unpublished GA/SA hyperparameters cannot be guaranteed.",
    }
    return model, info



# -----------------------------
# Per-trial vector formula export
# -----------------------------

def _fmt_float(x: float, decimals: int = 4) -> str:
    """Compact numeric formatting for formula strings."""
    try:
        x = float(x)
    except Exception:
        return str(x)
    if abs(x) < 0.5 * 10 ** (-decimals):
        x = 0.0
    return f"{x:.{decimals}f}"


def _named_vector(names: List[str], values: Array, decimals: int = 4, max_terms: Optional[int] = None) -> str:
    """
    Format a named vector as [x1:0.1234, x2:-0.5678, ...].
    If max_terms is set, keep the largest absolute values.
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    pairs = list(zip(names, values))

    if max_terms is not None and 0 < max_terms < len(pairs):
        order = np.argsort(-np.abs(values))[:max_terms]
        pairs = [(names[i], values[i]) for i in order]

    return "[" + ", ".join(f"{name}:{_fmt_float(val, decimals)}" for name, val in pairs) + "]"


def _plain_vector(values: Array, decimals: int = 0) -> str:
    """Format feature vector values, usually -1/+1."""
    values = np.asarray(values).reshape(-1)
    if decimals == 0:
        return "[" + ", ".join(str(int(v)) for v in values) + "]"
    return "[" + ", ".join(_fmt_float(float(v), decimals) for v in values) + "]"


def _make_inference_formula(
    trial_index: int,
    x: Array,
    feature_names: List[str],
    f_vec: Array,
    alpha_pos: Array,
    alpha_neg: Array,
    beta: float,
    gamma: float,
    log_z_ratio: float,
    z: float,
    p_pos: float,
    pred: float,
    true_y: float,
    choice: Optional[float],
    decimals: int = 4,
    max_terms: Optional[int] = None,
    alpha_pos_after: Optional[Array] = None,
    alpha_neg_after: Optional[Array] = None,
    gamma_after: Optional[float] = None,
) -> str:
    """
    Build one CSV formula cell describing the model's inference step.

    Eq. 2 form:
        P(y=1|x_t)=1/[1+exp(b_t + w_t·f_t)]
        w_t = beta * (alpha_- - alpha_+)
        b_t = log Z_+ - log Z_- + gamma_t
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    f_vec = np.asarray(f_vec, dtype=float).reshape(-1)
    alpha_pos = np.asarray(alpha_pos, dtype=float).reshape(-1)
    alpha_neg = np.asarray(alpha_neg, dtype=float).reshape(-1)

    alpha_diff = alpha_neg - alpha_pos
    w_vec = beta * alpha_diff
    b = log_z_ratio + gamma

    x_str = _plain_vector(x, decimals=0)
    f_str = _named_vector(feature_names, f_vec, decimals=0, max_terms=max_terms)
    alpha_diff_str = _named_vector(feature_names, alpha_diff, decimals=decimals, max_terms=max_terms)
    w_str = _named_vector(feature_names, w_vec, decimals=decimals, max_terms=max_terms)

    choice_text = "NA" if choice is None else ("+1" if choice == 1 else "-1")
    pred_text = "+1" if pred == 1 else "-1"
    true_text = "+1" if true_y == 1 else "-1"

    formula = (
        f"trial {trial_index}: "
        f"x_t={x_str}; "
        f"f_t={f_str}; "
        f"a_t=alpha_- - alpha_+={alpha_diff_str}; "
        f"w_t=beta*a_t={w_str}; "
        f"b_t=logZ_+ - logZ_- + gamma_t={_fmt_float(log_z_ratio, decimals)}+{_fmt_float(gamma, decimals)}={_fmt_float(b, decimals)}; "
        f"z_t=b_t+w_t·f_t={_fmt_float(z, decimals)}; "
        f"P_t(y=+1|x_t)=1/(1+exp(z_t))={_fmt_float(p_pos, decimals)}; "
        f"yhat_t={pred_text}; y_true={true_text}; choice={choice_text}"
    )

    if alpha_pos_after is not None and alpha_neg_after is not None and gamma_after is not None:
        after_diff = np.asarray(alpha_neg_after, dtype=float).reshape(-1) - np.asarray(alpha_pos_after, dtype=float).reshape(-1)
        after_w = beta * after_diff
        formula += (
            f"; after feedback: "
            f"a_(t+1)=alpha_- - alpha_+={_named_vector(feature_names, after_diff, decimals=decimals, max_terms=max_terms)}; "
            f"w_(t+1)=beta*a_(t+1)={_named_vector(feature_names, after_w, decimals=decimals, max_terms=max_terms)}; "
            f"gamma_(t+1)={_fmt_float(gamma_after, decimals)}"
        )

    return formula


def run_session_with_formula_trace(
    model: FeatureMixtureStrict,
    stimuli: Array,
    true_labels: Array,
    choices: Optional[Array] = None,
    update: bool = True,
    decimals: int = 4,
    max_terms: Optional[int] = None,
    include_after_update: bool = False,
) -> Tuple[Dict[str, Array], List[str]]:
    """
    Run the model trial by trial and return both normal prediction arrays and
    a list of per-trial vector-formula strings.

    The formula is generated before feedback update, because that is the model's
    inference state for the current trial. If include_after_update=True, the
    formula also appends the post-feedback weight vector.
    """
    X = np.asarray(stimuli, dtype=float)
    if X.ndim == 1:
        X = X[None, :]

    y = check_pm1(true_labels, "true_labels").reshape(-1)
    r = None if choices is None else check_pm1(choices, "choices").reshape(-1)
    F = model.features(X)

    feature_names = [feature_name(s) for s in model.subsets]

    p1s, preds, lls, pcors, pchs, gams = [], [], [], [], [], []
    alpha_pos_before_trials, alpha_neg_before_trials = [], []
    formulas: List[str] = []

    for t, (x, f_vec, yt) in enumerate(zip(X, F, y), start=1):
        # Snapshot before update.
        alpha_pos_before = model.alpha_pos.copy()
        alpha_neg_before = model.alpha_neg.copy()
        gamma_before = float(model.gamma)
        alpha_pos_before_trials.append(alpha_pos_before)
        alpha_neg_before_trials.append(alpha_neg_before)

        log_z_ratio = model._logZ(1) - model._logZ(-1)
        z = log_z_ratio + model.beta * float(f_vec @ (model.alpha_neg - model.alpha_pos)) + model.gamma
        p1 = float(sigmoid_neg_logodds(np.asarray([z]))[0])
        pred = 1.0 if p1 >= 0.5 else -1.0

        ch = pred if r is None else float(r[t - 1])
        pcor = p1 if yt == 1.0 else 1.0 - p1
        pch = p1 if ch == 1.0 else 1.0 - p1

        p1s.append(p1)
        preds.append(pred)
        pcors.append(pcor)
        pchs.append(pch)
        lls.append(math.log(max(pch, 1e-12)))
        gams.append(gamma_before)

        if update:
            model.learn_from_feedback_features(x, f_vec, yt)

        if include_after_update:
            formula = _make_inference_formula(
                trial_index=t,
                x=x,
                feature_names=feature_names,
                f_vec=f_vec,
                alpha_pos=alpha_pos_before,
                alpha_neg=alpha_neg_before,
                beta=model.beta,
                gamma=gamma_before,
                log_z_ratio=log_z_ratio,
                z=z,
                p_pos=p1,
                pred=pred,
                true_y=float(yt),
                choice=ch,
                decimals=decimals,
                max_terms=max_terms,
                alpha_pos_after=model.alpha_pos.copy(),
                alpha_neg_after=model.alpha_neg.copy(),
                gamma_after=float(model.gamma),
            )
        else:
            formula = _make_inference_formula(
                trial_index=t,
                x=x,
                feature_names=feature_names,
                f_vec=f_vec,
                alpha_pos=alpha_pos_before,
                alpha_neg=alpha_neg_before,
                beta=model.beta,
                gamma=gamma_before,
                log_z_ratio=log_z_ratio,
                z=z,
                p_pos=p1,
                pred=pred,
                true_y=float(yt),
                choice=ch,
                decimals=decimals,
                max_terms=max_terms,
            )

        formulas.append(formula)

    result = {
        "p_pos": np.asarray(p1s),
        "predicted_label": np.asarray(preds),
        "p_correct": np.asarray(pcors),
        "p_observed_choice": np.asarray(pchs),
        "log_likelihood": np.asarray(lls),
        "gamma_before_update": np.asarray(gams),
        "alpha_pos_before_choice": np.asarray(alpha_pos_before_trials),
        "alpha_neg_before_choice": np.asarray(alpha_neg_before_trials),
        "alpha_pos_minus_alpha_neg_before_choice": (
            np.asarray(alpha_pos_before_trials) - np.asarray(alpha_neg_before_trials)
        ),
    }
    return result, formulas


# -----------------------------
# CSV runner
# -----------------------------

def sti_to_bits(sti_id: int, n_bits: int = 5, msb_first: bool = True) -> Array:
    val = int(sti_id) - 1
    bits = np.array([(val >> i) & 1 for i in range(n_bits)], dtype=int)
    if msb_first:
        bits = bits[::-1]
    return np.where(bits == 1, 1.0, -1.0)


def prepare_dataframe(df: pd.DataFrame, n_bits: int, msb_first: bool):
    df = df.copy()
    missing = {"stiID", "category", "choice"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    feature_cols = [f"feature{j+1}" for j in range(n_bits)]
    if set(feature_cols).issubset(df.columns):
        X = df[feature_cols].to_numpy(dtype=float)
        check_pm1(X, "feature columns")
    else:
        X = np.vstack([sti_to_bits(s, n_bits=n_bits, msb_first=msb_first) for s in df["stiID"]])

    y = np.where(df["category"].to_numpy() == 1, 1.0, -1.0)
    r = np.where(df["choice"].to_numpy() == 1, 1.0, -1.0)
    for j in range(n_bits):
        df[f"x{j+1}"] = X[:, j].astype(int)
    df["category_pm1"] = y.astype(int)
    df["choice_pm1"] = r.astype(int)
    df["correct"] = (y == r).astype(int)
    if n_bits >= 4:
        df["x2_times_x4"] = (X[:, 1] * X[:, 3]).astype(int)
        df["rule_match_x2x4"] = (df["x2_times_x4"].to_numpy() == y).astype(int)
    return df, X, y, r


def load_csv(csv_path: Path, n_bits: int, msb_first: bool):
    return prepare_dataframe(pd.read_csv(csv_path), n_bits, msb_first)


def model_feature_to_csv_column(name: str) -> str:
    if not name.startswith("x"):
        raise ValueError(f"Unexpected feature name: {name}")
    return "*".join(f"X{idx}" for idx in name[1:])


def oral_feature_matrix(df: pd.DataFrame, feature_names: Sequence[str]) -> Array:
    cols = [model_feature_to_csv_column(name) for name in feature_names]
    values = []
    for col in cols:
        if col in df.columns:
            values.append(pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float))
        else:
            values.append(np.full(len(df), np.nan))
    return np.column_stack(values)


def save_alpha_trajectories(
    outdir: Path,
    df: pd.DataFrame,
    feature_names: Sequence[str],
    alpha_pos: Array,
    alpha_neg: Array,
    alpha_diff: Array,
) -> Dict[str, str]:
    id_cols = {
        "iSub": df["iSub"].to_numpy() if "iSub" in df.columns else np.full(len(df), ""),
        "iTrial": df["iTrial"].to_numpy() if "iTrial" in df.columns else np.arange(1, len(df) + 1),
        "stiID": df["stiID"].to_numpy(),
    }
    wide_paths = {}
    for key, arr in [
        ("alpha_pos", alpha_pos),
        ("alpha_neg", alpha_neg),
        ("alpha_diff", alpha_diff),
    ]:
        wide = pd.DataFrame(arr, columns=feature_names)
        for col, values in reversed(list(id_cols.items())):
            wide.insert(0, col, values)
        path = outdir / f"{key}_trajectory_wide.csv"
        wide.to_csv(path, index=False, encoding="utf-8-sig")
        wide_paths[f"{key}_trajectory_wide_csv"] = str(path)

    rows = []
    for t in range(len(df)):
        for j, feature in enumerate(feature_names):
            rows.append({
                "iSub": id_cols["iSub"][t],
                "iTrial": id_cols["iTrial"][t],
                "stiID": id_cols["stiID"][t],
                "feature": feature,
                "order": len(feature) - 1,
                "alpha_pos": alpha_pos[t, j],
                "alpha_neg": alpha_neg[t, j],
                "alpha_pos_minus_alpha_neg": alpha_diff[t, j],
            })
    long_path = outdir / "alpha_trajectory_long.csv"
    pd.DataFrame(rows).to_csv(long_path, index=False, encoding="utf-8-sig")
    wide_paths["alpha_trajectory_long_csv"] = str(long_path)
    return wide_paths


def save_oral_trajectories(
    outdir: Path,
    df: pd.DataFrame,
    feature_names: Sequence[str],
    oral_weights: Array,
) -> Dict[str, str]:
    id_data = {
        "iSub": df["iSub"].to_numpy() if "iSub" in df.columns else np.full(len(df), ""),
        "iTrial": df["iTrial"].to_numpy() if "iTrial" in df.columns else np.arange(1, len(df) + 1),
        "stiID": df["stiID"].to_numpy(),
    }
    wide = pd.DataFrame(oral_weights, columns=feature_names)
    for col, values in reversed(list(id_data.items())):
        wide.insert(0, col, values)
    wide_path = outdir / "oral_feature_weights_wide.csv"
    wide.to_csv(wide_path, index=False, encoding="utf-8-sig")

    rows = []
    for t in range(len(df)):
        for j, feature in enumerate(feature_names):
            rows.append({
                "iSub": id_data["iSub"][t],
                "iTrial": id_data["iTrial"][t],
                "stiID": id_data["stiID"][t],
                "feature": feature,
                "order": len(feature) - 1,
                "oral_weight": oral_weights[t, j],
            })
    long_path = outdir / "oral_feature_weights_long.csv"
    pd.DataFrame(rows).to_csv(long_path, index=False, encoding="utf-8-sig")
    return {
        "oral_feature_weights_wide_csv": str(wide_path),
        "oral_feature_weights_long_csv": str(long_path),
    }


def plot_feature_trajectories(
    trials: Array,
    values: Array,
    feature_names: Sequence[str],
    target_feature: str,
    ylabel: str,
    title: str,
    path: Path,
):
    fig, ax = plt.subplots(figsize=(11, 6))
    target_idx = feature_names.index(target_feature) if target_feature in feature_names else None
    for j, name in enumerate(feature_names):
        if j == target_idx:
            continue
        ax.plot(trials, values[:, j], color="0.75", linewidth=0.7, alpha=0.75)
    if target_idx is not None:
        ax.plot(trials, values[:, target_idx], color="#d62728", linewidth=2.8, label=target_feature)
    ax.axhline(0, color="0.2", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Trial")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if target_idx is not None:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def row_cosine(a: Array, b: Array) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return np.nan
    aa, bb = a[mask], b[mask]
    denom = np.linalg.norm(aa) * np.linalg.norm(bb)
    if denom == 0:
        return np.nan
    return float(np.dot(aa, bb) / denom)


def row_pearson(a: Array, b: Array) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if np.sum(mask) < 2:
        return np.nan
    aa, bb = a[mask], b[mask]
    if np.std(aa) == 0 or np.std(bb) == 0:
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def save_similarity_outputs(
    outdir: Path,
    df: pd.DataFrame,
    trials: Array,
    model_weights: Array,
    oral_weights: Array,
    feature_names: Sequence[str],
    target_feature: str,
) -> Dict[str, str]:
    cosine = np.asarray([row_cosine(m, o) for m, o in zip(model_weights, oral_weights)])
    pearson = np.asarray([row_pearson(m, o) for m, o in zip(model_weights, oral_weights)])
    sim_df = pd.DataFrame({
        "iSub": df["iSub"].to_numpy() if "iSub" in df.columns else "",
        "iTrial": trials,
        "stiID": df["stiID"].to_numpy(),
        "cosine_similarity": cosine,
        "pearson_correlation": pearson,
    })
    sim_path = outdir / "model_oral_similarity.csv"
    sim_df.to_csv(sim_path, index=False, encoding="utf-8-sig")

    sim_png = outdir / "model_oral_similarity.png"
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(trials, cosine, color="black", linewidth=1.8, label="cosine")
    ax.plot(trials, pearson, color="#1f77b4", linewidth=1.2, linestyle="--", label="Pearson r")
    ax.axhline(0, color="0.2", linewidth=0.8, alpha=0.6)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Similarity")
    ax.set_title("Similarity between model and oral feature-weight distributions")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(sim_png, dpi=200)
    plt.close(fig)

    target_png = outdir / f"target_{target_feature}_model_vs_oral.png"
    target_idx = feature_names.index(target_feature)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(trials, model_weights[:, target_idx], color="#d62728", linewidth=2.6, label=f"model {target_feature}")
    ax.plot(trials, oral_weights[:, target_idx], color="black", linewidth=1.5, linestyle="--", label=f"oral {target_feature}")
    ax.axhline(0, color="0.2", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Weight")
    ax.set_title(f"Target feature {target_feature}: model vs oral weights")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(target_png, dpi=200)
    plt.close(fig)

    return {
        "model_oral_similarity_csv": str(sim_path),
        "model_oral_similarity_png": str(sim_png),
        "target_feature_model_vs_oral_png": str(target_png),
    }


def sort_subjects(subjects: Sequence[Any]) -> List[str]:
    return sorted({str(s) for s in subjects}, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))


def subject_id_values(df: pd.DataFrame) -> List[str]:
    if "iSub" not in df.columns:
        return ["all"]
    return sort_subjects(df["iSub"].dropna().astype(str).unique())


def id_columns(df: pd.DataFrame) -> Dict[str, Array]:
    return {
        "iSub": df["iSub"].to_numpy() if "iSub" in df.columns else np.full(len(df), "all"),
        "iSession": df["iSession"].to_numpy() if "iSession" in df.columns else np.full(len(df), ""),
        "iTrial": df["iTrial"].to_numpy() if "iTrial" in df.columns else np.arange(1, len(df) + 1),
        "stiID": df["stiID"].to_numpy(),
    }


def make_wide_df(df: pd.DataFrame, values: Array, feature_names: Sequence[str]) -> pd.DataFrame:
    wide = pd.DataFrame(values, columns=feature_names)
    for col, data in reversed(list(id_columns(df).items())):
        wide.insert(0, col, data)
    return wide


def make_alpha_long_df(df: pd.DataFrame, feature_names: Sequence[str], alpha_pos: Array, alpha_neg: Array, alpha_diff: Array) -> pd.DataFrame:
    ids = id_columns(df)
    rows = []
    for t in range(len(df)):
        for j, feature in enumerate(feature_names):
            rows.append({
                "iSub": ids["iSub"][t],
                "iSession": ids["iSession"][t],
                "iTrial": ids["iTrial"][t],
                "stiID": ids["stiID"][t],
                "feature": feature,
                "order": len(feature) - 1,
                "alpha_pos": alpha_pos[t, j],
                "alpha_neg": alpha_neg[t, j],
                "alpha_pos_minus_alpha_neg": alpha_diff[t, j],
            })
    return pd.DataFrame(rows)


def make_oral_long_df(df: pd.DataFrame, feature_names: Sequence[str], oral_weights: Array) -> pd.DataFrame:
    ids = id_columns(df)
    rows = []
    for t in range(len(df)):
        for j, feature in enumerate(feature_names):
            rows.append({
                "iSub": ids["iSub"][t],
                "iSession": ids["iSession"][t],
                "iTrial": ids["iTrial"][t],
                "stiID": ids["stiID"][t],
                "feature": feature,
                "order": len(feature) - 1,
                "oral_weight": oral_weights[t, j],
            })
    return pd.DataFrame(rows)


def make_similarity_df(df: pd.DataFrame, model_weights: Array, oral_weights: Array) -> pd.DataFrame:
    ids = id_columns(df)
    return pd.DataFrame({
        "iSub": ids["iSub"],
        "iSession": ids["iSession"],
        "iTrial": ids["iTrial"],
        "stiID": ids["stiID"],
        "cosine_similarity": [row_cosine(m, o) for m, o in zip(model_weights, oral_weights)],
        "pearson_correlation": [row_pearson(m, o) for m, o in zip(model_weights, oral_weights)],
    })


def plot_subject_learning_curves(runs: Sequence[Dict[str, Any]], path: Path):
    fig, axes = plt.subplots(len(runs), 1, figsize=(10, 3.2 * len(runs)), squeeze=False)
    for ax, run in zip(axes[:, 0], runs):
        df = run["trial_predictions"]
        trial_col = "iTrial" if "iTrial" in df.columns else "trial_index"
        ax.plot(df[trial_col], df["rolling_accuracy_16"], label="Observed rolling accuracy")
        ax.plot(df[trial_col], df["rolling_model_p_correct_16"], label="Model rolling P(correct)")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"Subject {run['subject_id']}")
        ax.set_ylabel("Accuracy / probability")
        ax.legend(frameon=False, loc="lower right")
    axes[-1, 0].set_xlabel("Trial")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_subject_feature_grid(
    runs: Sequence[Dict[str, Any]],
    value_key: str,
    target_feature: str,
    ylabel: str,
    title: str,
    path: Path,
):
    fig, axes = plt.subplots(len(runs), 1, figsize=(11, 3.6 * len(runs)), squeeze=False)
    for ax, run in zip(axes[:, 0], runs):
        values = run[value_key]
        feature_names = run["feature_names"]
        trials = run["trial_values"]
        target_idx = feature_names.index(target_feature)
        for j, _name in enumerate(feature_names):
            if j != target_idx:
                ax.plot(trials, values[:, j], color="0.75", linewidth=0.65, alpha=0.7)
        ax.plot(trials, values[:, target_idx], color="#d62728", linewidth=2.8, label=target_feature)
        ax.axhline(0, color="0.2", linewidth=0.8, alpha=0.6)
        ax.set_title(f"Subject {run['subject_id']}")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, loc="upper right")
    axes[-1, 0].set_xlabel("Trial")
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_subject_similarity_grid(runs: Sequence[Dict[str, Any]], path: Path):
    fig, axes = plt.subplots(len(runs), 1, figsize=(10, 3.4 * len(runs)), squeeze=False)
    for ax, run in zip(axes[:, 0], runs):
        sim = run["similarity"]
        ax.plot(sim["iTrial"], sim["cosine_similarity"], color="black", linewidth=1.8, label="cosine")
        ax.plot(sim["iTrial"], sim["pearson_correlation"], color="#1f77b4", linewidth=1.2, linestyle="--", label="Pearson r")
        ax.axhline(0, color="0.2", linewidth=0.8, alpha=0.6)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(f"Subject {run['subject_id']}")
        ax.set_ylabel("Similarity")
        ax.legend(frameon=False, loc="lower right")
    axes[-1, 0].set_xlabel("Trial")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_subject_target_grid(runs: Sequence[Dict[str, Any]], target_feature: str, path: Path):
    fig, axes = plt.subplots(len(runs), 1, figsize=(10, 3.4 * len(runs)), squeeze=False)
    for ax, run in zip(axes[:, 0], runs):
        idx = run["feature_names"].index(target_feature)
        trials = run["trial_values"]
        ax.plot(trials, run["alpha_diff_trials"][:, idx], color="#d62728", linewidth=2.6, label=f"model {target_feature}")
        ax.plot(trials, run["oral_weights"][:, idx], color="black", linewidth=1.5, linestyle="--", label=f"oral {target_feature}")
        ax.axhline(0, color="0.2", linewidth=0.8, alpha=0.6)
        ax.set_title(f"Subject {run['subject_id']}")
        ax.set_ylabel("Weight")
        ax.legend(frameon=False, loc="upper right")
    axes[-1, 0].set_xlabel("Trial")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_one_subject(subject_id: str, subject_df: pd.DataFrame, source_csv: Path, args: argparse.Namespace) -> Dict[str, Any]:
    max_order = args.max_order if args.max_order is not None else args.n_bits
    sort_cols = [c for c in ["iSession", "iBlock", "iTrial"] if c in subject_df.columns]
    if sort_cols:
        subject_df = subject_df.sort_values(sort_cols).reset_index(drop=True)
    df, X, y, r = prepare_dataframe(subject_df, args.n_bits, msb_first=not args.lsb_first)

    if args.fit:
        cfg = FitConfig(
            population_size=args.population_size,
            generations=args.generations,
            anneal_steps=args.anneal_steps,
            seed=args.seed,
        )
        initial_model, fit_info = fit_paper_style(
            X, y, r, n_bits=args.n_bits, max_order=max_order,
            memory_window=1, fit_first_n=args.fit_first_n,
            config=cfg, verbose=not args.quiet
        )
    else:
        initial_model = FeatureMixtureStrict(
            n_bits=args.n_bits, max_order=max_order,
            beta=args.beta, eta=args.eta, gamma=0.0,
            memory_window=1, seed=args.seed
        )
        fit_info = {
            "mode": "no_fit_replay",
            "gamma0_fixed_to_zero": True,
            "alpha_update": "normalize(beta * alpha + eta * gradient)",
        }

    run_model = initial_model.copy()
    formula_max_terms = None if args.formula_max_terms <= 0 else args.formula_max_terms
    res, formula_strings = run_session_with_formula_trace(
        run_model,
        X,
        y,
        choices=r,
        update=True,
        decimals=args.formula_decimals,
        max_terms=formula_max_terms,
        include_after_update=args.include_after_update,
    )

    trial_values = df["iTrial"].to_numpy() if "iTrial" in df.columns else np.arange(1, len(df) + 1)
    formula_df = pd.DataFrame({
        "iSub": df["iSub"].to_numpy() if "iSub" in df.columns else np.full(len(df), subject_id),
        "iSession": df["iSession"].to_numpy() if "iSession" in df.columns else np.full(len(df), args.isession),
        "iTrial": trial_values,
        "stiID": df["stiID"].to_numpy(),
        "formula": formula_strings,
    })

    feature_names = [feature_name(s) for s in run_model.subsets]
    alpha_pos_trials = res["alpha_pos_before_choice"]
    alpha_neg_trials = res["alpha_neg_before_choice"]
    alpha_diff_trials = res["alpha_pos_minus_alpha_neg_before_choice"]
    oral_weights = oral_feature_matrix(df, feature_names)
    target_feature = "x24" if args.n_bits >= 4 else feature_names[0]

    df["model_p_category1"] = res["p_pos"]
    df["model_p_correct"] = res["p_correct"]
    df["model_p_observed_choice"] = res["p_observed_choice"]
    df["model_pred_pm1"] = res["predicted_label"].astype(int)
    df["model_pred_category"] = np.where(res["predicted_label"] == 1, 1, 2)
    df["model_pred_matches_choice"] = (res["predicted_label"] == r).astype(int)
    df["gamma_before_update"] = res["gamma_before_update"]

    trial_col = "iTrial" if "iTrial" in df.columns else "trial_index"
    if trial_col == "trial_index":
        df["trial_index"] = np.arange(1, len(df) + 1)

    df["rolling_accuracy_16"] = df["correct"].rolling(16, min_periods=16).mean()
    df["rolling_model_p_correct_16"] = df["model_p_correct"].rolling(16, min_periods=16).mean()

    feat_rows = []
    for i, s in enumerate(run_model.subsets):
        sep = float(run_model.alpha_pos[i] - run_model.alpha_neg[i])
        feat_rows.append({
            "iSub": subject_id,
            "feature": feature_name(s),
            "subset_zero_based": str(s),
            "subset_one_based": str(tuple(j + 1 for j in s)),
            "alpha_pos_final": float(run_model.alpha_pos[i]),
            "alpha_neg_final": float(run_model.alpha_neg[i]),
            "alpha_pos_minus_alpha_neg_final": sep,
            "abs_separation_final": abs(sep),
        })
    feat_df = pd.DataFrame(feat_rows).sort_values("abs_separation_final", ascending=False)

    task_feature = target_feature if args.n_bits >= 4 else None
    task_rank = None
    if task_feature in feat_df["feature"].to_numpy():
        task_rank = int(np.where(feat_df["feature"].to_numpy() == task_feature)[0][0] + 1)

    future = {}
    if args.fit_first_n is not None and args.fit_first_n < len(df):
        s = slice(args.fit_first_n, len(df))
        future = {
            "fit_first_n": int(args.fit_first_n),
            "future_choice_agreement": float(np.mean(res["predicted_label"][s] == r[s])),
            "future_mean_p_correct": float(np.mean(res["p_correct"][s])),
            "future_negative_log_likelihood": float(-np.sum(res["log_likelihood"][s])),
        }

    summary = {
        "iSub": subject_id,
        "source_csv": str(source_csv),
        "n_trials": int(len(df)),
        "n_unique_stimuli": int(df["stiID"].nunique()),
        "n_bits": args.n_bits,
        "max_order": max_order,
        "gamma0_fixed_to_zero": True,
        "memory_window": 1,
        "alpha_update": "normalize(beta * alpha + eta * gradient)",
        "observed_accuracy": float(df["correct"].mean()),
        "model_negative_log_likelihood": float(-np.sum(res["log_likelihood"])),
        "chance_negative_log_likelihood": float(len(df) * math.log(2)),
        "model_choice_agreement": float(df["model_pred_matches_choice"].mean()),
        "mean_model_p_correct": float(df["model_p_correct"].mean()),
        "mean_model_p_observed_choice": float(df["model_p_observed_choice"].mean()),
        "task_relevant_feature": task_feature,
        "task_relevant_feature_rank_by_abs_separation": task_rank,
        "rule_match_x2x4": float(df["rule_match_x2x4"].mean()) if "rule_match_x2x4" in df else None,
        "fit_info": fit_info,
        "future_prediction_metrics": future,
    }
    return {
        "subject_id": subject_id,
        "summary": summary,
        "trial_predictions": df,
        "formula_steps": formula_df,
        "final_feature_weights": feat_df,
        "alpha_pos_wide": make_wide_df(df, alpha_pos_trials, feature_names),
        "alpha_neg_wide": make_wide_df(df, alpha_neg_trials, feature_names),
        "alpha_diff_wide": make_wide_df(df, alpha_diff_trials, feature_names),
        "alpha_long": make_alpha_long_df(df, feature_names, alpha_pos_trials, alpha_neg_trials, alpha_diff_trials),
        "oral_wide": make_wide_df(df, oral_weights, feature_names),
        "oral_long": make_oral_long_df(df, feature_names, oral_weights),
        "similarity": make_similarity_df(df, alpha_diff_trials, oral_weights),
        "feature_names": feature_names,
        "trial_values": trial_values,
        "alpha_diff_trials": alpha_diff_trials,
        "oral_weights": oral_weights,
        "target_feature": target_feature,
    }


def concat_run_df(runs: Sequence[Dict[str, Any]], key: str) -> pd.DataFrame:
    return pd.concat([run[key] for run in runs], ignore_index=True)


def write_combined_outputs(runs: Sequence[Dict[str, Any]], outdir: Path, source_csv: Path, args: argparse.Namespace) -> Dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {
        "trial_predictions_csv": outdir / "trial_predictions.csv",
        "final_feature_weights_csv": outdir / "final_feature_weights.csv",
        "formula_steps_csv": outdir / "formula_steps.csv",
        "alpha_pos_trajectory_wide_csv": outdir / "alpha_pos_trajectory_wide.csv",
        "alpha_neg_trajectory_wide_csv": outdir / "alpha_neg_trajectory_wide.csv",
        "alpha_diff_trajectory_wide_csv": outdir / "alpha_diff_trajectory_wide.csv",
        "alpha_trajectory_long_csv": outdir / "alpha_trajectory_long.csv",
        "oral_feature_weights_wide_csv": outdir / "oral_feature_weights_wide.csv",
        "oral_feature_weights_long_csv": outdir / "oral_feature_weights_long.csv",
        "model_oral_similarity_csv": outdir / "model_oral_similarity.csv",
        "learning_curve_png": outdir / "learning_curve.png",
        "model_alpha_diff_over_trials_png": outdir / "model_alpha_diff_over_trials.png",
        "oral_feature_weights_over_trials_png": outdir / "oral_feature_weights_over_trials.png",
        "model_oral_similarity_png": outdir / "model_oral_similarity.png",
        "target_x24_model_vs_oral_png": outdir / "target_x24_model_vs_oral.png",
        "summary_json": outdir / "summary.json",
    }

    concat_run_df(runs, "trial_predictions").to_csv(paths["trial_predictions_csv"], index=False, encoding="utf-8-sig")
    concat_run_df(runs, "final_feature_weights").to_csv(paths["final_feature_weights_csv"], index=False, encoding="utf-8-sig")
    concat_run_df(runs, "formula_steps").to_csv(paths["formula_steps_csv"], index=False, encoding="utf-8-sig")
    concat_run_df(runs, "alpha_pos_wide").to_csv(paths["alpha_pos_trajectory_wide_csv"], index=False, encoding="utf-8-sig")
    concat_run_df(runs, "alpha_neg_wide").to_csv(paths["alpha_neg_trajectory_wide_csv"], index=False, encoding="utf-8-sig")
    concat_run_df(runs, "alpha_diff_wide").to_csv(paths["alpha_diff_trajectory_wide_csv"], index=False, encoding="utf-8-sig")
    concat_run_df(runs, "alpha_long").to_csv(paths["alpha_trajectory_long_csv"], index=False, encoding="utf-8-sig")
    concat_run_df(runs, "oral_wide").to_csv(paths["oral_feature_weights_wide_csv"], index=False, encoding="utf-8-sig")
    concat_run_df(runs, "oral_long").to_csv(paths["oral_feature_weights_long_csv"], index=False, encoding="utf-8-sig")
    concat_run_df(runs, "similarity").to_csv(paths["model_oral_similarity_csv"], index=False, encoding="utf-8-sig")

    target_feature = "x24" if args.n_bits >= 4 else runs[0]["target_feature"]
    plot_subject_learning_curves(runs, paths["learning_curve_png"])
    plot_subject_feature_grid(
        runs,
        "alpha_diff_trials",
        target_feature,
        "alpha_pos - alpha_neg",
        "Model feature weights over trials",
        paths["model_alpha_diff_over_trials_png"],
    )
    plot_subject_feature_grid(
        runs,
        "oral_weights",
        target_feature,
        "oral feature weight",
        "Oral-coded feature weights over trials",
        paths["oral_feature_weights_over_trials_png"],
    )
    plot_subject_similarity_grid(runs, paths["model_oral_similarity_png"])
    plot_subject_target_grid(runs, target_feature, paths["target_x24_model_vs_oral_png"])

    summary = {
        "source_csv": str(source_csv),
        "fit_subjects": [run["subject_id"] for run in runs],
        "n_subjects": len(runs),
        "n_bits": args.n_bits,
        "max_order": args.max_order if args.max_order is not None else args.n_bits,
        "optimizer_config": {
            "fit": args.fit,
            "population_size": args.population_size,
            "generations": args.generations,
            "anneal_steps": args.anneal_steps,
            "fit_first_n": args.fit_first_n,
            "seed": args.seed,
        },
        "runs": [run["summary"] for run in runs],
        "outputs": {key: str(path) for key, path in paths.items() if key != "summary_json"},
    }
    paths["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=None,
        help="Combined processed CSV. Defaults to data_exp5/processed/Rule1_processed.csv.",
    )
    parser.add_argument("--outdir", default="results/Cohen_model/Exp5")
    parser.add_argument("--n-bits", type=int, default=5)
    parser.add_argument("--max-order", type=int, default=None)
    parser.add_argument("--lsb-first", action="store_true")
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--fit-first-n", type=int, default=None)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--population-size", type=int, default=40)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--anneal-steps", type=int, default=120)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--subjects", nargs="*", default=None, help="iSub values to fit. Defaults to all subjects in the CSV.")
    parser.add_argument("--isession", type=int, default=1, help="Value for the first column of formula CSV.")
    parser.add_argument("--formula-csv-name", default="formula_steps.csv", help="Formula CSV filename.")
    parser.add_argument("--formula-decimals", type=int, default=4, help="Decimals used in formula strings.")
    parser.add_argument("--formula-max-terms", type=int, default=31, help="Max number of feature terms shown in each vector; 31 shows all 5-bit features.")
    parser.add_argument("--include-after-update", action="store_true", help="Append post-feedback weight vector to each formula.")
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else DEFAULT_EXP5_CSV
    outdir = Path(args.outdir)
    full_df = pd.read_csv(csv_path)
    subjects = sort_subjects(args.subjects) if args.subjects else subject_id_values(full_df)
    runs = []
    for subject_id in subjects:
        subject_df = full_df if subject_id == "all" else full_df[full_df["iSub"].astype(str) == subject_id].copy()
        if subject_df.empty:
            raise ValueError(f"No rows found for subject {subject_id}")
        runs.append(run_one_subject(subject_id, subject_df, csv_path, args))

    summary = write_combined_outputs(runs, outdir, csv_path, args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

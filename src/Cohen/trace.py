"""Trace for the Cohen–Schneidman reimplementation."""
from __future__ import annotations

from typing import Optional, Tuple, List, Dict
import numpy as np
from .binary import FeatureMixtureStrict
from .features import validate_trial_lengths
from .features import check_pm1, feature_name, sigmoid_neg_logodds

Array = np.ndarray


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
    make_formulas: bool = True,
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
    validate_trial_lengths(X, y, r)

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
        lls.append(float(-np.logaddexp(0.0, ch * z)))
        gams.append(gamma_before)

        if update:
            model.learn_from_feedback_features(x, f_vec, yt)

        if not make_formulas:
            continue

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
                choice=None if r is None else ch,
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
                choice=None if r is None else ch,
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

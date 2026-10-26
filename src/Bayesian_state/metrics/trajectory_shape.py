"""Translation-tolerant descriptive learning shapes for complete choice paths.

The sigmoid is an observation-level descriptor, not a replacement cognitive
model. Its BIC gain and labels are descriptive gates, not significance tests.
Every path remains in the continuous feature distribution, including flat,
decreasing, complex and censored paths. Absolute onset is recorded separately.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from scipy.spatial.distance import cdist


def sequence_score(probability: np.ndarray) -> dict:
    """Average independent estimates of sequence likelihood in log space."""
    p = np.asarray(probability, dtype=float)
    if p.ndim != 2 or min(p.shape) < 1 or not np.isfinite(p).all() or np.any((p <= 0) | (p > 1)):
        raise ValueError('Require positive repeat-by-trial probabilities')
    logs = np.log(p).sum(axis=1)
    r, t = p.shape
    value = float(-(logsumexp(logs) - np.log(r)) / t)
    weights = np.exp(logs - logsumexp(logs))
    deleted = np.array([-(logsumexp(np.delete(logs, i)) - np.log(r-1))/t
                        for i in range(r)]) if r > 1 else np.array([])
    se = float(np.sqrt((r-1)/r * np.square(deleted-deleted.mean()).sum())) if r > 1 else None
    return dict(nll=value, mcse=se, repeats=r, trials=t,
                maximum_system_weight=float(weights.max()), effective_systems=float(1/(weights@weights)),
                log_likelihoods=logs.tolist(), deleted_nll=deleted.tolist(),
                pooled_trial_nll=float(-np.log(p.mean(axis=0)).mean()),
                minimum_probability=float(p.min()))


def paired_difference(a: np.ndarray, b: np.ndarray) -> dict:
    """NLL(a)-NLL(b), with entire paired filters as jackknife units."""
    if np.shape(a) != np.shape(b) or len(a) < 2:
        raise ValueError('Require aligned independent systems')
    x, y = sequence_score(a), sequence_score(b)
    d = np.array(x['deleted_nll']) - np.array(y['deleted_nll'])
    se = np.sqrt((len(d)-1)/len(d) * np.square(d-d.mean()).sum())
    return dict(difference=x['nll']-y['nll'], mcse=float(se))


def _blocks(y: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    starts = np.arange(0, len(y), width)
    sizes = np.minimum(width, len(y)-starts)
    means = np.add.reduceat(y, starts)/sizes
    return means, sizes


def describe_path(correct: np.ndarray, policy: dict) -> dict:
    """Fit a four-parameter sigmoid and preserve unsupported transitions."""
    y = np.asarray(correct, dtype=float)
    if y.ndim != 1 or len(y) < 8 or not np.isin(y, [0, 1]).all():
        raise ValueError('Require at least eight binary observed outcomes')
    t = np.arange(len(y), dtype=float)
    n = len(y)
    mean = float(y.mean())
    p0 = np.clip(mean, .001, .999)
    null_nll = float(-np.sum(y*np.log(p0)+(1-y)*np.log1p(-p0)))

    def objective(z):
        low, high, center, logs = z
        scale = np.exp(logs)
        u = expit((t-center)/scale)
        p = np.clip(low+(high-low)*u, 1e-12, 1-1e-12)
        loss = -np.sum(y*np.log(p)+(1-y)*np.log1p(-p))
        common = (p-y)/(p*(1-p))
        du = (high-low)*u*(1-u)
        gradient = np.array([common@(1-u), common@u,
                             common@(-du/scale), common@(-du*(t-center)/scale)])
        return loss, gradient

    edge = max(8, n//8)
    low, high = np.clip([y[:edge].mean(), y[-edge:].mean()], .01, .99)
    fits = []
    for fraction in policy['center_starts']:
        for scale in (2., max(4., n/8)):
            fit = minimize(objective, [low, high, fraction*(n-1), np.log(scale)], jac=True,
                           method='L-BFGS-B', bounds=[(.001,.999),(.001,.999),(-n/4,1.25*n),
                                                   (np.log(.25),np.log(2*n))],
                           options={'maxiter': policy['optimizer_iterations'], 'ftol': 1e-9})
            if np.isfinite(fit.fun) and np.isfinite(fit.x).all():
                fits.append(fit)
    if not fits:
        raise ValueError('No finite descriptive fit')
    best = min(fits, key=lambda z: z.fun)
    lo, hi, center, logs = map(float, best.x)
    scale = np.exp(logs)
    width = float(2*np.log(9)*scale)
    left, right = center-width/2, center+width/2
    bic_gain = float(2*(null_nll-best.fun)-3*np.log(n))
    gain = hi-lo
    supported = bool(bic_gain >= policy['bic_gain'] and abs(gain) >= policy['minimum_gain'])
    trend = lo+(hi-lo)*expit((t-center)/scale) if supported else np.full(n, mean)
    if not supported:
        lo = hi = mean
        gain = 0.
    coarse, sizes = _blocks(y, policy['residual_block'])
    expected, _ = _blocks(trend, policy['residual_block'])
    dispersion = float(np.mean(sizes*np.square(coarse-expected)/np.maximum(expected*(1-expected), .02)))
    roughness = []
    for w in policy['windows']:
        b, _ = _blocks(y, w)
        roughness.append(float(np.sqrt(np.mean(np.diff(b)**2))) if len(b)>1 else 0.)
    endpoint = int(np.ceil(right))
    available = max(0, n-max(0, endpoint)) if supported and gain > 0 else 0
    observed_followup = min(policy['followup'], available)
    post = float(y[max(0,endpoint):max(0,endpoint)+observed_followup].mean()) if observed_followup else mean
    censored = bool(supported and (left < 0 or right >= n or available < policy['followup']))
    if dispersion > policy['complex_dispersion']:
        label = 'complex_fluctuation'
    elif not supported:
        label = 'stable_high' if mean >= policy['high_level'] else 'no_clear_increase'
    elif gain < 0:
        label = 'decreasing'
    elif censored:
        label = 'transition_censored'
    elif post < hi-policy['maintenance_drop']:
        label = 'increase_then_relapse'
    elif width <= policy['abrupt_width']:
        label = 'abrupt_maintained'
    elif width >= policy['gradual_width']:
        label = 'gradual_maintained'
    else:
        label = 'intermediate_maintained'
    # Onset is deliberately absent. Missing follow-up has an explicit indicator;
    # no zero-filled width is mistaken for an observed instantaneous transition.
    levels = np.array([lo, hi, post]) * np.sqrt(.35/3)
    transition = np.array([max(0.,gain), np.log1p(width)/np.log1p(8*n) if supported else 0.,
                           float(supported), float(censored)]) * np.sqrt(.35/4)
    variation = np.array([*roughness, min(dispersion,10.)/10]) * np.sqrt(.30/(len(roughness)+1))
    features = np.concatenate([levels, transition, variation])
    return dict(trials=n, mean_accuracy=mean, low_level=lo, high_level=hi, gain=gain,
                onset=center if supported else None, transition_width=width if supported else None,
                transition_start=left if supported else None, transition_end=right if supported else None,
                bic_gain=bic_gain, supported_transition=supported, censored=censored,
                post_level=post, followup_trials=observed_followup, dispersion=dispersion,
                roughness=roughness, label=label, optimizer_success=bool(best.success), features=features.tolist())


def describe_ensemble(choices: np.ndarray, categories: np.ndarray, policy: dict) -> list[dict]:
    x, c = np.asarray(choices), np.asarray(categories)
    if x.ndim != 2 or x.shape[1:] != c.shape:
        raise ValueError('Require aligned path-by-trial choices and categories')
    return [describe_path(row == c, policy) for row in x]


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if not 0 <= successes <= total or total < 1:
        raise ValueError('Invalid binomial counts')
    z = 1.959963984540054
    p = successes/total
    middle = (p+z*z/(2*total))/(1+z*z/total)
    half = z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/(1+z*z/total)
    return float(max(0,middle-half)), float(min(1,middle+half))


def shape_check(samples: np.ndarray, observed: np.ndarray, alpha: float = .05) -> dict:
    """Fixed-reference discrepancy, independently simulated calibration paths.

    This is a working compatibility check on selected descriptors. The Wilson
    interval measures simulation counting error, not parameter uncertainty.
    """
    x, y = np.asarray(samples, float), np.asarray(observed, float)
    if x.ndim != 2 or len(x) < 8 or y.shape != x.shape[1:] or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Need finite features and at least eight paths')
    reference, calibration = x[:len(x)//2], x[len(x)//2:]
    # A local discrepancy detects low-mass gaps between modes. Mean distance
    # to every reference path can accept such gaps merely because they are
    # central. Several neighbors prevent a single closest path from deciding.
    neighbors=max(2,int(np.sqrt(len(reference))))
    actual = float(np.sort(np.linalg.norm(reference-y, axis=1))[:neighbors].mean())
    distances = np.sort(cdist(calibration, reference),axis=1)[:,:neighbors].mean(axis=1)
    k = int(np.count_nonzero(distances >= actual))
    lo, hi = wilson_interval(k, len(calibration))
    status = 'compatible' if lo > alpha else 'incompatible' if hi < alpha else 'undetermined'
    return dict(status=status, rank_tail=(k+1)/(len(calibration)+1), tail_count=k, neighbors=neighbors,
                calibration_paths=len(calibration), reference_paths=len(reference),
                tail_mc_low=lo, tail_mc_high=hi, observed_discrepancy=actual,
                reference_q95=float(np.quantile(distances,.95)), alpha=alpha)

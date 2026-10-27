"""Provisional descriptive labels, independent of the frozen shape score.

Fit binary records to constant, step, linear-ramp and 3/4-stage curves.
Stage boundaries are searched at every trial with exact dynamic programming;
the best unrestricted fit at each stage count is then screened for amplitude.
The ramp uses deterministic multistart constrained optimization. BIC is a
descriptive comparison, with no calibrated error-rate or psychological claim.
"""
from __future__ import annotations

import hashlib
from copy import deepcopy

import numpy as np
from scipy.optimize import minimize
from scipy.special import xlogy

LABELS = ['no_clear_change', 'abrupt_improvement', 'gradual_improvement',
          'decreasing', 'multiple_changes', 'shape_undetermined']
SHORT = ['No clear change', 'Abrupt improvement', 'Gradual improvement',
         'Decreasing', 'Multiple changes', 'Shape undetermined']
CHINESE = dict(zip(LABELS, ['未检出明确变化', '突变式改善', '渐进式改善',
                          '下降', '多阶段变化', '形态未定']))


def _segment_costs(y: np.ndarray) -> np.ndarray:
    """Bernoulli MLE losses for every half-open interval [start, end)."""
    prefix = np.r_[0., np.cumsum(y)]
    size = np.arange(len(prefix))[None, :] - np.arange(len(prefix))[:, None]
    count = prefix[None, :] - prefix[:, None]
    valid = size > 0
    total = np.maximum(size, 1)
    hits = np.maximum(count, 0)
    misses = np.maximum(size-count, 0)
    loss = -(xlogy(hits, hits/total) + xlogy(misses, misses/total))
    return np.where(valid, loss, np.inf)


def _piecewise_fits(y: np.ndarray, minimum: int, maximum: int) -> list[dict]:
    n = len(y)
    costs = _segment_costs(y)
    previous = np.full(n+1, np.inf)
    previous[0] = 0.
    parents = np.full((maximum+1, n+1), -1, dtype=int)
    fits = []
    prefix = np.r_[0., np.cumsum(y)]
    for segments in range(1, maximum+1):
        current = np.full(n+1, np.inf)
        for end in range(segments*minimum, n+1):
            starts = np.arange((segments-1)*minimum, end-minimum+1)
            values = previous[starts] + costs[starts, end]
            index = int(np.argmin(values))
            current[end] = values[index]
            parents[segments, end] = starts[index]
        if np.isfinite(current[n]):
            boundaries = [n]
            for k in range(segments, 0, -1):
                boundaries.append(int(parents[k, boundaries[-1]]))
            boundaries.reverse()
            levels = [(prefix[b]-prefix[a])/(b-a) for a, b in zip(boundaries[:-1], boundaries[1:])]
            fits.append(dict(family='constant' if segments==1 else 'step' if segments==2 else 'stages',
                             segments=segments, boundaries=boundaries, levels=levels,
                             nll=float(current[n]), parameters=2*segments-1, converged=True))
        previous = current
    return fits


def _ramp_loss(z: np.ndarray, y: np.ndarray, x: np.ndarray) -> tuple[float, np.ndarray]:
    low, high, start, end = z
    width = max(end-start, 1e-12)
    u = np.clip((x-start)/width, 0., 1.)
    p = low + (high-low)*u
    active = (x > start) & (x < end)
    common = (p-y)/(p*(1-p))
    da = np.where(active, (x-end)/width**2, 0.)
    db = np.where(active, -(x-start)/width**2, 0.)
    gradient = np.array([common@(1-u), common@u,
                         common@((high-low)*da), common@((high-low)*db)])
    return float(-np.sum(y*np.log(p)+(1-y)*np.log1p(-p))), gradient


def _ramp_fit(y: np.ndarray, policy: dict, step: dict | None) -> dict:
    n = len(y)
    x = np.linspace(0., 1., n)
    edge = min(policy['minimum_segment'], max(1, n//4))
    low, high = np.clip([y[:edge].mean(), y[-edge:].mean()], .01, .99)
    starts = [(low, high, a, b) for a, b in policy['ramp_starts']]
    if step is not None:
        center = step['boundaries'][1]/n
        levels = np.clip(step['levels'], .01, .99)
        starts.append((*levels, max(0., center-.05), min(1., center+.05)))
    minimum_width = 1/(n-1)
    constraint = {'type':'ineq', 'fun':lambda z:z[3]-z[2]-minimum_width,
                  'jac':lambda z:np.array([0., 0., -1., 1.])}
    epsilon = policy['probability_epsilon']
    bounds = [(epsilon, 1-epsilon)]*2 + [(0., 1.)]*2
    fits = []
    for initial in starts:
        result = minimize(_ramp_loss, initial, args=(y, x), jac=True, method='SLSQP',
                          bounds=bounds, constraints=[constraint],
                          options={'maxiter':policy['ramp_iterations'], 'ftol':1e-8})
        if (np.isfinite(result.fun) and np.isfinite(result.x).all()
                and result.x[3]-result.x[2] >= minimum_width-1e-8):
            fits.append(result)
    if not fits:
        return dict(family='ramp', parameters=4, converged=False, nll=None)
    best = min(fits, key=lambda fit:fit.fun)
    low, high, start, end = map(float, best.x)
    return dict(family='ramp', parameters=4, converged=bool(best.success), nll=float(best.fun),
                levels=[low, high], start=start*(n-1), end=end*(n-1),
                starts_attempted=len(starts), starts_converged=sum(bool(f.success) for f in fits))


def fitted_probability(fit: dict, n: int) -> np.ndarray:
    """Reconstruct the saved descriptive fit on observed trials only."""
    if fit['family']=='ramp':
        low, high = fit['levels']
        return low+(high-low)*np.clip((np.arange(n)-fit['start'])/(fit['end']-fit['start']), 0., 1.)
    p = np.empty(n)
    for a, b, level in zip(fit['boundaries'][:-1], fit['boundaries'][1:], fit['levels']):
        p[a:b] = level
    return p


def select_comparison_set(fits: list[dict], policy: dict) -> tuple[list[dict], dict]:
    """Require additional evidence before admitting multi-stage competitors."""
    eligible = [f for f in fits if f['amplitude_eligible'] and f['bic'] is not None]
    audit = dict(selection_rule=policy.get('selection_rule', 'flat_class_comparison'))
    if audit['selection_rule']!='hierarchical_single_then_multiple':
        return eligible, audit
    single = [f for f in eligible if f['family'] in ['step', 'ramp']]
    multiple = [f for f in eligible if f['family']=='stages']
    single_best = min(single, key=lambda f:f['bic']) if single else None
    multiple_best = min(multiple, key=lambda f:f['bic']) if multiple else None
    gain = (single_best['bic']-multiple_best['bic']
            if single_best is not None and multiple_best is not None else None)
    admitted = bool(multiple_best is not None and
                    (single_best is None or gain>=policy['multiple_bic_gain']))
    audit.update(multiple_admitted=admitted, multiple_gain_over_single=gain,
                 single_best_bic=None if single_best is None else single_best['bic'],
                 multiple_best_bic=None if multiple_best is None else multiple_best['bic'])
    return (eligible if admitted else single), audit


def nearest_same_label(distances: np.ndarray, labels: list[str], observed_label: str) -> tuple[int | None, int]:
    """Return the closest eligible whole path; an empty category stays empty."""
    d = np.asarray(distances, dtype=float)
    if d.ndim!=1 or len(labels)!=len(d) or not np.isfinite(d).all():
        raise ValueError('Need aligned finite distances and labels')
    indices = np.flatnonzero(np.asarray(labels)==observed_label)
    return (int(indices[np.argmin(d[indices])]), len(indices)) if len(indices) else (None, 0)


def classify_path(correct: np.ndarray, policy: dict, *, fitted_models: list[dict] | None = None) -> dict:
    """Classify one binary path and retain fits, uncertainty and observations."""
    y = np.asarray(correct, dtype=float)
    if y.ndim != 1 or len(y)<2*policy['minimum_segment'] or not np.isin(y, [0, 1]).all():
        raise ValueError('Need a binary path of at least two minimum-length segments')
    n = len(y)
    if fitted_models is None:
        fits = _piecewise_fits(y, policy['minimum_segment'], policy['maximum_segments'])
        step = next((f for f in fits if f['family']=='step'), None)
        fits.append(_ramp_fit(y, policy, step))
    else:
        # The runner verifies outcome identity and every fit-setting value.
        fits = deepcopy(fitted_models)
    flat = fits[0]
    for fit in fits:
        fit['bic'] = None if fit['nll'] is None else 2*fit['nll']+fit['parameters']*np.log(n)
        levels = fit.get('levels', [])
        changes = np.diff(levels)
        fit['amplitude_eligible'] = bool(len(changes) and np.all(np.abs(changes)>=policy['minimum_change']))
        fit['class'] = ('no_clear_change' if fit['family']=='constant' else
                        'multiple_changes' if fit['family']=='stages' else
                        'decreasing' if len(changes) and changes[0]<0 else
                        'abrupt_improvement' if fit['family']=='step' else 'gradual_improvement')
    eligible, selection_audit = select_comparison_set(fits, policy)
    ranked = sorted(eligible, key=lambda f:f['bic'])
    best = ranked[0] if ranked else flat
    gain = float(flat['bic']-best['bic'])
    by_class = {}
    for fit in ranked:
        by_class.setdefault(fit['class'], fit)
    alternatives = [f for label, f in by_class.items() if label!=best['class']]
    gap = float(min(f['bic'] for f in alternatives)-best['bic']) if alternatives else None
    label, reason = 'no_clear_change', 'change_not_detected'
    selected = flat
    if gain >= policy['change_bic_gain']:
        selected = best
        if not best['converged']:
            label, reason = 'shape_undetermined', 'best_fit_not_converged'
        elif gap is not None and gap < policy['class_bic_gap']:
            label, reason = 'shape_undetermined', 'competing_shapes_close'
        else:
            label, reason = best['class'], 'best_shape_separated'
    elif (selection_audit.get('multiple_best_bic') is not None and
          flat['bic']-selection_audit['multiple_best_bic']>=policy['change_bic_gain']):
        label, reason = 'shape_undetermined', 'change_structure_unresolved'
    # A failed ramp that might improve on the winner prevents a definite label.
    ramp = fits[-1]
    if ramp['nll'] is None or (not ramp['converged'] and ramp['bic'] <= selected['bic']+policy['class_bic_gap']):
        label, reason = 'shape_undetermined', 'ramp_optimization_unresolved'
    p = fitted_probability(selected, n)
    residual_width = policy['display_block']
    starts = np.arange(0, n, residual_width)
    sizes = np.minimum(residual_width, n-starts)
    observed = np.array([y[a:a+residual_width].mean() for a in starts])
    expected = np.array([p[a:a+residual_width].mean() for a in starts])
    dispersion = float(np.mean(sizes*(observed-expected)**2/np.maximum(expected*(1-expected), .02)))
    roughness = {}
    for width in policy['roughness_windows']:
        blocks = np.array([y[a:a+width].mean() for a in range(0, n, width)])
        roughness[str(width)] = float(np.sqrt(np.mean(np.diff(blocks)**2))) if len(blocks)>1 else None
    center = endpoint = None
    if selected['family']=='step':
        center = endpoint = float(selected['boundaries'][1])
    elif selected['family']=='ramp':
        center = (selected['start']+selected['end'])/2
        endpoint = selected['end']
    begin = (int(endpoint) if selected['family']=='step' else int(np.floor(endpoint))+1) if endpoint is not None else n
    followup = max(0, n-begin) if endpoint is not None else None
    followup_level = float(y[begin:begin+policy['followup_window']].mean()) if followup else None
    return dict(version=policy['version'], label=label, reason=reason, trials=n,
                outcome_sha256=hashlib.sha256(y.astype('uint8').tobytes()).hexdigest(),
                mean_accuracy=float(y.mean()), final64_accuracy=float(y[-min(64,n):].mean()),
                best_family=selected['family'], best_supported_class=best['class'],
                selected_fit_index=fits.index(selected), selection_audit=selection_audit,
                change_bic_gain=gain, class_bic_gap=gap, competing_classes=list(by_class),
                change_center=center, change_end=endpoint, followup_trials=followup,
                followup_accuracy=followup_level, followup_accuracy_trials=min(followup,policy['followup_window']) if followup else 0,
                observation_limited=bool(endpoint is not None and followup<policy['followup_window']),
                dispersion=dispersion, roughness=roughness, fits=fits)

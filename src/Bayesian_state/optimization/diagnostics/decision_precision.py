"""Finite-particle numerical selection diagnostics, not parameter uncertainty."""
from __future__ import annotations
import numpy as np

def mixture_nll(probabilities: np.ndarray, observed: np.ndarray, mask: np.ndarray) -> float:
    """Average seed probabilities before the log, on the original score mask."""
    mean = probabilities.mean(axis=0)
    selected = mean[np.arange(len(observed)), observed]
    return float(-np.log(np.clip(selected[mask], 1e-12, 1.)).mean())


def decision_diagnostics(arrays: dict[str, dict], selected: str, tolerance: float,
                         alpha: float, replicates: int, seed: int) -> dict:
    """Upper numerical regret for a PRESELECTED point versus the whole bank.

    Resample complete PF seeds, preserving the temporal covariance and common
    random numbers across candidates. The maximum is formed inside each
    bootstrap draw: poor-vs-poor differences are never a stopping requirement.
    """
    if selected not in arrays or not 0 < alpha < 1 or tolerance <= 0:
        raise ValueError('Invalid selected candidate, alpha or tolerance')
    ids = sorted(arrays); first = arrays[ids[0]]
    y, mask, seeds = first['observed'], first['mask'], first['seeds']
    if len(seeds) < 2 or len(np.unique(seeds)) != len(seeds) or not mask.any():
        raise ValueError('Need distinct seeds and at least one scored trial')
    probabilities = []
    for pid in ids:
        a = arrays[pid]
        for key in ('observed', 'mask', 'seeds'):
            np.testing.assert_array_equal(a[key], first[key])
        p = a['probabilities']
        if p.ndim != 3 or p.shape[:2] != (len(seeds), len(y)) or not np.isfinite(p).all() or (p < 0).any():
            raise ValueError('Invalid probability array')
        np.testing.assert_allclose(p.sum(axis=-1), 1., atol=1e-10)
        probabilities.append(p[:, np.arange(len(y)), y][:, mask])
    weights = np.random.default_rng(seed).multinomial(
        len(seeds), np.full(len(seeds), 1/len(seeds)), size=replicates)/len(seeds)
    boot = np.stack([-np.log(np.clip(weights@q, 1e-12, 1.)).mean(axis=1) for q in probabilities])
    scores = np.array([-np.log(np.clip(q.mean(axis=0), 1e-12, 1.)).mean() for q in probabilities])
    index = ids.index(selected)
    differences = boot[index]-boot
    regret = differences.max(axis=0)  # Includes self, so regret >= 0.
    lower, upper = np.quantile(regret, [alpha, 1-alpha])
    pairs = {}
    for j, pid in enumerate(ids):
        if pid != selected:
            pairs[pid] = {'selected_minus_candidate': float(scores[index]-scores[j]),
                          'interval95': np.quantile(differences[j], [.025, .975]).tolist()}
    old_width = max((float(np.diff(np.quantile(boot[i]-boot[j], [.025, .975]))[0]/2)
                     for i in range(len(ids)) for j in range(i)), default=0.)
    # Old rule is reported for comparison only; this does not rewrite old reports.
    old_winner = int(scores.argmin())
    old_regret = float(np.quantile(boot[old_winner]-boot.min(axis=0), .95))
    status = ('acceptable_within_bank' if upper <= tolerance else
              'selected_point_inferior' if lower > tolerance else 'unresolved')
    return {'selected': selected, 'scores': dict(zip(ids, scores.tolist())),
            'point_regret': float(scores[index]-scores.min()),
            'regret_lower': float(lower), 'regret_upper': float(upper),
            'alpha': alpha, 'tolerance': tolerance, 'status': status,
            'selected_minus_candidates': pairs,
            'old_all_pair_halfwidth': old_width, 'old_winner_regret95': old_regret,
            'old_score_gate_pass': old_width <= tolerance and old_regret <= tolerance,
            'scope': 'Approximate numerical-seed bootstrap within a fixed bank at finite R. Not parameter uncertainty or integration convergence.'}

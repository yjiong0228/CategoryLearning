"""Independently audit saved belief-comparison predictions and summarize them.

Use --study COMPLETED_NUMERICAL_STUDY --output NEW. This entry reads frozen
nominees and saved numerical arrays; it never fits parameters or reruns paths.
All labels are checked against the original prepared data before scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ...metrics.autonomous_fit import behavior_features, energy_u


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arrays(path: Path) -> dict:
    meta = read(path.with_suffix('.json'))
    if digest(path.with_suffix('.npz')) != meta['predictions_sha256']:
        raise ValueError('Saved prediction hash differs')
    with np.load(path.with_suffix('.npz')) as z:
        return {k: z[k] for k in z.files}


def nll(probability: np.ndarray) -> float:
    if probability.ndim != 2 or not np.isfinite(probability).all() or np.any(probability <= 0):
        raise ValueError('Require finite positive repeat-by-trial probabilities')
    return float(-np.log(probability.mean(axis=0)).mean())


def jackknife_se(deleted: np.ndarray) -> float:
    n = len(deleted)
    return float(np.sqrt((n-1)/n * np.square(deleted-deleted.mean()).sum()))


def load_case(study: Path, nominee: dict, jobs: list[dict]) -> dict:
    sid, arm = nominee['subject'], nominee['arm']
    matching = [j for j in jobs if j['subject'] == sid and j['model'] == arm]
    folder = Path(matching[0]['folder'])
    meta = read(folder/'prepared.json')
    parts = []
    for name in ('training.npz', 'sealed_test.npz'):
        if digest(folder/name) != meta['array_sha256'][name]:
            raise ValueError('Prepared behavioral data changed')
        with np.load(folder/name) as z:
            parts.append({k: z[k] for k in z.files})
    observed = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    prediction = arrays(study/'evaluate'/f'subject_{sid}'/arm/nominee['candidate']['id'])
    p = prediction['probabilities']
    if p.ndim != 3 or p.shape[1] != meta['trials'] or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError('Invalid full-sequence probability matrix')
    np.testing.assert_allclose(p.sum(axis=2), 1., atol=1e-12, rtol=0)
    for key in ('choices', 'categories'):
        # The shared training aggregator concatenates all arrays over repeats;
        # evaluation labels consequently repeat once per single-filter chunk.
        value = prediction[key]
        repetitions, remainder = divmod(len(value), len(observed[key]))
        if remainder or repetitions not in (1, len(p)):
            raise ValueError('Unexpected evaluation label dimensions')
        np.testing.assert_array_equal(value, np.tile(observed[key], repetitions))
    index = np.arange(meta['trials'])
    np.testing.assert_allclose(prediction['observed_probability'], p[:, index, observed['choices']-1])
    np.testing.assert_allclose(prediction['correct_probability'], p[:, index, observed['categories']-1])
    autonomous_parts = [arrays(Path(j['target'])) for j in matching]
    for part, job in zip(autonomous_parts, matching):
        np.testing.assert_array_equal(part['categories'], observed['categories'])
        np.testing.assert_array_equal(part['seeds'], job['seeds'])
    autonomous = {k: np.concatenate([a[k] for a in autonomous_parts])
                  for k in autonomous_parts[0] if k != 'categories'}
    if len(set(autonomous['seeds'].tolist())) != len(autonomous['seeds']):
        raise ValueError('Autonomous paths must have distinct seeds')
    np.testing.assert_allclose(autonomous['probabilities'].sum(axis=2), 1., atol=1e-12, rtol=0)
    np.testing.assert_array_equal(autonomous['feedback'] == 1,
                                  autonomous['choices'] == observed['categories'][None, :])
    return dict(meta=meta, observed=observed, prediction=prediction, autonomous=autonomous)


def run(study: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new summary directory')
    nomination = read(study/'nominees.json')
    if nomination['status'] != 'frozen_before_test' or nomination['smoke']:
        raise ValueError('Require frozen scientific pilot estimates')
    jobs = read(study/'autonomous_jobs.json')['jobs']
    rows, cases, energies = [], {}, {}
    for nominee in nomination['nominees']:
        sid, arm = nominee['subject'], nominee['arm']
        case = cases[sid, arm] = load_case(study, nominee, jobs)
        meta, obs, pred, auto = [case[k] for k in ('meta', 'observed', 'prediction', 'autonomous')]
        cut = meta['training_trials']
        later = slice(cut, None)
        p = pred['observed_probability'][:, later]
        n_categories = 2 if meta['condition'] == 1 else 4
        features = behavior_features(auto['choices'][:, later], obs['categories'][later], n_categories=n_categories)
        empirical = behavior_features(obs['choices'][None, later], obs['categories'][later], n_categories=n_categories)[0]
        energy = energy_u(features, empirical)
        energies[sid, arm] = (features, empirical)
        rows.append(dict(subject=sid, arm=arm, condition=meta['condition'], candidate=nominee['candidate']['id'],
            training_trials=cut, test_trials=meta['test_trials'], paths=len(auto['choices']),
            training_nll=nll(pred['observed_probability'][:, :cut]), test_nll=nll(p),
            test_nll_mcse=jackknife_se(np.array([nll(np.delete(p, i, axis=0)) for i in range(len(p))])),
            observed_late_accuracy=float((obs['choices'][later] == obs['categories'][later]).mean()),
            generated_late_accuracy=float((auto['choices'][:, later] == obs['categories'][None, later]).mean()),
            late_energy=energy['value'], late_energy_mcse=energy['mc_se'],
            late_target_present=float(auto['present'][:, later].mean()),
            late_target_contribution=float(auto['contribution'][:, later].mean())))
    comparisons = []
    table = pd.DataFrame(rows)
    comparisons_to_run = [(sid, arm) for sid, arm in sorted(cases) if arm != 'SUPPORT']
    for sid, arm in comparisons_to_run:
        baseline, belief = cases[sid, 'SUPPORT'], cases[sid, arm]
        cut = baseline['meta']['training_trials']
        np.testing.assert_array_equal(baseline['prediction']['seeds'], belief['prediction']['seeds'])
        np.testing.assert_array_equal(baseline['autonomous']['seeds'], belief['autonomous']['seeds'])
        p, q = [c['prediction']['observed_probability'][:, cut:] for c in (baseline, belief)]
        deleted = np.array([nll(np.delete(q, i, axis=0))-nll(np.delete(p, i, axis=0)) for i in range(len(p))])
        a, empirical = energies[sid, 'SUPPORT']
        b, _ = energies[sid, arm]
        energy_deleted = np.array([energy_u(np.delete(b, i, axis=0), empirical)['value']
                                  - energy_u(np.delete(a, i, axis=0), empirical)['value'] for i in range(len(a))])
        comparisons.append(dict(subject=sid, arm=arm, test_nll_difference=nll(q)-nll(p),
            test_nll_difference_mcse=jackknife_se(deleted),
            late_energy_difference=energy_u(b, empirical)['value']-energy_u(a, empirical)['value'],
            late_energy_difference_mcse=jackknife_se(energy_deleted)))
    output.mkdir(parents=True)
    table.to_csv(output/'summary.csv', index=False)
    pd.DataFrame(comparisons).to_csv(output/'comparison.csv', index=False)
    with (output/'completion.json').open('x') as stream:
        json.dump(dict(status='complete', cases=len(rows), source_study=str(study.resolve()),
            nominee_sha256=digest(study/'nominees.json'), summary_sha256=digest(output/'summary.csv'),
            source_sha256=digest(Path(__file__)), scope='development pilot; Monte Carlo errors only',
            validated='data hashes, identity/order, pre-choice probabilities, normalized matrices, complete autonomous paths'),
            stream, indent=2)
    print(table.to_string(index=False))
    print(pd.DataFrame(comparisons).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.study, args.output)

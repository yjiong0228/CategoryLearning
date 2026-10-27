"""Retain target-belief/precision distributions and whole-history functionals.

Correct-rule belief includes zero mass when the rule is absent. Precision
histograms retain joint presence mass and are conditioned only for display.
Whole-history occupancy uses coherent ancestors, never independent trial draws.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .model_1024_individual_states import read, write, digest
from ...evaluation.independent_particle_systems import evidence_weights


def weighted_histogram(values: np.ndarray, weights: np.ndarray, edges: np.ndarray,
                       mask: np.ndarray | None = None) -> np.ndarray:
    """Trial-wise weighted bins, with a closed final endpoint."""
    values, weights, edges = np.asarray(values), np.asarray(weights), np.asarray(edges)
    if (values.ndim != 2 or values.shape != weights.shape or np.any(weights < 0)
            or not np.isfinite(values).all() or not np.isfinite(weights).all()
            or np.any(np.diff(edges) <= 0) or values.min() < edges[0]-1e-10
            or values.max() > edges[-1]+1e-10):
        raise ValueError('Invalid histogram values, weights or support')
    bins = len(edges)-1
    index = np.clip(np.searchsorted(edges, values, side='right')-1, 0, bins-1)
    index += np.arange(len(values))[:, None]*bins
    if mask is not None:
        weights = weights*np.asarray(mask, bool)
    return np.bincount(index.ravel(), weights=weights.ravel(), minlength=len(values)*bins).reshape(-1, bins)


def weighted_quantiles(values: np.ndarray, weights: np.ndarray, quantiles=(.025, .5, .975)) -> np.ndarray:
    order = np.argsort(values); x = values[order]; w = weights[order]
    if np.any(w < 0) or not np.isclose(w.sum(), 1., atol=1e-10, rtol=0):
        raise ValueError('Require normalized path weights')
    return x[np.minimum(np.searchsorted(w.cumsum(), quantiles), len(x)-1)]


def run(source: Path, summary: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new distribution directory')
    output.mkdir(parents=True)
    cases = {c['id']: c for c in read(source/'protocol.json')['cases']}
    selected = [s for s in read(summary/'selected.json') if 'file' in s]
    features = []
    for s in selected:
        case = cases[s['case']]
        if len(case['target_rules']) != 1:
            raise ValueError('Precision distribution requires one registered target rule')
        with np.load(s['file']) as z:
            logz = z['system_log_evidence']
            expected_means = {key: float(z['full_'+field].mean()) for key, field in [
                ('present_fraction', 'target_present'), ('mean_belief', 'target_belief'),
                ('expected_executed_fraction', 'target_execution')]}
        jobs = [read(source/'jobs'/(name+'.json')) for name in s['jobs']]
        n = case['trials']; belief_edges = np.linspace(0, 1, 21); precision_edges = np.linspace(0, 25, 26)
        result = dict(belief_edges=belief_edges, precision_edges=precision_edges)
        for mode in ('filtered', 'full'):
            result[mode+'_belief_histogram'] = np.zeros((n, 20))
            result[mode+'_precision_joint_histogram'] = np.zeros((n, 25))
        filtering = evidence_weights(logz); terminal = evidence_weights(logz[:, -1])
        path_weights, occupancy, average_belief, execution_mean, execution_noise = [], [], [], [], []
        for r, job in enumerate(jobs):
            file = Path(job['individual_target']).with_suffix('.npz')
            if digest(file) != read(file.with_suffix('.json'))['array_sha256']:
                raise ValueError('State archive changed')
            with np.load(file) as z:
                target = np.isin(z['particle_rule'], case['target_rules'])
                present = target.any(axis=2)
                belief = (target*z['particle_belief']).sum(axis=2)
                precision = (target*z['particle_precision']).sum(axis=2)
                execution = (target*z['particle_execution']).sum(axis=2)
                filtered_weights = z['filtered_weights']; full_weights = z['full_weights']
                indices = z['full_indices']
            for mode, w, outer in [('filtered', filtered_weights, filtering[r]),
                                   ('full', full_weights, np.full(n, terminal[r]))]:
                result[mode+'_belief_histogram'] += weighted_histogram(belief, w, belief_edges)*outer[:, None]
                result[mode+'_precision_joint_histogram'] += weighted_histogram(precision, w, precision_edges, present)*outer[:, None]
            select = (np.arange(n)[:, None], indices.T)
            occupancy.append(present[select].mean(axis=0))
            average_belief.append(belief[select].mean(axis=0))
            q = execution[select]
            execution_mean.append(q.mean(axis=0))
            # At zero inertia, execution is independent across trials given
            # the other-state history. At positive inertia q is one-hot.
            execution_noise.append((q*(1-q)).sum(axis=0)/n**2)
            path_weights.append(filtered_weights[-1]*terminal[r])
        w = np.concatenate(path_weights)
        for key, values in [('present_fraction', occupancy), ('mean_belief', average_belief),
                            ('expected_executed_fraction', execution_mean)]:
            x = np.concatenate(values); mean = float(w@x); variance = float(w@np.square(x-mean))
            np.testing.assert_allclose(mean, expected_means[key], atol=3e-12, rtol=0)
            if key=='expected_executed_fraction':
                variance += float(w@np.concatenate(execution_noise))
                lo, median, hi = (np.nan,)*3
            else:
                lo, median, hi = weighted_quantiles(x, w)
            features.append(dict(subject=case['subject'], candidate=case['candidate']['id'], functional=key,
                mean=mean, sd=float(np.sqrt(variance)), q025=lo, median=median, q975=hi,
                numerical_status=s['status'], parameter_status=case['parameter_status']))
            result['path_'+key] = x
        result['path_weights'] = w
        for mode in ('filtered', 'full'):
            np.testing.assert_allclose(result[mode+'_belief_histogram'].sum(axis=1), 1., atol=3e-12, rtol=0)
            present_mass = result[mode+'_precision_joint_histogram'].sum(axis=1)
            with np.load(s['file']) as z:
                np.testing.assert_allclose(present_mass, z[mode+'_target_present'], atol=3e-12, rtol=0)
            result[mode+'_precision_conditional_histogram'] = np.divide(
                result[mode+'_precision_joint_histogram'], present_mass[:, None],
                out=np.full_like(result[mode+'_precision_joint_histogram'], np.nan), where=present_mass[:, None]>0)
        with (output/(case['id']+'.npz')).open('xb') as stream:
            np.savez_compressed(stream, **result)
        print(f'Distributions {case["id"]}', flush=True)
    pd.DataFrame(features).to_csv(output/'whole_history_functionals.csv', index=False)
    write(output/'completion.json', dict(status='complete', candidates=len(selected),
        source_sha256=digest(Path(__file__)), interval_status='conditional model quantiles; no calibrated coverage claim',
        zero_inertia_execution='mean and total variance integrate conditional execution randomness'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'summary', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); run(args.source, args.summary, args.output)

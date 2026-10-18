"""Audit actual computation under the shared, deduplicated search policy.

The exchange bank is the union of each arm's leading candidates and a common
reference. Its cardinality depends on overlap. Reconstruct that union before
allowing an exchange-only difference in actual filter work between two fits.
All other stage budgets remain equal. This module does not change fitting.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .model_0923_h_validation import digest, load, nll
from ..utils.seeding import stable_seed

ROOT = Path(__file__).resolve().parents[3]


def expected_exchange(initial: dict, screened: dict, refined: dict, count: int) -> dict:
    """Reconstruct the declared union independently from saved stage rankings."""
    if count < 1 or set(initial) != set(screened) or set(initial) != set(refined):
        raise ValueError('Need the same complete arms and a positive exchange count')
    anchors = [rows[0] for rows in initial.values()]
    if any(row['id'] != anchors[0]['id'] or row['hyperparams'] != anchors[0]['hyperparams'] for row in anchors):
        raise ValueError('Arms must share the original reference point')
    bank = {}
    for arm in initial:
        # Refinement replaces the earlier score when an elite is rescored.
        unique = {row['id']: row for row in screened[arm]+refined[arm]}
        leading = sorted(unique.values(), key=lambda row:(row['mean_nll'], row['id']))
        if len(leading) < count:
            raise ValueError('Insufficient leading candidates')
        for row in [initial[arm][0], *leading[:count]]:
            if row['id'] in bank and bank[row['id']] != row['hyperparams']:
                raise ValueError('Candidate identity changes parameters')
            bank[row['id']] = row['hyperparams']
    return bank


def audit_exchange(fit: Path, subjects: list[int]) -> tuple[pd.DataFrame, dict[str, str]]:
    """Verify completed source scores, exchange identities, seeds and work units."""
    protocol = load(fit/'protocol.json')
    settings = protocol['settings']
    if protocol['smoke'] or not set(subjects) <= set(settings['subjects']):
        raise ValueError('Need declared scientific fit subjects')
    inputs = {}
    def bind(path: Path) -> None:
        inputs[str(path.resolve())] = digest(path)
    bind(fit/'protocol.json')
    for name, expected in protocol['source_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Executing fitting source changed')
        bind(ROOT/name)
    stages = {}
    for stage in ('screen', 'refine', 'exchange'):
        path = fit/(stage+'_completion.json')
        completed = load(path); bind(path)
        if completed['status'] != 'complete' or completed['stage'] != stage:
            raise ValueError('Exchange audit requires completed preceding stages')
        groups = {(g['subject'], g['model']):g['rows'] for g in completed['groups']}
        if len(groups) != len(completed['groups']):
            raise ValueError('Duplicate stage group')
        stages[stage] = groups
    rows, exchange_probabilities = [], {}
    for subject in subjects:
        arms = list(settings['arms'])
        initial = {g['model']:g['rows'] for g in protocol['initial_banks'] if g['subject'] == subject}
        if set(initial) != set(arms):
            raise ValueError('Incomplete shared initial banks')
        folder = ROOT/settings['data']/f'subject_{subject}'
        meta = load(folder/'prepared.json'); bind(folder/'prepared.json')
        trials = meta['training_trials']
        if meta['subject'] != subject:
            raise ValueError('Wrong prepared participant')
        for stage, groups in stages.items():
            budget = settings['budgets'][stage]
            seeds = [stable_seed(dict(protocol=settings['protocol'], base=settings['seed'],
                stage=stage, subject=subject, repeat=i)) for i in range(budget['seeds'])]
            for arm in arms:
                group = groups.get((subject,arm), [])
                if not group or len({r['id'] for r in group}) != len(group):
                    raise ValueError('Missing or duplicate candidate scores')
                for row in group:
                    path = fit/stage/f'subject_{subject}'/arm/(row['id']+'.json')
                    record = load(path); arrays = path.with_suffix('.npz')
                    if (Path(row['score_file']).resolve() != path.resolve()
                            or record['candidate']['id'] != row['id']
                            or record['candidate']['hyperparams'] != row['hyperparams']
                            or record['subject'] != subject or record['model'] != arm
                            or record['particles'] != budget['particles'] or record['trials'] != trials
                            or digest(arrays) != record['predictions_sha256']):
                        raise ValueError('Stage score identity, budget or arrays changed')
                    with np.load(arrays) as z:
                        p = z['observed_probability']
                        np.testing.assert_array_equal(z['seeds'], seeds)
                    np.testing.assert_array_equal(record['seeds'], seeds)
                    if p.shape != (len(seeds), trials) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
                        raise ValueError('Invalid full-prefix choice probabilities')
                    np.testing.assert_allclose(nll(p), [row['mean_nll'], record['mean_nll']], rtol=0, atol=1e-12)
                    if stage == 'exchange':
                        exchange_probabilities[subject,arm,row['id']] = p
                    bind(path); bind(arrays)
        expected = expected_exchange(initial,
            {arm:stages['screen'][subject,arm] for arm in arms},
            {arm:stages['refine'][subject,arm] for arm in arms}, settings['exchange_per_arm'])
        for arm in arms:
            observed = {r['id']:r['hyperparams'] for r in stages['exchange'][subject,arm]}
            if observed != expected:
                raise ValueError('Exchange bank differs from the declared deduplicated union')
            budget = settings['budgets']['exchange']
            seeds = [stable_seed(dict(protocol=settings['protocol'], base=settings['seed'],
                stage='exchange', subject=subject, repeat=i)) for i in range(budget['seeds'])]
            jobs = sorted((fit/'jobs').glob(f'exchange_S{subject}_{arm}_*.json'))
            actual_pairs = []
            for file in jobs:
                job = load(file); bind(file)
                point = job['candidate']
                if (job['phase'] != 'training' or job['model'] != arm or job['smoke']
                        or point['id'] not in expected or point['hyperparams'] != expected[point['id']]
                        or job['particles'] != budget['particles'] or len(job['seeds']) != 1
                        or job['source_sha256'] != protocol['source_sha256']
                        or Path(job['folder']).resolve() != folder.resolve()):
                    raise ValueError('Exchange worker changed the declared candidate or budget')
                actual_pairs.append((point['id'], job['seeds'][0]))
                result = Path(job['target']).with_suffix('.json')
                record = load(result); arrays = result.with_suffix('.npz')
                if (record['candidate'] != point or record['seeds'] != job['seeds']
                        or record['subject'] != subject or record['model'] != arm
                        or record['trials'] != trials or record['particles'] != job['particles']
                        or any(p['num_threads'] != 1 for p in record['numeric_pools'])
                        or digest(arrays) != record['predictions_sha256']):
                    raise ValueError('Exchange worker result or thread budget changed')
                with np.load(arrays) as z:
                    np.testing.assert_array_equal(z['seeds'], job['seeds'])
                    np.testing.assert_array_equal(z['observed_probability'],
                        exchange_probabilities[subject,arm,point['id']][[seeds.index(job['seeds'][0])]])
                    np.testing.assert_allclose(nll(z['observed_probability']), record['mean_nll'], rtol=0, atol=1e-12)
                bind(result); bind(arrays)
                if settings.get('structural_intervention') == 'retained_rule_precision':
                    sidecar = result.with_suffix('.precision.json')
                    execution = load(sidecar); bind(sidecar)
                    if (job.get('retain_precision') is not True
                            or execution.get('retained_precision') is not True
                            or execution.get('structural_intervention') != 'retained_rule_precision'):
                        raise ValueError('Training lacks the declared precision-retention execution')
                elif job.get('retain_precision', False):
                    raise ValueError('Original fitting worker unexpectedly retains precision')
            required_pairs = {(point,seed) for point in expected for seed in seeds}
            if len(actual_pairs) != len(required_pairs) or set(actual_pairs) != required_pairs:
                raise ValueError('Missing, repeated or unmatched exchange filter runs')
            rows.append(dict(subject=subject, model=arm, stage='exchange', scored_points=len(expected),
                maximum_points=1+len(arms)*settings['exchange_per_arm'],
                filter_runs=len(required_pairs), state_updates=len(required_pairs)*budget['particles']*trials))
    return pd.DataFrame(rows), inputs


def budget_contrasts(old: pd.DataFrame, new: pd.DataFrame,
                     old_exchange: pd.DataFrame, new_exchange: pd.DataFrame) -> pd.DataFrame:
    """Permit only the independently reconstructed exchange-cardinality change."""
    keys = ['subject','model','stage']; values = ['filter_runs','state_updates']
    left, right = [x.set_index(keys)[values].sort_index() for x in (old,new)]
    if not left.index.is_unique or not right.index.is_unique:
        raise ValueError('Duplicate stage budget rows')
    pd.testing.assert_index_equal(left.index, right.index)
    ordinary = left.index.get_level_values('stage') != 'exchange'
    pd.testing.assert_frame_equal(left.loc[ordinary], right.loc[ordinary])
    for actual, expected in ((left,old_exchange), (right,new_exchange)):
        pd.testing.assert_frame_equal(actual.loc[~ordinary],
            expected.set_index(keys)[values].sort_index(), check_dtype=False)
    result = left.add_prefix('reset_').join(right.add_prefix('retain_')).reset_index()
    for value in values:
        result['delta_'+value] = result['retain_'+value]-result['reset_'+value]
    result['actual_compute_equal'] = result.delta_state_updates.eq(0) & result.delta_filter_runs.eq(0)
    return result

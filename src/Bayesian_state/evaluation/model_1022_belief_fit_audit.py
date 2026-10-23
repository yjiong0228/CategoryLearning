"""Audit completed stages of a belief comparison without changing nominations.

Use --study FIT --output NEW --stages screen refine_1. Every selected stage
must be complete. Checks cover matched starting/confirmation banks, equal
candidate budgets, data boundaries, source snapshots, independent seed banks,
and scores recomputed from saved probabilities. The first declared fitting
stage must evaluate the complete starting bank with its exact parameter values,
including workflows that start with confirmation. Seed errors describe numerical
uncertainty only; the two-seed screening estimates remain coarse diagnostics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from ..utils.seeding import stable_seed
from ..workflows.analysis.model_1022_belief_summary import read, digest, nll, jackknife_se

ROOT = Path(__file__).resolve().parents[3]


def run(study: Path, output: Path, stages: list[str]) -> None:
    if output.exists():
        raise FileExistsError('Use a new fitting audit directory')
    protocol = read(study / 'protocol.json')
    source_files = [Path(__file__).resolve(),
        ROOT / 'src/Bayesian_state/workflows/analysis/model_1022_belief_summary.py',
        ROOT / 'src/Bayesian_state/utils/seeding.py']
    source_hashes = {str(path.relative_to(ROOT)): digest(path) for path in source_files}
    settings, policy = protocol['settings'], protocol['policy']
    if protocol['smoke']:
        raise ValueError('Scientific fit audit requires full training prefixes')
    expected_order = [s for s in settings['budgets'] if s != 'evaluate']
    if not stages or stages != expected_order[:len(stages)]:
        raise ValueError('Audit stages must form a completed prefix of the declared fitting stages')
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve()
        sha = digest(path)
        if expected is not None and sha != expected:
            raise ValueError('Saved fitting artifact changed: ' + str(path))
        inputs[str(path)] = sha

    bind(study / 'protocol.json')
    bind(study / 'starting_banks.json')
    for name, sha in protocol['source_sha256'].items():
        bind(study / 'source' / name, sha)
    starting = read(study / 'starting_banks.json')
    for name, sha in starting['inputs_sha256'].items():
        bind(name, sha)
    initial = {(r['subject'], r['arm']): {p['id']: p['hyperparams'] for p in r['candidates']}
               for r in starting['banks']}
    keys = {(s, a) for s in settings['subjects'] for a in policy['variants']}
    if set(initial) != keys or len(starting['banks']) != len(keys):
        raise ValueError('Starting banks omit a declared participant/model case')
    declared_count = policy.get('initial_candidates', policy.get('expected_candidates_per_subject'))
    for group in starting['banks']:
        bank = initial[group['subject'], group['arm']]
        if (len(bank) != len(group['candidates'])
                or (declared_count is not None and len(bank) != declared_count)):
            raise ValueError('Starting candidate count differs from the declared budget')
    seen, seed_banks, points, budgets = {key: set() for key in keys}, {}, [], []
    for stage in stages:
        completion = study / (stage + '_completion.json')
        bind(completion)
        saved = read(completion)
        groups = {(g['subject'], g['model']): g['rows'] for g in saved['groups']}
        if saved['status'] != 'complete' or set(groups) != keys:
            raise ValueError('Require a complete stage for every declared case')
        if saved['points'] != sum(len(rows) for rows in groups.values()):
            raise ValueError('Stage candidate count differs from saved groups')
        for sid in settings['subjects']:
            if stage in ('screen', 'confirmation'):
                banks = [{r['id'] for r in groups[sid, arm]} for arm in policy['variants']]
                if any(bank != banks[0] for bank in banks[1:]):
                    raise ValueError('Shared-stage parameter banks differ across mechanisms')
        for key, rows in groups.items():
            sid, arm = key
            ids = {r['id'] for r in rows}
            if len(ids) != len(rows):
                raise ValueError('A candidate was counted twice within a stage')
            if stage == expected_order[0]:
                actual = {r['id']: r['hyperparams'] for r in rows}
                if actual != initial[key]:
                    raise ValueError('First fitting stage differs from the declared starting bank')
            if stage == 'screen':
                if len(rows) != policy['initial_candidates'] or ids != set(initial[key]):
                    raise ValueError('Initial candidate budget or identity changed')
            elif stage.startswith('refine'):
                if (len(rows) != policy['refinement_candidates'] + policy['elites']
                        or len(ids - seen[key]) != policy['refinement_candidates']):
                    raise ValueError('Adaptive candidate budget changed')
            budget = settings['budgets'][stage]
            expected_seeds = [stable_seed(dict(protocol=settings['protocol'], base=settings['seed'],
                                              stage=stage, subject=sid, repeat=i)) for i in range(budget['seeds'])]
            previous = set().union(*(set(v) for (s, _), v in seed_banks.items() if s == sid))
            if previous.intersection(expected_seeds):
                raise ValueError('Fitting stages reuse a numerical seed bank')
            meta_path = ROOT / settings['data'] / f'subject_{sid}' / 'prepared.json'
            bind(meta_path)
            meta = read(meta_path)
            bind(meta_path.parent / 'training.npz', meta['array_sha256']['training.npz'])
            for row in rows:
                score_file = Path(row['score_file'])
                score = read(score_file)
                bind(score_file)
                bind(score_file.with_suffix('.npz'), score['predictions_sha256'])
                with np.load(score_file.with_suffix('.npz')) as z:
                    p, seeds = z['observed_probability'], z['seeds']
                np.testing.assert_array_equal(seeds, expected_seeds)
                if (p.shape != (budget['seeds'], meta['training_trials'])
                        or score['particles'] != budget['particles'] or score['trials'] != meta['training_trials']
                        or score['subject'] != sid or score['condition'] != meta['condition']
                        or score['candidate']['id'] != row['id']
                        or np.any(p > 1.) or score['candidate']['hyperparams'] != row['hyperparams']):
                    raise ValueError('Scoring budget, parameters or training boundary changed')
                value = nll(p)
                np.testing.assert_allclose([row['mean_nll'], score['mean_nll']], value, atol=1e-14, rtol=0)
                deleted = np.array([nll(np.delete(p, i, 0)) for i in range(len(p))])
                points.append(dict(stage=stage, subject=sid, arm=arm, candidate=row['id'], training_nll=value,
                    training_nll_mcse=jackknife_se(deleted), particles=budget['particles'], seeds=len(p),
                    individual_filter_nll_mean=float(-np.log(p).mean()),
                    individual_filter_nll_range=float(np.ptp(-np.log(p).mean(1))),
                    probability_pooling_gain=float(-np.log(p).mean())-value))
            seen[key].update(ids)
            budgets.append(dict(stage=stage, subject=sid, arm=arm, candidates=len(rows),
                particles=budget['particles'], seeds=budget['seeds'], training_trials=meta['training_trials']))
        for sid in settings['subjects']:
            seed_banks[sid, stage] = [stable_seed(dict(protocol=settings['protocol'], base=settings['seed'],
                stage=stage, subject=sid, repeat=i)) for i in range(settings['budgets'][stage]['seeds'])]
    output.mkdir(parents=True)
    for name, sha in source_hashes.items():
        if digest(ROOT / name) != sha:
            raise ValueError('Fitting audit source changed during evaluation')
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    pd.DataFrame(points).to_csv(output / 'candidate_scores.csv', index=False)
    pd.DataFrame(budgets).to_csv(output / 'budgets.csv', index=False)
    with (output / 'completion.json').open('x') as stream:
        json.dump(dict(status='complete', stages=stages, study=str(study.resolve()), inputs_sha256=inputs,
            source_sha256=source_hashes, cases=len(keys), candidate_evaluations=len(points),
            nominations_changed=False, later_choices_read=False,
            scope='completed fitting stages only; seed errors exclude parameter and participant uncertainty'), stream, indent=2)
    print(f'Validated {len(points)} candidate evaluations across {len(keys)} cases and {len(stages)} stages')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stages', nargs='+', required=True)
    args = parser.parse_args()
    run(args.study, args.output, args.stages)

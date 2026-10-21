"""Compare a completed entry diagnostic with its frozen COMPACT point.

Use --source PROBE --generation ORIGINAL_GENERATION --output NEW. Shared
scoring routines preserve the original training boundary and paired seeds.
The diagnostic changes entry support or precision at fixed parameters; it is a development
comparison. This summary performs no fitting and does not alter primary nominees.
The source protocol's comparison_label is CENTERED (default), PRECISION or
CENTERED_PRECISION. The original centering result keeps its existing labels.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .model_0923_h_validation import block_paths, distribution_scores, nll, jackknife_scores, jackknife_se
from .model_0923_h_generation_uncertainty import path_jackknife
from ..workflows.analysis.model_1021_simplification_retention_probe import load_bank, retained_after_discovery
from ..workflows.analysis.model_0923_h_validation import read, write, digest


def summarize(source: Path, generation: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new centered-entry comparison directory')
    done, protocol = read(source/'completion.json'), read(source/'protocol.json')
    label = protocol.get('comparison_label', 'CENTERED')
    if label not in ('CENTERED', 'PRECISION', 'CENTERED_PRECISION'):
        raise ValueError('Unknown entry diagnostic label')
    if done['status'] != 'complete' or done['smoke']:
        raise ValueError('Require the complete scientific diagnostic')
    fit = Path(protocol['fit'])
    if (digest(fit/'nominees.json') != protocol['nomination_sha256']
            or read(generation/'protocol.json')['nomination_sha256'] != protocol['nomination_sha256']):
        raise ValueError('Primary nomination changed or generation belongs to another fit')
    frozen = read(fit/'nominees.json')
    nominee, = [r for r in frozen['nominees'] if (r['subject'], r['model']) == (205, 'COMPACT')]
    key = nominee['candidate']['id']
    root = Path(__file__).resolve().parents[3]
    folder = root/frozen['settings']['data']/'subject_205'
    meta = read(folder/'prepared.json')
    bindings = {str(p.resolve()): digest(p) for p in (source/'completion.json', source/'protocol.json',
                fit/'nominees.json', generation/'protocol.json', Path(__file__))}
    observed, categories = [], []
    for name in ('training.npz', 'sealed_test.npz'):
        file = folder/name
        if digest(file) != meta['array_sha256'][name]:
            raise ValueError('Human data boundary changed')
        bindings[str(file.resolve())] = digest(file)
        with np.load(file) as z:
            observed.append(z['choices'] == z['categories'])
            categories.append(z['categories'])
    observed, categories = np.concatenate(observed), np.concatenate(categories)
    cut, length = meta['training_trials'], len(observed)
    banks = {
        'COMPACT': (generation/'autonomous/evaluate/subject_205/COMPACT'/(key+'.json'),
                    Path(nominee['candidate']['score_file'])),
        label: (source/'autonomous'/(key+'.json'), source/'training'/(key+'.json')),
    }
    scores, curves, lifetimes, jacks, seeds = [], [], [], {}, {}
    for arm, (gfile, tfile) in banks.items():
        gr, a = load_bank(gfile)
        tr, t = load_bank(tfile)
        for file in (gfile, tfile):
            bindings[str(file.resolve())] = digest(file)
            bindings[str(file.with_suffix('.npz').resolve())] = digest(file.with_suffix('.npz'))
        if any(r['candidate']['hyperparams'] != nominee['candidate']['hyperparams'] for r in (gr, tr)):
            raise ValueError('Fixed parameters changed')
        if tr['particles'] != 512 or tr['trials'] != cut or a['choices'].shape != (128, length):
            raise ValueError('Numerical budget or trial counts changed')
        if t['observed_probability'].shape != (8, cut):
            raise ValueError('Training filters or boundary changed')
        np.testing.assert_array_equal(a['categories'], categories)
        np.testing.assert_allclose(a['probabilities'].sum(-1), 1., atol=1e-12, rtol=0)
        seeds[arm] = a['seeds'], t['seeds']
        realized = a['choices'] == categories[None, :]
        retained = retained_after_discovery(a['present'])
        for i, seed in enumerate(a['seeds']):
            lifetimes.append(dict(arm=arm, seed=int(seed), discovered=bool(a['present'][i].any()),
                retained_after_discovery=retained[i], target_removals=int(a['target_capacity_dropped'][i].sum())))
        for phase, first, last in [('training', 0, cut), ('test', cut, length), ('whole', 0, length)]:
            x = block_paths(realized, first, last)
            y = block_paths(observed[None, :], first, last)[0]
            jk = dict(path_jackknife(x, y), training_nll=jackknife_scores(t['observed_probability']))
            jacks[arm, phase] = jk
            late = max(first, last-128)
            scores.append(dict(arm=arm, phase=phase, **distribution_scores(x, y),
                training_nll=nll(t['observed_probability']), training_nll_mc_se=jackknife_se(jk['training_nll']),
                energy_mc_se=jackknife_se(jk['energy']),
                target_ever=float(a['present'][:, :last].any(1).mean()),
                target_late_present=float(a['present'][:, late:last].mean()),
                target_late_contribution=float(a['contribution'][:, late:last].mean()),
                target_capacity_drops=float(a['target_capacity_dropped'][:, first:last].sum(1).mean()),
                mean_search=float(a['search'][:, first:last].mean()),
                late_accuracy=float(realized[:, late:last].mean())))
            if phase == 'whole':
                curves.extend(dict(arm=arm, trial=min((i+1)*32, length), training_trials=cut,
                    observed=float(y[i]), autonomous=float(x[:, i].mean())) for i in range(len(y)))
    for i in (0, 1):
        np.testing.assert_array_equal(seeds['COMPACT'][i], seeds[label][i])
    table = pd.DataFrame(scores)
    differences = []
    for phase, group in table.groupby('phase'):
        group = group.set_index('arm')
        row = dict(phase=phase)
        for metric in ('training_nll', 'energy', 'crps'):
            row['delta_'+metric] = group.loc[label, metric]-group.loc['COMPACT', metric]
            row['delta_'+metric+'_mc_se'] = jackknife_se(jacks[label, phase][metric]-jacks['COMPACT', phase][metric])
        differences.append(row)
    output.mkdir(parents=True)
    for name, value in dict(scores=table, contrasts=pd.DataFrame(differences), curves=pd.DataFrame(curves),
                             paths=pd.DataFrame(lifetimes)).items():
        value.to_csv(output/(name+'.csv'), index=False)
    write(output/'completion.json', dict(status='complete', source=str(source.resolve()),
        inputs_sha256=bindings, artifacts_sha256={p.name: digest(p) for p in output.glob('*.csv')},
        scope='S205 development diagnostic at original parameters; no refit or oral parameter selection'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'generation', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    summarize(args.source, args.generation, args.output)

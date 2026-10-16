"""Summarize every declared prototype, path and training-only filter replicate."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .model_0923_h_validation import load, digest, block_paths, distribution_scores, jackknife_se
from .model_0923_h_generation_uncertainty import path_jackknife
from ..workflows.analysis.model_0923_h_validation import write


def run(sources: list[Path], output: Path) -> None:
    if output.exists(): raise FileExistsError('Use a new summary destination')
    groups, origins, hashes = {}, {}, {}
    for source in sources:
        protocol = load(source / 'protocol.json'); complete = load(source / 'completion.json')
        if complete['status'] != 'complete' or complete['smoke']: raise ValueError('Need complete scientific runs')
        if digest(source / 'protocol.json') != complete['protocol_sha256']: raise ValueError('Protocol changed')
        hashes[str(source / 'protocol.json')] = digest(source / 'protocol.json')
        for job in protocol['jobs']:
            file = Path(job['target']).with_suffix('.npz')
            record = load(file.with_suffix('.json'))
            if digest(file) != record['predictions_sha256']: raise ValueError('Prediction artifact changed')
            hashes[str(file)] = record['predictions_sha256']
            key = (job['stage'], job['subject'], job['model'], job['arm'])
            with np.load(file) as z: groups.setdefault(key, []).append({k: z[k] for k in z.files})
            origins[key] = job
    summary, curves, paths, filters = [], [], [], []
    scores, reference_paths = {}, {}
    for key, chunks in groups.items():
        stage, subject, model, arm = key
        job = origins[key]; folder = Path(job['folder']); meta = load(folder / 'prepared.json')
        source_files = ['training.npz'] + ([] if stage.startswith('score') else ['sealed_test.npz'])
        observed, categories = [], []
        for name in source_files:
            if digest(folder / name) != meta['array_sha256'][name]: raise ValueError('Observed source changed')
            with np.load(folder / name) as z:
                observed.extend(z['choices']); categories.extend(z['categories'])
        y = np.asarray(observed) == np.asarray(categories)
        seeds = np.concatenate([c['seeds'] for c in chunks])
        if len(np.unique(seeds)) != len(seeds): raise ValueError('Duplicate simulation/filter seed')
        common = dict(stage=stage, subject=subject, model=model, arm=arm)
        if stage.startswith('score'):
            p = np.concatenate([c['observed_probability'] for c in chunks])
            floor = np.finfo(float).tiny
            nll = float(-np.log(np.maximum(p.mean(axis=0), floor)).mean())
            jk = np.array([-np.log(np.maximum(np.delete(p, i, axis=0).mean(axis=0), floor)).mean()
                           for i in range(len(p))])
            scores[key] = dict(nll=nll, jk=jk, seeds=seeds)
            for seed, values in zip(seeds, p):
                filters.append(dict(**common, seed=int(seed), nll=float(-np.log(np.maximum(values, floor)).mean())))
            summary.append(dict(**common, nll=nll, nll_mc_se=jackknife_se(jk), replicates=len(seeds), trials=len(y)))
            continue
        a = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]
             if k not in ('categories', 'onset', 'oracle')}
        np.testing.assert_array_equal(chunks[0]['categories'], categories)
        n = len(y); cut = meta['training_trials']; first = cut if stage == 'oracle' else 0
        accuracy = (a['choice'] == np.asarray(categories)[None, :]).astype(float)
        blocks = block_paths(accuracy, first, n); observed_blocks = block_paths(y[None, :], first, n)[0]
        dist = distribution_scores(blocks, observed_blocks)
        jk = path_jackknife(blocks, observed_blocks)['energy']
        late_start = max(first, n - 128)
        late = accuracy[:, late_start:].mean(axis=1)
        early = accuracy[:, :min(128, n)].mean(axis=1)
        active_count = a['present'][:, late_start:].sum(axis=1)
        beta = np.divide(a['beta'][:, late_start:].sum(axis=1), active_count,
                         out=np.zeros(len(seeds)), where=active_count > 0)
        discovered = a['present'].any(axis=1)
        target = int(chunks[0]['oracle'])
        common_row = dict(**common, replicates=len(seeds), trials=n, first=first,
            early_accuracy=float(early.mean()), late_accuracy=float(late.mean()),
            observed_early=float(y[:min(128, n)].mean()), observed_late=float(y[late_start:].mean()),
            late_mc_se=float(late.std(ddof=1)/np.sqrt(len(seeds))),
            ever_target=float(discovered.mean()), target_late_presence=float(a['present'][:, late_start:].mean()),
            target_late_contribution=float(a['contribution'][:, late_start:].mean()),
            target_late_beta=float(a['beta'][:, late_start:].sum() / max(1, active_count.sum())),
            search_count=float(a['search'].sum(axis=1).mean()), reject_count=float(a['rejected'].sum(axis=1).mean()),
            target_proposed=float((a['proposed'] == target).sum(axis=1).mean()),
            target_rejected=float(((a['proposed'] == target) & (a['rejected'] > 0)).sum(axis=1).mean()),
            energy_mc_se=jackknife_se(jk), **dist)
        summary.append(common_row)
        reference_paths[key] = dict(late=late, jk=jk, seeds=seeds, energy=dist['energy'])
        for b, start in enumerate(range(first, n, 32)):
            curves.append(dict(**common, trial=start+1, observed=float(observed_blocks[b]),
                mean=float(blocks[:, b].mean()), lower=float(np.quantile(blocks[:, b], .1)),
                upper=float(np.quantile(blocks[:, b], .9))))
        for i, seed in enumerate(seeds):
            paths.append(dict(**common, seed=int(seed), early_accuracy=float(early[i]), late_accuracy=float(late[i]),
                target_seen=bool(discovered[i]), first_target=int(np.argmax(a['present'][i]))+1 if discovered[i] else None,
                target_late_presence=float(a['present'][i, late_start:].mean()),
                target_late_beta=float(beta[i]) if active_count[i] else None))
    frame = pd.DataFrame(summary)
    for i, row in frame.iterrows():
        key = (row.stage, row.subject, row.model, row.arm)
        basekey = (row.stage, row.subject, row.model, 'BASE')
        bank = scores if row.stage.startswith('score') else reference_paths
        item, base = bank[key], bank[basekey]
        np.testing.assert_array_equal(item['seeds'], base['seeds'])
        if row.stage.startswith('score'):
            frame.loc[i, 'delta_nll'] = item['nll'] - base['nll']
            frame.loc[i, 'delta_nll_mc_se'] = jackknife_se(item['jk'] - base['jk'])
        else:
            diff = item['late'] - base['late']
            frame.loc[i, 'delta_late'] = float(diff.mean())
            frame.loc[i, 'delta_late_mc_se'] = float(diff.std(ddof=1)/np.sqrt(len(diff)))
            frame.loc[i, 'delta_energy'] = item['energy'] - base['energy']
            frame.loc[i, 'delta_energy_mc_se'] = jackknife_se(item['jk'] - base['jk'])
    output.mkdir(parents=True)
    frame.to_csv(output / 'summary.csv', index=False)
    pd.DataFrame(curves).to_csv(output / 'curves.csv', index=False)
    pd.DataFrame(paths).to_csv(output / 'paths.csv', index=False)
    pd.DataFrame(filters).to_csv(output / 'filters.csv', index=False)
    write(output / 'manifest.json', dict(sources=[str(s) for s in sources], rows=len(frame),
        source_sha256=hashes, inclusion='all declared arms, subjects, frozen points, paths and seeds',
        data_use='development only; all score jobs use training trials',
        summary_sha256={p.name: digest(p) for p in output.glob('*.csv')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); run(args.source, args.output)


if __name__ == '__main__': main()

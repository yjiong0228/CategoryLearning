"""Independently audit complete mechanism/parameter cross results from arrays.

No production scoring implementation is imported. --source is the completed
cross, --pipeline its completed source-bound launcher, and --output a new audit
directory. Reconstruct every aggregate from its original chunks; independently
recompute window scores, paired full-replicate errors, feedback strata and path
summaries. The fixed 80% interval definition matches the declared comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def loss(probability: np.ndarray) -> np.ndarray:
    p = np.asarray(probability, float)
    if p.ndim != 2 or 0 in p.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Expected finite filter-by-trial probabilities')
    return -np.log(np.maximum(p.mean(0), np.finfo(float).tiny))


def mc_error(values: np.ndarray) -> float:
    """Jackknife standard error from complete-replicate deletion estimates."""
    x = np.asarray(values, float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('Expected at least two finite deletion estimates')
    return float(np.sqrt((len(x) - 1) / len(x) * np.square(x - x.mean()).sum()))


def distribution(samples: np.ndarray, observed: np.ndarray) -> dict:
    x, y = np.asarray(samples, float), np.asarray(observed, float)
    if (x.ndim != 2 or y.ndim != 1 or 0 in x.shape or x.shape[1:] != y.shape
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError('Expected finite path-by-block samples and matching observed blocks')
    difference = x - y
    pairs = x[:, None, :] - x[None, :, :]
    lower, upper = np.quantile(x, [.1, .9], axis=0)
    return dict(energy=float((np.linalg.norm(difference, axis=1).mean()
                             - .5 * np.linalg.norm(pairs, axis=2).mean()) / np.sqrt(len(y))),
                crps=float(np.abs(difference).mean() - .5 * np.abs(pairs).mean()),
                interval_score=float((upper - lower + 10 * np.maximum(lower - y, 0)
                                      + 10 * np.maximum(y - upper, 0)).mean()),
                interval_coverage=float(((lower <= y) & (y <= upper)).mean()),
                interval_width=float((upper - lower).mean()),
                realized_mean_mae=float(np.abs(x.mean(0) - y).mean()))


def audit(source: Path, pipeline: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new independent cross-audit directory')
    inputs, checks = {}, 0

    def bind(path):
        path = Path(path).resolve(); digest = file_hash(path)
        if str(path) in inputs and inputs[str(path)] != digest:
            raise ValueError('Bound audit input changed')
        inputs[str(path)] = digest
        return path

    def read(path):
        return json.loads(bind(path).read_text())

    def check(a, b, exact=False):
        nonlocal checks
        if exact: np.testing.assert_array_equal(a, b)
        else: np.testing.assert_allclose(a, b, rtol=0, atol=1e-12, equal_nan=True)
        checks += np.asarray(a).size

    done, protocol = read(source / 'completion.json'), read(source / 'protocol.json')
    launcher, plan = read(pipeline / 'completion.json'), read(pipeline / 'plan.json')
    if (done['status'] != 'complete' or done['smoke'] or done['nominees_changed']
            or launcher['status'] != 'complete' or Path(launcher['output']).resolve() != source.resolve()
            or launcher['completion_sha256'] != file_hash(source / 'completion.json')
            or done['protocol_sha256'] != file_hash(source / 'protocol.json')):
        raise ValueError('Require complete unchanged cross computation and launcher')
    for mapping in (done['inputs_sha256'], plan['source_sha256']):
        for path, digest in mapping.items():
            if file_hash(bind(path)) != digest: raise ValueError('Executed source or input changed')
    for key, value in plan['expected'].items(): check(done[key], value, True)
    for name, digest in done['artifacts'].items():
        if file_hash(bind(source / name)) != digest: raise ValueError('Saved score table changed')
    tables = {name: pd.read_csv(source / f'{name}.csv') for name in
              ('scores', 'contrasts', 'feedback_scores', 'paths')}
    saved = read(source / 'banks.json')
    planned = {b['cell']: b for b in protocol['banks']}
    if (saved['status'] != 'complete' or saved['smoke'] or len(saved['rows']) != 5
            or len(planned) != 5 or {r['cell'] for r in saved['rows']} != set(planned)):
        raise ValueError('Missing or duplicated cross cells')
    banks = {}
    for row in saved['rows']:
        cell = row['cell']; bank = planned[cell]; pair = {}
        for key in ('mechanism', 'parameter', 'candidate'):
            if row[key] != bank[key]: raise ValueError('Cell identity changed')
        for role, items in bank['roles'].items():
            chunks, seeds = [], []
            for item in items:
                job = item['job']; target = Path(job['target'])
                record = read(target.with_suffix('.json')); precision = read(target.with_suffix('.precision.json'))
                if record['candidate'] != bank['candidate'] or record['model'] != 'SEARCH' or job['phase'] != role:
                    raise ValueError('Chunk parameter or proposal identity changed')
                automatic = role == 'autonomous'
                copies = len(job['seeds']) * (1 if automatic else job['particles'])
                check(precision['transitions'], copies * record['trials'], True)
                if not precision['retained_precision']: raise ValueError('Lost precision retention')
                if bank['mechanism'] == 'ONLINE':
                    online = read(target.with_suffix('.online.json'))
                    if (not online['enabled'] or not online['evidence_matches_retained_state']
                            or not online.get('exact_transition_timing', online.get('completed_history_timing_exact'))):
                        raise ValueError('ONLINE execution changed')
                    check(online['reconstructions'], copies * (record['trials'] - 1), True)
                path = bind(target.with_suffix('.npz'))
                if file_hash(path) != record['predictions_sha256']: raise ValueError('Chunk array changed')
                with np.load(path) as z: arrays = {k: z[k].copy() for k in z.files}
                check(arrays['seeds'], job['seeds'], True)
                chunks.append(arrays); seeds.extend(job['seeds'])
            target = bind(Path(row[role]['target']).with_suffix('.npz'))
            if file_hash(target) != row[role]['record']['predictions_sha256']:
                raise ValueError('Aggregate array changed')
            with np.load(target) as z: arrays = {k: z[k].copy() for k in z.files}
            shared = {'categories'} if role == 'autonomous' else {'categories', 'choices'}
            for key, value in arrays.items():
                if key in shared:
                    for chunk in chunks: check(value, chunk[key], True)
                else: check(value, np.concatenate([c[key] for c in chunks]), True)
            check(arrays['seeds'], seeds, True)
            if len(set(seeds)) != len(seeds): raise ValueError('Duplicate random replicate')
            check(arrays['probabilities'].sum(-1), 1.)
            if not np.isfinite(arrays['probabilities']).all() or (arrays['probabilities'] < 0).any():
                raise ValueError('Invalid predicted probabilities')
            pair[role] = arrays
        meta = row['evaluate']['record']
        if meta['condition'] != 2: raise ValueError('This audit uses the declared binary-feedback task')
        c, a = pair['evaluate'], pair['autonomous']
        check(c['categories'], a['categories'], True)
        check(a['feedback'], (a['choices'] == a['categories']).astype(float), True)
        observed = np.take_along_axis(c['probabilities'], (c['choices'] - 1)[None, :, None], axis=2)[..., 0]
        check(c['observed_probability'], observed, True)
        pair['meta'] = meta; banks[cell] = pair
    anchor = next(iter(banks.values()))
    width = protocol['settings']['features']['block_width']; summaries = {}
    for cell, bank in banks.items():
        c, a, meta = bank['evaluate'], bank['autonomous'], bank['meta']
        for key in ('seeds', 'choices', 'categories'): check(c[key], anchor['evaluate'][key], True)
        check(a['seeds'], anchor['autonomous']['seeds'], True)
        check(meta['training_trials'], anchor['meta']['training_trials'], True)
        cut, n = meta['training_trials'], len(c['choices']); human = c['choices'] == c['categories']
        generated = a['choices'] == a['categories']; p = c['observed_probability']; losses = loss(p)
        for phase, start, stop in [('training', 0, cut), ('test', cut, n), ('whole', 0, n)]:
            blocks = lambda values: np.stack([values[..., i:min(i + width, stop)].mean(-1)
                                               for i in range(start, stop, width)], axis=-1)
            x, y = blocks(generated), blocks(human)
            estimates = dict(choice_nll=float(losses[start:stop].mean()), **distribution(x, y),
                correct_loss=float((losses[start:stop] * human[start:stop]).mean()),
                error_loss=float((losses[start:stop] * ~human[start:stop]).mean()))
            score = tables['scores']; selected = score[(score.cell == cell) & (score.phase == phase)]
            if len(selected) != 1: raise ValueError('Missing score row')
            for metric, value in estimates.items(): check(selected.iloc[0][metric], value)
            summaries[cell, phase] = (estimates, x, y, p[:, start:stop])
        late = max(0, n - 128)
        rows = tables['paths'][tables['paths'].cell == cell].set_index('seed').loc[a['seeds']]
        direct = dict(late_accuracy=generated[:, late:].mean(1), target_ever=a['present'].any(1),
            target_late_present=a['present'][:, late:].mean(1), target_late_contribution=a['contribution'][:, late:].mean(1),
            target_exits=a['target_exited'].sum(1), target_capacity_drops=a['target_capacity_dropped'].sum(1),
            search_events=a['search'].sum(1))
        for key, value in direct.items(): check(rows[key], value)
    pairs = [('ONLINE_RETAIN', 'RETAIN_RETAIN'), ('ONLINE_ONLINE', 'RETAIN_ONLINE'),
             ('RETAIN_ONLINE', 'RETAIN_RETAIN'), ('ONLINE_ONLINE', 'ONLINE_RETAIN'), ('ONLINE_NEAR', 'ONLINE_ONLINE')]
    if len(tables['contrasts']) != 15 or len(tables['feedback_scores']) != 60:
        raise ValueError('Incomplete contrast or feedback table')
    for new, old in pairs:
        for phase, start, stop in [('training', 0, cut), ('test', cut, n), ('whole', 0, n)]:
            u, x, y, p = summaries[new, phase]; v, z, _, q = summaries[old, phase]
            table = tables['contrasts']; rows = table[(table['new'] == new) & (table.reference == old) & (table.phase == phase)]
            if len(rows) != 1: raise ValueError('Missing contrast')
            row = rows.iloc[0]
            deleted_paths = [np.array([distribution(np.delete(x, i, axis=0), y)[key]
                                       - distribution(np.delete(z, i, axis=0), y)[key] for i in range(len(x))])
                             for key in ('energy', 'crps')]
            deleted_filters = [loss(np.delete(p, i, axis=0)).mean() - loss(np.delete(q, i, axis=0)).mean()
                               for i in range(len(p))]
            for key, deleted in zip(('choice_nll', 'energy', 'crps'), [deleted_filters, *deleted_paths]):
                check(row['delta_' + key], u[key] - v[key]); check(row['delta_' + key + '_mc_se'], mc_error(deleted))
            table = tables['feedback_scores']; group = table[(table['new'] == new) & (table.reference == old) & (table.phase == phase)]
            actual = anchor['evaluate']['choices'][start:stop] == anchor['evaluate']['categories'][start:stop]
            for label, mask in [('all', np.ones(stop-start, bool)), ('incorrect', ~actual),
                                ('partial', np.zeros(stop-start, bool)), ('correct', actual)]:
                rows = group[group.feedback_group == label]
                if len(rows) != 1: raise ValueError('Missing feedback stratum')
                row = rows.iloc[0]; count = int(mask.sum()); check(row.trials, count, True)
                if count:
                    difference = loss(p)[mask] - loss(q)[mask]
                    check(row.retained_choice_nll, loss(q)[mask].mean()); check(row.online_choice_nll, loss(p)[mask].mean())
                    check(row.delta_choice_nll, difference.mean()); check(row.delta_total_nll, difference.sum())
                    check(row.contribution_to_window_mean, difference.sum() / (stop-start))
                    deleted = [(loss(np.delete(p, i, axis=0))[mask] - loss(np.delete(q, i, axis=0))[mask]).mean()
                               for i in range(len(p))]
                    check(row.delta_choice_nll_mc_se, mc_error(deleted))
                else:
                    check(row.delta_choice_nll, np.nan); check(row.delta_total_nll, 0.)
    for path, digest in inputs.items():
        if file_hash(Path(path)) != digest: raise ValueError('Audit input changed during verification')
    output.mkdir(parents=True)
    result = dict(status='complete', numerical_checks=checks, inputs_sha256=inputs,
                  source_sha256=file_hash(Path(__file__)), cells=5, original_nominations_changed=False,
                  scope='Raw chunk reconstruction, task and seed pairing, all saved scores and complete-replicate errors')
    with (output / 'validation.json').open('x') as stream: json.dump(result, stream, indent=2)
    print(dict(status='complete', numerical_checks=checks, inputs=len(inputs)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'pipeline', 'output'): parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(); audit(args.source, args.pipeline, args.output)

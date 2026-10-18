"""Separate target discovery, later absence, and use in saved capacity probes.

Run --source COMPLETED_PROBE --output NEW_DIRECTORY. All parameter cells and
all autonomous paths are retained. First presence is recorded before the
current choice; a path with no discovery receives the horizon+1 censor value.
Within each window, pre-discovery absence, post-discovery absence and presence
partition every path-trial. Conditional contribution pools present path-trials,
so presence times conditional contribution recovers unconditional contribution.
Paired whole-path jackknife errors describe simulation error at fixed parameters.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .model_0923_recalled_evidence_study import digest, read, write
from ...evaluation.model_0923_h_validation import jackknife_se

CAPACITY = 'engine.modules.hypo_transitions_mod.kwargs.max_capacity'
MEMORY = 'engine.modules.memory_mod.kwargs.gamma'


def path_windows(arrays: dict[str, np.ndarray], width: int) -> pd.DataFrame:
    """Validate saved transitions and partition absence without losing censored paths."""
    present = np.asarray(arrays['present'])
    if present.ndim != 2 or not np.isin(present, [0, 1]).all():
        raise ValueError('Presence must be a binary path-by-trial array')
    p, n = present.shape
    if not p or not n or width < 1:
        raise ValueError('Need paths, trials and a positive window width')
    present = present.astype(bool)
    for key in ('contribution', 'beta', 'target_exited', 'target_capacity_dropped'):
        value = np.asarray(arrays[key])
        if value.shape != (p, n) or not np.isfinite(value).all():
            raise ValueError(f'Invalid path array: {key}')
    for key in ('target_exited', 'target_capacity_dropped'):
        if not np.isin(arrays[key], [0, 1]).all():
            raise ValueError('Transition flags must be binary')
    contribution = arrays['contribution']
    if np.any(contribution < 0) or np.any(contribution > 1) or np.any(contribution[~present] != 0):
        raise ValueError('Target contribution must be a probability and zero while absent')
    if np.any(arrays['beta'] < 0) or np.any(arrays['beta'][~present] != 0):
        raise ValueError('Target precision must be nonnegative and zero while absent')
    seeds = arrays['seeds']
    if seeds.shape != (p,) or len(set(seeds)) != p:
        raise ValueError('Require one unique seed for each path')
    previous = np.concatenate([np.zeros((p, 1), bool), present[:, :-1]], axis=1)
    loss = previous & ~present
    if np.any(loss & ~(arrays['target_exited'].astype(bool) | arrays['target_capacity_dropped'].astype(bool))):
        raise ValueError('A target disappearance has no recorded transition cause')
    ever = present.any(axis=1)
    first = np.where(ever, present.argmax(axis=1), n)
    before = np.arange(n)[None, :] < first[:, None]
    after_absent = ~present & ~before
    np.testing.assert_array_equal(before.astype(int) + after_absent + present, np.ones((p, n)))
    rows = []
    for window, start, stop in [('first', 0, min(width, n)), ('late', max(0, n-width), n), ('whole', 0, n)]:
        sl = slice(start, stop)
        for i, seed in enumerate(seeds):
            rows.append(dict(seed=int(seed), window=window, start=start+1, stop=stop,
                ever_discovered=int(ever[i]), first_presence_censored=int(first[i]+1),
                pre_discovery_absence=float(before[i, sl].mean()),
                post_discovery_absence=float(after_absent[i, sl].mean()),
                presence=float(present[i, sl].mean()), contribution=float(contribution[i, sl].mean()),
                precision_mass=float(arrays['beta'][i, sl].mean()),
                capacity_drops=int(arrays['target_capacity_dropped'][i, sl].sum()),
                autonomous_exits=int(arrays['target_exited'][i, sl].sum()),
                loss_events=int(loss[i, sl].sum()),
                entries=int((present & ~previous)[i, sl].sum())))
    return pd.DataFrame(rows)


METRICS = ['ever_discovered', 'first_presence_censored', 'pre_discovery_absence',
           'post_discovery_absence', 'presence', 'contribution', 'capacity_drops',
           'autonomous_exits', 'loss_events', 'entries',
           'contribution_when_present', 'precision_when_present']


def summarize_paths(frame: pd.DataFrame) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Pool path-trials for conditional quantities and retain paired delete-one estimates."""
    if len(frame) < 3 or frame.seed.nunique() != len(frame):
        raise ValueError('Need at least three unique whole paths per window')
    estimates, deletes = {}, {}
    n = len(frame)
    for metric in METRICS:
        if metric.endswith('_when_present'):
            key = 'contribution' if metric.startswith('contribution') else 'precision_mass'
            numerator, denominator = frame[key].to_numpy(), frame.presence.to_numpy()
            estimates[metric] = float(numerator.sum()/denominator.sum()) if denominator.sum() else np.nan
            den = denominator.sum()-denominator
            values = np.full(n, np.nan)
            np.divide(numerator.sum()-numerator, den, out=values, where=den>0)
            deletes[metric] = values
        else:
            values = frame[metric].to_numpy(dtype=float)
            estimates[metric] = float(values.mean())
            deletes[metric] = (values.sum()-values)/(n-1)
    return estimates, deletes


def run(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new path-diagnosis output directory')
    completion, protocol = read(source/'completion.json'), read(source/'protocol.json')
    if completion['status'] != 'complete' or completion.get('smoke') is not False:
        raise ValueError('Need a complete scientific capacity probe')
    if digest(source/'summary.json') != completion['summary_sha256']:
        raise ValueError('Probe summary changed')
    summary = read(source/'summary.json')
    hashes = {str((source/name).resolve()): digest(source/name)
              for name in ('completion.json', 'protocol.json', 'summary.json')}
    for name, sha in summary['inputs'].items():
        if digest(Path(name)) != sha:
            raise ValueError('Probe input changed')
        hashes[name] = sha
    for name, sha in summary['artifacts'].items():
        if digest(source/name) != sha:
            raise ValueError('Probe table changed')
        hashes[str((source/name).resolve())] = sha
    groups, metadata = {}, {}
    files = sorted((source/'jobs').glob('*.json'))
    if len(files) != completion['jobs']:
        raise ValueError('Incomplete saved job bank')
    for file in files:
        job = read(file); target = Path(job['target']).with_suffix('.npz')
        record = read(target.with_suffix('.json'))
        if digest(target) != record['predictions_sha256']:
            raise ValueError('Path bank changed')
        if job['model'] != 'SEARCH' or job['smoke'] is not False:
            raise ValueError('Require scientific SEARCH paths')
        if record['candidate'] != job['candidate']:
            raise ValueError('Saved parameter identity differs from the job')
        with np.load(target) as data:
            arrays = {key: data[key] for key in data.files}
        np.testing.assert_array_equal(arrays['seeds'], job['seeds'])
        point = job['candidate']; key = (record['subject'], point['id'])
        meta = dict(subject=key[0], point=key[1], anchor=job['anchor'],
                    capacity=point['hyperparams'][CAPACITY], memory=point['hyperparams'][MEMORY])
        if key in metadata and metadata[key] != meta:
            raise ValueError('Inconsistent point identity')
        metadata[key] = meta
        frame = path_windows(arrays, protocol['settings']['window'])
        groups.setdefault(key, []).append(frame.assign(**meta))
        hashes.update({str(f.resolve()): digest(f) for f in (file, target, target.with_suffix('.json'))})
    expected = pd.read_csv(source/'scores.csv')
    if set(metadata) != set(zip(expected.subject, expected.point)):
        raise ValueError('Parameter cells differ from the completed probe')
    paths = pd.concat([pd.concat(parts) for parts in groups.values()], ignore_index=True)
    rows, paired, banks = [], [], {}
    for (subject, point, window), frame in paths.groupby(['subject', 'point', 'window']):
        frame = frame.sort_values('seed')
        if len(frame) != protocol['settings']['paths']:
            raise ValueError('Incomplete path bank')
        values, deletes = summarize_paths(frame)
        base = dict(metadata[subject, point], window=window, paths=len(frame),
                    start=int(frame.start.iloc[0]), stop=int(frame.stop.iloc[0]))
        banks[subject, point, window] = (frame.seed.to_numpy(), values, deletes)
        for metric, value in values.items():
            se = jackknife_se(deletes[metric]) if np.isfinite(deletes[metric]).all() else np.nan
            rows.append(dict(base, metric=metric, value=value, mcse=se))
    for (subject, point, window), (seeds, values, deletes) in banks.items():
        anchors = [key[1] for key, meta in metadata.items() if key[0] == subject and meta['anchor']]
        if len(anchors) != 1:
            raise ValueError('Need one original anchor per subject')
        reference = anchors[0]; rseeds, rvalues, rdeletes = banks[subject, reference, window]
        np.testing.assert_array_equal(seeds, rseeds)
        for metric, value in values.items():
            differences = deletes[metric]-rdeletes[metric]
            se = jackknife_se(differences) if np.isfinite(differences).all() else np.nan
            paired.append(dict(metadata[subject, point], reference=reference, window=window,
                               metric=metric, difference=value-rvalues[metric], mcse=se))
    output.mkdir(parents=True)
    for name, frame in [('paths', paths), ('scores', pd.DataFrame(rows)), ('contrasts', pd.DataFrame(paired))]:
        frame.to_csv(output/f'{name}.csv', index=False)
    write(output/'manifest.json', dict(status='complete', input_sha256=hashes,
        source_sha256={str(Path(__file__).resolve()): digest(Path(__file__))},
        artifacts={p.name: digest(p) for p in output.glob('*.csv')},
        python=sys.version, numpy=np.__version__, pandas=pd.__version__,
        scope='All saved capacity/memory cells; autonomous paths; no fitting or oral selection',
        timing='Target presence and transition at choice, before current feedback',
        censoring='Never-discovered paths retained, first_presence_censored=horizon+1',
        conditional_estimand='Pooled present path-trials, not the average of path-conditional means',
        uncertainty='Paired whole-path jackknife MCSE; fixed parameters; conditional precision is descriptive'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); run(args.source, args.output)

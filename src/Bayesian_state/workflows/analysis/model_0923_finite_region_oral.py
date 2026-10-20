"""Compare frozen finite-H region reasons with the two existing oral readings.

Run --config YAML --output NEW_DIRECTORY [--smoke]. Inputs are completed behavior
fits, exact particle state replays and geometric basis projections. Every frozen
evaluation seed is retained. Oral constraints are loaded after these checks and
never update parameters or states. Labels refer to each geometric region, while
the learned A rows determine how regions support the actual response.

Seed joint masses are pooled before conditioning on the current response. Whole
filter jackknife estimates quantify numerical uncertainty and paired arm
differences. Complete reports, parsed fragments and unscored reports remain
separate. Smoke scores the first eight trials of the first declared subject with
all original seeds; it validates postprocessing rather than a shortened fit.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

from ...evaluation.model_0923_h_validation import digest, load, jackknife_se
from ...evaluation.oral.decision_basis_scoring import (
    DOMAINS, LABELS, label_masses, pool_joint, pooled_mean_and_mcse,
)
from ...hypothesis_space.spaces.structural_0923 import build_relational_0923_space
from .model_0923_decision_basis_alignment import align_identities
from .model_0923_h_validation import write
from .model_0923_region_association_oral import tokens_key

ROOT = Path(__file__).resolve().parents[4]
KEYS = ['reading', 'subject', 'window', 'scope', 'feedback_stratum', 'domain']
TIMINGS = ('conditional', 'prechoice')


def score_bank(frame: pd.DataFrame, prior: np.ndarray, joint: np.ndarray,
               probability: np.ndarray, seeds: np.ndarray, cut: int,
               reading: str, subject: int, arm: str) -> tuple[pd.DataFrame, list[dict], dict]:
    """Keep identical report masks while pooling complete filter replicates."""
    n = len(frame)
    if (probability.shape != (len(seeds), n) or len(seeds) < 2
            or len(np.unique(seeds)) != len(seeds) or not np.all(np.diff(seeds) > 0)
            or prior.shape != (len(seeds), n, 2, 3) or joint.shape != prior.shape
            or not 0 < cut <= n or not np.isfinite(probability).all()
            or np.any(probability <= 0) or np.any(probability > 1)):
        raise ValueError('Require aligned complete seed banks and a valid training prefix')
    np.testing.assert_array_equal(frame.state_trial, np.arange(n))
    valid = np.isfinite(joint).all(axis=(0, 2, 3))
    if (not np.array_equal(np.isfinite(prior).all(axis=(0, 2, 3)), valid)
            or not np.isnan(joint[:, ~valid]).all() or not np.isnan(prior[:, ~valid]).all()
            or not valid[frame.scope.isin(['complete', 'parsed_fragment'])].all()):
        raise ValueError('Report coverage must agree across seeds and timings')
    np.testing.assert_allclose(prior[:, valid].sum(-1), 1., atol=1e-12, rtol=0)
    np.testing.assert_allclose(joint[:, valid].sum(-1),
                               np.repeat(probability[:, valid, None], 2, axis=-1), atol=1e-12, rtol=0)
    score = np.full((n, 2, 3), np.nan)
    score[valid] = pool_joint(joint[:, valid], probability[:, valid])
    trial = frame.copy()
    trial['reading'], trial['arm'], trial['training_trials'] = reading, arm, cut
    trial['phase'] = np.where(np.arange(n) < cut, 'training', 'test')
    trial['filter_seeds'] = len(seeds)
    trial['mean_choice_probability'] = probability.mean(0)
    for d, domain in enumerate(DOMAINS):
        for k, label in enumerate(LABELS):
            trial[f'conditional_{domain}_{label}'] = score[:, d, k]
            trial[f'prechoice_{domain}_{label}'] = prior.mean(0)[:, d, k]
    ordinal = np.arange(n)
    windows = dict(whole=ordinal >= 0, training=ordinal < cut, test=ordinal >= cut,
                   first64=ordinal < 64, last64=ordinal >= max(0, n-64))
    strata = dict(all=np.ones(n, bool), correct=frame.feedback.eq(1).to_numpy(),
                  incorrect=frame.feedback.eq(0).to_numpy())
    summary, deleted = [], {}
    for window, window_mask in windows.items():
        for scope in ('complete', 'parsed_fragment'):
            for stratum, feedback_mask in strata.items():
                mask = window_mask & feedback_mask & frame.scope.eq(scope).to_numpy()
                if not mask.any():
                    continue
                for d, domain in enumerate(DOMAINS):
                    row = dict(reading=reading, subject=subject, arm=arm, window=window,
                               scope=scope, feedback_stratum=stratum, domain=domain,
                               reports=int(mask.sum()), filter_seeds=len(seeds))
                    estimates = []
                    for timing, values, den in (('conditional', joint, probability),
                                               ('prechoice', prior, np.ones_like(probability))):
                        point, se, jackknife = pooled_mean_and_mcse(values[:, :, d], den, mask)
                        estimates.append(jackknife)
                        for k, label in enumerate(LABELS):
                            row[timing+'_'+label] = point[k]
                            row[timing+'_'+label+'_mcse'] = se[k]
                    deleted[tuple(row[k] for k in KEYS)] = np.stack(estimates)
                    summary.append(row)
    return trial, summary, deleted


def paired_contrasts(summary: pd.DataFrame, deleted: dict) -> pd.DataFrame:
    """Compare REHEARSE with FROZEN using matching masks and seed deletions."""
    a = summary[summary.arm.eq('REHEARSE')].set_index(KEYS).sort_index()
    b = summary[summary.arm.eq('FROZEN')].set_index(KEYS).sort_index()
    if not a.index.equals(b.index):
        raise ValueError('Mechanisms must retain the same report masks')
    np.testing.assert_array_equal(a[['reports', 'filter_seeds']], b[['reports', 'filter_seeds']])
    result = a[['reports', 'filter_seeds']].copy()
    for timing_index, timing in enumerate(TIMINGS):
        for k, label in enumerate(LABELS):
            name = timing+'_'+label
            result['delta_'+name] = a[name]-b[name]
            result['delta_'+name+'_mcse'] = [jackknife_se(
                deleted['REHEARSE', *key][timing_index, :, k] -
                deleted['FROZEN', *key][timing_index, :, k]) for key in a.index]
    return result.reset_index()


def run(config: Path, output: Path, smoke: bool = False) -> None:
    settings = yaml.safe_load(config.read_text())
    if set(settings) != {'fit', 'states', 'basis', 'readings'} or set(settings['readings']) != {'midpoint', 'relative'}:
        raise ValueError('Declare completed behavior inputs and both frozen oral readings')
    if output.exists():
        raise FileExistsError('Use a new finite-region oral output directory')
    fit, states, basis = (ROOT/settings[k] for k in ('fit', 'states', 'basis'))
    bindings = {}

    def bind(path, expected=None):
        path = Path(path).resolve(); value = digest(path)
        if expected is not None and value != expected:
            raise ValueError('Input changed: '+str(path))
        if str(path) in bindings and bindings[str(path)] != value:
            raise ValueError('Input changed during analysis: '+str(path))
        bindings[str(path)] = value
        return path

    def read(path):
        return load(bind(path))

    def verify():
        for p, sha in bindings.items():
            if digest(Path(p)) != sha:
                raise ValueError('Analysis input changed: '+p)

    fd, sd, bd = [read(p/'completion.json') for p in (fit, states, basis)]
    if any(d['status'] != 'complete' for d in (fd, sd, bd)) or fd['smoke'] or sd['smoke']:
        raise ValueError('Require completed full behavior states and projection')
    for p, done in ((fit, fd), (states, sd), (basis, bd)):
        for name, sha in done['artifacts_sha256'].items():
            bind(p/name, sha)
    for p, sha in bd['inputs_sha256'].items():
        bind(p, sha)
    protocol, nominated = read(fit/'protocol.json'), read(fit/'nominees.json')
    bind(fit/'nominees.json', sd['nomination_sha256'])
    if nominated['status'] != 'frozen_before_later_choices' or bd['oral_used']:
        raise ValueError('Require behavior-only frozen nominees and bases')
    groups = defaultdict(list); folders = {}
    for row in bd['rows']:
        s, arm, seed = row['subject'], row['arm'], row['seed']
        if not any((n['subject'], n['arm'], n['point']) == (s, arm, row['point']) for n in nominated['nominees']):
            raise ValueError('Basis parameters differ from frozen nominee')
        file = Path(row['output']); bind(file, row['arrays_sha256'])
        name = file.name.removesuffix('.basis.npz')
        job = read(states/'jobs'/(name+'.json')); state_record = read(Path(job['target']).with_suffix('.regions.json'))
        if (row['source_state_sha256'] != state_record['arrays_sha256'] or job['seed'] != seed
                or (job['subject'], job['arm'], job['point']) != (s, arm, row['point'])):
            raise ValueError('Basis does not match its exact state replay')
        folders[s] = Path(job['training']).parent
        groups[s, arm].append(row)
    expected = {(n['subject'], n['arm']) for n in nominated['nominees']}
    if set(groups) != expected or bd['groups'] != sum(map(len, groups.values())):
        raise ValueError('Incomplete mechanism groups')
    for (s, _), rows in groups.items():
        rows.sort(key=lambda r: r['seed'])
        if [r['seed'] for r in rows] != sorted(protocol['seeds'][str(s)]['evaluation']):
            raise ValueError('Incomplete or repeated evaluation seeds')
    space = build_relational_0923_space()
    if bd['catalogue_signature'] != json.loads(json.dumps(space.signature)):
        raise ValueError('Basis catalogue changed')
    counts = np.array([[len(c.components) for c in h.categories] for h in space])
    sources = [Path(__file__), ROOT/'src/Bayesian_state/evaluation/oral/decision_basis_scoring.py',
        ROOT/'src/Bayesian_state/workflows/analysis/model_0923_decision_basis_alignment.py',
        ROOT/'src/Bayesian_state/workflows/analysis/model_0923_region_association_oral.py']
    bind(config)
    for p in sources:
        bind(p)
    verify(); output.mkdir(parents=True); (output/'banks').mkdir()
    for p in sources:
        target = output/'source'/p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, target)
    shutil.copyfile(config, output/'config.yaml')
    write(output/'protocol.json', dict(settings=settings, smoke=smoke, input_bindings=bindings.copy(),
        behavior_inputs_checked_before_oral=True, normalization='Pool seed joint masses before conditioning',
        uncertainty='Delete complete filter seeds, with paired seed deletions for arm differences',
        scope='Five original chronological windows; full/fragment reports; both readings; all feedback strata',
        python=sys.version, versions={p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'PyYAML')}))
    subjects = sorted(folders)[:1] if smoke else sorted(folders)
    trials, summaries, deleted_all, query_records = [], [], {}, []
    for reading, paths in settings['readings'].items():
        audit, cache = ROOT/paths['audit'], ROOT/paths['queries']
        am, cm, cp = read(audit/'manifest.json'), read(cache/'completion.json'), read(cache/'protocol.json')
        if am['status'] != 'complete' or cm['status'] != 'complete' or cm['smoke']:
            raise ValueError('Require completed frozen semantic audits and queries')
        bind(cache/'protocol.json', cm['protocol_sha256'])
        for p, sha in cp['source_sha256'].items():
            if '/hypothesis_space/' in p or p.endswith(('/constraint_alignment.py', '/structure_0923.py', '/report_space.py', '/oral_coding.py')):
                bind(ROOT/p, sha)
        bind(audit/'trials.csv', am['artifacts']['trials.csv']); bind(cache/'queries.csv', cm['artifacts']['queries.csv'])
        oral, queries = pd.read_csv(audit/'trials.csv'), pd.read_csv(cache/'queries.csv')
        lookup = {(tokens_key(r.tokens), int(r.region)): str(r.query_id) for r in queries.itertuples()}
        if len(lookup) != len(queries):
            raise ValueError('Duplicate semantic query')
        cached = {}
        for s in subjects:
            length = groups[s, 'FROZEN'][0]['trials']; n = min(8, length) if smoke else length
            folder = folders[s]; meta = read(folder/'prepared.json')
            bind(Path(meta['paths']['learning_data']), meta['learning_data_sha256'])
            for name, sha in meta['array_sha256'].items(): bind(folder/name, sha)
            frame, cut = align_identities(oral, dict(folder=str(folder)), n)
            with np.load(folder/'training.npz') as a, np.load(folder/'sealed_test.npz') as b:
                frame['feedback'] = np.concatenate([a['feedback'], b['feedback']])[:n]
            labels = np.full((n, len(space), 4, int(counts.max()), 2), -1, np.int8)
            for t, row in enumerate(frame.itertuples()):
                if pd.isna(row.query_id): continue
                token = tokens_key(row.tokens)
                for g in range(4):
                    key = token, g
                    if key not in cached:
                        code = lookup[key]; p = cache/'queries'/(code+'.npz')
                        rec = read(p.with_suffix('.json')); bind(cache/'jobs'/('Q_'+code+'.json'), rec['job_sha256'])
                        bind(p, rec['arrays_sha256'])
                        if tokens_key(json.dumps(rec['tokens'])) != token or rec['region'] != g:
                            raise ValueError('Cached semantic identity changed')
                        with np.load(p) as z: lab = z['labels'].copy()
                        valid = np.arange(int(counts.max()))[None, :] < counts[:, g, None]
                        if lab.shape != (*valid.shape, 2) or not np.isin(lab[valid], [0, 1, 2]).all() or not (lab[~valid] == -1).all():
                            raise ValueError('Cached geometric label shape or padding changed')
                        cached[key] = lab
                        query_records.append(dict(reading=reading, query_id=code, region=g, tokens=json.dumps(token)))
                    labels[t, :, g] = cached[key]
            flattened = (n, len(space)*4, int(counts.max()))
            for arm in ('FROZEN', 'REHEARSE'):
                prior, joint, probabilities = [], [], []
                rows = groups[s, arm]; seeds = np.array([r['seed'] for r in rows], dtype=np.int64)
                for row in rows:
                    with np.load(row['output']) as z:
                        np.testing.assert_array_equal(z['choices'][:n], frame.choice)
                        for key, dest in (('prechoice_component_mass', prior), ('choice_component_joint_mass', joint)):
                            dest.append(label_masses(z[key][:n].reshape(flattened), labels.reshape(*flattened, 2)))
                        probabilities.append(z['choice_probability'][:n])
                prior, joint, probability = map(np.stack, (prior, joint, probabilities))
                trial, summary, deleted = score_bank(frame, prior, joint, probability, seeds, cut, reading, s, arm)
                trials.append(trial); summaries.extend(summary)
                deleted_all.update({(arm, *k): value for k, value in deleted.items()})
                with (output/'banks'/f'S{s}_{arm}_{reading}.npz').open('xb') as f:
                    np.savez_compressed(f, prechoice=prior, joint=joint, probability=probability, seeds=seeds,
                        choices=frame.choice.to_numpy(), state_trial=frame.state_trial.to_numpy())
    trial = pd.concat(trials, ignore_index=True); summary = pd.DataFrame(summaries)
    contrasts = paired_contrasts(summary, deleted_all)
    reason_groups = trial.groupby(['reading', 'iSub', 'arm', 'phase', 'scope', 'choice', 'text'], dropna=False).agg(
        reports=('state_trial', 'size'), correct=('feedback', 'sum'),
        conditional_support=('conditional_functional_support', 'mean'),
        prechoice_support=('prechoice_functional_support', 'mean')).reset_index()
    for name, frame in [('trial_scores', trial), ('summary', summary), ('contrasts', contrasts),
                         ('reason_groups', reason_groups), ('queries', pd.DataFrame(query_records))]:
        frame.to_csv(output/(name+'.csv'), index=False)
    verify()
    write(output/'completion.json', dict(status='complete', smoke=smoke, subjects=subjects,
        behavior_parameters_updated=False, trials=len(trial), summary_rows=len(summary),
        protocol_sha256=digest(output/'protocol.json'), bindings=bindings,
        artifacts={str(p.relative_to(output)): digest(p) for p in output.rglob('*')
                   if p.is_file() and 'source' not in p.relative_to(output).parts}))
    print(contrasts.query('window == "test" and scope == "complete" and feedback_stratum == "all" and domain == "functional"')
          [['reading', 'subject', 'reports', 'delta_conditional_support', 'delta_prechoice_support']].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.config, args.output, args.smoke)

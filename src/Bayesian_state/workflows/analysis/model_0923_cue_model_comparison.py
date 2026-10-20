"""Compare existing model predictions at stimulus-defined cue recurrence pairs.

Use --config YAML --output NEW_DIRECTORY. The frozen oral chronology defines
diagnostic trial pairs; every observed pair for the configured subjects is kept.
Finite-H FROZEN and REHEARSE use their original behavior-selected parameters,
all evaluation filters and all autonomous paths. GLOBAL is a separate fixed-
catalogue reference with its original parameters, not an equal-budget fit arm.

Conditional predictions at source and target each use the observed history up
to that trial. Their product is not treated as a two-event probability. For
autonomous paths, a matching source choice selects paths before examining their
target responses. Ratios pool counts across whole paths; jackknife uncertainty
resamples paths, preserving dependent events within a path. Oral wording is not
generated, so these are choice recurrence diagnostics, not oral likelihoods.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ...evaluation.model_0923_h_validation import digest, load
from .model_0923_h_validation import write
from .model_0923_finite_region_fit import load_trials

ROOT = Path(__file__).resolve().parents[4]
KEYS = ['subject', 'phase', 'event', 'text', 'source_choice', 'source_feedback']


def path_ratio(numerator: np.ndarray, denominator: np.ndarray) -> tuple[float, float]:
    """Pooled event ratio and delete-one-path jackknife standard error.

    Zero-opportunity paths remain in the arrays. An empty total or an empty
    delete-one denominator yields an undefined estimate or uncertainty.
    """
    a, b = np.asarray(numerator, float), np.asarray(denominator, float)
    if (a.ndim != 1 or a.shape != b.shape or not np.isfinite(a).all()
            or not np.isfinite(b).all() or np.any(a < 0) or np.any(a > b) or np.any(b < 0)):
        raise ValueError('Require aligned nonnegative path counts with numerator <= denominator')
    total = b.sum()
    value = float(a.sum()/total) if total else np.nan
    if len(a) < 2 or np.any(total-b <= 0):
        return value, np.nan
    deleted = (a.sum()-a)/(total-b)
    se = np.sqrt((len(a)-1)/len(a)*np.square(deleted-deleted.mean()).sum())
    return value, float(se)


def pair_values(probabilities: np.ndarray, choices: np.ndarray, source: np.ndarray,
                target: np.ndarray, reference: np.ndarray) -> dict[str, np.ndarray]:
    """Read paired observations; the target never selects the matching source set."""
    p, y = np.asarray(probabilities), np.asarray(choices)
    source, target, reference = map(np.asarray, (source, target, reference))
    if (p.ndim != 2 or p.shape[1] != 4 or y.shape != (len(p),)
            or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1)
            or not np.isin(y, [1, 2, 3, 4]).all()
            or source.ndim != 1 or source.shape != target.shape or source.shape != reference.shape
            or not all(np.issubdtype(v.dtype, np.integer) for v in (source, target, reference))
            or np.any(source < 0) or np.any(target <= source) or np.any(target >= len(p))
            or not np.isin(reference, [1, 2, 3, 4]).all()):
        raise ValueError('Require valid probabilities, choices and aligned ordered trial pairs')
    np.testing.assert_allclose(p.sum(1), 1., atol=1e-12, rtol=0)
    first = y[source] == reference
    second = y[target] == reference
    return dict(source_match=first, target_match=second, joint_match=first & second,
                source_probability=p[source, reference-1], target_probability=p[target, reference-1],
                matched_target_expected=first*p[target, reference-1])


def summarize(events: pd.DataFrame, conditional: pd.DataFrame,
              autonomous: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Keep physical target-category strata and all independent simulation units."""
    condition_rows, auto_rows, path_rows = [], [], []
    for values, group in events.groupby(KEYS, sort=True):
        identity = dict(zip(KEYS, values))
        for relation in ('all', 'same', 'different'):
            same = group.target_category.eq(group.source_choice)
            part = group if relation == 'all' else group[same if relation == 'same' else ~same]
            if part.empty:
                continue
            base = dict(identity, target_category_relation=relation, pairs=len(part),
                observed_same_choice=int(part.same_choice.sum()),
                observed_same_wording=int(part.same_wording.sum()),
                observed_same_wording_and_choice=int(part.same_wording_and_choice.sum()),
                source_cue_false=int(part.source_cue_status.eq('false').sum()))
            selected = conditional[conditional.event_id.isin(part.event_id)]
            for arm, frame in selected.groupby('arm', sort=True):
                pooled = frame.groupby('event_id')[['source_probability', 'target_probability', 'target_observed_probability']].mean()
                per_filter = frame.groupby('seed')[['source_probability', 'target_probability']].mean()
                assert len(pooled) == len(part)
                row = dict(base, arm=arm, filters=len(per_filter),
                    source_nll=float(-np.log(np.maximum(pooled.source_probability, 1e-300)).mean()),
                    target_nll=float(-np.log(np.maximum(pooled.target_observed_probability, 1e-300)).mean()))
                for key in ('source_probability', 'target_probability'):
                    row[key] = float(pooled[key].mean())
                    row[key+'_mc_se'] = float(per_filter[key].std(ddof=1)/np.sqrt(len(per_filter))) if len(per_filter) > 1 else np.nan
                condition_rows.append(row)
            selected = autonomous[autonomous.event_id.isin(part.event_id)]
            for arm, frame in selected.groupby('arm', sort=True):
                totals = frame.groupby('seed')[['source_match', 'target_match', 'joint_match', 'matched_target_expected']].sum()
                assert frame.groupby('seed').size().eq(len(part)).all()
                source = totals.source_match.to_numpy()
                row = dict(base, arm=arm, paths=len(totals), matching_sources=int(source.sum()),
                    source_match_fraction=float(source.mean()/len(part)),
                    joint_match_fraction=float(totals.joint_match.mean()/len(part)))
                for key, column in [('next_choice_rate', 'joint_match'), ('next_choice_expected', 'matched_target_expected')]:
                    row[key], row[key+'_mc_se'] = path_ratio(totals[column].to_numpy(), source)
                auto_rows.append(row)
                for seed, record in totals.iterrows():
                    path_rows.append(dict(base, arm=arm, seed=int(seed), **record.to_dict()))
    return pd.DataFrame(condition_rows), pd.DataFrame(auto_rows), pd.DataFrame(path_rows)


def run(config: Path, output: Path) -> None:
    settings = yaml.safe_load(config.read_text())
    if set(settings) != {'cue_events', 'finite_fit', 'catalogue_reference', 'subjects', 'diagnostic_rules'}:
        raise ValueError('Declare chronology, frozen fits, subjects and diagnostic rule IDs')
    if output.exists():
        raise FileExistsError('Use a new cue-model comparison directory')
    bindings = {}
    def bind(path, expected=None):
        path = Path(path).resolve(); h = digest(path)
        if (expected is not None and h != expected) or (str(path) in bindings and bindings[str(path)] != h):
            raise ValueError('Source changed: '+str(path))
        bindings[str(path)] = h
        return path
    bind(config); bind(__file__)
    cue, fit, catalogue = (ROOT/settings[k] for k in ('cue_events', 'finite_fit', 'catalogue_reference'))
    cd, fd, gd = (load(bind(p/'completion.json')) for p in (cue, fit, catalogue))
    if any(d['status'] != 'complete' for d in (cd, fd, gd)) or fd['smoke'] or gd['smoke']:
        raise ValueError('Require completed full sources')
    pro = load(bind(fit/'protocol.json', fd['protocol_sha256']))
    nominees = load(bind(fit/'nominees.json', fd['nomination_sha256']))
    if nominees['status'] != 'frozen_before_later_choices':
        raise ValueError('Require behavior-selected frozen nominees')
    if (not settings['subjects'] or len(set(settings['subjects'])) != len(settings['subjects'])
            or not set(settings['subjects']) <= set(pro['subjects']) & set(cd['subjects'])):
        raise ValueError('Subjects must be distinct and shared by the completed sources')
    for record in (cd['inputs_sha256'], pro['inputs_sha256'], gd['bindings']):
        for p, h in record.items():
            bind(p, h)
    events = pd.read_csv(bind(cue/'events.csv', cd['artifacts']['events.csv']))
    trials = pd.read_csv(bind(cue/'trials.csv', cd['artifacts']['trials.csv']))
    events['event_id'] = np.arange(len(events))
    events = events[events.subject.isin(settings['subjects'])].copy()
    observed = events[events.status.eq('observed')].copy()
    conditional_rows, autonomous_rows, rule_rows = [], [], []
    for subject in settings['subjects']:
        pairs = observed[observed.subject.eq(subject)].copy()
        source = pairs.source_ordinal.to_numpy(int)-1
        target = pairs.target_ordinal.to_numpy(int)-1
        reference = pairs.source_choice.to_numpy(int)
        raw = trials[trials.iSub.eq(subject)].sort_values('ordinal_trial')
        np.testing.assert_array_equal(raw.ordinal_trial, np.arange(1, len(raw)+1))
        for arm in ('FROZEN', 'REHEARSE'):
            for stage in ('evaluation', 'autonomous'):
                files = sorted((fit/'jobs').glob(f'{stage}_S{subject}_{arm}_*.json'))
                expected_seeds = pro['seeds'][str(subject)][stage]
                jobs = [load(bind(p, fd['artifacts_sha256'][str(p.relative_to(fit))])) for p in files]
                if sorted(j['seed'] for j in jobs) != sorted(expected_seeds):
                    raise ValueError('Incomplete original filter/path bank')
                for file, job in zip(files, jobs):
                    data = load_trials(job)
                    for key, column in [('choices', 'choice'), ('feedback', 'feedback'), ('categories', 'category')]:
                        np.testing.assert_array_equal(data[key], raw[column])
                    np.testing.assert_array_equal(data['stimulus'], raw[[f'feature{i}' for i in range(1, 5)]])
                    np.testing.assert_array_equal(raw.phase, np.where(np.arange(len(raw)) < job['training_trials'], 'training', 'later'))
                    record_file = Path(job['target']).with_suffix('.json')
                    record = load(bind(record_file, fd['artifacts_sha256'][str(record_file.relative_to(fit))]))
                    if record['status'] != 'complete' or record['job_sha256'] != digest(file):
                        raise ValueError('Job completion mismatch')
                    arrays = Path(job['target']).with_suffix('.npz')
                    with np.load(bind(arrays, record['arrays_sha256'])) as saved:
                        p = saved['probabilities']
                        y = saved['choices'] if stage == 'evaluation' else saved['generated_choices']
                        f = saved['feedback'] if stage == 'evaluation' else saved['generated_feedback']
                        if stage == 'evaluation':
                            np.testing.assert_array_equal(y, data['choices']); np.testing.assert_array_equal(f, data['feedback'])
                        else:
                            np.testing.assert_array_equal(f, (y == data['categories']).astype(float))
                        values = pair_values(p, y, source, target, reference)
                        frame = pd.DataFrame(dict(event_id=pairs.event_id.to_numpy(), subject=subject, arm=arm, seed=job['seed'], **values))
                        if stage == 'evaluation':
                            frame['target_observed_probability'] = p[target, data['choices'][target]-1]
                            conditional_rows.append(frame)
                        else:
                            frame['source_generated_choice'] = y[source]; frame['target_generated_choice'] = y[target]
                            frame['source_generated_feedback'] = f[source]; frame['target_generated_feedback'] = f[target]
                            autonomous_rows.append(frame)
        name = f'chunks/evaluation_S{subject}_GLOBAL.npz'
        with np.load(bind(catalogue/name, gd['artifacts'][name])) as saved:
            p, y, f = saved['probabilities'], saved['choices'], saved['feedback']
            np.testing.assert_array_equal(y, raw.choice); np.testing.assert_array_equal(f, raw.feedback)
            np.testing.assert_array_equal(saved['categories'], raw.category)
            q, a, w = saved['geometry'], saved['associations'], saved['rule_weights']
            rule_prediction = np.einsum('thg,thgc->thc', q, a)
            np.testing.assert_allclose(np.einsum('th,thc->tc', w, rule_prediction), p, atol=1e-12, rtol=0)
            frame = pd.DataFrame(dict(event_id=pairs.event_id.to_numpy(), subject=subject, arm='GLOBAL', seed=-1,
                **pair_values(p, y, source, target, reference)))
            frame['target_observed_probability'] = p[target, y[target]-1]; conditional_rows.append(frame)
            for rule in settings['diagnostic_rules']:
                if type(rule) is not int or not 0 <= rule < w.shape[1]:
                    raise ValueError('Diagnostic rule outside the source catalogue')
                rule_rows.append(pd.DataFrame(dict(event_id=pairs.event_id.to_numpy(), subject=subject, rule=rule,
                    source_weight=w[source, rule], target_weight=w[target, rule],
                    source_rule_probability=rule_prediction[source, rule, reference-1],
                    target_rule_probability=rule_prediction[target, rule, reference-1])))
    conditional = pd.concat(conditional_rows, ignore_index=True)
    autonomous = pd.concat(autonomous_rows, ignore_index=True)
    cs, ats, ps = summarize(observed, conditional, autonomous)
    output.mkdir(parents=True)
    tables = dict(events=events, conditional_events=conditional, autonomous_events=autonomous,
        catalogue_rule_events=pd.concat(rule_rows, ignore_index=True),
        conditional_summary=cs, autonomous_summary=ats, autonomous_path_summary=ps)
    for name, table in tables.items():
        table.to_csv(output/(name+'.csv'), index=False)
    for p, h in bindings.items():
        if digest(Path(p)) != h:
            raise ValueError('Input changed during comparison: '+p)
    write(output/'completion.json', dict(status='complete', settings=settings, event_rows=len(events),
        observed_pairs=len(observed), conditional_rows=len(conditional), autonomous_rows=len(autonomous),
        inputs_sha256=bindings, nominees=nominees['nominees'],
        scope='Development posthoc choice recurrence; no oral generation, continuous latent retention or causal feedback effect inferred',
        uncertainty='Independent filters or whole autonomous paths; describes simulation error, not participant uncertainty',
        fitted_arms='FROZEN and REHEARSE retain original equal-grid behavior fits; GLOBAL retains its separate catalogue-reference parameters',
        oral_parameter_fitting=False, artifacts={p.name: digest(p) for p in output.glob('*.csv')}))
    print(f'{len(observed)} observed pairs; {len(conditional)} conditional rows; {len(autonomous)} autonomous rows', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)

"""Describe report recurrence at stimulus-defined opportunities after feedback.

Use --config YAML --output NEW_DIRECTORY. Literal single-clause extrema reports
define origins. For every origin retain the immediate next trial, the first
future stimulus satisfying its weak extremum, and the first satisfying its
strict extremum. Targets depend on stimuli only, within the same session and
training/evaluation phase. A missing opportunity is censored, not non-recurrence.

All original trials, compound wording at targets, feedback levels and source
cue truth are retained. Exact wording and response recurrence are descriptive
observables; they do not establish uninterrupted internal-rule retention or a
causal effect of feedback. Oral data never estimate model parameters here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ...evaluation.model_0923_h_validation import digest, load
from .model_0923_h_validation import write

ROOT = Path(__file__).resolve().parents[4]


def extremum_gap(stimulus: np.ndarray, dimension: int, direction: str) -> np.ndarray:
    """Signed margin against all other features; zero preserves exact physical ties."""
    x = np.asarray(stimulus, float)
    if (x.ndim != 2 or x.shape[1] != 4 or not np.isfinite(x).all()
            or type(dimension) is not int or not 0 <= dimension < 4 or direction not in ('max', 'min')):
        raise ValueError('Require finite four-feature stimuli and a declared extremum')
    others = np.delete(x, dimension, axis=1)
    return x[:, dimension]-others.max(1) if direction == 'max' else others.min(1)-x[:, dimension]


def next_opportunity(eligible: np.ndarray, sessions: np.ndarray,
                     phases: np.ndarray, origin: int) -> tuple[int | None, str]:
    """Return the first eligible later trial before either boundary, or censor it."""
    eligible, sessions, phases = map(np.asarray, (eligible, sessions, phases))
    if (eligible.ndim != 1 or eligible.dtype != np.bool_ or sessions.shape != eligible.shape
            or phases.shape != eligible.shape or type(origin) is not int or not 0 <= origin < len(eligible)):
        raise ValueError('Require aligned eligibility, session, phase and a valid origin')
    for target in range(origin+1, len(eligible)):
        if sessions[target] != sessions[origin]: return None, 'session_boundary'
        if phases[target] != phases[origin]: return None, 'phase_boundary'
        if eligible[target]: return target, 'observed'
    return None, 'sequence_end'


def summarize(events: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Use observed opportunities as recurrence denominators, with censoring counts."""
    rows = []
    for key, group in events.groupby(keys, dropna=False, sort=True):
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        observed = group[group.status == 'observed']
        row.update(origins=len(group), opportunities=len(observed), censored=int((group.status != 'observed').sum()))
        for name in ('same_wording', 'mentions_literal_phrase', 'same_choice', 'same_wording_and_choice'):
            row[name+'_count'] = int(observed[name].sum())
            row[name+'_fraction'] = float(observed[name].mean()) if len(observed) else np.nan
        row['median_gap_trials'] = float(observed.gap_trials.median()) if len(observed) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def run(config: Path, output: Path) -> None:
    settings = yaml.safe_load(config.read_text())
    if set(settings) != {'oral_audit', 'data', 'subjects', 'templates'}:
        raise ValueError('Declare existing oral audit, prepared data, subjects and literal templates')
    subjects = settings['subjects']; templates = settings['templates']
    if not subjects or len(set(subjects)) != len(subjects) or not templates:
        raise ValueError('Require distinct subjects and declared templates')
    for text, cue in templates.items():
        if (not isinstance(text, str) or not text.endswith('。') or set(cue) != {'feature', 'direction'}
                or cue['feature'] not in ('head', 'neck', 'leg', 'tail') or cue['direction'] not in ('max', 'min')):
            raise ValueError('Require literal single-clause extrema templates')
    output = output.resolve()
    if output.exists(): raise FileExistsError('Use a new oral cue persistence directory')
    bindings = {}
    def bind(path, expected=None):
        p = Path(path).resolve(); h = digest(p)
        if expected is not None and h != expected: raise ValueError('Input changed: '+str(p))
        if str(p) in bindings and bindings[str(p)] != h: raise ValueError('Input changed during analysis')
        bindings[str(p)] = h
        return p
    bind(config); bind(__file__)
    audit = ROOT/settings['oral_audit']; done = load(bind(audit/'manifest.json'))
    if done['status'] != 'complete': raise ValueError('Require a completed frozen oral audit')
    oral_file = bind(audit/'trials.csv', done['artifacts']['trials.csv'])
    for path, h in done['input_sha256'].items(): bind(ROOT/path, h)
    oral = pd.read_csv(oral_file, keep_default_na=False)
    events, all_trials, coverage = [], [], []
    for subject in subjects:
        folder = ROOT/settings['data']/f'subject_{subject}'
        meta = load(bind(folder/'prepared.json'))
        if meta['subject'] != subject: raise ValueError('Prepared subject mismatch')
        parts = []
        for name in ('training.npz', 'sealed_test.npz'):
            with np.load(bind(folder/name, meta['array_sha256'][name])) as z:
                parts.append({k: z[k] for k in ('stimulus', 'choices', 'feedback', 'categories')})
        data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        raw = pd.read_csv(bind(meta['paths']['learning_data'], meta['learning_data_sha256']), keep_default_na=False)
        raw = raw[raw.iSub == subject].sort_values(['iSession', 'iTrial']).reset_index(drop=True)
        report = oral[oral.iSub == subject].sort_values('ordinal_trial').reset_index(drop=True)
        n = len(raw)
        if n != meta['trials'] or len(report) != n: raise ValueError('Incomplete subject sequence')
        for name in ('iSession', 'iTrial', 'choice', 'text', 'condition', *(f'feature{i}_name' for i in range(1, 5))):
            np.testing.assert_array_equal(raw[name], report[name])
        for name, column in [('choices', 'choice'), ('feedback', 'feedback'), ('categories', 'category')]:
            np.testing.assert_array_equal(data[name], raw[column])
        np.testing.assert_array_equal(data['stimulus'], raw[[f'feature{i}' for i in range(1, 5)]])
        np.testing.assert_array_equal(report.ordinal_trial, np.arange(1, n+1))
        feature_names = raw[[f'feature{i}_name' for i in range(1, 5)]].to_numpy()
        if (set(feature_names[0]) != {'head', 'neck', 'leg', 'tail'}
                or np.any(feature_names != feature_names[0])):
            raise ValueError('Require a stable named feature order within each subject')
        sessions = raw.iSession.to_numpy(); phases = np.where(np.arange(n) < meta['training_trials'], 'training', 'later')
        gaps = {text: extremum_gap(data['stimulus'], list(feature_names[0]).index(cue['feature']), cue['direction'])
                for text, cue in templates.items()}
        raw['ordinal_trial'] = np.arange(1, n+1); raw['phase'] = phases; raw['scope'] = report.scope
        raw['literal_template_origin'] = raw.text.isin(templates)
        all_trials.append(raw)
        for phase in ('training', 'later'):
            group = raw[raw.phase == phase]
            coverage.append(dict(subject=subject, condition=meta['condition'], phase=phase, trials=len(group),
                nonempty=int(group.text.str.len().gt(0).sum()), literal_origins=int(group.literal_template_origin.sum())))
        for origin in np.flatnonzero(raw.literal_template_origin):
            origin = int(origin); text = raw.text.iloc[origin]; cue = templates[text]; gap = gaps[text]
            status = 'strict' if gap[origin] > 0 else 'tie' if gap[origin] == 0 else 'false'
            for event, eligible in [('next_trial', np.ones(n, bool)), ('next_weak_cue', gap >= 0), ('next_strict_cue', gap > 0)]:
                target, observed = next_opportunity(eligible, sessions, phases, origin)
                row = dict(subject=subject, condition=meta['condition'], phase=phases[origin], session=int(sessions[origin]),
                    source_trial=int(raw.iTrial.iloc[origin]), source_ordinal=origin+1, text=text,
                    feature=cue['feature'], direction=cue['direction'], source_cue_status=status,
                    source_cue_gap=float(gap[origin]), source_choice=int(data['choices'][origin]),
                    source_category=int(data['categories'][origin]), source_feedback=float(data['feedback'][origin]),
                    event=event, status=observed)
                if target is not None:
                    next_text = raw.text.iloc[target]; same = bool(next_text == text)
                    same_choice = bool(data['choices'][target] == data['choices'][origin])
                    row.update(target_ordinal=target+1, target_trial=int(raw.iTrial.iloc[target]),
                        target_text=next_text, target_scope=report.scope.iloc[target], gap_trials=target-origin,
                        target_choice=int(data['choices'][target]), target_category=int(data['categories'][target]),
                        target_feedback=float(data['feedback'][target]), target_cue_gap=float(gap[target]),
                        same_wording=same, mentions_literal_phrase=bool(text[:-1] in next_text),
                        same_choice=same_choice, same_wording_and_choice=same and same_choice,
                        same_task_category=bool(data['categories'][target] == data['categories'][origin]),
                        intervening_trials=target-origin-1,
                        intervening_incorrect=int((data['feedback'][origin+1:target] == 0.).sum()),
                        intervening_partial=int((data['feedback'][origin+1:target] == .5).sum()),
                        intervening_correct=int((data['feedback'][origin+1:target] == 1.).sum()))
                events.append(row)
    events = pd.DataFrame(events); output.mkdir(parents=True)
    common = ['subject', 'condition', 'phase', 'event', 'source_feedback']
    tables = dict(events=events, trials=pd.concat(all_trials, ignore_index=True), coverage=pd.DataFrame(coverage),
        summary=summarize(events, common),
        cue_summary=summarize(events, common+['text', 'source_choice']),
        truth_summary=summarize(events, common+['source_cue_status']))
    for name, table in tables.items(): table.to_csv(output/(name+'.csv'), index=False)
    for path, h in bindings.items():
        if digest(Path(path)) != h: raise ValueError('Input changed while computing recurrence')
    write(output/'completion.json', dict(status='complete', subjects=subjects,
        origins=int(tables['coverage'].literal_origins.sum()), opportunities=int((events.status == 'observed').sum()),
        event_rows=len(events), inputs_sha256=bindings,
        policy='First stimulus-defined target, same session and fitting phase; exact ties retained; all feedback levels separate',
        scope='Development description of literal recurrence; no latent retention, causal feedback effect or oral likelihood inferred',
        oral_parameter_fitting=False, artifacts={p.name: digest(p) for p in output.glob('*.csv')}))
    print(tables['coverage'].to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); run(args.config, args.output)

"""Audit bare long/short contrast descriptions against observed stimuli.

Run --oral-audit COMPLETED_DEVELOPMENT_AUDIT --output NEW_DIRECTORY. Preserve
all audited report identities and explicitly record eligibility. Eligible reports
contain both long and short singleton midpoint facets without an explicit
numeric, body, comparative or ordinal reference. Compare the existing absolute
reading with an exploratory within-stimulus group ordering. No coding, model
state, parameter or raw data is changed. Relative regions have larger volume;
stimulus compatibility is a diagnostic, not a likelihood for the oral report.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .model_0923_decision_basis_alignment import ROOT, digest, read

EXPLICIT_REFERENCE = re.compile(r'半|分之|[0-9０-９]|中点|范围|躯干|身体|身子|比(?!较)|长于|短于|大于|小于|高于|低于|超过|最|第[一二三四]|一样|等长|相等|持平')


def contrast_dimensions(text: str, tokens: str, scope: str) -> tuple[str, list[int], list[int]]:
    """Classify existing audited predicates without rewriting their meaning."""
    if scope != 'complete': return 'outside_complete_report_scope', [], []
    if EXPLICIT_REFERENCE.search(str(text)): return 'explicit_reference_or_ordering', [], []
    values = json.loads(tokens)
    if not isinstance(values, list): raise ValueError('Audited tokens must be a list')
    long, short = set(), set()
    for token in values:
        if not isinstance(token, str) or not token.startswith('facet:'):
            return 'other_predicate', [], []
        row = np.asarray(json.loads(token[6:]), float)
        if row.ndim != 2 or row.shape[1] != 5 or not np.isfinite(row).all():
            raise ValueError('Invalid audited facet')
        if row.shape != (1,5): return 'other_predicate', [], []
        nonzero = np.flatnonzero(row[0,:4])
        if len(nonzero) != 1: return 'other_predicate', [], []
        dimension = int(nonzero[0]); coefficient = row[0,dimension]
        if coefficient not in (-1.,1.) or row[0,4] != coefficient*.5:
            return 'other_predicate', [], []
        (long if coefficient < 0 else short).add(dimension)
    if long & short: raise ValueError('Complete report assigns both polarities to one dimension')
    if not long or not short: return 'single_polarity_or_empty', [], []
    return 'eligible', sorted(long), sorted(short)


def stimulus_diagnostics(stimulus: np.ndarray, long: list[int], short: list[int]) -> dict:
    """Compare two fixed readings and state their unit-cube region volumes."""
    x = np.asarray(stimulus, float)
    if (x.shape != (4,) or not np.isfinite(x).all() or np.any((x < 0) | (x > 1))
            or not long or not short or set(long)&set(short)
            or len(set(long)) != len(long) or len(set(short)) != len(short)
            or not set(long+short) <= set(range(4))):
        raise ValueError('Need disjoint nonempty groups and a normalized four-dimensional stimulus')
    minimum_long = float(x[long].min()); maximum_short = float(x[short].max())
    relative = minimum_long-maximum_short
    absolute = min(minimum_long-.5, .5-maximum_short)
    used = x[long+short]; tolerance = 1e-12
    n = len(long)+len(short)
    return dict(minimum_reported_long=minimum_long, maximum_reported_short=maximum_short,
        absolute_margin=absolute, relative_margin=relative,
        absolute_match=bool(absolute >= -tolerance), relative_match=bool(relative >= -tolerance),
        relative_only=bool(relative >= -tolerance and absolute < -tolerance),
        relative_tie=bool(abs(relative) <= tolerance),
        all_reported_below_midpoint=bool(np.all(used < .5-tolerance)),
        all_reported_above_midpoint=bool(np.all(used > .5+tolerance)),
        absolute_unit_cube_volume=.5**n,
        relative_unit_cube_volume=1/math.comb(n,len(long)))


def join_stimuli(oral: pd.DataFrame, behavior: pd.DataFrame) -> pd.DataFrame:
    """Retain the exact participant/session/trial and feature-order mapping."""
    keys = ['iSub','iSession','iTrial']
    selected = behavior[behavior.iSub.isin(oral.iSub.unique())].copy()
    if 'miss' in selected: selected = selected[selected['miss'].eq(0)].copy()
    if oral.duplicated(keys).any() or selected.duplicated(keys).any(): raise ValueError('Repeated trial identity')
    selected['behavior_ordinal'] = selected.groupby('iSub').cumcount()+1
    identity = ['condition','choice']+[f'feature{k}_name' for k in range(1,5)]
    features = [f'feature{k}' for k in range(1,5)]
    joined = oral.merge(selected[keys+identity+features+['category','behavior_ordinal']], on=keys,
        how='outer', validate='one_to_one', suffixes=('','_behavior'), indicator=True)
    if not joined['_merge'].eq('both').all(): raise ValueError('Audited and behavioral trials differ')
    for name in identity: np.testing.assert_array_equal(joined[name], joined[name+'_behavior'])
    np.testing.assert_array_equal(joined.ordinal_trial, joined.behavior_ordinal)
    return joined.drop(columns=['_merge', 'behavior_ordinal', *[name+'_behavior' for name in identity]])


def summarize(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count compatibility and bound a common threshold without estimating it."""
    summaries, profiles = [], []
    for subject, group in table.groupby('iSub', sort=True):
        eligible = group[group.eligible]
        row = dict(subject=int(subject), all_reports=len(group), complete_reports=int(group.scope.eq('complete').sum()),
                   eligible_reports=len(eligible))
        for name in ('absolute_match','relative_match','relative_only','relative_tie',
                     'all_reported_below_midpoint','all_reported_above_midpoint'):
            row[name] = int(eligible[name].sum()) if len(eligible) else 0
        summaries.append(row)
        for (choice,long,short), part in eligible.groupby(['choice','long_dimensions','short_dimensions'],sort=True):
            lower = float(part.maximum_reported_short.max()); upper = float(part.minimum_reported_long.min())
            profiles.append(dict(subject=int(subject), choice=int(choice), long_dimensions=long, short_dimensions=short,
                reports=len(part), first_trial=int(part.ordinal_trial.min()), last_trial=int(part.ordinal_trial.max()),
                absolute_matches=int(part.absolute_match.sum()), relative_matches=int(part.relative_match.sum()),
                common_threshold_lower=lower, common_threshold_upper=upper,
                common_strict_threshold_exists=bool(lower < upper), example=part.text.iloc[0]))
    return pd.DataFrame(summaries), pd.DataFrame(profiles)


def run(audit: Path, output: Path) -> None:
    if output.exists(): raise FileExistsError('Use a new contrast-reference audit directory')
    manifest = read(audit/'manifest.json')
    if manifest['status'] != 'complete' or manifest['config']['sample_role'] != 'development':
        raise ValueError('Require a completed development oral audit')
    inputs = {str((audit/'manifest.json').resolve()):digest(audit/'manifest.json')}
    for folder, hashes in [(ROOT,manifest['input_sha256']), (audit,manifest['artifacts'])]:
        for name, expected in hashes.items():
            file = folder/name
            if digest(file) != expected: raise ValueError('Oral audit or original input changed')
            inputs[str(file.resolve())] = expected
    oral = pd.read_csv(audit/'trials.csv')
    if set(oral.iSub.unique()) != set(manifest['config']['subjects']): raise ValueError('Audit participant scope changed')
    processed = ROOT/manifest['config']['processed']
    table = join_stimuli(oral, pd.read_csv(processed))
    rows = []
    for row in table.to_dict('records'):
        reason, long, short = contrast_dimensions(row['text'],row['tokens'],row['scope'])
        result = dict(row, eligible=reason=='eligible', eligibility_reason=reason,
            long_dimensions=json.dumps(long), short_dimensions=json.dumps(short))
        if reason == 'eligible':
            result.update(stimulus_diagnostics(np.array([row[f'feature{k}'] for k in range(1,5)]),long,short))
        rows.append(result)
    table = pd.DataFrame(rows)
    summary, profiles = summarize(table)
    output.mkdir(parents=True)
    for name, frame in [('all_reports',table),('eligible_reports',table[table.eligible]),
                        ('subjects',summary),('profiles',profiles)]:
        frame.to_csv(output/(name+'.csv'),index=False)
    (output/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8">'
        '<style>body{max-width:1250px;margin:30px auto;font:16px/1.6 sans-serif}td,th{padding:5px}</style>'
        '<h1>裸长短对照描述的参照核对</h1><p>全部原报告保留。检查范围为已有完整编码、同时含裸长与短、'
        '且没有明确数字、身体、比较或顺序参照的报告。原编码保持记录值。</p>'
        '<p>绝对读法要求每个长部位至少为0.5、短部位至多为0.5；相对读法要求所有长组部位至少与所有短组部位一样长。'
        '精确平局另行记录。相对区域更大，刺激匹配数只提供参照线索。表中的单位立方体体积描述几何范围，'
        '实际任务的刺激分布由实验抽样决定。共同阈值区间只检查跨试次一致性，未估计行为参数。</p>'
        +summary.to_html(index=False)+
        '<p><a href="all_reports.csv">全部报告及纳入原因</a> · <a href="eligible_reports.csv">全部可比较报告</a> · '
        '<a href="profiles.csv">按被试与所选类别划分的判断组</a></p>')
    for name, expected in inputs.items():
        if digest(Path(name)) != expected: raise ValueError('Audit input changed during analysis')
    result = dict(status='complete', oral_audit=str(audit.resolve()), subjects=sorted(table.iSub.unique().tolist()),
        input_sha256=inputs, source_sha256={str(Path(__file__).relative_to(ROOT)):digest(Path(__file__))},
        versions={name:version(name) for name in ('numpy','pandas')},
        data_use='development semantic diagnostic; no recoding, model fitting or parameter selection',
        eligibility_regex=EXPLICIT_REFERENCE.pattern, eligibility_counts=table.eligibility_reason.value_counts().to_dict(),
        interpretation='Two fixed readings; relative region includes the absolute region. Compatibility alone does not identify intended meaning.',
        geometry='On independent uniform unit-cube coordinates: absolute volume=2^(-m), relative volume=1/choose(m,n_long). Actual task sampling need not be uniform.',
        artifacts={file.name:digest(file) for file in output.iterdir()})
    (output/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oral-audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.oral_audit,args.output)

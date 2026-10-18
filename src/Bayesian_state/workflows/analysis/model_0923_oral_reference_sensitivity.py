"""Build a fixed relative-reading sensitivity audit from an existing oral audit.

Run --oral-audit COMPLETED_DEVELOPMENT_AUDIT --output NEW_DIRECTORY --workers N.
All eligible bare long/short groups receive the same exploratory relative
interpretation. Eligibility uses only text and previously audited predicates;
this workflow reads no stimulus values, model states or fit scores. Explicit
references and every other report retain their original encoding. The original
audit remains the absolute-reading reference. Both audits can be passed to the
existing decision-basis alignment workflow without changing any model state.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy

from . import model_0923_oral_constraint_audit as audit_tools
from . import model_0923_oral_contrast_reference as reference
from ...model.modules.readout.report_space import facet_token

READING = 'bare-long-short-groups-relative-sensitivity-v1'


def relative_tokens(long: list[int], short: list[int]) -> tuple[str, ...]:
    """Encode every long > short pair, preserving dimensions and conjunction."""
    if (not long or not short or set(long) & set(short)
            or len(set(long)) != len(long) or len(set(short)) != len(short)
            or not set(long+short) <= set(range(4))):
        raise ValueError('Need disjoint nonempty dimension groups')
    tokens = []
    for longer in long:
        for shorter in short:
            row = np.zeros((1,4)); row[0,longer] = -1.; row[0,shorter] = 1.
            tokens.append(facet_token(row, np.zeros(1)))
    return tuple(sorted(tokens))


def change_reading(trials: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Apply one global reading to every text-eligible report, before scoring."""
    result = trials.copy(deep=True)
    for name in ('tokens', 'query_id', 'n_predicates', 'flags'):
        result['original_'+name] = result[name]
    result['reference_sensitivity_changed'] = False
    result['reference_sensitivity_reason'] = ''
    queries = {}
    for index, row in result.iterrows():
        reason, long, short = reference.contrast_dimensions(row.text, row.tokens, row.scope)
        result.at[index,'reference_sensitivity_reason'] = reason
        if reason != 'eligible': continue
        tokens = relative_tokens(long, short)
        payload = dict(condition=int(row.condition), choice=int(row.choice), tokens=list(tokens))
        query = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]
        queries[query] = dict(query_id=query, **payload)
        result.at[index,'tokens'] = json.dumps(tokens)
        result.at[index,'query_id'] = query
        result.at[index,'n_predicates'] = len(tokens)
        flags = set(json.loads(row['flags'])) - {'absolute_threshold_convention'}
        result.at[index,'flags'] = json.dumps(sorted(flags | {READING}))
        result.at[index,'reference_sensitivity_changed'] = True
    return result, list(queries.values())


def assemble(trials: pd.DataFrame, old_queries: pd.DataFrame, old_matches: pd.DataFrame,
             calculated: list[tuple[dict, list[dict]]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Join verdicts by semantic query; retain shared explicit-reference rows."""
    summaries = [old_queries]
    components = [old_matches]
    for summary, matches in calculated:
        summaries.append(pd.DataFrame([summary])); components.append(pd.DataFrame(matches))
    summary = pd.concat(summaries, ignore_index=True)
    matches = pd.concat(components, ignore_index=True)
    if summary.query_id.duplicated().any() or matches.duplicated(['query_id','rule','component']).any():
        raise ValueError('Repeated catalogue query or component')
    used = set(trials.query_id.dropna())
    if not used <= set(summary.query_id): raise ValueError('Missing sensitivity query verdicts')
    summary = summary[summary.query_id.isin(used)].reset_index(drop=True)
    matches = matches[matches.query_id.isin(used)].reset_index(drop=True)
    columns = [name for name in summary if name not in ('query_id','condition','choice','tokens')]
    base = trials.drop(columns=columns)
    result = base.merge(summary[['query_id',*columns]], on='query_id', how='left', validate='many_to_one', sort=False)
    for name in columns: result[name] = result[name].astype(trials[name].dtype)
    np.testing.assert_array_equal(result[['iSub','iSession','iTrial']], trials[['iSub','iSession','iTrial']])
    unchanged = ~trials.reference_sensitivity_changed
    pd.testing.assert_frame_equal(result.loc[unchanged,columns].reset_index(drop=True),
                                  trials.loc[unchanged,columns].reset_index(drop=True))
    return result, summary, matches


def run(audit: Path, output: Path, requested_workers: int) -> None:
    if output.exists(): raise FileExistsError('Use a new sensitivity audit directory')
    if requested_workers < 1: raise ValueError('workers must be positive')
    read, digest, root = reference.read, reference.digest, reference.ROOT
    original = read(audit/'manifest.json')
    if (original['status'] != 'complete' or original['config']['sample_role'] != 'development'
            or original.get('semantic_sensitivity') is not None):
        raise ValueError('Require a completed original development audit')
    inputs = {str((audit/'manifest.json').resolve()):digest(audit/'manifest.json')}
    for base, hashes in [(root, original['input_sha256']), (audit, original['artifacts'])]:
        for name, sha in hashes.items():
            if digest(base/name) != sha: raise ValueError('Original audit or input changed')
            inputs[str((base/name).resolve())] = sha
    # Reused catalogue verdicts and newly calculated verdicts use identical math.
    for name, sha in original['source_sha256'].items():
        if digest(root/name) != sha: raise ValueError('Catalogue or encoding dependencies changed')
    before = pd.read_csv(audit/'trials.csv')
    if set(before.iSub.unique()) != set(original['config']['subjects']):
        raise ValueError('Declared development participants changed')
    trials, queries = change_reading(before)
    old_queries = pd.read_csv(audit/'constraint_queries.csv')
    old_matches = pd.read_csv(audit/'component_matches.csv')
    known = old_queries.set_index('query_id')
    jobs = []
    for query in queries:
        if query['query_id'] in known.index:
            row = known.loc[query['query_id']]
            if (row.condition != query['condition'] or row.choice != query['choice']
                    or json.loads(row.tokens) != query['tokens']):
                raise ValueError('Query identifier collision')
        else:
            jobs.append(dict(query, tolerance=original['config']['tolerance']))
    workers = min(requested_workers, 128, len(os.sched_getaffinity(0)), len(jobs)) if jobs else 0
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    sources = dict(original['source_sha256'])
    for file in (Path(__file__), Path(reference.__file__), root/'src/Bayesian_state/workflows/analysis/model_0923_decision_basis_alignment.py'):
        sources[str(file.relative_to(root))] = digest(file)
    manifest = dict(status='running', protocol=original['protocol'], encoder=original['encoder']+'+'+READING,
        config=original['config'], source_audit=str(audit.resolve()), input_sha256=inputs, source_sha256=sources,
        semantic_sensitivity=READING, changed_reports=int(trials.reference_sensitivity_changed.sum()),
        eligibility_regex=reference.EXPLICIT_REFERENCE.pattern, workers=workers, new_queries=len(jobs),
        distinct_changed_queries=len(queries), data_use='development fixed-reading sensitivity; behavior-only model states unchanged',
        interpretation='All text-eligible bare groups receive all long>short predicates; no selection using stimuli or model scores.',
        role_assumption=original['role_assumption'], component_handling=original['component_handling'],
        timing_for_future_state_alignment=original['timing_for_future_state_alignment'],
        started_utc=datetime.now(timezone.utc).isoformat(),
        versions=dict(python=platform.python_version(), numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__))
    output.mkdir(parents=True); audit_tools.write_json(output/'manifest.json',manifest)
    print(f'{len(trials)} reports; {manifest["changed_reports"]} sensitivity readings; {len(jobs)} new queries; {workers} workers',flush=True)
    calculated = []
    if workers == 1:
        calculated = list(map(audit_tools.audit_query, jobs))
    elif jobs:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            for number, item in enumerate(pool.map(audit_tools.audit_query,jobs,chunksize=1),1):
                calculated.append(item)
                if number % 25 == 0: print(f'Calculated {number}/{len(jobs)} queries',flush=True)
    trials, queries_frame, matches = assemble(trials,old_queries,old_matches,calculated)
    unchanged = ~trials.reference_sensitivity_changed
    pd.testing.assert_frame_equal(trials.loc[unchanged,before.columns].reset_index(drop=True),
                                  before.loc[unchanged].reset_index(drop=True))
    for name in ['iSub','iSession','iTrial','ordinal_trial','text','scope','status','choice','condition',
                 *[f'feature{k}_name' for k in range(1,5)]]:
        pd.testing.assert_series_equal(trials[name],before[name])
    coverage = trials.groupby('iSub').agg(reports=('iSub','size'),changed=('reference_sensitivity_changed','sum'))
    for name, frame in [('trials',trials),('constraint_queries',queries_frame),('component_matches',matches),('subjects',coverage.reset_index())]:
        frame.to_csv(output/(name+'.csv'),index=False)
    (output/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8">'
        '<style>body{max-width:1100px;margin:30px auto;font:16px/1.6 sans-serif}td,th{padding:5px}</style>'
        '<h1>裸长短的固定相对读法敏感性检验</h1><p>所有符合文本条件的长短组统一转换为长组每项大于短组每项。'
        '明确含“一半”的报告与其他未纳入报告保持原编码。绝对读法仍保存在原审查目录；本结果为探索性的另一种固定解释。</p>'
        '<p>纳入条件只读取原话及原约束，未读取刺激数值或模型得分。两种读法允许的范围不同，'
        '应同时报告显式谓词支持、几何支持、完整报告共同分母及仅受影响报告的结果。参数与状态仅由行为确定。</p>'
        +coverage.to_html()+'<p><a href="trials.csv">全部报告与原编码</a> · <a href="constraint_queries.csv">约束查询</a> · '
        '<a href="component_matches.csv">全部分量判定</a></p></html>')
    for name, sha in {**inputs, **sources}.items():
        if digest(root/name) != sha: raise ValueError('Sensitivity source changed during analysis')
    manifest.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),trials=len(trials),
        scope_counts=trials.scope.value_counts().to_dict(), unique_queries=len(queries_frame),
        artifacts={p.name:digest(p) for p in output.iterdir() if p.name!='manifest.json'})
    audit_tools.write_json(output/'manifest.json',manifest)
    print('Sensitivity audit complete',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oral-audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=128)
    args=parser.parse_args();run(args.oral_audit,args.output,args.workers)

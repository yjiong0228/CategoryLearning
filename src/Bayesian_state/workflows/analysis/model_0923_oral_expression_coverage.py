"""Separate single-predicate coverage from complete-report representability.

Use --absolute-audit AUDIT --relative-audit AUDIT --output NEW_DIRECTORY.
Both inputs are completed development-corpus audits. The existing frozen
tokens and rule catalogue are used unchanged; no behavioral state is loaded.
Componentwise maxima are catalogue ceilings, never fitted alignment scores.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from src.Bayesian_state.model.modules.readout.report_space import component_tokens
from src.Bayesian_state.workflows.analysis.model_0923_oral_constraint_audit import catalogue


def token_coverage(tokens: tuple[str, ...], selected: list[frozenset], all_choices: list[frozenset]) -> dict:
    """Count explicit predicates without choosing a favorable cognitive state."""
    report=frozenset(tokens)
    if not report or len(report)!=len(tokens) or not selected or not all_choices:
        raise ValueError('Require unique nonempty report predicates and catalogue components')
    union=frozenset().union(*selected);any_label=frozenset().union(*all_choices)
    counts=[len(report&component) for component in selected]
    covered=len(report&union);every_label=len(report&any_label)
    whole=max(counts)==len(report)
    return dict(predicates=len(report),individually_covered=covered,
        individually_covered_any_choice=every_label,max_predicates_in_one_component=max(counts),
        max_fraction_in_one_component=max(counts)/len(report),
        all_individually_covered=covered==len(report),whole_structural_support=whole,
        combination_limited=covered==len(report) and not whole,
        any_uncovered_selected_choice=covered<len(report),
        any_uncovered_all_choices=every_label<len(report),
        missing_tokens_selected_choice=json.dumps(sorted(report-union),ensure_ascii=False),
        missing_tokens_all_choices=json.dumps(sorted(report-any_label),ensure_ascii=False))


def run(absolute: Path, relative: Path, output: Path) -> dict:
    if output.exists():raise FileExistsError('Use a new expression coverage output')
    inputs={};checks=0;frames=[];query_rows=[];catalogue_rows=[];expected_identity=None

    def bind(path: Path) -> str:
        sha=hashlib.sha256(path.read_bytes()).hexdigest();inputs[str(path.resolve())]=sha;return sha

    components={}
    for categories in (2,4):
        space=catalogue(categories);counts=Counter()
        for choice in range(1,categories+1):
            own=[]
            for rule in space:
                for index,component in enumerate(rule.categories[choice-1].components):
                    predicates=frozenset(component_tokens(component.A,component.b));counts[len(predicates)]+=1
                    own.append(dict(rule=rule.index,component=index,predicates=predicates))
            components[categories,choice]=own
        catalogue_rows.extend(dict(categories=categories,rules=len(space),predicates=number,components=count)
                              for number,count in sorted(counts.items()))
    for reading,folder in [('midpoint',absolute),('relative',relative)]:
        manifest=json.loads((folder/'manifest.json').read_text());bind(folder/'manifest.json')
        if manifest['status']!='complete' or manifest['config']['sample_role']!='development':
            raise ValueError('Completed development-corpus audits required')
        for name in ('trials.csv','component_matches.csv'):
            if bind(folder/name)!=manifest['artifacts'][name]:raise ValueError('Oral audit artifact changed')
        trials=pd.read_csv(folder/'trials.csv');matches=pd.read_csv(folder/'component_matches.csv')
        identity=trials[['iSub','iSession','iTrial','condition','choice','text','scope']].fillna('').sort_values(['iSub','iSession','iTrial']).reset_index(drop=True)
        if expected_identity is None:expected_identity=identity
        else:pd.testing.assert_frame_equal(identity,expected_identity)
        complete=trials[trials.scope=='complete'];lookup={}
        for query,group in complete.groupby('query_id',sort=False):
            row=group.iloc[0];tokens=tuple(json.loads(row.tokens));categories=2 if row.condition==1 else 4
            own=components[categories,int(row.choice)]
            all_choices=[x['predicates'] for c in range(1,categories+1) for x in components[categories,c]]
            result=token_coverage(tokens,[x['predicates'] for x in own],all_choices)
            previous=matches[matches.query_id==query].set_index(['rule','component'])
            if len(previous)!=len(own) or previous.index.has_duplicates:raise ValueError('Incomplete prior component audit')
            for item in own:
                count=len(frozenset(tokens)&item['predicates']);saved=previous.loc[item['rule'],item['component']]
                if int(saved.explicit_predicates)!=count or (saved.structural=='support')!=(count==len(tokens)):
                    raise ValueError('Current catalogue differs from frozen explicit-predicate audit')
                checks+=2
            if result['whole_structural_support']!=bool(row.structural_support_any_component>0):
                raise ValueError('Whole-report coverage changed')
            checks+=1
            lookup[query]=result
            query_rows.append(dict(reading=reading,query_id=query,condition=int(row.condition),choice=int(row.choice),
                reports=len(group),tokens=row.tokens,**result))
        extra=pd.DataFrame([dict(query_id=query,**result) for query,result in lookup.items()])
        joined=trials.merge(extra,on='query_id',how='left',validate='many_to_one')
        # Metrics belong only to complete reports, even when a fragment happens
        # to share the same query. Preserve every original row and scope.
        joined.loc[joined.scope!='complete',list(next(iter(lookup.values())).keys())]=np.nan
        joined['reading']=reading;frames.append(joined)
    all_trials=pd.concat(frames,ignore_index=True);summary=[]
    for (reading,subject),group in all_trials.groupby(['reading','iSub'],sort=True):
        complete=group[group.scope=='complete']
        summary.append(dict(reading=reading,subject=int(subject),condition=int(group.condition.iloc[0]),
            total_reports=len(group),complete_reports=len(complete),
            single_predicate_reports=int((complete.predicates==1).sum()),
            whole_structural_reports=int(complete.whole_structural_support.sum()),
            whole_functional_reports=int((complete.functional_support_any_component>0).sum()),
            all_individually_covered_reports=int(complete.all_individually_covered.sum()),
            combination_limited_reports=int(complete.combination_limited.sum()),
            any_uncovered_selected_choice_reports=int(complete.any_uncovered_selected_choice.sum()),
            any_uncovered_all_choices_reports=int(complete.any_uncovered_all_choices.sum()),
            mean_component_coverage_ceiling=float(complete.max_fraction_in_one_component.mean())))
    summary=pd.DataFrame(summary)
    np.testing.assert_array_equal(summary.whole_structural_reports+summary.combination_limited_reports+
        summary.any_uncovered_selected_choice_reports,summary.complete_reports)
    for name,sha in inputs.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=sha:raise ValueError('Input changed during coverage audit')
    output.mkdir(parents=True)
    for name,frame in [('trials',all_trials),('queries',pd.DataFrame(query_rows)),('summary',summary),('catalogue',pd.DataFrame(catalogue_rows))]:
        frame.to_csv(output/(name+'.csv'),index=False)
    page=('<!doctype html><html lang="zh"><meta charset="utf-8"><title>口述表达范围诊断</title>'
        '<style>body{max-width:1400px;margin:30px auto;font:16px/1.6 sans-serif}td,th{padding:5px}</style>'
        '<h1>单项判断能否表达，与整句能否共同表达</h1><p>全部原报告及两种固定读法保留。'
        '下表仅描述目录的表达范围，未读取模型状态或重新编码。组合受限表示各项判断分别有显式支持，'
        '但没有一个局部分量同时支持整组判断。所选类别下缺少单项支持另行列出；跨类别检查仅用于定位标签限制。'
        '分量覆盖上限是整个目录中能同时支持的最多判断比例，不能作为实际模型成绩。</p>'
        +summary.to_html(index=False,float_format=lambda x:f'{x:.4f}')
        +'<p>当前完整报告评分要求同一依据同时支持所有判断。二分类分量都只有一个显式谓词，'
        '因此多谓词报告的完整显式支持必为零。报告内容角色与目录表达能力仍需结合原话核对。</p>'
        '<p><a href="trials.csv">全部原报告与诊断</a> · <a href="queries.csv">去重查询</a> · '
        '<a href="catalogue.csv">分量谓词数量</a></p></html>')
    (output/'index.html').write_text(page)
    shutil.copy2(__file__,output/'audit_source.py')
    result=dict(status='complete',numeric_checks=checks,readings=2,subjects=summary.subject.nunique(),
        original_reports=len(expected_identity),report_rows=len(all_trials),complete_report_rows=int((all_trials.scope=='complete').sum()),
        inputs_sha256=inputs,source_sha256={str(Path(__file__).resolve()):bind(Path(__file__))},
        data_use='Frozen oral tokens and catalogue only; no model weights, feedback, current stimulus or parameter estimation.',
        limitation='Structural predicate coverage; no inference of core versus supplementary utterance roles.',
        artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()})
    (output/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--absolute-audit',type=Path,required=True)
    parser.add_argument('--relative-audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=run(args.absolute_audit,args.relative_audit,args.output)
    print({k:v for k,v in result.items() if k not in ['inputs_sha256','source_sha256','artifacts']})

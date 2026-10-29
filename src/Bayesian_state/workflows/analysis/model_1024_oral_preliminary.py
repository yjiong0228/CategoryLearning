"""Make a provisional reason-content comparison from saved individual states.

Run from the repository root with --config CONFIG --output NEW_DIRECTORY.
Raw transcripts, encodings, parameters and particle probabilities stay intact.
This reads existing arrays; it launches no particle filters or simulations.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault('MPLCONFIGDIR', '/tmp/model1024_oral_mpl')
import numpy as np
import pandas as pd
import scipy
import yaml

from src.oral_coding import Recording_Processor_Center
from ...evaluation.oral.partial_reason_alignment import (
    FEATURE_METRICS, PREDICATE_METRICS, component_geometry, feature_coverage,
    predicate_coverage, stimulus_choice_reference, unresolved_reason_role)
from .model_0923_decision_basis_alignment import align_identities, make_labels
from .model_0923_oral_constraint_audit import catalogue
from .model_1024_individual_oral_alignment import subject_features, constraint_text, SCOPE_LABELS

METRICS = (*FEATURE_METRICS, 'whole_rule_reason_feature_coverage',
           'whole_rule_basis_feature_coverage', 'whole_rule_exact_feature_set',
           *PREDICATE_METRICS, 'whole_encoded_conjunction_support')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(headers: list[str], rows: list[list]) -> str:
    return '\n'.join(['| '+' | '.join(headers)+' |', '|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(map(str, row))+' |' for row in rows])


def review_html(frame: pd.DataFrame, output: Path) -> None:
    columns = [('subject', '被试'), ('trial', '有效试次'), ('iSession', '场次'), ('iTrial', '原试次'),
        ('text', '原文：本题选择理由'), ('choice', '所选类别'), ('stimulus_description', '当前刺激：归一化长度'),
        ('reason_features', '理由涉及的部位'), ('scope', '现有语义编码'),
        ('decoded_constraints', '暂定数值解释'), ('unparsed', '尚未解析文字'),
        ('representative_basis', 'P1下最可能的局部规则内容'),
        ('feature_range', '理由部位被覆盖：候选范围'), ('basis_range', '局部规则部位被说到：候选范围'),
        ('predicate_range', '已编码判断被支持：候选范围'), ('conflict_range', '已编码判断冲突：候选范围'),
        ('reference_features', '只看当前刺激与选项：部位覆盖'), ('review_flags', '复核提示')]
    rows=[]
    for row in frame.to_dict('records'):
        row['scope']=SCOPE_LABELS.get(row['scope'],row['scope'])
        cells=''.join('<td>'+html.escape('' if pd.isna(row.get(key,'')) else str(row.get(key,'')))+'</td>' for key,_ in columns)
        rows.append(f'<tr data-subject="{row["subject"]}" data-trial="{row["trial"]}">{cells}</tr>')
    content='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>口述理由初步对照</title>
<style>body{font:14px sans-serif;margin:20px;color:#202020}table{border-collapse:collapse}th,td{padding:8px;border:1px solid #ddd;vertical-align:top;min-width:75px}th{position:sticky;top:0;background:#e8f0f5}td:nth-child(5),td:nth-child(8),td:nth-child(10){min-width:260px}tr:nth-child(even){background:#fafafa}select,input{font:inherit;padding:6px}</style>
<h1>口述理由与局部规则内容：初步对照</h1><p>默认显示S205。所有记录保留，原文尚待人工复核。概率由行为推断，口述没有改变参数或状态权重。
表内使用本题选择后、反馈前的过滤结果；参数使用完整行为记录估计。候选范围没有置信区间含义。
理由部位覆盖与局部规则部位覆盖采用不同分母，两个百分比需合看；关系的长短方向另看判断支持与冲突。
P1仅为固定展示候选，前三项规则内容按原概率列出。S129四组均为诊断参数，尚未确立。
解析片段的判断分数仅适用于已解析部分。数字属于内容对照，尚不代表因果理由或口述生成概率。</p>
<p><a href="report.md">阅读方法与结果</a> · <a href="trial_review.csv">下载全部逐条表格</a></p>
<label>被试 <select id="sid"><option value="">全部</option>'''
    content+=''.join(f'<option value="{sid}"'+(' selected' if sid==205 else '')+f'>S{sid}</option>' for sid in sorted(frame.subject.unique()))
    content+='''</select></label> <label>试次从 <input id="lo" type="number" min="1" value="1" style="width:70px"></label>
<label>到 <input id="hi" type="number" value="2000" style="width:70px"></label>
<label>搜索 <input id="q"></label><span id="count"></span><table><thead><tr>'''
    content+=''.join('<th>'+label+'</th>' for _,label in columns)+'</tr></thead><tbody>'+''.join(rows)+'''</tbody></table>
<script>const rows=[...document.querySelectorAll('tbody tr')];function filter(){const sid=document.getElementById('sid').value,lo=+document.getElementById('lo').value,hi=+document.getElementById('hi').value,q=document.getElementById('q').value.toLowerCase();let n=0;for(const r of rows){let ok=(!sid||r.dataset.subject===sid)&&+r.dataset.trial>=lo&&+r.dataset.trial<=hi&&r.textContent.toLowerCase().includes(q);r.hidden=!ok;n+=ok;}document.getElementById('count').textContent='显示 '+n+' 条';}for(const id of ['sid','lo','hi','q'])document.getElementById(id).oninput=filter;filter();</script></html>'''
    (output/'review.html').write_text(content)


def run(config: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new output directory')
    started=time.time(); cpu=time.process_time(); inputs={}
    def register(path: Path) -> Path:
        inputs[str(path)]=digest(path);return path
    def read(path: Path):
        return json.loads(register(path).read_text())
    policy=yaml.safe_load(register(config).read_text())
    if (policy['primary_mode']!='filtered' or policy['secondary_mode']!='full'
            or policy['new_particle_filters']!=0 or policy['parameter_selection_changes']
            or policy['budget']['workers']!=1 or policy['budget']['numeric_threads']!=1):
        raise ValueError('This entry point supports only the registered single-process saved-state analysis')
    output.mkdir(parents=True)
    (output/'registration.json').write_text(json.dumps(dict(policy=policy,started_unix=started,
        user_authorization='Initial provisional oral analysis requested; reason-for-current-choice instruction confirmed',
        scope='All six existing subjects, 13 retained and four diagnostic candidates; no new filtering',
        semantic_role='Content coverage only; local geometry is not a causal attention state'),ensure_ascii=False,indent=2))
    source=Path(policy['state_source']); analysis=Path(policy['state_analysis'])
    selected=[s for s in read(analysis/'selected.json') if s['subject'] in policy['subjects']]
    protocol=read(source/'protocol.json')
    cases={c['id']:c for c in protocol['cases']};numerical=pd.read_csv(register(analysis/'numerical_checks.csv'))
    old=Path(policy['previous_oral']);matchfile=old.parent/'oral_midpoint/component_matches.csv'
    matches=pd.read_csv(register(matchfile),usecols=['query_id','rule','component','structural','functional'])
    diagnostics=read(Path(protocol['policy']['precision_summary'])/'diagnostics.json')
    diagnostics+=read(source/'summaries/initial_completed/diagnostics.json')
    scores=[];reviews=[];summaries=[];resolution=[];sources=[]
    for sid in policy['subjects']:
        own=[s for s in selected if s['subject']==sid];case=cases[own[0]['case']]
        previous=pd.read_csv(register(old/f'S{sid}_review.csv'))
        frame,_=align_identities(previous,case['template'],case['trials'])
        raw=pd.read_csv(register(Path(policy['raw_directory'])/f'Task2_{sid}_rec.csv'))
        joined=frame[['iSession','iTrial','text']].merge(raw,on=['iSession','iTrial'],how='outer',validate='one_to_one',suffixes=('_old',''),indicator=True)
        if not joined['_merge'].eq('both').all() or not joined.text.fillna('').eq(joined.text_old.fillna('')).all():
            raise ValueError('Raw transcript changed: refresh independent content coding before scoring')
        frame['text']=frame.text.fillna('');features=subject_features(frame)
        mask=Recording_Processor_Center(parts=features).process_use(frame).filter(like='oraluse').to_numpy(bool)
        role=frame.text.apply(unresolved_reason_role).to_numpy()
        feature_valid=mask.any(axis=1)&~role
        tokens=[tuple(json.loads(value)) for value in frame.tokens]
        predicate_valid=np.array([bool(t) and scope in ('complete','parsed_fragment') and not bad
            for t,scope,bad in zip(tokens,frame.scope,role)])
        numeric_complete=np.array([ok and scope=='complete' and all(t.startswith('facet:') for t in ts)
            for ok,scope,ts in zip(predicate_valid,frame.scope,tokens)])
        frame['reason_features']=['、'.join(np.array(features)[m]) for m in mask]
        frame['reason_feature_count']=mask.sum(1);frame['feature_eligible']=feature_valid
        frame['predicate_primary_eligible']=numeric_complete;frame['reason_role_review']=role
        frame['semantic_conventions']=frame['flags'].fillna('')
        frame['review_flags']=[('理由角色需复核；' if bad else '')+('含参照或容差约定；' if 'convention' in flags or 'uncertain' in flags else '')+
            ('部分文字未解析；' if scope!='complete' else '') for bad,flags,scope in zip(role,frame['semantic_conventions'],frame.scope)]
        folder=Path(case['template']['folder']);parts=[]
        register(folder/'prepared.json')
        for name in ('training.npz','sealed_test.npz'):
            with np.load(register(folder/name)) as z:parts.append(z['stimulus'])
        x=np.concatenate(parts);choice=frame.choice.to_numpy(int);n=len(frame)
        frame['stimulus_description']=['；'.join(f'{name}={value:g}' for name,value in zip(features,row)) for row in x]
        space=catalogue(2 if case['condition']==1 else 4)
        with np.load(register(Path(own[0]['file']))) as z:counts=z['component_counts'];h,m=z['filtered_component'].shape[1:]
        np.testing.assert_array_equal(counts,[[len(c.components) for c in rule.categories] for rule in space])
        labels=make_labels(frame,matches,h,m)[...,1]
        values=np.zeros((n,h,m,len(METRICS)));reference=np.full((n,h,m),np.nan)
        uniform=np.zeros_like(reference);reference_available=np.zeros(n,bool);definitions={};cache={}
        for category in np.unique(choice):
            entries,local,whole=component_geometry(space,int(category));idx=np.flatnonzero(choice==category)
            reference[idx],reference_available[idx]=stimulus_choice_reference(x[idx],entries,h,m)
            for hh,mm,A,b in entries:
                uniform[idx,hh,mm]=1/h/counts[hh,category-1]
                definitions[int(category),hh,mm]=constraint_text(A,b,features)
            for t in idx:
                if time.time()-started>policy['budget']['wall_seconds']:
                    raise TimeoutError('Read-only preliminary analysis reached its registered wall budget')
                key=(int(category),tuple(mask[t]),tokens[t],bool(feature_valid[t]),bool(predicate_valid[t]))
                if key not in cache:
                    v=np.zeros((h,m,len(METRICS)))
                    if feature_valid[t]:
                        f=feature_coverage(mask[t],local);w=feature_coverage(mask[t],whole)
                    for k,(hh,mm,A,b) in enumerate(entries):
                        if feature_valid[t]:v[hh,mm,:6]=np.r_[f[k],w[k]]
                        if predicate_valid[t]:v[hh,mm,6:10]=predicate_coverage(tokens[t],A,b)
                    cache[key]=v
                values[t]=cache[key]
                values[t,...,10]=(labels[t]==0).astype(float)
        baseline={}
        for name,mass in [('catalogue',uniform),('stimulus_choice',reference)]:
            baseline[name]=np.einsum('thm,thmk->tk',np.nan_to_num(mass),values,optimize=True)
            if name=='stimulus_choice':baseline[name][~reference_available]=np.nan
        metadata_columns=['trial','iSession','iTrial','text','choice','stimulus_description','scope','decoded_constraints','unparsed',
            'reason_features','reason_feature_count','feature_eligible','predicate_primary_eligible','review_flags','semantic_conventions',
            'manual_corrected_text','manual_intended_meaning','manual_notes']
        review=frame[metadata_columns].copy();review['subject']=sid
        for s in own:
            file=register(Path(s['file']))
            if digest(file)!=s['sha256']:raise ValueError('Selected state archive changed')
            with np.load(file) as z:arrays={mode:z[mode+'_component'] for mode in ('filtered','full')}
            lower=None;lower_arrays=None
            if s['parameter_status']=='retained':
                candidates=[r for r in diagnostics if r['case']==s['case'] and r['particles']==s['comparison_particles'] and not r['confirmation']]
                lower=max(candidates,key=lambda r:r['systems'])
                lowfile=register(Path(lower['pooled_file']))
                if digest(lowfile)!=lower['pooled_sha256']:raise ValueError('Lower-resolution state changed')
                with np.load(lowfile) as z:lower_arrays={mode:z[mode+'_component'] for mode in ('filtered','full')}
            sources.append(dict(case=s['case'],file=str(file),sha256=digest(file),particles=s['particles'],systems=s['systems'],
                lower_file=lower['pooled_file'] if lower else None,lower_particles=lower['particles'] if lower else None,
                lower_systems=lower['systems'] if lower else None))
            for mode,mass in arrays.items():
                if not np.isfinite(mass).all() or (mass<0).any():raise ValueError('Invalid component probability')
                np.testing.assert_allclose(mass.sum((1,2)),1,atol=2e-10,rtol=0)
                current=np.einsum('thm,thmk->tk',mass,values,optimize=True)
                current[~feature_valid,:6]=np.nan;current[~predicate_valid,6:]=np.nan
                same=numerical[numerical.case.eq(s['case'])&numerical['mode'].eq(mode)&numerical.measure.eq('execution')].iloc[0]
                for k,metric in enumerate(METRICS):
                    eligible=feature_valid if k<6 else numeric_complete
                    if lower_arrays is not None:
                        low=np.einsum('thm,thm->t',lower_arrays[mode],values[...,k],optimize=True)
                        delta=np.abs(current[:,k]-low)
                        resolution.append(dict(subject=sid,number=s['number'],mode=mode,metric=metric,
                            reports=int(eligible.sum()),mean_difference=float(np.mean(delta[eligible])) if eligible.any() else None,
                            q95_difference=float(np.quantile(delta[eligible],.95)) if eligible.any() else None,
                            independent_half_check_for_new_metric=False,whole_execution_distribution_checked=bool(same.usable)))
                    groups=[('whole',np.ones(n,bool)),('first64',np.arange(n)<64),('last64',np.arange(n)>=n-64)]
                    groups += [(f'block_{i//policy["block_trials"]+1}',(np.arange(n)>=i)&(np.arange(n)<i+policy['block_trials'])) for i in range(0,n,policy['block_trials'])]
                    for window,in_window in groups:
                        use=eligible&in_window;common=use&reference_available
                        mean=lambda a,mask:float(a[mask].mean()) if mask.any() else None
                        summaries.append(dict(subject=sid,case=s['case'],number=s['number'],mode=mode,metric=metric,window=window,
                            parameter_status=s['parameter_status'],window_trials=int(in_window.sum()),reports=int(use.sum()),
                            mean=mean(current[:,k],use),catalogue_mean=mean(baseline['catalogue'][:,k],use),
                            reference_reports=int(common.sum()),model_on_reference=mean(current[:,k],common),
                            stimulus_choice_mean=mean(baseline['stimulus_choice'][:,k],common),
                            gain_over_stimulus_choice=mean(current[:,k]-baseline['stimulus_choice'][:,k],common)))
                for t in range(n):
                    scores.append(dict(subject=sid,case=s['case'],number=s['number'],mode=mode,trial=t+1,
                        feature_eligible=bool(feature_valid[t]),predicate_primary_eligible=bool(numeric_complete[t]),
                        parameter_status=s['parameter_status'],**dict(zip(METRICS,current[t]))))
                if s['number']==1 and mode=='filtered':
                    descriptions=[]
                    for t in range(n):
                        ranked=np.argsort(mass[t].ravel())[-3:][::-1]
                        descriptions.append('；'.join(f'R{hh}：{definitions[int(choice[t]),hh,mm]}（{mass[t,hh,mm]:.1%}）'
                            for hh,mm in [np.unravel_index(i,(h,m)) for i in ranked] if mass[t,hh,mm]>0))
                    review['representative_basis']=descriptions
            print(f'S{sid} P{s["number"]}: saved-state oral projection complete',flush=True)
        own_scores=pd.DataFrame(scores);own_scores=own_scores[own_scores.subject.eq(sid)&own_scores['mode'].eq('filtered')]
        for metric,column in [('reason_feature_coverage','feature_range'),('basis_feature_coverage','basis_range'),
                              ('predicate_supported','predicate_range'),('predicate_conflicting','conflict_range')]:
            ranges=own_scores.groupby('trial')[metric].agg(['min','max'])
            review[column]=[f'{a:.1%}–{b:.1%}' if np.isfinite(a) else '未评分' for a,b in ranges.to_numpy()]
        review['reference_features']=[f'{a:.1%}' if ok and available else '未评分' for a,ok,available in
            zip(baseline['stimulus_choice'][:,0],feature_valid,reference_available)]
        reviews.append(review)
        np.savez_compressed(output/f'S{sid}_reference_scores.npz',**baseline,available=reference_available)
    all_reviews=pd.concat(reviews,ignore_index=True);summary=pd.DataFrame(summaries)
    all_reviews.to_csv(output/'trial_review.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(scores).to_csv(output/'candidate_trial_scores.csv.gz',index=False,compression='gzip')
    summary.to_csv(output/'summary.csv',index=False)
    pd.DataFrame(resolution).to_csv(output/'particle_sensitivity.csv',index=False)
    (output/'state_sources.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2))
    review_html(all_reviews,output)
    if any(digest(Path(k))!=v for k,v in inputs.items()):raise ValueError('Input changed during analysis')
    source_files=[Path(__file__),Path('src/Bayesian_state/evaluation/oral/partial_reason_alignment.py'),
        Path('src/Bayesian_state/evaluation/oral/constraint_alignment.py'),Path('src/oral_coding.py')]
    manifest=dict(status='provisional_content_analysis_complete',rows=len(all_reviews),candidates=len(selected),
        raw_changed=False,encoder_changed=False,new_filters=0,oral_used_for_state_inference=False,
        instruction=policy['report_instruction'],primary_condition='after current choice before feedback, filtered; parameters fit on complete record',
        score_scope='descriptive geometric content; no report likelihood, causal-reason claim or formal validation',
        independent_new_metric_half_bank_check=False,inputs=inputs,sources={str(p):digest(p) for p in source_files},
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__),
        wall_seconds=time.time()-started,cpu_seconds=time.process_time()-cpu,
        artifacts={p.name:digest(p) for p in output.iterdir() if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps(dict(rows=len(all_reviews),candidates=len(selected),seconds=time.time()-started),ensure_ascii=False))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/shared/model1024_oral_preliminary.yaml'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.config,args.output)


if __name__=='__main__':
    main()

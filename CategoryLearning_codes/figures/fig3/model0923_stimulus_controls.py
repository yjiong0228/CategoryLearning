"""Show audited dynamic and prefix-boundary-frozen oral support; PNG draft.

Use --output NEW_DIRECTORY. All four development participants, both readings,
and both proposal views for 208 are retained. These are fixed-parameter
diagnostics of choice-unweighted marginal rule contributions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
CASES=[(107,'SEARCH'),(207,'SEARCH'),(208,'POOL'),(208,'SEARCH'),(306,'SEARCH')]


def run(output: Path) -> None:
    if output.exists():raise FileExistsError('Use a new stimulus-control figure directory')
    result=ROOT/'results/model_0923';inputs={};frames=[];contrasts=[];checks=0
    def bind(path,expected=None):
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        if expected is not None and sha!=expected:raise ValueError('Changed audited figure source')
        inputs[str(path.resolve())]=sha
    audit_file=result/'oral_stimulus_controls_audit_20261019_v1/validation.json'
    bind(audit_file);audit=json.loads(audit_file.read_text())
    if audit['status']!='complete':raise ValueError('Require independent raw-state and kernel verification')
    for subject,proposal in CASES:
        for reading in ('midpoint','relative'):
            middle=proposal.lower()+'_' if subject==208 else ''
            version=2 if (subject,proposal,reading)==(208,'POOL','midpoint') else 1
            folder=result/f'stimulus_control_s{subject}_{middle}{reading}_20261019_v{version}'
            bind(folder/'manifest.json',audit['input_sha256'][str((folder/'manifest.json').resolve())])
            manifest=json.loads((folder/'manifest.json').read_text())
            for name in ('scores.csv','trials.csv','contrasts.csv'):bind(folder/name,manifest['artifacts'][name])
            scores=pd.read_csv(folder/'scores.csv');trials=pd.read_csv(folder/'trials.csv');difference=pd.read_csv(folder/'contrasts.csv')
            for row in scores.itertuples():
                data=trials[trials.arm.eq(row.arm)&trials.scope.eq('complete')]
                start=manifest['training_trials'] if row.window=='test' else max(manifest['training_trials'],int(trials.state_trial.max())+1-64)
                data=data[data.state_trial>=start]
                if len(data)!=row.reports:raise ValueError('Figure changed the report denominator')
                for label in ('support','conflict','undetermined'):
                    np.testing.assert_allclose(data[row.metric+'_'+row.domain+'_'+label].mean(),getattr(row,label),rtol=0,atol=1e-12)
                    checks+=1
            for row in difference.itertuples():
                selected=scores[scores.window.eq(row.window)&scores.domain.eq(row.domain)&scores.metric.eq(row.metric)].set_index('arm')
                for label in ('support','conflict','undetermined'):
                    np.testing.assert_allclose(selected.loc[row.arm,label]-selected.loc[row.reference_arm,label],getattr(row,label+'_difference'),rtol=0,atol=1e-12)
                    checks+=1
            frames.append(scores.assign(reading=reading,proposal=proposal));contrasts.append(difference.assign(reading=reading,proposal=proposal))
    scores=pd.concat(frames,ignore_index=True);differences=pd.concat(contrasts,ignore_index=True)
    output.mkdir(parents=True)
    contract=dict(conclusion='ONLINE has higher later oral support in 207, 208 and 306; much of that support exists at the prefix boundary, and 208 loses less support thereafter.',
        archetype='quantitative grid',backend='python',dimensions_mm=[183,145],export='PNG only, 300 dpi discussion draft',
        mapping={'x':'support, fraction of choice-unweighted whole-report basis mass','y':'participant and proposal view',
                 'color':'BASE or ONLINE','open_circle':'boundary_frozen','filled_diamond':'common_kernel_dynamic','bars':'support_mcse'},
        panels='Identical support comparison under absolute and relative readings; identical full probability scale.',
        sample='Four development participants; 208 POOL and SEARCH are sensitivity views of one person. All complete later reports retained.',
        uncertainty='Delete-one-complete-filter MCSE, four seeds at 128 particles. This is simulation error, not participant confidence intervals.',
        data_use='Original behavioral parameters; current choice only queries category; common independent perception kernel; no oral fitting.',
        exclusions='Display focuses on later structural whole-report support and boundary freezing. All seven controls, both domains and later windows remain in source tables.',
        limitations='107 whole-report support is constrained by catalogue expression coverage. Frozen rule contributions do not constitute a full autonomous state simulator.',
        adaptation='Style-only inheritance of local font and palette conventions; paired support plot built for this contrast.')
    (output/'figure_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    scores.to_csv(output/'summary_source.csv',index=False);differences.to_csv(output/'difference_source.csv',index=False)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':7,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False,
        'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7.204724,5.708661))
    fig.subplots_adjust(left=.14,right=.97,bottom=.14,top=.79,wspace=.43)
    colors={'BASE':'#858B92','ONLINE':'#407B94'}
    for col,reading in enumerate(('midpoint','relative')):
        ax=axes[col]
        for i,(subject,proposal) in enumerate(CASES):
            selected=scores[scores.subject.eq(subject)&scores.proposal.eq(proposal)&scores.reading.eq(reading)&
                scores.window.eq('test')&scores.domain.eq('structural')].set_index(['arm','metric'])
            for arm,offset in [('BASE',-.22),('ONLINE',.22)]:
                frozen=selected.loc[arm,'boundary_frozen'];dynamic=selected.loc[arm,'common_kernel_dynamic'];y=i*1.5+offset
                ax.plot([frozen.support,dynamic.support],[y,y],color=colors[arm],lw=1.1,zorder=1)
                for row,marker,size,fill in [(frozen,'o',5.2,'white'),(dynamic,'D',3.6,colors[arm])]:
                    ax.errorbar(row.support,y,xerr=row.support_mcse,fmt=marker,markersize=size,markerfacecolor=fill,
                        markeredgecolor=colors[arm],color=colors[arm],capsize=1.8,elinewidth=.65,zorder=2)
            ax.axhline(i*1.5+.7,color='#E9EBED',lw=.6,zorder=0)
        ax.set_yticks(np.arange(len(CASES))*1.5,[f'{s}\n{p}' for s,p in CASES],fontsize=6.5)
        ax.set_ylim(len(CASES)*1.5-.6,-.65);ax.set_xlim(-.025,1.025);ax.set_xticks([0,.25,.5,.75,1.])
        ax.set_xlabel('Later whole-report support');ax.tick_params(axis='y',length=0)
        ax.set_title(('Absolute reading','Relative reading')[col],fontsize=8,pad=10)
        ax.text(-.33,1.06,'ab'[col],transform=ax.transAxes,fontweight='bold',fontsize=9)
    fig.legend([Line2D([0],[0],color=colors[a],lw=2) for a in ('BASE','ONLINE')],['BASE','ONLINE'],
        loc='upper center',bbox_to_anchor=(.56,.97),ncol=2,fontsize=7)
    fig.legend([Line2D([0],[0],marker='o',linestyle='none',markerfacecolor='white',markeredgecolor='#3B4147',markersize=5),
                Line2D([0],[0],marker='D',linestyle='none',color='#3B4147',markersize=3.6)],
        ['Prefix-boundary frozen','Dynamic'],loc='upper center',bbox_to_anchor=(.56,.915),ncol=2,fontsize=7)
    fig.text(.14,.065,'Later complete reports (top to bottom): 35, 264, 119, 119, 255',fontsize=6)
    fig.text(.14,.035,'Four development participants; 208 has two proposal views. Bars: filter MCSE (4 seeds).',fontsize=6)
    fig.savefig(output/'stimulus_controls.png',dpi=300,facecolor='white');plt.close(fig)
    note=('空心圆固定前段末端的规则作答贡献，实心菱形使用每题动态贡献；两者都读取当前刺激，'
        '共用独立知觉采样。误差线为四次完整滤波的模拟标准误，每次128粒子。'
        '两种读法使用同一批后段完整报告，208的两行属于同一被试的提议敏感性。'
        '当前选择只指定所选类别，本题选择似然未用于重加权。行为参数沿用既有结果，口述未用于估参。')
    interpretation=('208的BASE动态支持低于末端冻结，ONLINE下降较小。207、306的ONLINE在前段末端已取得较高支持，'
        '后段动态与冻结接近；较高绝对支持与继续学习的增益需要分开解释。'
        '107整句支持接近0，保留其目录表达范围限制；逐判断内容覆盖见已有补充结果。'
        '本批后段的公共知觉核动态评分与原粒子读出一致，全程读出偏差保存在原对照结果中。')
    display=scores[scores.domain.eq('structural')&scores.window.eq('test')&scores.metric.eq('dynamic_minus_boundary')]
    (output/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>规则贡献冻结与动态对照</title>'
        '<style>body{max-width:1100px;margin:30px auto;font:16px/1.6 sans-serif}img{max-width:100%}</style>'
        '<h1>保留当前刺激后的冻结与动态规则贡献</h1><p>'+note+'</p><img src="stimulus_controls.png"><p>'+interpretation+'</p>'
        +display[['subject','proposal','reading','arm','reports','support','support_mcse']].to_html(index=False,float_format=lambda v:f'{v:.6f}')
        +'<p><a href="summary_source.csv">全部控制分数</a> · <a href="difference_source.csv">全部方案差值</a></p></html>')
    shutil.copy2(__file__,output/'figure_source.py')
    for path,h in inputs.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=h:raise ValueError('Figure input changed')
    (output/'validation.json').write_text(json.dumps(dict(status='complete',source_checks=checks,input_sha256=inputs,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        qa_exceptions=['PNG-only repository convention','300 dpi discussion draft; no submission claim']),indent=2)+'\n')
    print({'status':'complete','source_checks':checks})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
